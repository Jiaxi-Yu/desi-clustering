#!/usr/bin/env python
"""
Script to create and spawn desipipe tasks to compute clustering measurements on box mocks.
To create and spawn the tasks on NERSC, use the following commands:
```bash
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
python desipipe_box_mocks.py         # create the list of tasks
desipipe tasks -q box_mocks          # check the list of tasks
desipipe spawn -q box_mocks --spawn  # spawn the jobs
desipipe queues -q box_mocks         # check the queue
```
"""
import os
from pathlib import Path
import functools

import numpy as np
from desipipe import Queue, Environment, TaskManager, spawn, setup_logging

from clustering_statistics import tools, box_tools

setup_logging()

queue = Queue('box_mocks')
queue.clear(kill=False)

output, error = 'slurm_outputs/box_mocks/slurm-%j.out', 'slurm_outputs/box_mocks/slurm-%j.err'
kwargs = {}
# environ = Environment('nersc-cosmodesi')
environ = Environment('nersc-cosmodesi', command='export PYTHONPATH=$HOME/LSScode/dr2-clustering-analysis/:$PYTHONPATH')
tm = TaskManager(queue=queue, environ=environ)
tm = tm.clone(scheduler=dict(max_workers=10), provider=dict(provider='nersc', time='02:00:00',
                            mpiprocs_per_worker=4, output=output, error=error, stop_after=1, constraint='gpu'))
tm80 = tm.clone(provider=dict(provider='nersc', time='02:00:00',
                            mpiprocs_per_worker=4, output=output, error=error, stop_after=1, constraint='gpu&hbm80g'))

def run_stats(tracer='LRG', version='ezmock', analysis='box', project='', zsnaps=None, los='z', imocks=[0], stats_dir=Path(os.getenv('SCRATCH')) / 'measurements', stats=['mesh2_spectrum'], **kwargs):
    # Everything inside this function will be executed on the compute nodes;
    # This function must be self-contained; and cannot rely on imports from the outer scope.
    import os
    import sys
    import functools
    from pathlib import Path
    import jax
    from jax import config
    config.update('jax_enable_x64', True)
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'
    try: jax.distributed.initialize()
    except RuntimeError: print('Distributed environment already initialized')
    else: print('Initializing distributed environment')
    from clustering_statistics import box_tools, setup_logging, compute_box_stats_from_options, fill_box_fiducial_options
    setup_logging()


    cache = {}
    if zsnaps is None:
        raise ValueError('Please provide znaps.')
    get_box_stats_fn = functools.partial(box_tools.get_box_stats_fn, stats_dir=stats_dir, project=project)
    for imock in imocks:
        for zsnap in zsnaps:
            options = dict(catalog=dict(version=version, tracer=tracer, zsnap=zsnap, los=los, imock=imock))
            options = fill_box_fiducial_options(options)
            compute_box_stats_from_options(stats, get_box_stats_fn=get_box_stats_fn, cache=cache, **options)


if __name__ == '__main__':

    stats = []
    # version  = 'ezmock'
    version = 'abacus-hf-v2'
    check_for_existing_measurements = False
    
    # run on interactive node
    mode = 'interactive'
    imocks2run = np.arange(25)
    # stats_dir  = Path(os.getenv('SCRATCH')) / 'cai-dr2-benchmarks' 
    
    # to run job
    # mode = 'slurm'
    # imocks2run = 1 + np.arange(1000)
    stats_dir  = tools.base_stats_dir

    # run 
    stats    = ['mesh2_spectrum', 'mesh3_spectrum']
    analysis = 'box'
    project  = f'{analysis}'
    # tracers  = ['ELG','QSO','LRG']
    tracers  = ['QSO_lorentzian']
    los = 'z'
    max_mocks_per_batch = 1
    
    for tracer in tracers:
        if tracer in ['QSO','ELG']:
            # only use zsnaps 1.400 and 0.950 for QSO and ELG respectively.
            zsnaps = box_tools.propose_box_fiducial('zsnaps', tracer, version=version)[:1]
        elif tracer in ['QSO_lorentzian']:
            zsnaps = box_tools.propose_box_fiducial('zsnaps', tracer, version=version)[:]
        elif tracer in ['LRG']:
            # only use zsnap 0.800 for LRG
            zsnaps = box_tools.propose_box_fiducial('zsnaps', tracer, version=version)[1:]
        elif tracer in ['BGS']:
            # only use zsnap 0.200 for BGS
            zsnaps = box_tools.propose_box_fiducial('zsnaps', tracer, version=version)
        cosmo  = box_tools.propose_box_fiducial('catalog', tracer, version=version)['cosmo']
        if check_for_existing_measurements:
            rerun = []
            for zsnap in zsnaps:
                if 'ezmock' in version:
                    # TODO: Fix catalog checks for EZmocks.
                    imocks = imocks2run
                else:
                    exists, missing = tools.checks_if_exists_and_readable(get_fn=functools.partial(box_tools.get_box_catalog_fn, tracer=tracer[0] if isinstance(tracer, (list, tuple)) else tracer, 
                                                                                                   version=version, zsnap=zsnap), test_if_readable=False, imock=imocks2run)[:2]
                    imocks = exists[1]['imock']
                for kind in stats:
                    stats_kws = dict(basis='sugiyama-diagonal', kind=kind, stats_dir=Path(str(stats_dir).replace('global','dvs_ro')),
                                     version=version, project=project, tracer=tracer, zsnap=zsnap,  cosmo=cosmo, los=los)
                    rexists, missing, unreadable = tools.checks_if_exists_and_readable(get_fn=functools.partial(box_tools.get_box_stats_fn, **stats_kws), test_if_readable=True, imock=imocks2run)
                    # print(missing)
                    rerun += [imock for imock in imocks if (imock in unreadable[1]['imock']) or (imock not in rexists[1]['imock'])]
            imocks = sorted(set(rerun))
        else:
            imocks = imocks2run
       
        def get_run_stats():
            _tm = tm
            return run_stats if mode == 'interactive' else _tm.python_app(run_stats)

        run_stats_kws = dict(tracer=tracer, stats_dir=stats_dir, project=project, version=version, stats=stats, analysis=analysis, zsnaps=zsnaps)
        if any('window' in stat or 'covariance' in stat for stat in stats):
            _imocks = [0]
            get_run_stats()(imocks=_imocks, **run_stats_kws)
        elif stats:
            batch_imocks = np.array_split(imocks, max(len(imocks) // max_mocks_per_batch, 1)) if len(imocks) else []
            for _imocks in batch_imocks:
                get_run_stats()(imocks=_imocks, **run_stats_kws)