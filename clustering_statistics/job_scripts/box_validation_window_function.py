#!/usr/bin/env python
"""
Script to create and spawn desipipe tasks to compute clustering measurements on box mocks.
To create and spawn the tasks on NERSC, use the following commands:
```bash
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
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
environ = Environment('nersc-cosmodesi')
tm = TaskManager(queue=queue, environ=environ)
tm = tm.clone(scheduler=dict(max_workers=10), provider=dict(provider='nersc', time='02:00:00',
                            mpiprocs_per_worker=4, output=output, error=error, stop_after=1, constraint='gpu'))
tm80 = tm.clone(provider=dict(provider='nersc', time='02:00:00',
                            mpiprocs_per_worker=4, output=output, error=error, stop_after=1, constraint='gpu&hbm80g'))


def run_stats(tracer='LRG', hod='base', version='abacus-hf-v2', analysis='box', project='', zsnaps=None, los='z', imocks=[0], stats_dir=Path(os.getenv('SCRATCH')) / 'measurements', stats=['mesh2_spectrum'], **kwargs):
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
            options = dict(catalog=dict(version=version, tracer=tracer, zsnap=zsnap, hod=hod, los=los, imock=imock))
            #options['mesh3_spectrum'] = dict(basis='sugiyama', ells=[(0, 0, 0), (0, 2, 2), (1, 1, 0), (1, 1, 2), (2, 2, 0), (2, 2, 2)], mask_edges=['edge1[:, 1] <= nyq / 2.', 'edge2[:, 1] <= nyq / 2.'], buffer_size=31, mattrs={'meshsize': 400})
            options['mesh3_spectrum'] = dict(basis='scoccimarro', ells=[0, 2], mattrs={'meshsize': 320}, buffer_size=34, edges={'step': 0.01}, mask_edges=['(mid3 >= jnp.abs(mid1 - mid2)) & (mid3 <= jnp.abs(mid1 + mid2))', 'edge1[:, 1] <= nyq * 2. / 3.', 'edge2[:, 1] <= nyq * 2. / 3.', 'edge2[:, 1] <= nyq * 2. / 3.'])
            options = fill_box_fiducial_options(options)
            compute_box_stats_from_options(stats, get_box_stats_fn=get_box_stats_fn, cache=cache, **options)


if __name__ == '__main__':

    stats = []
    version  = 'abacus-hf-v2'

    # to run job
    mode = 'interactive'
    imocks = np.arange(25)
    stats_dir  = tools.base_stats_dir

    # run 
    stats    = ['mesh2_spectrum', 'mesh3_spectrum'][1:]
    analysis = 'full_shape'
    project  = f'{analysis}/box_window_function_validation'
    tracers  = ['LRG', 'ELG', 'QSO'][1:]
    los = 'z'
    max_mocks_per_batch = 50
    
    for tracer in tracers:
        zsnaps = box_tools.propose_box_fiducial('zsnaps', tracer, version=version)
        hod = {'LRG': 'base_B', 'ELG': 'base_conf_nfwexp', 'QSO': 'base'}[tracer]

        def get_run_stats():
            _tm = tm
            return run_stats if mode == 'interactive' else _tm.python_app(run_stats)

        run_stats_kws = dict(tracer=tracer, stats_dir=stats_dir, project=project, version=version, hod=hod, stats=stats, analysis=analysis, zsnaps=zsnaps)
        if any('window' in stat or 'covariance' in stat for stat in stats):
            _imocks = [0]
            get_run_stats()(imocks=_imocks, **run_stats_kws)
        elif stats:
            batch_imocks = np.array_split(imocks, max(len(imocks) // max_mocks_per_batch, 1)) if len(imocks) else []
            for _imocks in batch_imocks:
                get_run_stats()(imocks=_imocks, **run_stats_kws)