#!/usr/bin/env python
"""
Script to create and spawn desipipe tasks to validate the analytical covariance matrices.
To create and spawn the tasks on NERSC, use the following commands:
```bash
salloc -N 1 -C "gpu&hbm80g" -t 04:00:00 --gpus 4 --qos interactive --account desi_g
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
python validation_analytic_covariance.py
```
"""
import os
from pathlib import Path
import functools

import numpy as np
from desipipe import Queue, Environment, TaskManager, spawn, setup_logging

from clustering_statistics import tools

setup_logging()

# to run job
mode = 'interactive'
#mode = 'slurm'

if mode == 'slurm':
    queue = Queue('covariance')
    queue.clear(kill=False)
    
    output, error = 'slurm_outputs/abacus_mocks/slurm-%j.out', 'slurm_outputs/abacus_mocks/slurm-%j.err'
    kwargs = {}
    environ = Environment('nersc-cosmodesi', command=['module unload desi-clustering desilike'])
    tm = TaskManager(queue=queue, environ=environ)
    tm = tm.clone(scheduler=dict(max_workers=20), provider=dict(provider='nersc', time='03:00:00',
                                mpiprocs_per_worker=4, output=output, error=error, constraint='gpu'))
    tm80 = tm.clone(provider=dict(provider='nersc', time='03:00:00',
                                mpiprocs_per_worker=4, output=output, error=error, stop_after=1, constraint='gpu&hbm80g'))
    tmw = tm.clone(scheduler=dict(max_workers=1), provider=dict(provider='nersc', time='00:10:00',
                    mpiprocs_per_worker=2250, nodes_per_worker=25, output=output, error=error, stop_after=1, constraint='cpu'))
    tmw = tm.clone(provider=dict(provider='nersc', time='04:00:00',
                                mpiprocs_per_worker=4, output=output, error=error, stop_after=1, constraint='gpu&hbm80g'))


def get_stats_fn(*args, extra='', onthefly=None, method=None, **kwargs):
    from clustering_statistics import tools
    extra = [txt for txt in [extra, onthefly, method] if txt]
    return tools.get_stats_fn(*args, extra='_'.join(extra), **kwargs)


def run_stats(tracer='LRG', project='', version='abacus-hf-lc-dr2-v1.9dv', imocks=[0], stats_dir=Path(os.getenv('SCRATCH')) / 'measurements', stats=['mesh2_spectrum'], weight='default-FKP', analysis='full_shape', regions=['ALL'], ibatch=None, zranges=None, get_stats_fn=get_stats_fn, **kwargs):
    # Everything inside this function will be executed on the compute nodes;
    # This function must be self-contained; and cannot rely on imports from the outer scope.
    import os
    import sys
    import functools
    from pathlib import Path
    import jax
    from jax import config
    import numpy as np
    config.update('jax_enable_x64', True)
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'
    try: jax.distributed.initialize()
    except RuntimeError: print('Distributed environment already initialized')
    else: print('Initializing distributed environment')
    from clustering_statistics import tools, setup_logging, compute_stats_from_options, fill_fiducial_options, postprocess_stats_from_options
    setup_logging()

    cache = {}
    if zranges is None:
        raise ValueError('Please provide zranges.')
    for imock in imocks:
        for region in regions:
            mesh2_spectrum = {'cut': None, 'auw': None}
            mesh3_spectrum = {'auw': None}
            window_mesh3_spectrum = {'ibatch': ibatch} if isinstance(ibatch, tuple) else {'computed_batches': ibatch}
            particle2_correlation = {'split_randoms': (2., 10)} #, 'battrs': dict(s=np.linspace(0., 40., 41), mu=(np.linspace(-1., 1., 201), 'midpoint'))}
            particle3_correlation = {'split_randoms': (2., 10), 'battrs': dict(s=np.linspace(0., 20., 21), pole=(list(range(6)), 'firstpoint'))}
            window_mesh2_spectrum = {'cut': None, 'split_randoms': (20, 2 if 'ELG' in tracer else 4)}
            window_mesh3_spectrum = {'split_randoms': (20, 2 if 'ELG' in tracer else 4), 'computed_batches': None} #[None]}
            options = dict(catalog=dict(version=version, tracer=tracer, zrange=zranges, region=region, weight=weight, imock=imock), 
                           mesh2_spectrum=mesh2_spectrum, window_mesh2_spectrum=window_mesh2_spectrum,
                           mesh3_spectrum=mesh3_spectrum, window_mesh3_spectrum=window_mesh3_spectrum,
                           particle2_correlation=particle2_correlation,
                           particle3_correlation=particle3_correlation)
            options = fill_fiducial_options(options, analysis=analysis)
            
            for itracer in options['catalog']:
                options['catalog'][itracer]['nran'] = 1
                options['catalog'][itracer]['zranges'] = zranges  # override fiducial zranges
                options['catalog'][itracer]['FKP_P0'] = None

            _get_stats_fn = functools.partial(get_stats_fn, stats_dir=stats_dir, project=project, onthefly=onthefly)
            compute_stats_from_options(stats, analysis=analysis, get_stats_fn=_get_stats_fn, cache=cache, **options)


def postprocess_stats(tracer='LRG', analysis='full_shape', project='', version='abacus-hf-dr2-v2-altmtl', onthefly=None, imocks=[150], stats_dir=Path(os.getenv('SCRATCH')) / 'measurements', weight='default-FKP', postprocess=['combine_regions'], zranges=None, get_stats_fn=get_stats_fn, **kwargs):
    from clustering_statistics import postprocess_stats_from_options
    if zranges is None:
        zranges = tools.propose_fiducial('zranges', tracer, analysis=analysis)
    options = dict(catalog=dict(version=version, tracer=tracer, zrange=zranges, weight=weight, imock=imocks[0]), imocks=imocks, combine_regions={'stats': ['covariance_mesh2_spectrum', 'covariance_particle2_correlation', 'covariance_recon_mesh2_spectrum', 'covariance_recon_particle2_correlation']}, mesh2_spectrum={}, window_mesh2_spectrum={}, mesh3_spectrum={}, window_mesh3_spectrum={})
    stats_dir_kws = dict(stats_dir=stats_dir, project=project)
    _get_stats_fn = functools.partial(get_stats_fn, stats_dir=stats_dir, project=project, onthefly=onthefly) #, method='smooth_particle')
    postprocess_stats_from_options(postprocess, analysis=analysis, get_stats_fn=_get_stats_fn, **options)



if __name__ == '__main__':

    stats, postprocess = [], []
    version = 'abacus-hf-lc-dr2-v1.9'
    check_for_existing_measurements = False

    #imocks = np.arange(25)
    #imocks = np.arange(5, 25)
    #imocks = np.arange(5, 9)
    #imocks = np.arange(1)
    imocks = [0]
    stats_dir = tools.base_stats_dir

    # run fiducial full_shape
    tracers = ['QSO']

    # run BGS
    #version = 'abacus-2ndgen-dr2-altmtl'
    #tracers = ['BGS']

    #stats = ['mesh2_spectrum']
    stats = ['mesh2_spectrum', 'window_mesh2_spectrum', 'covariance_mesh2_spectrum'][1:]
    #stats = ['particle3_correlation']
    #stats = []
    postprocess = ['combine_regions'][:0]
    analysis = 'full_shape'
    project = f'{analysis}/lightcone_validation'
    weight = 'default-FKP'
    #weight = 'default'
    regions = ['ALL']
    max_mocks_per_batch = 5

    onthefly = None
    #onthefly = 'complete-nozfail'
    #onthefly = 'complete-renorm'
    #onthefly = 'complete-downsample'
    #onthefly = 'complete-samenz'
    #onthefly = 'complete-fixnz'
    #onthefly = 'complete'
    
    for tracer in tracers:
        tracer = tools.get_full_tracer(tracer, version=version)
        zranges = [(0.8, 2.1), (0.8, 1.6), (1.6, 2.1)]
        def get_run_stats():
            if mode == 'interactive':
                return run_stats
            _tm = tm80
            if tracer in ['LRG']:
                _tm = tm80
            if any('window_mesh3' in stat for stat in stats):
                _tm = tmw
            return _tm.python_app(run_stats)

        run_stats_kws = dict(tracer=tracer, stats_dir=stats_dir, project=project, version=version, analysis=analysis, onthefly=onthefly, zranges=zranges, regions=regions, weight=weight, postprocess=postprocess)
        if True:
            if any('window' in stat or 'covariance' in stat for stat in stats):
                _imocks = [0]
                nbatches = 1
                tasks = []
                for ibatch in range(nbatches):
                    task = get_run_stats()(imocks=_imocks, ibatch=(ibatch, nbatches), stats=stats, **run_stats_kws)
                    tasks.append(task)
                if nbatches > 1:
                    # Add dependence on other tasks
                    get_run_stats()(imocks=_imocks, ibatch=nbatches, tasks=tasks, stats=stats, **run_stats_kws)
            elif any('covariance' in stat for stat in stats):
                get_run_stats()(imocks=[0], stats=stats, **run_stats_kws)
            elif stats:
                batch_imocks = np.array_split(imocks, max((len(imocks) + max_mocks_per_batch - 1) // max_mocks_per_batch, 1)) if len(imocks) else []
                for _imocks in batch_imocks:
                    get_run_stats()(imocks=_imocks, stats=stats, **run_stats_kws)
        if postprocess:
            postprocess_stats(imocks=imocks, **run_stats_kws)