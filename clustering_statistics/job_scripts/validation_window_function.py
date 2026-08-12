#!/usr/bin/env python
"""
Script to create and spawn desipipe tasks to compute clustering measurements on abacus mocks.
To create and spawn the tasks on NERSC, use the following commands:
```bash
salloc -N 1 -C "gpu&hbm80g" -t 04:00:00 --gpus 4 --qos interactive --account desi_g
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
python desipipe_abacus_mocks.py          # create the list of tasks
desipipe tasks -q abacus_mocks           # check the list of tasks
desipipe spawn -q abacus_mocks --spawn   # spawn the jobs
desipipe queues -q abacus_mocks          # check the queue
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
    queue = Queue('window_function2')
    queue.clear(kill=False)

    output, error = 'slurm_outputs/abacus_mocks/slurm-%j.out', 'slurm_outputs/abacus_mocks/slurm-%j.err'
    kwargs = {}
    environ = Environment('nersc-cosmodesi', command=['module unload desi-clustering cucount jaxpower'])
    tm = TaskManager(queue=queue, environ=environ)
    tm = tm.clone(scheduler=dict(max_workers=20), provider=dict(provider='nersc', time='03:00:00',
                                mpiprocs_per_worker=4, output=output, error=error, constraint='gpu'))
    tm80 = tm.clone(provider=dict(provider='nersc', time='03:00:00',
                                mpiprocs_per_worker=4, output=output, error=error, stop_after=1, constraint='gpu&hbm80g'))
    tmw = tm.clone(scheduler=dict(max_workers=1), provider=dict(provider='nersc', time='00:10:00',
                    mpiprocs_per_worker=2250, nodes_per_worker=25, output=output, error=error, stop_after=1, constraint='cpu'))
    tmw = tm.clone(provider=dict(provider='nersc', time='04:00:00',
                                mpiprocs_per_worker=4, output=output, error=error, stop_after=1, constraint='gpu&hbm80g'))

def get_stats_fn(kind='mesh2_spectrum', extra='', onthefly=None, method=None, **kwargs):
    from clustering_statistics import tools
    #if 'window' not in kind: method = None
    extra = [txt for txt in [extra, onthefly, method] if txt]
    return tools.get_stats_fn(kind=kind, extra='_'.join(extra), **kwargs)


def run_stats(tracer='LRG', project='', version='abacus-hf-dr2-v2-altmtl', onthefly=None, imocks=[150], stats_dir=Path(os.getenv('SCRATCH')) / 'measurements', stats=['mesh2_spectrum'], weight='default-FKP', analysis='full_shape', regions=['NGC','SGC'], ibatch=None, reuse_raw=False, window_method='smooth_particle', zranges=None, get_stats_fn=get_stats_fn, **kwargs):
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
            mesh2_spectrum = {'cut': True if 'full_shape' in analysis else None,
                              'auw': None}
            window_mesh2_spectrum = {'cut': True if 'full_shape' in analysis else None}
            #mesh3_spectrum = {'auw': None}
            mesh3_spectrum = {'basis': 'scoccimarro', 'ells': [0, 2], 'buffer_size': 5 if 'LRG' in tracer else 0}
            particle2_correlation = {'split_randoms': (2., 10), 'battrs': dict(s=np.linspace(0., 40., 41), mu=(np.linspace(-1., 1., 201), 'midpoint'))}
            particle3_correlation = {'split_randoms': (2., 10), 'battrs': dict(s=np.linspace(0., 20., 21), pole=(list(range(6)), 'firstpoint'))}
            # 'smooth_particle' computes the window correlation from RRR particle counts, which is
            # what gives access to the HIGHER-ORDER window multipoles.
            method = window_method
            window_mesh2_spectrum = {'cut': True if 'full_shape' in analysis else None, 'method': method, 'split_randoms': (20, 2 if 'ELG' in tracer else 4)}
            window_mesh3_spectrum = {'method': method, 'split_randoms': (20, 2 if 'ELG' in tracer else 4)}
            if 'particle' in method: window_mesh3_spectrum |= {'ellwmax': 2}
            # window_mesh3_spectrum |= {'ellwmax': window_ellwmax}
            # window_mesh3_spectrum |= {'ellmax': window_ellmax}
            # ibatch splits the window MULTIPOLES over jobs: a tuple computes one batch of raw
            # correlations only, an int assembles the matrix from the batches computed before.
            window_mesh3_spectrum |= {'ibatch': ibatch} if isinstance(ibatch, tuple) else {'computed_batches': ibatch}
            # reuse_raw rebuilds the MATRIX ONLY, from the unbatched raw correlation already on disk
            if reuse_raw:
                window_mesh3_spectrum |= {'computed_batches': [None]}
            options = dict(catalog=dict(version=version, tracer=tracer, zrange=zranges, region=region, weight=weight, imock=imock),
                           mesh2_spectrum=mesh2_spectrum, window_mesh2_spectrum=window_mesh2_spectrum,
                           mesh3_spectrum=mesh3_spectrum, window_mesh3_spectrum=window_mesh3_spectrum,
                           particle2_correlation=particle2_correlation,
                           particle3_correlation=particle3_correlation)
            options = fill_fiducial_options(options, analysis=analysis)

            for itracer in options['catalog']:
                options['catalog'][itracer]['zranges'] = zranges  # override fiducial zranges
                options['catalog'][itracer]['expand'] = {'parent_randoms_fn': tools.get_catalog_fn(kind='parent_randoms', version='data-dr2-v2', tracer=itracer, nran=options['catalog'][itracer]['nran'])}
                if onthefly is not None and onthefly.startswith('complete'):
                    options['catalog'][itracer]['complete'] = {'with_completeness': 'nocomp' not in onthefly, 'with_tracer_cuts': True}
                elif onthefly == 'reshuffle':
                    options['catalog'][itracer]['reshuffle'] = {'merged_data_fn': tools.get_catalog_fn(kind='data', **(options['catalog'][itracer] | dict(region='ALL')))}

            _get_stats_fn = functools.partial(get_stats_fn, stats_dir=stats_dir, project=project, onthefly=onthefly, method=method)
            compute_stats_from_options(stats, analysis=analysis, get_stats_fn=_get_stats_fn, cache=cache, **options)


def postprocess_stats(tracer='LRG', analysis='full_shape', project='', version='abacus-hf-dr2-v2-altmtl', onthefly=None, imocks=[150], stats_dir=Path(os.getenv('SCRATCH')) / 'measurements', weight='default-FKP', postprocess=['combine_regions'], zranges=None, get_stats_fn=get_stats_fn, **kwargs):
    from clustering_statistics import postprocess_stats_from_options
    if zranges is None:
        zranges = tools.propose_fiducial('zranges', tracer, analysis=analysis)
    options = dict(catalog=dict(version=version, tracer=tracer, zrange=zranges, weight=weight, imock=imocks[0]), imocks=imocks, combine_regions={'stats': ['window_mesh2_spectrum', 'window_mesh3_spectrum']}, mesh2_spectrum={}, window_mesh2_spectrum={}, mesh3_spectrum={}, window_mesh3_spectrum={})
    stats_dir_kws = dict(stats_dir=stats_dir, project=project)
    _get_stats_fn = functools.partial(get_stats_fn, stats_dir=stats_dir, project=project, onthefly=onthefly, method='smooth_particle')
    postprocess_stats_from_options(postprocess, analysis=analysis, get_stats_fn=_get_stats_fn, **options)



if __name__ == '__main__':

    stats, postprocess = [], []
    version = 'abacus-hf-dr2-v2-altmtl'
    # version = 'abacus-2ndgen-dr2-complete'
    # version = 'abacus-2ndgen-dr2-altmtl'
    check_for_existing_measurements = False

    imocks = np.arange(25)
    #imocks = np.arange(5, 25)
    #imocks = np.arange(5, 9)
    #imocks = np.arange(1)
    #imocks = [0]
    stats_dir = tools.base_stats_dir

    # run fiducial full_shape
    tracers = ['LRG', 'ELG', 'QSO']
    #tracers = ['LRG']
    #tracers = ['QSO']

    # run BGS
    #version = 'abacus-2ndgen-dr2-altmtl'
    #tracers = ['BGS']

    # run data_splits for lensing group with full_shape setup
    #stats = ['mesh2_spectrum', 'mesh3_spectrum']
    #stats = ['window_mesh2_spectrum', 'window_mesh3_spectrum']
    #stats = ['mesh2_spectrum', 'window_mesh2_spectrum']
    stats = ['mesh3_spectrum', 'window_mesh3_spectrum']
    #stats = ['particle3_correlation']
    postprocess = ['combine_regions'][:0]
    analysis = 'full_shape'
    project = f'{analysis}/cutsky_window_function_validation'
    weight = 'default-FKP'
    #weight = 'default'
    regions = ['NGC', 'SGC']
    max_mocks_per_batch = 5

    #onthefly = None
    #onthefly = 'complete-nozfail'
    #onthefly = 'complete-renorm'
    #onthefly = 'complete-downsample'
    #onthefly = 'complete-samenz'
    #onthefly = 'complete-fixnz'
    onthefly = 'complete'

    for tracer in tracers:
        tracer = tools.get_full_tracer(tracer, version=version)
        if 'png' in analysis:
            # do not compute measurements for overlapping redshifts
            zranges = tools.propose_fiducial('zranges', tracer, analysis=analysis)[:1]
        else:
            zranges = tools.propose_fiducial('zranges', tracer, analysis=analysis)[:1]

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
            if any('window' in stat for stat in stats):
                _imocks = [0]
                nbatches = 1
                if nbatches > 1:
                    # Compute the window multipoles in batches, then assemble the matrix
                    tasks = []
                    for ibatch in range(nbatches):
                        task = get_run_stats()(imocks=_imocks, ibatch=(ibatch, nbatches), stats=stats, **run_stats_kws)
                        tasks.append(task)
                    # Add dependence on other tasks
                    get_run_stats()(imocks=_imocks, ibatch=nbatches, tasks=tasks, stats=stats, **run_stats_kws)
                else:
                    # ibatch=None: correlations and matrix in one go. Batching here would return
                    # the raw correlations only, and nothing would assemble them into a matrix.
                    get_run_stats()(imocks=_imocks, ibatch=None, stats=stats, **run_stats_kws)
            elif any('covariance' in stat for stat in stats):
                get_run_stats()(imocks=[0], stats=stats, **run_stats_kws)
            elif stats:
                batch_imocks = np.array_split(imocks, max((len(imocks) + max_mocks_per_batch - 1) // max_mocks_per_batch, 1)) if len(imocks) else []
                for _imocks in batch_imocks:
                    get_run_stats()(imocks=_imocks, stats=stats, **run_stats_kws)
        if postprocess:
            postprocess_stats(imocks=imocks, **run_stats_kws)