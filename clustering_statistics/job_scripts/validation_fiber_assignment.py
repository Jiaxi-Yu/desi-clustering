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
    queue = Queue('abacus_mocks5')
    queue.clear(kill=False)

    output, error = 'slurm_outputs/abacus_mocks/slurm-%j.out', 'slurm_outputs/abacus_mocks/slurm-%j.err'
    kwargs = {}
    environ = Environment('nersc-cosmodesi', command=['module unload desi-clustering'])
    tm = TaskManager(queue=queue, environ=environ)
    tm = tm.clone(scheduler=dict(max_workers=20), provider=dict(provider='nersc', time='03:00:00',
                                mpiprocs_per_worker=4, output=output, error=error, constraint='gpu'))
    tm80 = tm.clone(provider=dict(provider='nersc', time='03:00:00',
                                mpiprocs_per_worker=4, output=output, error=error, stop_after=1, constraint='gpu&hbm80g'))
    tmw = tm.clone(scheduler=dict(max_workers=1), provider=dict(provider='nersc', time='00:10:00',
                    mpiprocs_per_worker=2250, nodes_per_worker=25, output=output, error=error, stop_after=1, constraint='cpu'))


def get_stats_fn(*args, extra='', onthefly=None, **kwargs):
    from clustering_statistics import tools
    extra = [txt for txt in [extra, onthefly] if txt]
    return tools.get_stats_fn(*args, extra='_'.join(extra), **kwargs)


def run_stats(tracer='LRG', project='', version='abacus-hf-dr2-v2-altmtl', onthefly=None, imocks=[150], stats_dir=Path(os.getenv('SCRATCH')) / 'measurements', stats=['mesh2_spectrum'], weight='default-FKP', analysis='full_shape', regions=['NGC', 'SGC'], cat_dir=None, ibatch=None, zranges=None, get_stats_fn=get_stats_fn, **kwargs):
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
            correction = any('close_pair_correction' in stat or 'window' in stat for stat in stats) # run AUW or theta-cut only when asking for close_pair_correction
            auw = correction and ('altmtl' in version and onthefly is None or 'data' in version)
            cut = correction
            mesh2_spectrum = {'cut': cut, 'auw': auw}
            window_mesh2_spectrum = {'cut': cut}
            mesh3_spectrum = {'auw': auw}
            window_mesh3_spectrum = {'ibatch': ibatch} if isinstance(ibatch, tuple) else {'computed_batches': ibatch}
            mode = 'smu'
            if mode == 'smu':
                particle2_correlation = {'split_randoms': (2., 10), 'battrs': dict(s=np.linspace(0., 40., 41), mu=(np.linspace(-1., 1., 201), 'midpoint'))}
                particle3_correlation = {'split_randoms': (2., 10), 'battrs': dict(s=np.linspace(0., 20., 21), pole=(list(range(6)), 'firstpoint'))}
            elif mode == 'theta':
                particle2_correlation = {'split_randoms': (2., 10), 'battrs': dict(theta=np.linspace(0., 0.3, 31))}
                particle3_correlation = {'split_randoms': (2., 10), 'battrs': dict(theta=np.linspace(0., 0.3, 31))}
            particle2_correlation |= {'auw': auw}
            particle3_correlation |= {'auw': auw}
            #particle3_correlation = {'split_randoms': (2., 10), 'battrs': dict(s=np.linspace(0., 20., 21), pole=(list(range(6)), 'firstpoint'))}
            options = dict(catalog=dict(version=version, tracer=tracer, zrange=zranges, region=region, weight=weight, imock=imock),
                           mesh2_spectrum=mesh2_spectrum, window_mesh2_spectrum=window_mesh2_spectrum,
                           mesh3_spectrum=mesh3_spectrum, window_mesh3_spectrum=window_mesh3_spectrum,
                           particle2_correlation=particle2_correlation,
                           particle3_correlation=particle3_correlation)
            options = fill_fiducial_options(options, analysis=analysis)

            for itracer in options['catalog']:
                #options['catalog'][itracer]['nran'] = 5
                #if 'BGS_BRIGHT' in itracer:
                #    options['catalog'][itracer]['tracer'] = 'BGS_BRIGHT'
                options['catalog'][itracer]['zranges'] = zranges # override fiducial zranges
                options['catalog'][itracer]['expand'] = {'parent_randoms_fn': tools.get_catalog_fn(kind='parent_randoms', version='data-dr2-v2', tracer=itracer, nran=options['catalog'][itracer]['nran'])}
                if onthefly is not None and onthefly.startswith('complete'):
                    options['catalog'][itracer]['complete'] = {'with_completeness': 'nocomp' not in onthefly, 'with_tracer_cuts': True}
                elif onthefly is not None and onthefly.startswith('altmtl'):
                    options['catalog'][itracer]['complete'] = {'altmtl': True}
                elif onthefly == 'reshuffle':
                    options['catalog'][itracer]['reshuffle'] = {'merged_data_fn': tools.get_catalog_fn(kind='data', **(options['catalog'][itracer] | dict(region='ALL')))}

            _get_stats_fn = functools.partial(get_stats_fn, stats_dir=stats_dir, project=project, onthefly=onthefly)
            _get_catalog_fn = tools.get_catalog_fn
            if cat_dir is not None:
                if 'auxiliary_data' in str(cat_dir):
                    def _get_catalog_fn(**kwargs):
                        imock = kwargs.get('imock')
                        out_dir = cat_dir / f'mock{imock:d}'
                        return tools.get_catalog_fn(cat_dir=out_dir, **kwargs)
                else:
                    def _get_catalog_fn(**kwargs):
                        _cat_dir = Path(cat_dir)
                        if 'full' in kwargs['kind']:
                            _cat_dir = _cat_dir.parent
                        return tools.get_catalog_fn(cat_dir=_cat_dir, ext=None, **kwargs)
            compute_stats_from_options(stats, analysis=analysis, get_catalog_fn=_get_catalog_fn, get_stats_fn=_get_stats_fn, cache=cache, **options)


def postprocess_stats(tracer='LRG', analysis='full_shape', project='', version='abacus-hf-dr2-v2-altmtl', onthefly=None, imocks=[150], stats_dir=Path(os.getenv('SCRATCH')) / 'measurements', weight='default-FKP', postprocess=['combine_regions'], zranges=None, get_stats_fn=get_stats_fn, **kwargs):
    from clustering_statistics import postprocess_stats_from_options
    if zranges is None:
        zranges = tools.propose_fiducial('zranges', tracer, analysis=analysis)
    options = dict(catalog=dict(version=version, tracer=tracer, zrange=zranges, weight=weight, imock=imocks[0]), imocks=imocks,
                   combine_regions={'stats': ['mesh2_spectrum', 'mesh3_spectrum', 'window_mesh2_spectrum', 'window_mesh3_spectrum', 'particle2_correlation', 'particle3_correlation']},
                   mesh2_spectrum={'cut': True, 'auw': True}, window_mesh2_spectrum={'cut': True},
                   mesh3_spectrum={'auw': True}, window_mesh3_spectrum={},
                   systematic_templates={'stats': ['mesh2_spectrum', 'mesh3_spectrum'], 'effects': ['auw'],
                                        'templates': {'auw': {'extra': 'auw'}, 'raw': {}}})
    _get_stats_fn = functools.partial(get_stats_fn, stats_dir=stats_dir, project=project, onthefly=onthefly)
    postprocess_stats_from_options(postprocess, analysis=analysis, get_stats_fn=_get_stats_fn, **options)



if __name__ == '__main__':

    check_for_existing_measurements = False
    stats, postprocess = [], []
    #version = 'abacus-hf-dr2-v2-altmtl'
    #version = 'glam-uchuu-v2-altmtl'
    version = 'abacus-hf-dr2-v2-altmtl'
    #version = 'glam-uchuu-v2-altmtl'
    #version = 'abacus-2ndgen-dr2-complete'
    #version = 'abacus-2ndgen-dr2-altmtl'
    #version = 'data-dr2-v2'
    #version = 'data-dr2-test-maskedfracz'
    #version = 'data-dr2-test-maskedfraczpNN'
    analysis = 'full_shape'
    cat_dir = None
    #compweight = 'tilelocid-LRG1'
    #compweight = 'tilelocid-LRG0'
    #cat_dir = tools.base_stats_dir / f'auxiliary_data/fiber_assignment_systematics_ELG_{compweight}' / version
    #cat_dir = tools.desi_dir / f'survey/catalogs/DA2/LSS/loa-v1/LSScats/test/maskedfraczpNN'

    project = f'{analysis}/fiber_assignment_systematics'
    #project = f'{analysis}/fiber_assignment_systematics_tests'
    #project = f'{analysis}/fiber_assignment_systematics_ELG_{compweight}'
    imocks = np.arange(10)
    #imocks = np.arange(14, 25)
    #imocks = np.arange(12, 25)
    #imocks = np.arange(5, 9)
    #imocks = np.arange(9)
    #imocks = np.arange(1, 2)
    if 'data' in version:
        imocks = [None]

    if version == 'glam-uchuu-v2-altmtl':
        check_for_existing_measurements = False
        imocks = np.loadtxt('../helper_scripts/glam-uchuu-v2-altmtl_dark-time_imocks_for_covariance.txt', dtype=int)[:25]
        #imocks = [150]

    stats_dir = tools.base_stats_dir

    # run fiducial full_shape
    tracers = ['BGS', 'LRG', 'ELG', 'QSO'][1:]
    #tracers = ['ELG', 'LRG']
    #tracers = ['LRG']
    #tracers = ['ELG', 'QSO'][:1]
    #tracers = ['LRG', 'QSO']
    # run BGS
    #version = 'abacus-2ndgen-dr2-altmtl'
    #tracers = ['BGS_BRIGHT']
    #tracers = ['BGS_BRIGHT-02']
    #tracers = ['BGS_ANY-02']

    # run data_splits for lensing group with full_shape setup
    #stats = ['mesh2_spectrum', 'mesh3_spectrum']
    #stats = ['mesh3_spectrum', 'close_pair_correction']
    #stats = ['mesh2_spectrum', 'window_mesh2_spectrum'][:1]
    #stats = ['window_mesh2_spectrum', 'window_mesh3_spectrum']
    #stats = ['mesh2_spectrum', 'mesh3_spectrum', 'particle2_correlation', 'particle3_correlation'][:2]
    #stats = ['particle2_correlation', 'particle3_correlation', 'close_pair_correction'][:2]
    #stats = ['particle2_correlation', 'close_pair_correction']
    #stats = ['particle2_correlation']
    stats = ['mesh2_spectrum', 'mesh3_spectrum', 'close_pair_correction'][:0]
    #stats = ['mesh3_spectrum', 'close_pair_correction']
    #postprocess = ['combine_regions']
    postprocess = ['systematic_templates']
    weight = 'default-FKP'
    #weight = 'default-FKP-bitwise-iip'
    #weight = 'default-FKP'
    #weight = 'default-FKP-noimsys'
    #weight = 'default'
    regions = ['NGC', 'SGC'][:1]
    #regions = ['SGCnoDES', 'DES']
    max_mocks_per_batch = 5

    onthefly = None
    #onthefly = 'complete-nozfail'
    #onthefly = 'complete-renorm'
    #onthefly = 'complete-downsample'
    #onthefly = 'complete-samenz'
    #onthefly = 'complete-fixnz'
    #onthefly = 'complete-fibered'
    #onthefly = 'complete'
    #onthefly = 'altmtl'

    for tracer in tracers:
        if 'BGS_BRIGHT-02' not in tracer:
            tracer = tools.get_full_tracer(tracer, version=version)
        if 'png' in analysis:
            # do not compute measurements for overlapping redshifts
            zranges = tools.propose_fiducial('zranges', tracer, analysis=analysis)[:1]
        else:
            zranges = tools.propose_fiducial('zranges', tracer, analysis=analysis)

        def get_run_stats():
            if mode == 'interactive':
                return run_stats
            _tm = tm80
            if tracer in ['LRG']:
                _tm = tm80
            if any('window_mesh3' in stat for stat in stats):
                _tm = tmw
            return _tm.python_app(run_stats)

        run_stats_kws = dict(tracer=tracer, stats_dir=stats_dir, project=project, version=version, analysis=analysis, onthefly=onthefly, zranges=zranges, regions=regions, weight=weight, postprocess=postprocess, cat_dir=cat_dir)

        if check_for_existing_measurements:
            exists, missing = tools.checks_if_exists_and_readable(get_fn=functools.partial(tools.get_catalog_fn, tracer=tracer[0] if isinstance(tracer, (list, tuple)) else tracer,
                                                                                           region='NGC', version=version), test_if_readable=False, imock=imocks)[:2]
            imocks = exists[1]['imock']
        if True:
            if any('window' in stat for stat in stats):
                _imocks = imocks[:1]
                nbatches = 1
                tasks = []
                for ibatch in range(nbatches):
                    task = get_run_stats()(imocks=_imocks, ibatch=(ibatch, nbatches), stats=stats, **run_stats_kws)
                    tasks.append(task)
                if nbatches >= 1:
                    # Add dependence on other tasks
                    get_run_stats()(imocks=_imocks, ibatch=nbatches, tasks=tasks, stats=stats, **run_stats_kws)
            elif any('covariance' in stat for stat in stats):
                get_run_stats()(imocks=[0], stats=stats, **run_stats_kws)
            elif stats:
                batch_imocks = np.array_split(imocks, max((len(imocks) + max_mocks_per_batch - 1) // max_mocks_per_batch, 1)) if len(imocks) > max_mocks_per_batch else [imocks]
                for _imocks in batch_imocks:
                    get_run_stats()(imocks=_imocks, stats=stats, **run_stats_kws)
        if postprocess:
            postprocess_stats(imocks=imocks, **run_stats_kws)