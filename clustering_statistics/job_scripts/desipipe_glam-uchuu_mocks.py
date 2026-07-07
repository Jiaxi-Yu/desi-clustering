"""
Script to create and spawn desipipe tasks to compute clustering measurements on glam-uchuu mocks.
To create and spawn the tasks on NERSC, use the following commands:
```bash
salloc -N 1 -C "gpu&hbm80g" -t 04:00:00 --gpus 4 --qos interactive --account desi_g
salloc -N 1 -C "gpu" -t 04:00:00 --gpus 4 --qos interactive --account desi_g
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
python desipipe_glam-uchuu_mocks.py         # create the list of tasks
desipipe tasks  -q glam-uchuu_mocks         # check the list of tasks
desipipe spawn  -q glam-uchuu_mocks --spawn # spawn the jobs
desipipe queues -q glam-uchuu_mocks         # check the queue
desipipe spawn  -q glam-uchuu_mocks --spawn --mode no_stream # add --mode no_stream if running >40 jobs in parallel
```
"""
import os
from pathlib import Path
import functools

import numpy as np
from desipipe import Queue, Environment, TaskManager, spawn, setup_logging

from clustering_statistics import tools

setup_logging()

queue = Queue('glam-uchuu_mocks')
queue.clear(kill=False)

output, error = 'slurm_outputs/glam-uchuu_mocks/slurm-%j.out', 'slurm_outputs/glam-uchuu_mocks/slurm-%j.err'
kwargs = {}
# environ = Environment('nersc-cosmodesi')
environ = Environment('nersc-cosmodesi', command='export PYTHONPATH=$HOME/LSScode/dr2-clustering-analysis/:$PYTHONPATH')
tm = TaskManager(queue=queue, environ=environ)
tm = tm.clone(scheduler=dict(max_workers=30), provider=dict(provider='nersc', time='02:00:00', mpiprocs_per_worker=4, 
                                                            output=output, error=error, stop_after=1, constraint='gpu'))
tm80 = tm.clone(provider=dict(provider='nersc', time='02:00:00', mpiprocs_per_worker=4, 
                              output=output, error=error, stop_after=1, constraint='gpu&hbm80g'))
tmw = tm.clone(scheduler=dict(max_workers=1), provider=dict(provider='nersc', time='00:10:00',
                mpiprocs_per_worker=2250, nodes_per_worker=25, output=output, error=error, stop_after=1, constraint='cpu'))

combine_region_sources = {
    'GCcomb': ['NGC', 'SGC'],
    'NS': ['N', 'S'],
    'GCcomb_noN': ['NGCnoN', 'SGC'],
    'GCcomb_noDES': ['NGC', 'SGCnoDES'],
}

def run_stats(tracer='LRG', project='', version='glam-uchuu-v2-altmtl', onthefly=None, imocks=[150], stats_dir=Path(os.getenv('SCRATCH')) / 'measurements', stats=['mesh2_spectrum'], weight='default-FKP', analysis='full_shape', regions=['NGC','SGC'], ibatch=None, postprocess=None, zranges=None, **kwargs):
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
    from clustering_statistics import tools, setup_logging, compute_stats_from_options, fill_fiducial_options, postprocess_stats_from_options
    setup_logging()

    cache = {}
    if zranges is None:
        raise ValueError('Please provide zranges.')
    for imock in imocks:
        for region in regions:
            mesh2_spectrum = {'cut': True if 'shape' in analysis else None, 
                              'auw': None}
                              # 'auw': True if 'altmtl' in version and onthefly is None and 'shape' in analysis else None}
            window_mesh2_spectrum = {'cut': True if 'shape' in analysis else None}
            
            options = dict(catalog=dict(version=version, tracer=tracer, zrange=zranges, region=region, weight=weight, imock=imock), 
                           mesh2_spectrum=mesh2_spectrum, window_mesh2_spectrum=window_mesh2_spectrum,
                           window_mesh3_spectrum={'ibatch': ibatch} if isinstance(ibatch, tuple) else {'computed_batches': ibatch})
            options = fill_fiducial_options(options, analysis=analysis)
            
            for itracer in options['catalog']:
                options['catalog'][itracer]['zranges'] = zranges # override fiducial zranges 
                options['catalog'][itracer]['expand']  = {'parent_randoms_fn': tools.get_catalog_fn(kind='parent_randoms', version='data-dr2-v2', tracer=itracer, nran=options['catalog'][itracer]['nran']), 'from_data': ['Z', 'WEIGHT_SYS', 'FRAC_TLOBS_TILES']}
                if onthefly == 'complete':
                    options['catalog'][itracer]['complete'] = {}
                elif onthefly == 'reshuffle':
                    merged_dir = tools.base_stats_dir / 'merged_catalogs' / version
                    options['catalog'][itracer]['reshuffle'] = {'merged_data_fn': tools.get_catalog_fn(kind='data', cat_dir=merged_dir, **(options['catalog'][itracer] | dict(region='ALL')))}                
            
            get_stats_fn = functools.partial(tools.get_stats_fn, stats_dir=stats_dir, project=project, extra=onthefly if onthefly else '')
            compute_stats_from_options(stats, analysis=analysis, get_stats_fn=get_stats_fn, cache=cache, **options)

    # postprocess
    if postprocess:
        postprocess_options = dict(catalog=dict(version=version, tracer=tracer, zrange=zranges, weight=weight, imock=imocks[0]), imocks=imocks, 
                                   combine_regions={'stats': stats}, mesh2_spectrum=mesh2_spectrum, window_mesh2_spectrum=window_mesh2_spectrum)
        postprocess_stats_from_options(postprocess, analysis=analysis, get_stats_fn=get_stats_fn, **postprocess_options)


def postprocess_stats(tracer='LRG', analysis='full_shape', project='', version='glam-uchuu-v2-altmtl', onthefly=None, imocks=[150], stats_dir=Path(os.getenv('SCRATCH')) / 'measurements', stats=['mesh2_spectrum'], weight='default-FKP', postprocess=['combine_regions'], zranges=None, **kwargs):
    from clustering_statistics import postprocess_stats_from_options
    if zranges is None:
        zranges = tools.propose_fiducial('zranges', tracer, analysis=analysis)
    options = dict(catalog=dict(version=version, tracer=tracer, zrange=zranges, weight=weight, imock=imocks[0]), imocks=imocks, combine_regions={'stats': stats}, mesh2_spectrum={'cut': True, 'auw': True}, window_mesh2_spectrum={'cut': True})
    stats_dir_kws = dict(stats_dir=stats_dir, project=project)
    if onthefly == 'complete':
        get_stats_fn = functools.partial(tools.get_stats_fn, extra='complete', **stats_dir_kws)
    elif onthefly == 'reshuffle':
        get_stats_fn = functools.partial(tools.get_stats_fn, extra='reshuffle', **stats_dir_kws)
    else:
        get_stats_fn = functools.partial(tools.get_stats_fn, **stats_dir_kws)

    postprocess_stats_from_options(postprocess, analysis=analysis, get_stats_fn=get_stats_fn, **options)



if __name__ == '__main__':

    stats, postprocess = [], []
    version  = 'glam-uchuu-v2-altmtl'
    # version  = 'glam-uchuu-bgs-altmtl'
    check_for_existing_measurements = True
    postprocess_only = False # If True, no measurements are performed and only postprocessing of existing measurements is handled.

    # run on interactive node
    # mode = 'interactive'
    # imocks2run = 150 + np.arange(1)
    # imocks2run = 10 + np.arange(1)
    # stats_dir  = Path(os.getenv('SCRATCH')) / 'cai-dr2-benchmarks' 

    # to run job
    mode = 'interactive'
    # mode = 'slurm'
    # imocks2run = np.arange(2000)
    imocks2run = 150 + np.arange(50)
    if version == 'glam-uchuu-v2-altmtl':
        #imocks2run = np.loadtxt('../helper_scripts/glam-uchuu-v2-altmtl_dark-time_imocks_for_covariance.txt', dtype=int)
        good_imocks = np.loadtxt('../helper_scripts/glam-uchuu-v2-altmtl_dark-time_imocks_for_covariance_v2.txt', dtype=int)
        imocks2run = imocks2run[np.isin(imocks2run, good_imocks)]
    stats_dir  = tools.base_stats_dir

    # run fiducial full_shape
    # stats       = ['mesh2_spectrum', 'mesh3_spectrum'] 
    # # stats = ['particle2_correlation']
    # postprocess = ['combine_regions']
    # analysis = 'full_shape'
    # project  = f'{analysis}/base'
    # weight   = 'default-FKP'
    # regions  = ['NGC','SGC']
    # # tracers  = ['QSO', 'ELG_LOPnotqso', 'LRG']
    # tracers = ['BGS_BRIGHT-21.35']
    # max_mocks_per_batch_qso = 20
    # max_mocks_per_batch_others = 20
    # postregions = ['GCcomb']

    # run data_splits for lensing group with full_shape setup 
    stats   = ['mesh2_spectrum']#, 'mesh3_spectrum']
    analysis = 'full_shape'
    project = f'{analysis}/data_splits'
    weight  = 'default-noimsys-FKP'
    # regions = ['NGC', 'SGC'] # already computed under full_shape/base/
    regions = ['N', 'NGCnoN', 'S', 'SGCnoDES'] # galactic and imaging regions
    regions = regions + ['ACT_DR6', 'PLANCK_PR4'] + [f'GAL0{i}' for i in [40, 60]] # lensing regions
    tracers = [ 'LRG', 'QSO', 'ELG_LOPnotqso']
    # tracers = ['BGS_BRIGHT-21.35']
    max_mocks_per_batch_qso = 50
    max_mocks_per_batch_others = 50
    # postprocess = ['combine_regions']
    # postregions = ['GCcomb', 'NS', 'GCcomb_noN', 'GCcomb_noDES'][1:]

    # run fiducial local_png
    # stats       = ['mesh2_spectrum']
    # postprocess = ['combine_regions']
    # analysis = 'local_png'
    # project  = f'{analysis}/base'
    # weight   = 'default-fkp-oqe'
    # regions  = ['NGC','SGC']
    # tracers  = ['LRG', 'ELGnotqso', 'QSO', ('LRG','QSO'), ('LRG','ELGnotqso'), ('ELGnotqso','QSO')]
    # max_mocks_per_batch_others = max_mocks_per_batch_qso = 50
    # postregions = ['GCcomb']

    onthefly = None

    for tracer in tracers:
        if tracer == 'QSO':
             max_mocks_per_batch = max_mocks_per_batch_qso # allow mocks to be processed since QSOs only have one zbin
        else:
             max_mocks_per_batch = max_mocks_per_batch_others
        if 'png' in analysis:
            # do not compute measurements for overlapping redshifts
            zranges = tools.propose_fiducial('zranges', tracer, analysis=analysis)[:1]
        else:
            zranges = tools.propose_fiducial('zranges', tracer, analysis=analysis)
        if check_for_existing_measurements:
            exists, missing = tools.checks_if_exists_and_readable(get_fn=functools.partial(tools.get_catalog_fn, tracer=tracer[0] if isinstance(tracer, (list, tuple)) else tracer,
                                                                                           region='NGC', version=version), test_if_readable=False, imock=imocks2run)[:2]
            catalog_imocks = exists[1]['imock']
            rerun_by_region = {region: [] for region in regions}
            for zrange in zranges[-1:]: # only check last zrange to speed up this step.
                for kind in stats:
                    for region in regions:
                        stats_kws = dict(basis='sugiyama-diagonal', kind=kind, stats_dir=Path(str(stats_dir).replace('global','dvs_ro')),
                                         tracer=tracer, region=region, weight=weight, zrange=zrange, version=version, project=project,
                                         extra=onthefly if onthefly else '')
                        rexists, missing, unreadable = tools.checks_if_exists_and_readable(get_fn=functools.partial(tools.get_stats_fn, **stats_kws), test_if_readable=True, imock=imocks2run)
                        rerun_by_region[region] += [imock for imock in catalog_imocks if (imock in unreadable[1]['imock']) or (imock not in rexists[1]['imock'])]
            rerun_by_region = {region: sorted(set(rerun)) for region, rerun in rerun_by_region.items()}
            imocks = sorted(set(imock for rerun in rerun_by_region.values() for imock in rerun))
        else:
            imocks = imocks2run
            rerun_by_region = {region: imocks for region in regions}

        def get_run_stats():
            _tm = tm80
            if tracer in ['LRG','BGS_BRIGHT-21.35']:
                _tm = tm
            if any('window_mesh3' in stat for stat in stats):
                _tm = tmw
            return run_stats if mode == 'interactive' else _tm.python_app(run_stats)

        run_stats_kws = dict(tracer=tracer, stats_dir=stats_dir, project=project, version=version, stats=stats, analysis=analysis, onthefly=onthefly, zranges=zranges, regions=regions, weight=weight, postprocess=postprocess)
        if not postprocess_only:
            if any('window' in stat for stat in stats):
                _imocks = [150]
                nbatches = 1
                tasks = []
                for ibatch in range(nbatches):
                    task = get_run_stats()(imocks=_imocks, ibatch=(ibatch, nbatches), **run_stats_kws)
                    tasks.append(task)
                if nbatches >= 1:
                    # Add dependence on other tasks
                    get_run_stats()(imocks=_imocks, ibatch=nbatches, tasks=tasks, **run_stats_kws)
            elif any('covariance' in stat for stat in stats):
                get_run_stats()(imocks=[150], **run_stats_kws)
            elif stats:
                for region, region_imocks in rerun_by_region.items():
                    batch_imocks = np.array_split(region_imocks, max(len(region_imocks) // max_mocks_per_batch, 1)) if len(region_imocks) else []
                    for _imocks in batch_imocks:
                        get_run_stats()(imocks=_imocks, **(run_stats_kws | dict(regions=[region])))
        else:
            # this handles the combination of measurements by region
            if check_for_existing_measurements:
                postprocess_rerun = []
                for zrange in zranges:
                    for kind in stats:
                        for region in postregions:
                            stats_kws = dict(basis='sugiyama-diagonal', kind=kind, stats_dir=Path(str(stats_dir).replace('global','dvs_ro')),
                                             tracer=tracer, region=region, weight=weight, zrange=zrange, version=version, project=project,
                                             extra=onthefly if onthefly else '')
                            rexists, missing, unreadable = tools.checks_if_exists_and_readable(get_fn=functools.partial(tools.get_stats_fn, **stats_kws), test_if_readable=True, imock=imocks2run)
                            postprocess_rerun += [imock for imock in imocks2run if (imock in unreadable[1]['imock']) or (imock not in rexists[1]['imock'])]
                imocks = sorted(set(postprocess_rerun))
            else:
                imocks = imocks2run
            postprocess_stats(imocks=imocks, **(run_stats_kws | dict(regions=postregions)))