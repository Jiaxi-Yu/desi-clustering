"""
Script to run a large batch of clustering measurements on an interactive GPU node.
To run on NERSC, use the following commands:
```bash
salloc -N 1 -C "gpu&hbm80g" -t 04:00:00 --gpus 4 --qos interactive --account desi_g
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
export PYTHONPATH=$HOME/LSScode/dr2-clustering-analysis/:$PYTHONPATH
srun -n 4 python interactive_job.py
```
"""
import os
from pathlib import Path
import functools

import numpy as np
from desipipe import setup_logging

from clustering_statistics import tools

setup_logging()

def run_stats(tracer='LRG', project='', version='holi-v3-altmtl', onthefly=None, imocks=[150], stats_dir=Path(os.getenv('SCRATCH')) / 'measurements', stats=['mesh2_spectrum'], weight='default-FKP', analysis='full_shape', do_jackknife=False, regions=['NGC','SGC'], ibatch=None, postprocess=None, zranges=None, profile_time=False, **kwargs):
    # Everything inside this function will be executed on the compute nodes;
    # This function must be self-contained; and cannot rely on imports from the outer scope.
    import os
    import sys
    from time import time
    import functools
    import logging
    from pathlib import Path
    from mpi4py import MPI
    
    import jax
    from jax import config
    config.update('jax_enable_x64', True)
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'
    try: jax.distributed.initialize()
    except RuntimeError: print('Distributed environment already initialized')
    else: print('Initializing distributed environment')
    from clustering_statistics import tools, setup_logging, compute_stats_from_options, fill_fiducial_options, postprocess_stats_from_options
    setup_logging()
    
    logger = logging.getLogger('timer')
    
    cache = {}
    if zranges is None:
        raise ValueError('Please provide zranges.')
    for imock in imocks:
        for region in regions:
            t0 = time()
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
            particle2_correlation |= {'jackknife': {'nsplits': 60}} if do_jackknife else {}
            particle3_correlation |= {'auw': auw}
            #particle3_correlation = {'split_randoms': (2., 10), 'battrs': dict(s=np.linspace(0., 20., 21), pole=(list(range(6)), 'firstpoint'))}
            options = dict(catalog=dict(version=version, tracer=tracer, zrange=zranges, region=region, weight=weight, imock=imock),
                           mesh2_spectrum=mesh2_spectrum, window_mesh2_spectrum=window_mesh2_spectrum,
                           mesh3_spectrum=mesh3_spectrum, window_mesh3_spectrum=window_mesh3_spectrum,
                           particle2_correlation=particle2_correlation,
                           particle3_correlation=particle3_correlation)
            options = fill_fiducial_options(options, analysis=analysis)
            
            for itracer in options['catalog']:
                options['catalog'][itracer]['zranges'] = zranges # override fiducial zranges 
                if version != 'uchuu-hf-reference' and version != 'abacus-hf-dr2-v1':
                    options['catalog'][itracer]['expand']  = {'parent_randoms_fn': tools.get_catalog_fn(kind='parent_randoms', version='data-dr2-v2', tracer=itracer, nran=options['catalog'][itracer]['nran']), 'from_data': ['Z', 'WEIGHT_SYS', 'FRAC_TLOBS_TILES']}
                if onthefly == 'complete':
                    options['catalog'][itracer]['complete'] = {}
                elif onthefly == 'reshuffle':
                    merged_dir = tools.base_stats_dir / 'merged_catalogs' / version
                    options['catalog'][itracer]['reshuffle'] = {'merged_data_fn': tools.get_catalog_fn(kind='data', cat_dir=merged_dir, **(options['catalog'][itracer] | dict(region='ALL')))}               
            
            get_stats_fn = functools.partial(tools.get_stats_fn, stats_dir=stats_dir, project=project, extra=onthefly if onthefly else '')
            compute_stats_from_options(stats, analysis=analysis, get_stats_fn=get_stats_fn, cache=cache, **options)
            _time = time() - t0
            if profile_time: 
                _tracer = tools.join_tracers(tracer)
                logger.info(f"For {_tracer} of {version} we computed {stats} for {region} in {_time:.2f} seconds.")
                # Creates file to log times
                mpicomm = MPI.COMM_WORLD
                if mpicomm.rank == 0:
                    with open(f"../helper_scripts/profiling_{analysis}_stats.txt", "a") as file:
                        file.write(f"For {_tracer} of {version} we computed {stats} for {region} in {_time:.2f} seconds.\n")

    # postprocess
    if postprocess:
        postprocess_options = dict(catalog=dict(version=version, tracer=tracer, zrange=zranges, weight=weight, imock=imocks[0]), imocks=imocks, 
                                   combine_regions={'stats': stats}, mesh2_spectrum=mesh2_spectrum, window_mesh2_spectrum=window_mesh2_spectrum)
        postprocess_stats_from_options(postprocess, analysis=analysis, get_stats_fn=get_stats_fn, **postprocess_options)


def postprocess_stats(tracer='LRG', analysis='full_shape', project='', version='holi-v3-altmtl', onthefly=None, imocks=[150], stats_dir=Path(os.getenv('SCRATCH')) / 'measurements', stats=['mesh2_spectrum'], weight='default-FKP', postprocess=['combine_regions'], zranges=None, **kwargs):
    from clustering_statistics import postprocess_stats_from_options
    if zranges is None:
        zranges = tools.propose_fiducial('zranges', tracer, analysis=analysis)
    options = dict(catalog=dict(version=version, tracer=tracer, zrange=zranges, weight=weight, imock=imocks[0]), imocks=imocks, combine_regions={'stats': stats}, mesh2_spectrum={'cut': True, 'auw': True}, window_mesh2_spectrum={'cut': True})
    stats_dir_kws = dict(stats_dir=stats_dir, project=project)
    if onthefly == 'complete':
        get_stats_fn = functools.partial(tools.get_stats_fn, extra='complete', **stats_dir_kws)
    elif onthefly == 'reshuffle':
        # get_stats_fn = functools.partial(tools.get_stats_fn, extra='reshuffle', **stats_dir_kws)
        get_stats_fn = functools.partial(tools.get_stats_fn, extra='reshuffle', **stats_dir_kws)
    else:
        get_stats_fn = functools.partial(tools.get_stats_fn, **stats_dir_kws)

    postprocess_stats_from_options(postprocess, analysis=analysis, get_stats_fn=get_stats_fn, **options)



if __name__ == '__main__':

    stats, postprocess = [], []
    # version  = 'glam-uchuu-v2-altmtl'
    # version  = 'holi-v3-altmtl'
    # version  = 'holi-bgs-altmtl'
    # version  = 'abacus-hf-dr2-v2-altmtl'
    # version = 'abacus-2ndgen-dr2-altmtl'
    # version = 'uchuu-hf-reference'
    # version = 'uchuu-hf-altmtl'
    # version = 'abacus-hf-dr2-v1'
    # version = 'holi-v4-altmtl'
    # version = 'glam-uchuu-v2-altmtl-maskedfraczpNN'
    version = 'holi-v4-altmtl-NN'
    check_for_existing_measurements = True

    # test run 
    # imocks2run = 150 + np.arange(1)
    # imocks2run = np.arange(1)
    # imocks2run = np.arange(25)
    # stats_dir  = Path(os.getenv('SCRATCH')) / 'cai-dr2-benchmarks' 

    # official run
    # imocks2run = [150] + np.arange(50)
    imocks2run = np.arange(1)
    # imocks2run = np.arange(1,500)
    # imocks2run = np.arange(500,1000)
    # if version == 'holi-v3-altmtl':
    #     # do not perform measurements on dubious mocks
    #     bad_imocks = np.loadtxt('../helper_scripts/dubious_holi-v3-altmtl.txt',dtype=int)
    #     imocks2run = imocks2run[~np.isin(imocks2run,bad_imocks)]
    #if version == 'glam-uchuu-v2-altmtl':
        # imocks2run = np.loadtxt('../helper_scripts/glam-uchuu-v2-altmtl_dark-time_imocks_for_covariance.txt', dtype=int)
    #    good_imocks = np.loadtxt('../helper_scripts/glam-uchuu-v2-altmtl_dark-time_imocks_for_covariance_v2.txt', dtype=int)
    #    imocks2run = imocks2run[np.isin(imocks2run, good_imocks)]
    stats_dir  = tools.base_stats_dir

    # run fiducial full_shape
    # stats       = ['mesh2_spectrum']#, 'particle2_correlation']
    # stats       = ['mesh3_spectrum', 'window_mesh3_spectrum']
    stats = ['mesh2_spectrum', 'mesh3_spectrum',
             'window_mesh2_spectrum', 'window_mesh3_spectrum', 
             'covariance_mesh2_spectrum', 'covariance_mesh3_spectrum']
    # stats = ['mesh2_spectrum','mesh3_spectrum']
    postprocess = ['combine_regions']
    analysis = 'full_shape'
    project = f'{analysis}/base'
    #project  = f'{analysis}/base'
    weight   = 'default-FKP'
    # weight   = 'default'
    regions  = ['NGC','SGC']
    # regions = ['NGC','SGC','N','NGCnoN','S','SGCnoDES','SnoDES','DES','ACT_DR6','PLANCK_PR4','GAL040','GAL060']
    # tracers  = ['LRG', 'ELG_LOPnotqso', 'QSO']
    tracers  = ['ELG_LOPnotqso']
    # tracers  = ['BGS_BRIGHT-21.35']
    #tracers = ['BGS_BRIGHT-02']
    max_mocks_per_batch = 1

    # run data_splits for lensing group with full_shape setup 
    # stats   = ['mesh2_spectrum']
    # postprocess = ['combine_regions']
    # analysis = 'full_shape'
    # project = f'{analysis}/data_splits'
    # weight  = 'default-FKP'
    # regions = ['N','NGCnoN','S','SGCnoDES','SnoDES','DES','ACT_DR6','PLANCK_PR4','GAL040','GAL060']
    # regions = ['GCcomb_noN', 'GCcomb_noDES']
    # tracers  = ['LRG', 'ELG_LOPnotqso', 'QSO']
    # imax_mocks_per_batch = 1 

    # run fiducial local_png
    # stats       = ['mesh2_spectrum']
    # postprocess = ['combine_regions']
    # analysis = 'local_png'
    # project  = f'{analysis}/base'
    # weight   = 'default-noimsys-fkp-oqe'
    # weight   = 'default-fkp'
    # regions  = ['NGC','SGC']
    # tracers  = ['LRG', 'ELGnotqso', 'QSO', ('LRG','QSO'), ('LRG','ELGnotqso'), ('ELGnotqso','QSO')]
    # tracers = ['ELGnotqso']
    # max_mocks_per_batch = 1

    # onthefly = 'complete'
    # onthefly = 'reshuffle'
    onthefly = None
    do_jackknife = False

    for tracer in tracers:
        if 'png' in analysis:
            # do not compute measurements for overlapping redshifts
            zranges = tools.propose_fiducial('zranges', tracer, analysis=analysis)[:1]
        else:
            #if 'QSO' == tracer:
            #    zranges = [(0.8,1.6),(1.6,2.1)] 
            #else:
            #    zranges = [(0.8,1.6)]
            zranges = tools.propose_fiducial('zranges', tracer, analysis=analysis)

        if check_for_existing_measurements:
            exists, missing = tools.checks_if_exists_and_readable(get_fn=functools.partial(tools.get_catalog_fn, tracer=tracer[0] if isinstance(tracer, (list, tuple)) else tracer,
                                                                                           region='NGC', version=version), test_if_readable=False, imock=imocks2run)[:2]
            imocks = exists[1]['imock']
            # print(tools.get_catalog_fn(tracer=tracer[0] if isinstance(tracer, (list, tuple)) else tracer, region='NGC', version=version, imock=imocks2run[0]))
            # print(imocks)
            rerun = []
            for zrange in zranges:
                for kind in stats:
                    stats_kws = dict(basis='sugiyama-diagonal', kind=kind, stats_dir=Path(str(stats_dir).replace('global','dvs_ro')),
                                     tracer=tracer, region=regions[-1], weight=weight, zrange=zrange, version=version, project=project,
                                     extra=onthefly if onthefly else '')
                    rexists, missing, unreadable = tools.checks_if_exists_and_readable(get_fn=functools.partial(tools.get_stats_fn, **stats_kws), test_if_readable=True, imock=imocks2run)
                    rerun += [imock for imock in imocks if (imock in unreadable[1]['imock']) or (imock not in rexists[1]['imock'])]
            imocks = sorted(set(rerun))
        else:
            imocks = imocks2run
        
        run_stats_kws = dict(tracer=tracer, stats_dir=stats_dir, project=project, version=version, stats=stats, analysis=analysis, onthefly=onthefly, zranges=zranges, do_jackknife=do_jackknife, regions=regions, weight=weight, postprocess=postprocess)
        batch_imocks = np.array_split(imocks, max(len(imocks) // max_mocks_per_batch, 1)) if len(imocks) else []
        for _imocks in batch_imocks:
            run_stats(imocks=_imocks, **run_stats_kws)
        #if postprocess:
        #   postprocess_stats(imocks=imocks, **run_stats_kws)