"""
Script to create and spawn desipipe tasks to compute clustering measurements on data for PNG key project.

To run interactively see __main__ section below.

To create and spawn the tasks on NERSC, use the following commands:
```
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

# create the list of tasks:
python desipipe_data_png.py  --blinded

# check the list of tasks (you can also check with desipipe tasks -q '*/*' to see all the queues and tasks):
desipipe tasks -q data_png  

# spawn i.e. launch the jobs:
desipipe spawn -q data_png --spawn  

# check the queue:
desipipe queues -q data_png  

# Restart some jobs that failed (for instance due to time limit):**
desipipe retry -q data_png --state RUNNING # FAILED # KILLED

# delete the queue and all the tasks:
desipipe delete -q data_png --force
```
"""

import logging

import os
from pathlib import Path
import itertools
import functools

import numpy as np


logger = logging.getLogger('Data PNG')


# disable jax warning:
logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)
logging.getLogger("jax._src.distributed").setLevel(logging.ERROR)
# Remove warning from jax
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def setup_queue():
    """Set up the desipipe queue and task manager."""
    from desipipe import Queue, Environment, TaskManager, spawn

    queue = Queue('data_png')
    queue.clear(kill=False)

    output, error = 'slurm_outputs/data_png/slurm-%j.out', 'slurm_outputs/data_png/slurm-%j.err'
    kwargs = {}
    environ = Environment('nersc-cosmodesi')
    tm = TaskManager(queue=queue, environ=environ)
    tm = tm.clone(scheduler=dict(max_workers=10), provider=dict(provider='nersc', time='01:30:00',
                                mpiprocs_per_worker=4, output=output, error=error, stop_after=1, constraint='gpu'))
    tm80 = tm.clone(provider=dict(provider='nersc', time='02:00:00',
                                mpiprocs_per_worker=4, output=output, error=error, stop_after=1, constraint='gpu&hbm80g'))
    return tm, tm80


def _update_regression_maps(options, weight):
    """ update the template / redshift range / photometric range as function of the weight used. """
    # work only for the auto now:
    if isinstance(weight, tuple):
        logger.error('NOT READY FOR CROSS-CORRELATION...')
        # see: https://github.com/cosmodesi/desi-clustering/blob/67b0926114af5d987f593ddbef725cb829bbe548/clustering_statistics/tools.py#L596
        sys.exit(3)

    else:
        if 'wsys-imlin_finezbin_allebvcmb' in weight: 
            logger.info(f'Update templates for {weight=}')
            photoregions = ["N", "S"]
            regression_zranges = [(0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0), (1.0, 1.1)]

            options['window_mesh2_spectrum_fm']['regression_maps'] = ['STARDENS', 'PSFSIZE_G', 'PSFSIZE_R', 'PSFSIZE_Z', 'GALDEPTH_G', 'GALDEPTH_R', 'GALDEPTH_Z', 'HI', 'PSFDEPTH_W1', 'EBV_DIFF_GR', 'EBV_DIFF_RZ', 'ZCMB']
            options['window_mesh2_spectrum_fm']['amr_regions_zranges'] = list(itertools.product(photoregions, regression_zranges))

        else:
            logger.error(f'Not ready for {weight=}...')
            sys.exit(3)

    return options
        

def run_stats(cat_dir=None, stats_dir=None, tracer='LRG', zranges=[0.4, 1.1], weights=['default-fkp'], 
              regions=['NGC','SGC'], stats=['mesh2_spectrum'], **kwargs):
    """" Everything inside this function will be executed on the compute nodes; This function must be self-contained; 
         and cannot rely on imports from the outer scope. """
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
    else: pass  # print('Initializing distributed environment')
    from clustering_statistics import tools, setup_logging, compute_stats_from_options

    # May need to add this for the desipipe now interactive is fine...
    #from mpi4py import MPI
    #setup_logging(level=(logging.INFO if MPI.COMM_WORLD.rank == 0 else logging.ERROR))

    cache = {}
    get_stats_fn = functools.partial(tools.get_stats_fn, stats_dir=stats_dir)
    for region in regions:
        for weight in weights:
            # options will be filled by default options in compute_stats_from_options following analysis='local_png'
            options = dict(catalog=dict(cat_dir=cat_dir, tracer=tracer, zrange=zranges, weight=weight, region=region, ext='fits'), 
                           mesh2_spectrum={'cut': False}, window_mesh2_spectrum={'cut': False})

            if 'window_mesh2_spectrum_fm' in stats:
                options['catalog']['nran'] = 1  # not enough memory to do with more randoms ... 
                options['catalog']['keep_columns'] = ['RA', 'DEC', 'Z', 'POSITION', 'NX', 'TARGETID', 'WEIGHT_FKP']  # add WEIGHT_FKP for the foward model

                options['window_mesh2_spectrum_fm'] = {}
                # for ell=0: 4 is the max that I can fit in memory with nran=1 / for ell=2 -> I need to go down to 3...
                options['window_mesh2_spectrum_fm']['batch_size'] = 3  

                if 'spectrum_regions_zranges' in kwargs:
                    options['window_mesh2_spectrum_fm']['spectrum_regions_zranges'] = kwargs['spectrum_regions_zranges']

                options['window_mesh2_spectrum_fm']['geo'] = kwargs.get('geo', True)
                options['window_mesh2_spectrum_fm']['ric'] = kwargs.get('ric', True)
                options['window_mesh2_spectrum_fm']['amr'] = kwargs.get('amr', True)
                options['window_mesh2_spectrum_fm']['ellsout'] = kwargs.get('ellsout', None)

                options['window_mesh2_spectrum_fm']['theory_rebin'] = 10 # reduce the number of points -> greatly improve the computation time.

                options['window_mesh2_spectrum_fm']['n_realizations'] = 10
                options['window_mesh2_spectrum_fm']['seeds'] = [50, 20, 77, 80, 97, 6, 52, 64, 76, 81]
                if tracer == 'QSO':
                    options['window_mesh2_spectrum_fm']['n_realizations'] += 5  # 10
                    options['window_mesh2_spectrum_fm']['seeds'] += [84, 31, 2, 79, 15]  #, 84, 43, 28, 98, 12]

                if 'wsys' in weight:
                    options = _update_regression_maps(options, weight)

            if 'shotnoise_mesh2_spectrum_fm' in stats:
                options['catalog']['nran'] = 2  # not enough memory to do with more randoms ... 
                options['catalog']['keep_columns'] = ['RA', 'DEC', 'Z', 'POSITION', 'NX', 'TARGETID', 'WEIGHT_FKP']  # add WEIGHT_FKP for the foward model

                options['shotnoise_mesh2_spectrum_fm'] = {}
                if 'spectrum_regions_zranges' in kwargs:
                    options['shotnoise_mesh2_spectrum_fm']['spectrum_regions_zranges'] = kwargs['spectrum_regions_zranges']
                options['shotnoise_mesh2_spectrum_fm']['n_realizations'] = 30
                options['shotnoise_mesh2_spectrum_fm']['seeds'] = [75, 79, 59, 37, 88, 69, 18, 30, 84, 84, 
                                                                   18, 10, 69, 92, 45, 99, 81, 3, 70, 7, 55,
                                                                   74, 55, 85, 60, 23, 82, 42, 65, 49]

            compute_stats_from_options(stats, get_stats_fn=get_stats_fn, cache=cache, analysis='local_png', **options)


def postprocess_stats(cat_dir=None, stats_dir=None, tracer='LRG', zranges=[0.4, 1.1], weights=['default-fkp'], 
                      stats=['mesh2_spectrum'], postprocess=['combine_regions'], regions=None, **kwargs):
    from clustering_statistics import postprocess_stats_from_options, tools
    for weight in weights:
        get_stats_fn = functools.partial(tools.get_stats_fn, stats_dir=stats_dir)
        options = dict(catalog=dict(cat_dir=cat_dir, tracer=tracer, zrange=zranges, weight=weight, ext='fits'), 
                       combine_regions={'stats': stats}, mesh2_spectrum={'cut': False}, window_mesh2_spectrum={'cut': False})

        if 'extra' in kwargs: options['extra'] = kwargs['extra']

        if 'window_mesh2_spectrum_fm' in stats:
            options['window_mesh2_spectrum_fm'] = {}
            options['window_mesh2_spectrum_fm']['ellsout'] = kwargs.get('ellsout', None)
            options['window_mesh2_spectrum_fm']['n_realizations'] = 10
            options['window_mesh2_spectrum_fm']['seeds'] = [50, 20, 77, 80, 97, 6, 52, 64, 76, 81]
            if tracer == 'QSO':
                options['window_mesh2_spectrum_fm']['n_realizations'] += 5  # 10
                options['window_mesh2_spectrum_fm']['seeds'] += [84, 31, 2, 79, 15]  #, 84, 43, 28, 98, 12]
            
            options['combine_window_mesh2_spectrum'] = {}
            options['combine_window_mesh2_spectrum']['effect'] = 'RIC+AMR' if kwargs['amr'] else 'RIC'

            for region in regions: 
                options['catalog']['region'] = region
                postprocess_stats_from_options(postprocess, get_stats_fn=get_stats_fn, analysis='local_png', **options)

        else:
            postprocess_stats_from_options(postprocess, get_stats_fn=get_stats_fn, analysis='local_png', **options)



def collect_argparser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--interactive', action='store_true', help='Whether to run in interactive mode (without spawning jobs with desipipe).')
    
    parser.add_argument('--tracers', nargs='+', type=str, default=None, help='Which tracer to run..')
    parser.add_argument('--zranges', nargs='+', type=float, default=None, help='Provide --zranges zmin1 zmax1 zmin2 zmax2 if you want to compute things on (zmin1,zmax1) and (zmin2,zmax2) and ect...')
    parser.add_argument('--regions', nargs='+', type=str, default=None, help='Which regions to compute..', 
                        choices=['NGC', 'SGC', 'N', 'NGCnoN', 'SGCnoDES', 'DES'] + ['ACT_DR6', 'PLANCK_PR4'] + [f'GAL0{i}' for i in [40, 60]])
    
    parser.add_argument('--blinded', action='store_true', help='Run with blinded data or not.')

    parser.add_argument('--todo', nargs='+', type=str, default=None, help='Which activities to do?', 
                        choices=['spectrum', 'window', 'covariance', 'geo', 'ric', 'amr', 'sn'])

    parser.add_argument('--ellsout', nargs='+', type=int, default=None, help='For which mulitpoles the forward model of the window function is computed. If None, compute for each mulitpoles available into the corresponding power spectrum.')

    args = parser.parse_args()
    logger.info(args)
    return args


if __name__ == '__main__':
    """
    # To run interactively on NERSC, use the following commands:
    salloc -N 1 -C "gpu&hbm80g" -t 04:00:00 --gpus 4 --qos interactive --account desi_g
    source /global/homes/e/edmondc/.bash_profile
    export HDF5_USE_FILE_LOCKING=TRUE

    # run power spectrum / analytical window / analytical covariance:
    srun -n 4 python desipipe_data_png.py --interactive --blinded --todo spectrum --tracer QSO --region NGC SGC NGCnoN SGCnoDES GAL040 --zranges 0.8 3.5 1.6 3.5
    srun -n 4 python desipipe_data_png.py --interactive --todo spectrum --tracer LRG_zcmb QSO LRG_zcmbxQSO --region NGC SGC

    # run forward model of the window for IC contributions:
    srun -n 4 python desipipe_data_png.py --interactive --blinded --todo geo --ellsout 0 2 --tracer QSO
    srun -n 4 python desipipe_data_png.py --interactive --blinded --todo ric --ellsout 0 2 --tracer QSO
    srun -n 4 python desipipe_data_png.py --interactive --blinded --todo ric amr --ellsout 0 2 --tracer LRGxELGnotqso

    # run shotnoise template forward model:
    srun -n 4 python desipipe_data_png.py --interactive --blinded --todo sn --tracer QSO

    """
    from clustering_statistics import setup_logging, tools
    from mpi4py import MPI
    setup_logging(level=(logging.INFO if MPI.COMM_WORLD.rank == 0 else logging.ERROR))

    args = collect_argparser()


    mode = 'interactive' if args.interactive else 'slurm'
    if mode == 'interactive':
        logger.info('Running in interactive mode')
    else:         
        logger.info("Create queue with jobs inside using desipipe. Don\'t forget to run `desipipe spawn -q data_png --spawn` to launch the jobs!")
        tm, tm80 = setup_queue()


    def get_run_stats():
        if mode == 'interactive':
            return run_stats
        else: 
            return tm80.python_app(run_stats)


    cat_dir = Path('/global/cfs/cdirs/desi/survey/catalogs/DA2/LSS/loa-v1/LSScats/v2/fNL/')
    #cat_dir = Path('/pscratch/sd/e/edmondc/catalogs/DA2/LSS/loa-v1/LSScats/v2/fNL/')
    stats_dir = Path(os.getenv('SCRATCH', '.')) / 'desi-clustering/dr2/summary_statistics/local_png/base/desi-data/loa-v1/v2/fNL/'

    if args.blinded:
        cat_dir = cat_dir / 'blinded'
        stats_dir = stats_dir / 'blinded'
    else:
        logger.error('NOT READY FOR UNBLINDED DATA YET')
        import sys
        sys.exit(3)
        logger.warning('HERE WE ARE !!')
    
    logger.info(f'cat_dir: {cat_dir}')
    logger.info(f'stats_dir: {stats_dir}')


    if args.tracers is not None:
        tracers = [tuple(tt.split('x')) if 'x' in tt else tt for tt in args.tracers] 
    else:
        tracers = ['LRG']

    tracers_available = ['LRG', 'LRG_zcmb', 'ELGnotqso', 'ELGnotqso_zcmb', 'QSO', 'QSO_zcmb', ('LRG', 'QSO'), ('LRG', 'ELGnotqso'), ('ELGnotqso', 'QSO'), ('LRG_zcmb', 'QSO_zcmb'), ('LRG_zcmb', 'ELGnotqso_zcmb'), ('ELGnotqso_zcmb', 'QSO_zcmb')]
    assert (tracers not in tracers_available), f'Please use one available tracer: {[tt if not isinstance(tt, tuple) else 'x'.join(tt) for tt in tracers_available]}'

    regions = args.regions if args.regions is not None else ['NGC', 'SGC']

    for tracer in tracers:
        # if len(zranges) is odd, it will not consider the last one.
        zranges = tools.propose_fiducial(kind='zranges', tracer=tracer, analysis='local_png')[:1] if args.zranges is None else list(zip(args.zranges[::2], args.zranges[1::2]))

        weights = ['default-fkp-oqe', 'default-fkp'][:1]
        
        # Choice of imaging systematics avaialble in the catalogs: https://desi.lbl.gov/trac/wiki/keyprojects/Y3-DR/LSScat/imaging_systematics
        #weights = ['default-fkp-oqe-wsys-imlin_newzbin2']
        if tracer == 'LRG_zcmb':
            weights += ['default-fkp-oqe-wsys-imlin_finezbin_allebvcmb']
        elif tracer == ('LRG_zcmb', 'QSO'):
            weights += [('default-fkp-oqe-wsys-imlin_finezbin_allebvcmb', 'default-fkp-oqe')]
        elif tracer == 'ELGnotqso':
            weights += ['default-fkp-oqe-wsys-imlin', 'default-fkp-oqe-wsys-imlin_finezbin_nodebv'][:1]
        elif tracer in [('LRG', 'ELGnotqso'), ('LRG_zcmb', 'ELGnotqso')]:
            weights += [('default-fkp-oqe', 'default-fkp-oqe-wsys-imlin')]         
        elif tracer == ('ELGnotqso', 'QSO'):
            weights += [('default-fkp-oqe-wsys-imlin', 'default-fkp-oqe')]


        logger.info(f'{tracer=}, {zranges=}, {weights=}')

        # Compute power spectrum / analytical window matrix / analytical covariance?
        analytical = ('spectrum' in args.todo) or  ('window' in args.todo) or ('covariance' in args.todo)
        # Computing window forward model?
        fm_window = ('geo' in args.todo) or ('ric' in args.todo) or ('amr' in args.todo)
        # Compute shotnoise template forward model ?
        fm_shotnoise = ('sn' in args.todo)

        # Compute power spectrum, window matrix, analytical covariance: 
        if analytical:
            stats_translator = {'spectrum': 'mesh2_spectrum', 'window': 'window_mesh2_spectrum', 'covariance': 'covariance_mesh2_spectrum'}
            stats = [stats_translator[key] for key in args.todo]
            postprocess = ['combine_regions']
            logger.info(f'Running stats {stats} and postprocess {postprocess}')
            get_run_stats()(cat_dir=cat_dir, stats_dir=stats_dir, tracer=tracer, zranges=zranges, weights=weights, regions=regions, stats=stats)            
            if postprocess:
                postprocess_stats(cat_dir=cat_dir, stats_dir=stats_dir, tracer=tracer, zranges=zranges, weights=weights, postprocess=postprocess, stats=stats)

        # Forward model RIC/AMR correction to the window matrix:
        if fm_window:
            assert not (('amr' in args.todo) and not ('ric' in args.todo)), 'Do not run AMR without RIC. Provide --ric --amr.'

            stats = ['window_mesh2_spectrum_fm']
            postprocess = ['combine_window_mesh2_spectrum', 'combine_regions'] if 'ric' in args.todo else []
            logger.info(f'Running {stats=} and {postprocess=}')

            # Collect which region and redshift range are necessary to run the imaging systematic weights (need the full redshift range / footprint ..):
            total_regions, total_zranges = tools.propose_fiducial(kind='window_mesh2_spectrum_fm', tracer=tracer)['total_region_zrange']
            logger.info(f"Use region={total_regions} and zrange={total_zranges} for reading the catalogs.")

            # Remark: in case of cross-correlation, the catalogs will be read with the same zranges !  This will trigger a warning in desiwing:
            # For instance with LRGxQSO -> LRG will be with 0.8 < z < 1.3 -> warning because there are some objects with z > 1.1, 
            # that will not be used to compute imaging systematic weights.
            # This is not a problem because the power spectrum will be computed only with z < z_required=1.1 < 1.3

            # Update options for forward window:
            kwargs = {'geo': 'geo' in args.todo, 'ric': 'ric' in args.todo, 'amr': 'amr' in args.todo, 'ellsout': args.ellsout,
                      #'amr_regions_zranges':  
                      'spectrum_regions_zranges': list(itertools.product(regions, zranges))}
            logger.info(f"window_mesh2_spectrum_fm kwargs: {kwargs}")

            get_run_stats()(cat_dir=cat_dir, stats_dir=stats_dir, tracer=tracer, zranges=total_zranges, weights=weights, 
                            regions=[total_regions], stats=stats, **kwargs)

            if 'combine_window_mesh2_spectrum' in postprocess:
                postprocess_stats(cat_dir=cat_dir, stats_dir=stats_dir, tracer=tracer, zranges=zranges, weights=weights, 
                                postprocess=['combine_window_mesh2_spectrum'], stats=stats, regions=regions, **kwargs)
            
            if "combine_regions" in postprocess:
                kwargs['extra'] = 'RIC+AMR' if args.amr else 'RIC'
                postprocess_stats(cat_dir=cat_dir, stats_dir=stats_dir, tracer=tracer, zranges=zranges, weights=weights, 
                                postprocess=['combine_regions'], stats=['window_mesh2_spectrum'], **kwargs)

        # Forward model shotnoise template:
        elif fm_shotnoise:
            stats = ['shotnoise_mesh2_spectrum_fm']
            postprocess = ['combine_regions']
            logger.info(f'Running {stats=} and {postprocess=}')

            # Collect which region and redshift range are necessary to run the imaging systematic weights (need the full redshift range / footprint ..):
            total_regions, total_zranges = tools.propose_fiducial(kind='shotnoise_mesh2_spectrum_fm', tracer=tracer)['total_region_zrange']
            logger.info(f"Use region={total_regions} and zrange={total_zranges} for reading the catalogs.")

            # Update options for forward shotnoise:
            kwargs = {'spectrum_regions_zranges': list(itertools.product(regions, zranges))}
            logger.info(f"shotnoise_mesh2_spectrum_fm kwargs: {kwargs}")

            get_run_stats()(cat_dir=cat_dir, stats_dir=stats_dir, tracer=tracer, zranges=total_zranges, weights=weights, 
                            regions=[total_regions], stats=stats, **kwargs)

            if "combine_regions" in postprocess:
                for extra in ['geometry', 'RIC+AMR']:
                    kwargs['extra'] = extra
                    postprocess_stats(cat_dir=cat_dir, stats_dir=stats_dir, tracer=tracer, zranges=zranges, weights=weights, 
                                      postprocess=['combine_regions'], stats=['shotnoise_mesh2_spectrum_fm'], **kwargs)