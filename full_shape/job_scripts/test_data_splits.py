"""
Script to run fits with region splits tests
To create and spawn the tasks on NERSC, use the following commands:
```bash
salloc -N 1 -C gpu -t 04:00:00 --gpus 4 --qos interactive -t 04:00:00 --account desi_g
salloc -N 1 -n 4 -c 32 -C cpu --qos interactive -t 04:00:00 --account desi
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
export PYTHONPATH=/global/u1/s/shengyu/Y3/blinded_data_splits/desi-clustering:$PYTHONPATH
srun -n 4 python test_data_splits.py 
```
"""

import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from full_shape import tools, setup_logging
setup_logging()

THEORY_MODELS = ['folpsD', 'folpsEFT', 'reptvelocileptors', 'comet']
COSMO_MODELS = ['base', 'base_ns-fixed', 'fixed']
PRIOR_BASES = ['physical', 'physical_aap', 'tcm_chudaykin_aap', 'standard']
SAMPLERS = ['emcee', 'zeus', 'mhmcmc', 'nuts', 'pocomc', 'nautilus', 'numpyro_nuts', 'numpyro_barker']
FOLPSD_DAMPINGS = ['exp', 'lor', 'vdg']
FOLPSD_DAMPING_METHODS = ['none', 'tree', 'tree-gtns']
KRANGES = {
    'mesh2_spectrum': [
        {'ells': 0, 'k': [0.02, 0.35, 0.01]},
        {'ells': 2, 'k': [0.02, 0.25, 0.01]},
        # {'ells': 4, 'k': [0.02, 0.301, 0.01]},
    ],
    'mesh3_spectrum': [
        {'ells': (0, 0, 0), 'k': [0.02, 0.20, 0.01]},
        {'ells': (2, 0, 2), 'k': [0.02, 0.03, 0.01]},
    ],
}

COEFF_FN = Path("/global/cfs/cdirs/desi/users/shengyu/repeats/DA2/loa-v1/nz/corr_coeff_region.csv")
COV_STAT_DIR = Path('/global/cfs/cdirs/desi/science/cai/desi-clustering/dr2/summary_statistics/full_shape/data_splits')
USE_SCALE_COV = False
BASE_COV_REGIONS = {'GCcomb'}
FOLPSD_DAMPING = 'vdg'
FOLPSD_DAMPING_METHOD = 'tree'

def _validate_theory_model(stats, theory_model):
    if theory_model == 'reptvelocileptors' and 'mesh3_spectrum' in stats:
        raise ValueError('theory model reptvelocileptors is only supported with mesh2_spectrum')

def _get_covariance_scale(tracer, region, coeff_fn=COEFF_FN):
    with Path(coeff_fn).open(newline='') as stream:
        for row in csv.DictReader(stream):
            if row['name'] == tracer and row['region'] == region:
                return float(row['cov_fac'])
    raise ValueError(
        f'No covariance scaling factor found for tracer={tracer!r}, region={region!r} '
        f'in coefficient table {coeff_fn}.'
    )

def _apply_kranges(observable_options):
    stat = observable_options['stat']['kind']
    if stat not in KRANGES:
        raise ValueError(f'Unknown stat {stat} for applying k-range selections.')
    observable_options['stat']['select'] = [
        {'ells': item['ells'], 'k': list(item['k'])}
        for item in KRANGES[stat]
    ]

def _build_likelihoods_options(stats, tracers, regions, version, covariance, stats_dir, project,
                               theory_model, cov_stats_dir=None, prior_basis='physical_aap',
                               weight='default-FKP', cut=False, auw=False, coeff_fn=COEFF_FN,
                               use_scale_covariance=USE_SCALE_COV,
                               folpsd_damping=FOLPSD_DAMPING,
                               folpsd_damping_method=FOLPSD_DAMPING_METHOD):
    _validate_theory_model(stats, theory_model)
    folpsd_damping_method = None if folpsd_damping_method == 'none' else folpsd_damping_method
    likelihoods = []
    for tracer in tracers:
        for region in regions:
            region_cov_stats_dir = cov_stats_dir
            if region_cov_stats_dir is None:
                use_base_covariance = use_scale_covariance or region in BASE_COV_REGIONS
                region_cov_stats_dir = COV_STAT_DIR if use_base_covariance else stats_dir
            likelihood_options = tools.generate_likelihood_options_helper(
                stats=stats,
                tracer=tracer,
                region=region,
                version=version,
                covariance=covariance,
                stats_dir=stats_dir,
                project=project,
            )
            likelihood_options['covariance'].update({
                'version': covariance,
                'project': '.',
            })
            if use_scale_covariance:
                likelihood_options['covariance'].update({
                    'region': 'GCcomb',
                    'scale': _get_covariance_scale(tracer, region, coeff_fn=coeff_fn),
                })
            else:
                likelihood_options['covariance'].update({'region': region})
            likelihood_options['covariance']['stats_dir'] = region_cov_stats_dir
            if auw and 'mesh3_spectrum' in stats:
                likelihood_options['covariance'].setdefault('stat_options', {})
                likelihood_options['covariance']['stat_options'].setdefault('mesh3_spectrum', {})
                likelihood_options['covariance']['stat_options']['mesh3_spectrum']['auw'] = False
            for observable_options in likelihood_options['observables']:
                stat = observable_options['stat']['kind']
                observable_options['catalog']['weight'] = weight
                # Theta-cut variants are only available for mesh2_spectrum; AUW is available for both S2 and S3.
                observable_options['catalog']['cut'] = bool(cut) if stat == 'mesh2_spectrum' else False
                observable_options['catalog']['auw'] = bool(auw)
                _apply_kranges(observable_options)
                observable_options.setdefault('theory', {})
                observable_options['theory']['model'] = theory_model
                observable_options['theory']['prior_basis'] = prior_basis
                if theory_model == 'folpsD':
                    observable_options['theory']['damping'] = folpsd_damping
                    if stat == 'mesh2_spectrum':
                        observable_options['theory']['damping_method'] = folpsd_damping_method
            likelihoods.append(likelihood_options)
    return likelihoods

def _build_run_options(stats, tracers, regions, version, covariance, stats_dir, theory_model,
                       project='', cov_stats_dir=None, cosmo_model='base',
                       template='direct', sampler='emcee', nchains=1, resume=False,
                       prior_basis='physical_aap', weight='default-FKP', cut=False, auw=False, coeff_fn=COEFF_FN,
                       use_scale_covariance=USE_SCALE_COV,
                       folpsd_damping=FOLPSD_DAMPING,
                       folpsd_damping_method=FOLPSD_DAMPING_METHOD):
    options = {}
    options['likelihoods'] = _build_likelihoods_options(
        stats=stats,
        tracers=tracers,
        regions=regions,
        version=version,
        covariance=covariance,
        stats_dir=stats_dir,
        project=project,
        cov_stats_dir=cov_stats_dir,
        theory_model=theory_model,
        prior_basis=prior_basis,
        weight=weight,
        cut=cut,
        auw=auw,
        coeff_fn=coeff_fn,
        use_scale_covariance=use_scale_covariance,
        folpsd_damping=folpsd_damping,
        folpsd_damping_method=folpsd_damping_method,
    )
    options['cosmology'] = {'template': template, 'model': cosmo_model,
                            'engine': 'eisenstein_hu' if 'comet' in theory_model else 'class'}
    options['sampler'] = tools.propose_fiducial_sampler_options(sampler=sampler)
    options['sampler']['nchains'] = nchains
    options['sampler']['resume'] = resume
    for section in ['init', 'run']:
        if 'nparallel' in options['sampler'][section]:
            options['sampler'][section]['nparallel'] = nchains
    return tools.fill_fiducial_options(options)

def _get_chain_fns(options, fits_dir, project):
    return [
        tools.get_fits_fn(kind='chain', fits_dir=fits_dir, project=project, **options, ichain=ichain)
        for ichain in range(options['sampler']['nchains'])
    ]

def _chain_exists(fn):
    fn = Path(fn)
    return fn.exists() and fn.stat().st_size > 0

def run_fit(actions=('profile',), template='direct',
            stats=['mesh2_spectrum'],
            version=None, cut=False, auw=False,
            tracers=None, regions=None, weight='default-FKP',
            stats_dir=None, fits_dir=None, project=None,
            covariance='holi-v3-altmtl', cov_stats_dir=None, cache_dir=None,
            theory_model='folpsD', cosmo_model='base', prior_basis='physical_aap',
            sampler='emcee', nchains=1,
            resume=False, coeff_fn=COEFF_FN, use_scale_covariance=USE_SCALE_COV,
            folpsd_damping=FOLPSD_DAMPING,
            folpsd_damping_method=FOLPSD_DAMPING_METHOD):
    # Everything inside this function will be executed on the compute nodes;
    # This function must be self-contained; and cannot rely on imports from the outer scope.
    import os
    from pathlib import Path
    import functools
    from mpi4py import MPI
    mpicomm = MPI.COMM_WORLD
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', str(mpicomm.rank))
    import jax
    from jax import config
    config.update('jax_enable_x64', True)
    print(f'[rank {mpicomm.rank}] host={os.uname().nodename} CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")} jax_devices={jax.devices()}', flush=True)
    from full_shape import run_fit_from_options, setup_logging
    from full_shape.tools import get_likelihood
    from desilike.samples import Profiles
    # You can pass region, version, covariance, ...
    options = _build_run_options(
        stats=stats,
        tracers=tracers,
        regions=regions,
        version=version,
        weight=weight,
        cut=cut,
        auw=auw,
        covariance=covariance,
        stats_dir=stats_dir,
        project=project,
        cov_stats_dir=cov_stats_dir,
        theory_model=theory_model,
        cosmo_model=cosmo_model,
        template=template,
        sampler=sampler,
        nchains=nchains,
        resume=resume,
        prior_basis=prior_basis,
        coeff_fn=coeff_fn,
        use_scale_covariance=use_scale_covariance,
        folpsd_damping=folpsd_damping,
        folpsd_damping_method=folpsd_damping_method,
    )
    get_fits_fn = functools.partial(tools.get_fits_fn, fits_dir=fits_dir, project=project)
    run_fit_from_options(actions, **options, get_fits_fn=get_fits_fn, cache_dir=cache_dir if cache_dir else fits_dir / '_cache')
    return 0
    if 'profile' in actions:
        likelihood = get_likelihood(likelihoods_options=options['likelihoods'],
                                    cosmology_options=options['cosmology'],
                                    cache_dir=cache_dir if cache_dir else fits_dir / '_cache')
        profiles = Profiles.load(get_fits_fn(kind='profiles', **options))
        likelihood(**profiles.bestfit.choice(input=True, index='argmax'))
        if mpicomm.rank == 0:
            plot_dir = get_fits_fn(kind='profiles', **options).parent
            for ilikelihood, sublikelihood in enumerate(likelihood.likelihoods):
                for iobservable, observable in enumerate(sublikelihood.observables):
                    plot_covariance = sublikelihood.covariance.at.observable.get(observables=observable.name)
                    plot_covariance = plot_covariance.at.observable.match(observable.data)
                    observable.covariance = plot_covariance
                    observable.plot(fn=plot_dir / f'plot_likelihood{ilikelihood}_observable{iobservable}.png')

########################################################################################################################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, default='data-dr2-v2', help='Blinded dataset to fit.')
    parser.add_argument('--todo', type=str, nargs='*', default=['profile'], choices=['build', 'profile', 'sample'],
                        help='Run build, profile, and / or sample. Defaults to profile.')
    parser.add_argument('--stats', type=str, nargs='*', default=['mesh2_spectrum', 'mesh3_spectrum'],
                        choices=['mesh2_spectrum', 'mesh3_spectrum'], help='Statistics to fit. Defaults to mesh2_spectrum.')
    parser.add_argument('--theory_model', type=str, default='folpsD',
                        choices=THEORY_MODELS, help='Theory model to fit. Defaults to folpsD.')
    parser.add_argument('--prior_basis', type=str, default='physical_aap',
                        choices=PRIOR_BASES, help='Nuisance-parameter prior basis. Defaults to physical_aap.')
    parser.add_argument('--folpsd_damping', type=str, default=FOLPSD_DAMPING,
                        choices=FOLPSD_DAMPINGS,
                        help=f'FoG damping kernel for folpsD fits. Defaults to {FOLPSD_DAMPING}.')
    parser.add_argument('--folpsd_damping_method', type=str, default=FOLPSD_DAMPING_METHOD,
                        choices=FOLPSD_DAMPING_METHODS,
                        help=f"FoG damping method for folpsD mesh2 fits. Use 'none' for desilike's None mode. Defaults to {FOLPSD_DAMPING_METHOD}.")
    parser.add_argument('--cosmo_params', type=str, default='base_ns-fixed',
                        choices=COSMO_MODELS,
                        help='Cosmology parameter setup to fit. base varies h, omega_cdm, omega_b, logA, n_s; '
                             'base_ns-fixed varies h, omega_cdm, omega_b, logA; '
                             'fixed varies only nuisance parameters. Defaults to base.')
    parser.add_argument('--sampler', type=str, default='pocomc',
                        choices=SAMPLERS, help='desilike sampler backend to use. Defaults to pocomc.')
    parser.add_argument('--fit_tracers', nargs='+', default=['LRG1'],
                        help='Tracers to fit. Defaults to LRG1.')
    parser.add_argument('--fit_regions', nargs='+', default=['GCcomb'],
                        help='Sky regions to include. Defaults to GCcomb.',
                        choices=['GCcomb', 'NS', 'GCcomb_noN', 'GCcomb_noDES', 'NGC', 'SGC', 'N', 'NGCnoN', 'S', 'SGCnoDES', 'ACT_DR6', 'PLANCK_PR4', 'GAL040', 'GAL060'])
    parser.add_argument('--weight', type=str, default='default-FKP',
                        help='Measurement weight type to read, e.g. default-FKP.')
    parser.add_argument('--thetacut', action='store_true',
                        help='Read theta-cut measurement variants with the _thetacut suffix.')
    parser.add_argument('--auw', action='store_true',
                        help='Read angular-upweighted measurement variants with the _auw suffix.')
    parser.add_argument('--fits_dir', type=str, default=None,
                        help='Base directory for fits results.')
    parser.add_argument('--cache_dir', type=str, default=None,
                        help='Base directory for cached prepared stats and emulators.')
    parser.add_argument('--project', type=str, default='',
                        help='Measurement project subdirectory under stats_dir.')
    parser.add_argument('--covariance', type=str, default='holi-v3-altmtl',
                        help='Covariance mock version. Defaults to holi-v3-altmtl.')
    parser.add_argument('--cov_stats_dir', type=str, default=None,
                        help='Directory for covariance summary statistics. By default, use full_shape/base for scaled/base-region covariances and stats_dir for direct split-region covariances.')
    parser.add_argument('--nchains', type=int, default=1,
                        help='Number of MCMC chains to run with desilike. Defaults to 1.')
    parser.add_argument('--resume', action='store_true',
                        help='Resume sampling from existing chain files in the derived fits directory.')
    parser.add_argument('--use_rsf', action='store_true',
                        help='Use the region scale factor table with GCcomb covariance.')
    args = parser.parse_args()

    check_for_existing_measurements = False

    stats_dir = Path('/global/cfs/cdirs/desi/science/cai/desi-clustering/dr2/summary_statistics/full_shape/data_splits')
    # fits_dir = Path('/global/cfs/cdirs/desi/science/cai/desi-clustering/dr2/fits/')
    fits_dir = Path(args.fits_dir) if args.fits_dir is not None else Path(os.getenv('SCRATCH')) / 'test'
    cache_dir = Path(args.cache_dir) if args.cache_dir is not None else fits_dir / '_cache'
    project = args.project or 'blinded_data'

    stats = args.stats
    version = args.version

    fit_tracers = args.fit_tracers
    fit_regions = args.fit_regions
    # fit_regions = ['GCcomb', 'GCcomb_noN', 'GCcomb_noDES'][:]
    # fit_regions += ['NGC', 'SGC', 'N', 'NGCnoN', 'SGCnoDES']
    # fit_regions += ['ACT_DR6', 'PLANCK_PR4'] + [f'GAL0{i}' for i in [40, 60]] #lensing regions

    covariance = args.covariance
    cov_stats_dir = Path(args.cov_stats_dir) if args.cov_stats_dir is not None else None
    _validate_theory_model(stats, args.theory_model)

    for tracer in fit_tracers:
        for region in fit_regions:
            if check_for_existing_measurements and 'sample' in args.todo:
                options = _build_run_options(
                    stats=stats,
                    tracers=[tracer],
                    regions=[region],
                    version=version,
                    weight=args.weight,
                    cut=args.thetacut,
                    auw=args.auw,
                    covariance=covariance,
                    stats_dir=stats_dir,
                    project=project,
                    cov_stats_dir=cov_stats_dir,
                    theory_model=args.theory_model,
                    cosmo_model=args.cosmo_params,
                    sampler=args.sampler,
                    nchains=args.nchains,
                    resume=args.resume,
                    prior_basis=args.prior_basis,
                    use_scale_covariance=args.use_rsf,
                    folpsd_damping=args.folpsd_damping,
                    folpsd_damping_method=args.folpsd_damping_method,
                )
                chain_fns = _get_chain_fns(options, fits_dir=fits_dir, project=project)
                if all(_chain_exists(fn) for fn in chain_fns):
                    print(f'Skipping {tracer} {region}: existing chain file(s) found:', flush=True)
                    for fn in chain_fns:
                        print(f'  {fn}', flush=True)
                    continue
            run_fit(actions=args.todo, stats=stats, #what to run
                    version=version, tracers=[tracer], regions=[region], # dataset settings
                    stats_dir=stats_dir, fits_dir=fits_dir, project=project, # IO settings
                    covariance=covariance, cov_stats_dir=cov_stats_dir, cache_dir=cache_dir,
                    theory_model=args.theory_model, # fitting model settings
                    cosmo_model=args.cosmo_params, sampler=args.sampler, nchains=args.nchains, # sampling settings
                    resume=args.resume, prior_basis=args.prior_basis,
                    weight=args.weight, cut=args.thetacut, auw=args.auw,
                    use_scale_covariance=args.use_rsf,
                    folpsd_damping=args.folpsd_damping,
                    folpsd_damping_method=args.folpsd_damping_method)
