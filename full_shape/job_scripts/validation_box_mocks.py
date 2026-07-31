"""
Script to run full-shape fits with cubic box mocks.

Examples
--------
python validation_box_mocks.py
python validation_box_mocks.py --tracer LRG --version abacus-2ndgen --zsnap 0.8 --stats mesh2_spectrum
"""
import argparse
import os
from pathlib import Path

from full_shape import setup_logging


setup_logging()

THEORY_MODELS = ['folpsD', 'folpsEFT', 'reptvelocileptors', 'comet']
COSMO_MODELS = ['base', 'base_ns-fixed', 'fixed']
PRIOR_BASES = ['physical', 'physical_aap', 'tcm_chudaykin_aap', 'standard']
SAMPLERS = ['emcee', 'zeus', 'mhmcmc', 'nuts', 'pocomc', 'nautilus', 'numpyro_nuts', 'numpyro_barker']
FOLPSD_DAMPINGS = ['exp', 'lor', 'vdg']
FOLPSD_DAMPING_METHODS = [
    'none', 'loop+ctr', 'tree+loop', 'tree+loop+ctr', 'tree+loop+ctr+sn', 'all',
]
BOX_STATS_DIR = Path('/global/cfs/cdirs/desicollab/science/cai/desi-clustering/dr2/summary_statistics/box')
EZMOCK_STATS_DIR = Path('/dvs_ro/cfs/cdirs/desi/science/cai/desi-clustering/dr2/summary_statistics/mock_challenge/ezmock')
FITS_DIR = Path(os.getenv('SCRATCH', '.')) / 'fits_box_mocks'
CACHE_DIR = Path(os.getenv('SCRATCH', '.')) / 'desi-clustering/full_shape/job_scripts/_cache'
TRACER = 'QSO_lorentzian'
VERSION = 'abacus-hf-v2'
ZSNAP = 1.475
STATS = ('mesh2_spectrum', 'mesh3_spectrum')
FOLPSD_DAMPING = 'vdg'
FOLPSD_DAMPING_METHOD = 'tree+loop'
GELMAN_RUBIN = 1.03
ESS = 700
KRANGES = {
    'mesh2_spectrum': [
        {'ells': 0, 'k': [0.02, 0.30, 0.01]},
        {'ells': 2, 'k': [0.02, 0.30, 0.01]},
    ],
    'mesh3_spectrum': [
        {'ells': (0, 0, 0), 'k': [0.02, 0.10, 0.01]},
        # {'ells': (2, 0, 2), 'k': [0.02, 0.03, 0.01]},
    ],
}


def _get_kranges():
    return {
        stat: [
            {'ells': item['ells'], 'k': list(item['k'])}
            for item in selects
        ]
        for stat, selects in KRANGES.items()
    }


def _parse_imocks(value):
    if value in [None, 'all', '*']:
        return '*'
    if isinstance(value, str) and ',' in value:
        return [int(item) for item in value.split(',') if item]
    return int(value) if isinstance(value, str) else value


def _build_likelihood_options(stats=STATS, tracer=TRACER, version=VERSION, zsnap=ZSNAP,
                              imocks='all', los=None, hod=None, cosmo=None,
                              stats_dir=BOX_STATS_DIR, covariance_stats_dir=EZMOCK_STATS_DIR,
                              theory_model='folpsD', prior_basis='physical_aap', emulator=None,
                              folpsd_damping=FOLPSD_DAMPING,
                              folpsd_damping_method=FOLPSD_DAMPING_METHOD):
    from clustering_statistics import box_tools as clustering_box_tools
    from full_shape.box_tools import (generate_box_likelihood_options_helper,
                                      get_default_box_mesh3_basis,
                                      get_lsstypes_covariance_defaults)

    stats = (stats,) if isinstance(stats, str) else tuple(stats)
    folpsd_damping_method = None if folpsd_damping_method == 'none' else folpsd_damping_method
    emulator = theory_model != 'comet' if emulator is None else emulator
    catalog_defaults = clustering_box_tools.propose_box_fiducial('catalog', tracer=tracer, version=version)
    los = catalog_defaults['los'] if los is None else los
    hod = catalog_defaults['hod'] if hod is None else hod
    cosmo = catalog_defaults['cosmo'] if cosmo is None else cosmo
    stat_options = (
        {'mesh3_spectrum': {'basis': get_default_box_mesh3_basis(version)}}
        if 'mesh3_spectrum' in stats else {}
    )
    covariance_defaults = get_lsstypes_covariance_defaults(tracer, stats=stats)
    likelihood_options = generate_box_likelihood_options_helper(
        tracer=tracer, zsnap=zsnap, stats=stats,
        cosmo=cosmo, hod=hod, los=los,
        version=version, imocks=_parse_imocks(imocks),
        stats_dir=Path(stats_dir),
        selects=_get_kranges(),
        stat_options=stat_options,
        window_mode='none',
        covariance_stats_dir=Path(covariance_stats_dir),
        covariance_version=None,
        covariance_tracer=covariance_defaults['tracer'],
        covariance_zsnap=covariance_defaults['zsnap'],
        covariance_imocks=covariance_defaults['imock'],
        covariance_stat_options=covariance_defaults.get('stat_options', {}),
        covariance_volume_rescaling={
            'enabled': True,
            'source_boxsize_gpch': 6.0,
            'target_boxsize_gpch': 2.0,
        },
        emulator=emulator,
    )
    for observable_options in likelihood_options['observables']:
        observable_options.setdefault('theory', {})
        observable_options['theory']['model'] = theory_model
        observable_options['theory']['prior_basis'] = prior_basis
        if theory_model == 'folpsD':
            observable_options['theory']['damping'] = folpsd_damping
            if 'mesh2_spectrum' in observable_options['stat']['kind']:
                observable_options['theory']['damping_method'] = folpsd_damping_method
    return likelihood_options


def _build_run_options(stats=STATS, tracer=TRACER, version=VERSION, zsnap=ZSNAP,
                       imocks='all', los=None, hod=None, cosmo=None,
                       stats_dir=BOX_STATS_DIR, covariance_stats_dir=EZMOCK_STATS_DIR,
                       theory_model='folpsD', prior_basis='physical_aap', cosmo_model='base',
                       template='direct', sampler='emcee', nchains=1, resume=False, emulator=None,
                       gelman_rubin=GELMAN_RUBIN, ess=ESS,
                       folpsd_damping=FOLPSD_DAMPING,
                       folpsd_damping_method=FOLPSD_DAMPING_METHOD):
    from full_shape import tools

    likelihood_options = _build_likelihood_options(
        stats=stats, tracer=tracer, version=version, zsnap=zsnap,
        imocks=imocks, los=los, hod=hod, cosmo=cosmo,
        stats_dir=stats_dir, covariance_stats_dir=covariance_stats_dir,
        theory_model=theory_model, prior_basis=prior_basis, emulator=emulator,
        folpsd_damping=folpsd_damping, folpsd_damping_method=folpsd_damping_method,
    )
    options = {
        'likelihoods': [likelihood_options],
        'cosmology': {
            'template': template,
            'model': cosmo_model,
            'engine': 'eisenstein_hu' if 'comet' in theory_model else 'class',
        },
        'sampler': {
            'sampler': sampler,
            'nchains': nchains,
            'resume': resume,
        },
    }
    options = tools.fill_fiducial_options(options)
    if gelman_rubin is not None and 'gelman_rubin' in options['sampler']['run']:
        options['sampler']['run']['gelman_rubin'] = gelman_rubin
    if ess is not None and 'ess' in options['sampler']['run']:
        options['sampler']['run']['ess'] = ess
    for section in ['init', 'run']:
        if 'nparallel' in options['sampler'][section]:
            options['sampler'][section]['nparallel'] = nchains
    return options


def _plot_profile_best_fit(options, get_stats_fn, get_fits_fn, cache_dir, mpicomm):
    from desilike import compile
    from desilike.samples import Profiles
    from full_shape.tools import get_likelihood

    likelihood = get_likelihood(
        likelihoods_options=options['likelihoods'],
        cosmology_options=options['cosmology'],
        get_stats_fn=get_stats_fn,
        cache_dir=cache_dir,
    )
    fn = get_fits_fn(kind='profiles', **options)
    profiles = Profiles.read(fn)
    best = profiles.choice(index='argmax', squeeze=True).select(input=True).best
    compile(likelihood)(**best)
    if mpicomm.rank == 0:
        plot_dir = fn.parent
        for ilikelihood, sublikelihood in enumerate(likelihood.likelihoods):
            for iobservable, observable in enumerate(sublikelihood.observables):
                observable.plot(fn=plot_dir / f'plot_likelihood{ilikelihood}_observable{iobservable}.png')


def _get_parser():
    parser = argparse.ArgumentParser(description='Run full-shape fits with cubic box mocks.')
    parser.add_argument('--tracer', default=TRACER)
    parser.add_argument('--version', default=VERSION)
    parser.add_argument('--zsnap', type=float, default=ZSNAP)
    parser.add_argument('--todo', type=str, nargs='*', default=['profile'],
                        choices=['build', 'profile', 'sample'],
                        help='Run build, profile, and / or sample. Defaults to profile.')
    parser.add_argument('--actions', dest='todo', type=str, nargs='*',
                        choices=['build', 'profile', 'sample'],
                        default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument('--stats', type=str, nargs='*', default=list(STATS),
                        choices=['mesh2_spectrum', 'mesh3_spectrum'],
                        help='Statistics to fit. Defaults to mesh2_spectrum mesh3_spectrum.')
    parser.add_argument('--theory_model', type=str, default='folpsD',
                        choices=THEORY_MODELS,
                        help='Theory model to fit. Defaults to folpsD.')
    parser.add_argument('--prior_basis', type=str, default='physical_aap',
                        choices=PRIOR_BASES,
                        help='Nuisance-parameter prior basis. Defaults to physical_aap.')
    parser.add_argument('--folpsd_damping', type=str, default=FOLPSD_DAMPING,
                        choices=FOLPSD_DAMPINGS,
                        help=f'FoG damping kernel for folpsD fits. Defaults to {FOLPSD_DAMPING}.')
    parser.add_argument('--folpsd_damping_method', type=str, default=FOLPSD_DAMPING_METHOD,
                        choices=FOLPSD_DAMPING_METHODS,
                        help=f"FoG damping method for folpsD mesh2 fits. Use 'none' for desilike's None mode. Defaults to {FOLPSD_DAMPING_METHOD}.")
    parser.add_argument('--cosmo_params', type=str, default='base',
                        choices=COSMO_MODELS,
                        help='Cosmology parameter setup to fit. Defaults to base.')
    parser.add_argument('--sampler', type=str, default='emcee',
                        choices=SAMPLERS,
                        help='desilike sampler backend to use. Defaults to emcee.')
    parser.add_argument('--imocks', default='all', help="Data phases: 'all', '*', one integer, or comma-separated integers.")
    parser.add_argument('--los', default=None)
    parser.add_argument('--hod', default=None)
    parser.add_argument('--cosmo', default=None)
    parser.add_argument('--stats_dir', type=Path, default=BOX_STATS_DIR,
                        help=f'Base directory for box clustering statistics. Defaults to {BOX_STATS_DIR}.')
    parser.add_argument('--box-stats-dir', dest='stats_dir', type=Path,
                        default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument('--covariance_stats_dir', type=Path, default=EZMOCK_STATS_DIR,
                        help=f'Base directory for lsstypes EZmock covariance statistics. Defaults to {EZMOCK_STATS_DIR}.')
    parser.add_argument('--covariance-stats-dir', dest='covariance_stats_dir', type=Path,
                        default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument('--fits_dir', type=Path, default=FITS_DIR,
                        help=f'Base directory for fits. Defaults to {FITS_DIR}.')
    parser.add_argument('--fits-dir', dest='fits_dir', type=Path,
                        default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument('--cache_dir', type=Path, default=CACHE_DIR,
                        help=f'Base directory for cached prepared stats and emulators. Defaults to {CACHE_DIR}.')
    parser.add_argument('--template', default='direct')
    parser.add_argument('--nchains', type=int, default=1,
                        help='Number of MCMC chains to run with desilike. Defaults to 1.')
    parser.add_argument('--gelman_rubin', type=float, default=GELMAN_RUBIN,
                        help=f'Gelman-Rubin convergence threshold for MCMC samplers. Defaults to {GELMAN_RUBIN}.')
    parser.add_argument('--ess', type=float, default=ESS,
                        help=f'Effective sample size convergence threshold for MCMC samplers. Defaults to {ESS}.')
    parser.add_argument('--resume', action='store_true',
                        help='Resume sampling from existing chain files in the derived fits directory.')
    return parser


def run_fit(actions=('profile',), template='direct', fits_dir=FITS_DIR,
            stats=STATS, tracer=TRACER, version=VERSION, zsnap=ZSNAP,
            imocks='all', los=None, hod=None, cosmo=None,
            stats_dir=BOX_STATS_DIR, covariance_stats_dir=EZMOCK_STATS_DIR,
            cache_dir=CACHE_DIR, theory_model='folpsD', prior_basis='physical_aap',
            cosmo_model='base', sampler='emcee', nchains=1, resume=False,
            emulator=None, local_safe_threads=False, gelman_rubin=GELMAN_RUBIN, ess=ESS,
            folpsd_damping=FOLPSD_DAMPING,
            folpsd_damping_method=FOLPSD_DAMPING_METHOD):
    # Everything inside this function will be executed on the compute nodes;
    # this function must be self-contained and cannot rely on imports from the outer scope.
    import os
    import functools
    from pathlib import Path
    if local_safe_threads:
        for name, value in {
            'OMP_NUM_THREADS': '1',
            'OPENBLAS_NUM_THREADS': '1',
            'VECLIB_MAXIMUM_THREADS': '1',
        }.items():
            os.environ.setdefault(name, value)
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'
    from desilike import distributed
    distributed.initialize()
    mpicomm = distributed.get_mpicomm()
    from jax import config
    config.update('jax_enable_x64', True)
    from clustering_statistics import box_tools as clustering_box_tools
    from full_shape import run_fit_from_options, setup_logging
    from full_shape.job_scripts.validation_box_mocks import _build_run_options, _plot_profile_best_fit
    from full_shape.tools import get_fits_fn
    setup_logging()
    options = _build_run_options(
        stats=stats, tracer=tracer, version=version, zsnap=zsnap,
        imocks=imocks, los=los, hod=hod, cosmo=cosmo,
        stats_dir=stats_dir, covariance_stats_dir=covariance_stats_dir,
        theory_model=theory_model, prior_basis=prior_basis, cosmo_model=cosmo_model,
        template=template, sampler=sampler, nchains=nchains, resume=resume,
        emulator=emulator, gelman_rubin=gelman_rubin, ess=ess,
        folpsd_damping=folpsd_damping, folpsd_damping_method=folpsd_damping_method,
    )
    get_fits_fn = functools.partial(get_fits_fn, fits_dir=fits_dir)
    cache_dir = Path(cache_dir)
    run_fit_from_options(
        actions, **options,
        get_stats_fn=clustering_box_tools.get_box_stats_fn,
        get_fits_fn=get_fits_fn,
        cache_dir=cache_dir,
    )
    if 'profile' in actions:
        _plot_profile_best_fit(
            options,
            get_stats_fn=clustering_box_tools.get_box_stats_fn,
            get_fits_fn=get_fits_fn,
            cache_dir=cache_dir,
            mpicomm=mpicomm,
        )


if __name__ == '__main__':
    args = _get_parser().parse_args()
    run_fit(
        actions=args.todo, template=args.template, fits_dir=args.fits_dir,
        stats=args.stats, tracer=args.tracer, version=args.version, zsnap=args.zsnap,
        imocks=args.imocks, los=args.los, hod=args.hod, cosmo=args.cosmo,
        stats_dir=args.stats_dir, covariance_stats_dir=args.covariance_stats_dir,
        cache_dir=args.cache_dir, theory_model=args.theory_model, prior_basis=args.prior_basis,
        cosmo_model=args.cosmo_params, sampler=args.sampler, nchains=args.nchains,
        resume=args.resume, emulator=False if args.no_emulator else None,
        local_safe_threads=args.local_safe_threads, gelman_rubin=args.gelman_rubin, ess=args.ess,
        folpsd_damping=args.folpsd_damping,
        folpsd_damping_method=args.folpsd_damping_method,
    )
