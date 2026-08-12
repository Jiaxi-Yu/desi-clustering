"""Run full-shape fits to the mean Abacus lightcone validation vector.

The data vector is the mean of all realizations in
``full_shape/lightcone_validation/abacus-hf-lc-dr2-v1.9``.  The covariance is
estimated independently from the ``full_shape/base/holi-v3-altmtl`` mocks.

Example NERSC commands::

    salloc -N 1 -C gpu -t 02:00:00 --gpus 4 --qos interactive --account desi_g
    source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
    python validation_abacus_lightcone_mocks.py --stats mesh2_spectrum mesh3_spectrum --todo build
    srun -n 4 python validation_abacus_lightcone_mocks.py \
        --stats mesh2_spectrum mesh3_spectrum --todo sample --nchains 4
"""

import os
from pathlib import Path

from full_shape.job_scripts import validation_abacus_mocks as validation


VERSION = 'abacus-hf-lc-dr2-v1.9'
COVARIANCE = 'holi-v3-altmtl'
STATS_DIR = Path('/global/cfs/cdirs/desi/science/cai/desi-clustering/dr2/summary_statistics')
PROJECT = 'full_shape/lightcone_validation'
COVARIANCE_PROJECT = 'full_shape/base'
DATA_REGION = 'ALL'
COVARIANCE_REGION = 'GCcomb'
TRACERS = ['QSO1']


def _get_parser():
    """Return the standard Abacus parser with lightcone-specific defaults."""
    parser = validation._get_parser()
    actions = {action.dest: action for action in parser._actions}

    actions['dataset'].choices = [VERSION]
    actions['dataset'].default = VERSION
    actions['dataset'].help = f'Lightcone dataset to fit. Defaults to {VERSION}.'

    actions['tracers'].choices = TRACERS
    actions['tracers'].help = 'Tracer(s) to fit. The current lightcone products support QSO1.'

    actions['stats_dir'].default = STATS_DIR
    actions['stats_dir'].help = f'Base directory for clustering statistics. Defaults to {STATS_DIR}.'

    actions['project'].default = PROJECT
    actions['project'].help = f'Data-vector project directory. Defaults to {PROJECT}.'

    actions['fits_dir'].help = (
        'Base directory for fits. Defaults to $SCRATCH/fits_abacus_lightcone_mocks '
        'or ./fits_abacus_lightcone_mocks.'
    )
    return parser


def _build_run_options(stats=('mesh2_spectrum',), tracers=TRACERS, **kwargs):
    """Build options with lightcone data and Holi covariance kept separate."""
    defaults = {
        'version': VERSION,
        'covariance': COVARIANCE,
        'stats_dir': STATS_DIR,
        'project': PROJECT,
        'theory_model': 'folpsD',
    }
    defaults.update(kwargs)
    options = validation._build_run_options(
        stats=list(stats), tracers=list(tracers), **defaults,
    )
    for likelihood_options in options['likelihoods']:
        for observable_options in likelihood_options['observables']:
            observable_options['catalog']['region'] = DATA_REGION
            observable_options['catalog']['imock'] = '*'
        likelihood_options['covariance'].update({
            'source': 'mock',
            'project': COVARIANCE_PROJECT,
            'region': COVARIANCE_REGION,
        })
    return options


def _run_fit(actions, options, fits_dir, cache_dir, local_safe_threads=False):
    """Run prebuilt lightcone options without rebuilding them in the shared script."""
    import functools

    if local_safe_threads:
        validation._apply_local_safe_threads()
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'

    from desilike import compile, distributed
    from desilike.samples import Profiles
    from jax import config

    from full_shape import run_fit_from_options
    from full_shape import tools

    try:
        distributed.initialize()
    except RuntimeError:
        print('Distributed environment already initialized')
    else:
        print('Initializing distributed environment')
    mpicomm = distributed.get_mpicomm()
    config.update('jax_enable_x64', True)

    get_fits_fn = functools.partial(tools.get_fits_fn, fits_dir=Path(fits_dir))
    cache_dir = Path(cache_dir)
    run_fit_from_options(
        actions, **options, get_fits_fn=get_fits_fn, cache_dir=cache_dir,
    )

    if 'profile' in actions:
        likelihood = tools.get_likelihood(
            likelihoods_options=options['likelihoods'],
            cosmology_options=options['cosmology'],
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
                    observable.plot(
                        fn=plot_dir / f'plot_likelihood{ilikelihood}_observable{iobservable}.png'
                    )


if __name__ == '__main__':
    args = _get_parser().parse_args()

    base_fits_dir = (
        Path(args.fits_dir) if args.fits_dir is not None
        else Path(os.getenv('SCRATCH', '.')) / 'fits_abacus_lightcone_mocks'
    )
    fits_dir = base_fits_dir / args.dataset
    stats = args.stats
    tracers = args.tracers or TRACERS
    kmax_overrides = validation._parse_kmax_overrides(args.kmax)
    validation._validate_theory_model(stats, args.theory_model)

    options = _build_run_options(
        version=args.dataset,
        covariance=COVARIANCE,
        stats_dir=Path(args.stats_dir),
        project=args.project,
        stats=stats,
        tracers=tracers,
        theory_model=args.theory_model,
        cosmo_model=args.cosmo_params,
        sampler=args.sampler,
        nchains=args.nchains,
        resume=args.resume,
        prior_basis=args.prior_basis,
        gelman_rubin=args.gelman_rubin,
        ess=args.ess,
        profile_iterations=args.profile_iterations,
        folpsd_damping=args.folpsd_damping,
        folpsd_damping_method=args.folpsd_damping_method,
        emulator=not args.no_emulator,
        kmax_overrides=kmax_overrides,
        mesh3_theory_dk=args.mesh3_theory_dk,
    )
    _run_fit(
        actions=args.todo,
        options=options,
        fits_dir=fits_dir,
        cache_dir=Path(args.cache_dir),
        local_safe_threads=args.local_safe_threads,
    )
