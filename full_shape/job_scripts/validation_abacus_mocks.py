"""
Script to run fits with Abacus mocks.
To create and spawn the tasks on NERSC, use the following commands:
```bash
salloc -N 1 -C gpu -t 02:00:00 --gpus 4 --qos interactive --account desi_g
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
python validation_abacus_mocks.py --dataset abacus-2ndgen-dr2-complete --tracers LRG1 --stats mesh2_spectrum mesh3_spectrum --todo build
srun -n 4 python validation_abacus_mocks.py --dataset abacus-2ndgen-dr2-complete --tracers LRG1 LRG2 LRG3 ELG2 --stats mesh2_spectrum mesh3_spectrum--todo sample --nchains 4
```
"""
import argparse
import os
from pathlib import Path

import numpy as np

from full_shape import tools, setup_logging


setup_logging()


THEORY_MODELS = ['folpsD', 'folpsEFT', 'reptvelocileptors', 'comet']
COSMO_MODELS = ['base', 'base_ns-fixed', 'fixed']
PRIOR_BASES = ['physical', 'physical_aap', 'tcm_chudaykin_aap', 'standard']
SAMPLERS = ['emcee', 'zeus', 'mhmcmc', 'nuts', 'pocomc', 'nautilus', 'numpyro_nuts', 'numpyro_barker']
FOLPSD_DAMPINGS = ['exp', 'lor', 'vdg']
FOLPSD_DAMPING_METHODS = [
    'none', 'loop+ctr', 'tree+loop', 'tree+loop+ctr', 'tree+loop+ctr+sn', 'all',
]
FOLPSD_DAMPING = 'vdg'
FOLPSD_DAMPING_METHOD = 'tree+loop'
GELMAN_RUBIN = 1.03
ESS = 700
PROFILE_ITERATIONS = 4
DEFAULT_STATS_DIR = Path('/global/cfs/cdirs/desicollab/science/cai/desi-clustering/dr2/summary_statistics')
DEFAULT_PROJECT = 'full_shape/base'
#DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / '_cache'
DEFAULT_CACHE_DIR = Path(os.environ['SCRATCH']) / 'desi-clustering/full_shape/job_scripts/_cache'

LOCAL_SAFE_THREAD_ENV = {
    'OMP_NUM_THREADS': '1',
    'OPENBLAS_NUM_THREADS': '1',
    'VECLIB_MAXIMUM_THREADS': '1',
}
KRANGES = {
    'mesh2_spectrum': [
        {'ells': 0, 'k': [0.02, 0.20, 0.01]},
        {'ells': 2, 'k': [0.02, 0.20, 0.01]},
    ],
    'mesh3_spectrum': [
        {'ells': (0, 0, 0), 'k': [0.02, 0.20, 0.01]},
        {'ells': (2, 0, 2), 'k': [0.02, 0.03, 0.01]},
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


def _normalize_ells(value):
    """Return an integer or tuple representation of a multipole selector."""
    if isinstance(value, (tuple, list)):
        values = tuple(int(item) for item in value)
        return values[0] if len(values) == 1 else values
    values = tuple(int(item.strip()) for item in str(value).split(',') if item.strip())
    if not values:
        raise ValueError('Multipole selector cannot be empty.')
    return values[0] if len(values) == 1 else values


def _parse_kmax_overrides(values):
    """Parse repeatable ``STAT:ELLS=KMAX`` command-line overrides."""
    overrides = []
    for value in values or []:
        try:
            selector, kmax = value.rsplit('=', 1)
            stat, ells = selector.split(':', 1)
            overrides.append({'stat': stat, 'ells': _normalize_ells(ells), 'kmax': float(kmax)})
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid kmax override {value!r}; expected STAT:ELLS=KMAX, "
                "for example mesh2_spectrum:0=0.15 or mesh3_spectrum:0,0,0=0.10."
            ) from exc
    return overrides


def _apply_kmax_overrides(kranges, overrides):
    """Apply validated overrides to a copied KRANGES mapping."""
    for override in overrides or []:
        stat, ells, kmax = override['stat'], _normalize_ells(override['ells']), float(override['kmax'])
        if stat not in kranges:
            raise ValueError(f"Unknown statistic {stat!r} in kmax override.")
        matches = [item for item in kranges[stat] if _normalize_ells(item['ells']) == ells]
        if not matches:
            raise ValueError(f"No {stat!r} selection with ells={ells!r}.")
        if len(matches) != 1:
            raise ValueError(f"Ambiguous {stat!r} selection with ells={ells!r}.")
        klim = matches[0]['k']
        if kmax <= klim[0]:
            raise ValueError(
                f"kmax={kmax} must be greater than kmin={klim[0]} for {stat}:{ells}; "
                "this cut would select no bins."
            )
        step = klim[2]
        nstep = round((kmax - klim[0]) / step)
        if not abs(klim[0] + nstep * step - kmax) < 1e-8:
            raise ValueError(f"kmax={kmax} is not on the {stat}:{ells} grid with step {step}.")
        klim[1] = kmax
    return kranges


def _validate_theory_model(stats, theory_model):
    if theory_model == 'reptvelocileptors' and 'mesh3_spectrum' in stats:
        raise ValueError('theory model reptvelocileptors is only supported with mesh2_spectrum')


def _apply_kranges(observable_options, kranges=None):
    stat = observable_options['stat']['kind']
    kranges = _get_kranges() if kranges is None else kranges
    if stat not in kranges:
        return
    observable_options['stat']['select'] = [
        {'ells': item['ells'], 'k': list(item['k'])}
        for item in kranges[stat]
    ]


def _build_likelihoods_options(stats, tracers, version, covariance, stats_dir,
                               project=DEFAULT_PROJECT, theory_model='folpsD',
                               prior_basis='physical_aap', emulator=True,
                               folpsd_damping=FOLPSD_DAMPING,
                               folpsd_damping_method=FOLPSD_DAMPING_METHOD,
                               kmax_overrides=None, mesh3_theory_dk=None,
                               region='GCcomb', imock=None,
                               covariance_project='full_shape/base', covariance_region=None):
    _validate_theory_model(stats, theory_model)
    if mesh3_theory_dk is not None and mesh3_theory_dk <= 0.:
        raise ValueError('mesh3_theory_dk must be positive.')
    folpsd_damping_method = None if folpsd_damping_method == 'none' else folpsd_damping_method
    kranges = _apply_kmax_overrides(_get_kranges(), kmax_overrides)
    likelihoods = []
    for tracer in tracers:
        # The BGS alternate-MTL covariance products live in their own version;
        # the other cut-sky tracers use the common holi-v3 products.
        tracer_covariance = covariance
        if 'BGS' in tracer and isinstance(covariance, str) and 'holi' in covariance:
            tracer_covariance = 'holi-bgs-altmtl'
        likelihood_options = tools.generate_likelihood_options_helper(
            stats=stats,
            tracer=tracer,
            version=version,
            covariance=tracer_covariance,
            stats_dir=stats_dir,
            project=project,
            region=region,
            imock=imock,
            emulator=emulator and theory_model != 'comet',
        )
        for observable_options in likelihood_options['observables']:
            _apply_kranges(observable_options, kranges=kranges)
            if 'mesh3_spectrum' in observable_options['stat']['kind'] and mesh3_theory_dk is not None:
                observable_options.setdefault('window', {})['theory_dk'] = float(mesh3_theory_dk)
            observable_options.setdefault('theory', {})
            observable_options['theory']['model'] = theory_model
            #observable_options['theory']['marg'] = False
            observable_options['theory']['prior_basis'] = prior_basis
            if theory_model == 'folpsD':
                observable_options['theory']['damping'] = folpsd_damping
                if 'mesh2_spectrum' in observable_options['stat']['kind']:
                    observable_options['theory']['damping_method'] = folpsd_damping_method
            else:
                observable_options['theory'].pop('damping', None)
                observable_options['theory'].pop('damping_method', None)
        # Covariance mocks are stored beneath full_shape/base independently
        # of the project used for the fitted data vectors.
        likelihood_options['covariance']['source'] = 'mock'
        likelihood_options['covariance']['project'] = covariance_project
        if covariance_region is not None:
            likelihood_options['covariance']['region'] = covariance_region
        likelihoods.append(likelihood_options)
    return likelihoods


def _build_run_options(stats, tracers, version, covariance, stats_dir,
                       project=DEFAULT_PROJECT, theory_model='folpsD',
                       cosmo_model='base', template='direct', sampler='emcee', nchains=1,
                       resume=False, prior_basis='physical_aap', emulator=True,
                       gelman_rubin=GELMAN_RUBIN, ess=ESS,
                       profile_iterations=PROFILE_ITERATIONS,
                       folpsd_damping=FOLPSD_DAMPING,
                       folpsd_damping_method=FOLPSD_DAMPING_METHOD,
                       kmax_overrides=None, mesh3_theory_dk=None,
                       region='GCcomb', imock=None,
                       covariance_project='full_shape/base', covariance_region=None):
    profile_iterations = int(profile_iterations)
    if profile_iterations <= 0:
        raise ValueError('profile_iterations must be positive.')
    options = {}
    options['likelihoods'] = _build_likelihoods_options(
        stats=stats,
        tracers=tracers,
        version=version,
        covariance=covariance,
        stats_dir=stats_dir,
        project=project,
        theory_model=theory_model,
        prior_basis=prior_basis,
        emulator=emulator,
        folpsd_damping=folpsd_damping,
        folpsd_damping_method=folpsd_damping_method,
        kmax_overrides=kmax_overrides,
        mesh3_theory_dk=mesh3_theory_dk,
        region=region,
        imock=imock,
        covariance_project=covariance_project,
        covariance_region=covariance_region,
    )
    options['cosmology'] = {'template': template, 'model': cosmo_model, 'engine': 'eisenstein_hu' if 'comet' in theory_model else 'class'}
    options['sampler'] = {
        'sampler': sampler,
        'nchains': nchains,
        'resume': resume,
    }
    options = tools.fill_fiducial_options(options)
    if theory_model != 'folpsD':
        for likelihood_options in options['likelihoods']:
            for observable_options in likelihood_options['observables']:
                observable_options['theory'].pop('damping', None)
                observable_options['theory'].pop('damping_method', None)
    options['profiler']['maximize']['niterations'] = profile_iterations
    sampler_kw = {'nparallel': nchains, 'gelman_rubin': gelman_rubin, 'ess': ess}
    for section in ['init', 'run']:
        for name in options['sampler'][section]:
            if name in sampler_kw:
                options['sampler'][section][name] = sampler_kw[name]
    return options


def _apply_local_safe_threads(environ=None):
    environ = os.environ if environ is None else environ
    for name, value in LOCAL_SAFE_THREAD_ENV.items():
        environ.setdefault(name, value)
    return environ


def run_fit(actions=('profile',), template='direct', version='abacus-2ndgen-dr2-complete',
            covariance='holi-v1-altmtl',
            stats_dir=DEFAULT_STATS_DIR,
            project='full_shape/base',
            fits_dir=Path(os.getenv('SCRATCH', '.')) / 'fits',
            cache_dir=DEFAULT_CACHE_DIR,
            stats=['mesh2_spectrum'], tracers=None, theory_model='folpsD',
            cosmo_model='base', sampler='emcee', nchains=1, resume=False,
            prior_basis='physical_aap', emulator=True, local_safe_threads=False,
            gelman_rubin=GELMAN_RUBIN, ess=ESS,
            profile_iterations=PROFILE_ITERATIONS,
            folpsd_damping=FOLPSD_DAMPING,
            folpsd_damping_method=FOLPSD_DAMPING_METHOD,
            kmax_overrides=None, mesh3_theory_dk=None,
            region='GCcomb', imock=None,
            covariance_project='full_shape/base', covariance_region=None):
    # Everything inside this function will be executed on the compute nodes;
    # This function must be self-contained; and cannot rely on imports from the outer scope.
    import os
    from pathlib import Path
    import functools
    if local_safe_threads:
        _apply_local_safe_threads()
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'
    from desilike import distributed
    try: distributed.initialize()
    except RuntimeError: print('Distributed environment already initialized')
    else: print('Initializing distributed environment')
    mpicomm = distributed.get_mpicomm()
    import jax
    from jax import config
    config.update('jax_enable_x64', True)
    from full_shape import run_fit_from_options, setup_logging
    from full_shape.tools import get_likelihood
    from desilike import compile
    from desilike.samples import Profiles
    # You can pass region, version, covariance, ...
    options = _build_run_options(
        stats=stats,
        tracers=tracers,
        version=version,
        covariance=covariance,
        stats_dir=stats_dir,
        project=project,
        theory_model=theory_model,
        cosmo_model=cosmo_model,
        template=template,
        sampler=sampler,
        nchains=nchains,
        resume=resume,
        prior_basis=prior_basis,
        emulator=emulator,
        gelman_rubin=gelman_rubin,
        ess=ess,
        profile_iterations=profile_iterations,
        folpsd_damping=folpsd_damping,
        folpsd_damping_method=folpsd_damping_method,
        kmax_overrides=kmax_overrides,
        mesh3_theory_dk=mesh3_theory_dk,
        region=region,
        imock=imock,
        covariance_project=covariance_project,
        covariance_region=covariance_region,
    )
    get_fits_fn = functools.partial(tools.get_fits_fn, fits_dir=fits_dir)
    cache_dir = Path(cache_dir)
    run_fit_from_options(actions, **options, get_fits_fn=get_fits_fn, cache_dir=cache_dir)
    if 'profile' in actions:
        likelihood = get_likelihood(likelihoods_options=options['likelihoods'],
                                    cosmology_options=options['cosmology'],
                                    cache_dir=cache_dir)
        fn = get_fits_fn(kind='profiles', **options)
        profiles = Profiles.read(fn)
        # Evaluate likelihood at dictionary of parameters
        best = profiles.choice(index='argmax', squeeze=True).select(input=True).best
        compile(likelihood)(**best)
        if mpicomm.rank == 0:
            plot_dir = fn.parent
            for ilikelihood, sublikelihood in enumerate(likelihood.likelihoods):
                for iobservable, observable in enumerate(sublikelihood.observables):
                    observable.plot(fn=plot_dir / f'plot_likelihood{ilikelihood}_observable{iobservable}.png')


def _get_parser():
    datasets = ['abacus-2ndgen-dr2-altmtl', 'abacus-2ndgen-dr2-complete', 'abacus-hf-dr2-v2-altmtl']
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=datasets, default='abacus-2ndgen-dr2-complete',
                        help='Dataset to fit. Defaults to abacus-2ndgen-dr2-complete.')
    parser.add_argument('--todo', type=str, nargs='*', default=['profile'],
                        choices=['build', 'profile', 'sample'],
                        help='Run build, profile, and / or sample. Defaults to profile.')
    parser.add_argument('--stats', type=str, nargs='*', default=['mesh2_spectrum'],
                        choices=['mesh2_spectrum', 'mesh3_spectrum'],
                        help='Statistics to fit. Defaults to mesh2_spectrum.')
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
                        help=(f'FoG damping method for folpsD mesh2 fits. none is an alias for '
                              f'tree+loop+ctr; all is an alias for tree+loop+ctr+sn. '
                              f'Defaults to {FOLPSD_DAMPING_METHOD}.'))
    parser.add_argument('--cosmo_params', type=str, default='base',
                        choices=COSMO_MODELS,
                        help='Cosmology parameter setup to fit. base varies h, omega_cdm, omega_b, logA, n_s; '
                             'base_ns-fixed varies h, omega_cdm, omega_b, logA; '
                             'fixed varies only nuisance parameters. Defaults to base.')
    parser.add_argument('--sampler', type=str, default='emcee',
                        choices=SAMPLERS,
                        help='desilike sampler backend to use. Defaults to emcee.')
    parser.add_argument('--tracers', action='extend', nargs='+', default=None,
                        help='Tracer(s) to fit. Pass one or more values after --tracers. Defaults to LRG1.')
    parser.add_argument('--fits_dir', type=str, default=None,
                        help='Base directory for fits. Defaults to $SCRATCH/fits_abacus_mocks or ./fits_abacus_mocks.')
    parser.add_argument('--stats_dir', type=str, default=DEFAULT_STATS_DIR,
                        help=f'Base directory for clustering statistics. Defaults to {DEFAULT_STATS_DIR}.')
    parser.add_argument('--project', type=str, default=DEFAULT_PROJECT,
                        help=f'Base directory for clustering statistics. Defaults to {DEFAULT_PROJECT}.')
    parser.add_argument('--cache_dir', type=str, default=DEFAULT_CACHE_DIR,
                        help=f'Base directory for cached prepared stats and emulators. Defaults to {DEFAULT_CACHE_DIR}.')
    parser.add_argument(
        '--kmax', action='append', default=[], metavar='STAT:ELLS=KMAX',
        help=('Override one selection kmax; repeat as needed. Examples: '
              'mesh2_spectrum:0=0.15, mesh3_spectrum:0,0,0=0.10.'),
    )
    parser.add_argument(
        '--mesh3-theory-dk', type=float, default=0.005, metavar='DK',
        help=('Fix the first-stage mesh3 window-theory spacing independently of the observable '
              'binning. Omit to retain the current dynamic behavior.'),
    )
    parser.add_argument('--nchains', type=int, default=1,
                        help='Number of MCMC chains to run with desilike. Defaults to 1.')
    parser.add_argument('--gelman_rubin', type=float, default=GELMAN_RUBIN,
                        help=f'Gelman-Rubin convergence threshold for MCMC samplers. Defaults to {GELMAN_RUBIN}.')
    parser.add_argument('--ess', type=float, default=ESS,
                        help=f'Effective sample size convergence threshold for MCMC samplers. Defaults to {ESS}.')
    parser.add_argument('--profile-iterations', type=int, default=PROFILE_ITERATIONS,
                        help=f'Independent profiling starts. Defaults to {PROFILE_ITERATIONS}.')
    parser.add_argument('--resume', action='store_true',
                        help='Resume sampling from existing chain files in the derived fits directory.')
    parser.add_argument('--no_emulator', action='store_true',
                        help='Disable Taylor emulators and evaluate the theory directly. '
                             'Useful for fixed-cosmology, nuisance-only fits.')
    parser.add_argument('--local_safe_threads', action='store_true',
                        help='Limit OpenMP/BLAS thread counts for local macOS CLASS/OpenMP crashes. '
                             'Defaults to off so cluster runs keep their normal threading.')
    return parser


if __name__ == '__main__':
    parser = _get_parser()
    args = parser.parse_args()

    base_fits_dir = Path(args.fits_dir) if args.fits_dir is not None else Path(os.getenv('SCRATCH', '.')) / 'fits_abacus_mocks'
    fits_dir = base_fits_dir / args.dataset
    version = args.dataset
    covariance = 'holi-v3-altmtl'
    stats_dir = Path(args.stats_dir)
    cache_dir = Path(args.cache_dir)
    stats = args.stats
    tracers = args.tracers or ['LRG1']
    kmax_overrides = _parse_kmax_overrides(args.kmax)
    _validate_theory_model(stats, args.theory_model)
    run_fit(actions=args.todo, version=version, covariance=covariance, stats_dir=stats_dir, project=args.project,
            fits_dir=fits_dir, cache_dir=cache_dir, stats=stats, tracers=tracers, theory_model=args.theory_model,
            cosmo_model=args.cosmo_params, sampler=args.sampler, nchains=args.nchains,
            resume=args.resume, prior_basis=args.prior_basis,
            gelman_rubin=args.gelman_rubin, ess=args.ess,
            profile_iterations=args.profile_iterations,
            folpsd_damping=args.folpsd_damping,
            folpsd_damping_method=args.folpsd_damping_method,
            local_safe_threads=args.local_safe_threads, emulator=not args.no_emulator,
            kmax_overrides=kmax_overrides,
            mesh3_theory_dk=args.mesh3_theory_dk)
