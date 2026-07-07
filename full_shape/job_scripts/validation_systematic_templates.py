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
DEFAULT_STATS_DIR = Path('/global/cfs/cdirs/desicollab/science/cai/desi-clustering/dr2/summary_statistics')
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


def _validate_theory_model(stats, theory_model):
    if theory_model == 'reptvelocileptors' and 'mesh3_spectrum' in stats:
        raise ValueError('theory model reptvelocileptors is only supported with mesh2_spectrum')


def _apply_kranges(observable_options):
    stat = observable_options['stat']['kind']
    if stat not in KRANGES:
        return
    observable_options['stat']['select'] = [
        {'ells': item['ells'], 'k': list(item['k'])}
        for item in KRANGES[stat]
    ]


def _build_likelihoods_options(stats, tracers, version, covariance, stats_dir, project, theory_model,
                               syst_templates=tuple(), emulator=True, prior_basis='physical_aap'):
    _validate_theory_model(stats, theory_model)
    likelihoods = []
    for tracer in tracers:
        likelihood_options = tools.generate_likelihood_options_helper(
            stats=stats,
            tracer=tracer,
            version=version,
            covariance=covariance,
            stats_dir=stats_dir,
            project=project,
            emulator=emulator or theory_model != 'comet',
        )
        for observable_options in likelihood_options['observables']:
            _apply_kranges(observable_options)
            observable_options['window']['templates'] = syst_templates
            observable_options.setdefault('theory', {})
            observable_options['theory']['model'] = theory_model
            #observable_options['theory']['marg'] = False
            observable_options['theory']['prior_basis'] = prior_basis
        likelihood_options['covariance']['source'] = 'mock'
        likelihood_options['covariance']['project'] = 'full_shape/base'
        likelihoods.append(likelihood_options)
    return likelihoods


def _build_run_options(stats, tracers, version, covariance, stats_dir, project, theory_model,
                       cosmo_model='base', template='direct', sampler='emcee', nchains=1,
                       resume=False, prior_basis='physical_aap', emulator=True, syst_templates=tuple()):
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
        syst_templates=syst_templates,
    )
    options['cosmology'] = {'template': template, 'model': cosmo_model, 'engine': 'eisenstein_hu' if 'comet' in theory_model else 'class'}
    options['sampler'] = tools.propose_fiducial_sampler_options(sampler=sampler)
    sampler_kw = {'nparallel': nchains}
    for section in ['init', 'run']:
        for name, value in options['sampler'][section].items():
            if name in sampler_kw:
                options['sampler'][section][name] = sampler_kw[name]
    options['sampler']['resume'] = resume
    return tools.fill_fiducial_options(options)


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
            syst_templates=tuple(), prior_basis='physical_aap', emulator=True, local_safe_threads=False):
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
        syst_templates=syst_templates,
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
    parser.add_argument('--dataset', choices=datasets, default='abacus-hf-dr2-v2-altmtl',
                        help='Dataset to fit..')
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
    parser.add_argument('--project', type=str, default='full_shape/fiber_assignment_systematics',
                        help=f'Base directory for clustering statistics.')
    parser.add_argument('--syst_templates', type=str, nargs='*', default=[], choices=['auw'],
                        help=f'Systematic templates.')
    parser.add_argument('--cache_dir', type=str, default=DEFAULT_CACHE_DIR,
                        help=f'Base directory for cached prepared stats and emulators. Defaults to {DEFAULT_CACHE_DIR}.')
    parser.add_argument('--nchains', type=int, default=1,
                        help='Number of MCMC chains to run with desilike. Defaults to 1.')
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
    _validate_theory_model(stats, args.theory_model)
    run_fit(actions=args.todo, version=version, covariance=covariance, stats_dir=stats_dir, project=args.project,
            fits_dir=fits_dir, cache_dir=cache_dir, stats=stats, tracers=tracers, theory_model=args.theory_model,
            syst_templates=args.syst_templates, cosmo_model=args.cosmo_params, sampler=args.sampler, nchains=args.nchains,
            resume=args.resume, prior_basis=args.prior_basis,
            local_safe_threads=args.local_safe_threads, emulator=not args.no_emulator)
