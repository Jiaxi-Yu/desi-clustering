import os
import shutil
from pathlib import Path
import functools
import argparse
import logging

from desilike.base import compile, copy, Prior, Posterior, get_params
from desilike.samples import MCSamples, Profiles
from desilike.profilers import Profiler
from desilike.samplers import Sampler
from desilike.distributed import get_mpicomm

from full_shape.tools import get_likelihood, get_prior, fill_fiducial_options, generate_likelihood_options_helper, setup_logging
from full_shape import tools
from clustering_statistics import tools as clustering_tools


logger = logging.getLogger('fit')


def print_priors(params):
    print(f"{'param':20} {'prior':50} {'reference':50} derived")
    for p in params:
        print(f"{p.name:20} {str(p.prior):50} {str(p.ref):50} {p.derived}")


def run_fit_from_options(actions,
                         get_stats_fn=clustering_tools.get_stats_fn,
                         get_fits_fn=tools.get_fits_fn,
                         cache_dir:str | Path=None, cache_mode: str='rw', **kwargs):
    """
    Build a likelihood from options and run fitting actions (profile / sample).

    This helper constructs desi-like likelihood(s) from provided options,
    instantiates the corresponding desilike :class:`ObservablesGaussianLikelihood`
    and then runs fitting.

    Parameters
    ----------
    actions : str or sequence[str]
        One or more actions to run. Supported values: 'profile' (maximize using a
        profiler) and 'sample' (run MCMC sampler).
    get_stats_fn : callable, optional
        Function used to locate/read measurement files (passed to likelihood builder).
    get_fits_fn : callable, optional
        Function that constructs file paths for fit outputs (used to name saved chains/profiles).
    cache_dir : str or pathlib.Path, optional
        Directory used for caching emulators and precomputed products.
    cache_mode : str, optional
        'rw' for read/write; 'r' for read-only.
    **kwargs :
        Top-level options dictionary consumed by fill_fiducial_options. Must include
        a 'likelihoods' entry; may include sampler/profiler configuration and init/run kwargs.

    """
    if isinstance(actions, str):
        actions = [actions]
    options = fill_fiducial_options(kwargs)
    likelihood = get_likelihood(likelihoods_options=options['likelihoods'],
                                cosmology_options=options['cosmology'],
                                get_stats_fn=get_stats_fn, cache_dir=cache_dir, cache_mode=cache_mode)
    mpicomm = get_mpicomm()
    if mpicomm.rank == 0:
        logger.info('priors:')
        print_priors(get_params(likelihood).select(varied=True, derived=False))
    fn = get_fits_fn(kind='config', **options, ext='yaml')
    tools.write_options(fn, options)
    profiles_fn = get_fits_fn(kind='profiles', **options)
    for action in actions:
        if action == 'build':
            pass  # likelihood already built and cached above
        elif action == 'profile':
            profiler_options = dict(options['profiler'])
            cls = tools.get_profiler_cls(profiler_options.pop('profiler', 'minuit'))
            likelihood_profiler = copy(likelihood)
            for param in get_params(likelihood_profiler).select(solved=True):
                param.update(derived='best')
            posterior = compile(Posterior(likelihood_profiler, prior=get_prior(likelihood_profiler)))
            kw = dict(profiler_options.get('init', {}))
            kernel = cls(**{name: kw.pop(name) for name in list(kw) if name not in ['rng', 'rescale', 'covariance']})
            profiler = Profiler(posterior, kernel=kernel, output_fn=profiles_fn, **kw)
            profiler.maximize(**profiler_options.get('maximize', {}))
            #profiler.covariance()
            #if mpicomm.rank == 0:
            #    print(profiler.profiles.to_stats(tablefmt='pretty'))
        elif action == 'sample':
            sampler_options = dict(options['sampler'])
            cls = tools.get_sampler_cls(sampler_options.pop('sampler', 'emcee'))
            resume = sampler_options.pop('resume', True)
            kw = sampler_options.get('init', {})
            output_dir = get_fits_fn(kind='samples', **options).parent
            if not resume and mpicomm.rank == 0:
                 for path in Path(output_dir).glob('*'):
                    if path.name != 'profiles.h5':
                        shutil.rmtree(path) if path.is_dir() else path.unlink()
            mpicomm.Barrier()
            likelihood_sampler = copy(likelihood)
            if kw.get('rescale', False):
                profiles = Profiles.read(profiles_fn).choice(index='argmax', squeeze=True)
                best, error, covariance = profiles.best, profiles.error, profiles.covariance
                kw['covariance'] = covariance
                #error = {param: covariance.std(param) for param in covariance.names()}
                for param in get_params(likelihood_sampler):
                    if param.name in error:
                        param.update(ref=dict(dist='norm', loc=best[param.name], scale=error[param.name]))
            if kw.get('prior', None) is not None:
                profiles = Profiles.read(profiles_fn).choice(index='argmax', squeeze=True)
                kw['prior'] = kw['prior'] * profiles.covariance
            posterior = compile(Posterior(likelihood_sampler, prior=get_prior(likelihood_sampler)))
            kernel = cls(**{name: kw.pop(name) for name in list(kw) if name not in ['rng', 'rescale', 'covariance', 'nparallel', 'prior', 'batch_size']})
            sampler = Sampler(posterior, kernel=kernel, output_dir=output_dir, **kw)
            sampler.run(**sampler_options.get('run', {}))
        else:
            raise NotImplementedError(f'{action} not implemented')



if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Fit DESI cutsky clustering statistics.',
    )
    parser.add_argument(
        '--actions', type=str, default='profile',
        choices=['build', 'profile', 'sample'], nargs='*',
        help='Run best fit (maximize) and / or sample. Use "build" to pre-warm caches without running inference.',
    )
    parser.add_argument(
        '--stats', type=str, nargs='*', default=['recon_particle2_correlation'],
        choices=['recon_particle2_correlation'],
        help='Statistics to fit.',
    )
    parser.add_argument(
        '--tracers', nargs='*', type=str, default=['LRG2'],
        help='Tracer labels (default: LRG2).',
    )
    parser.add_argument(
        '--region', type=str, default='GCcomb',
        help='Sky region (default: GCcomb).',
    )
    parser.add_argument(
        '--data', type=str, default='abacus-hf-dr2-v2-altmtl',
        help='Data product identifier (default: abacus-hf-dr2-v2-altmtl).',
    )
    parser.add_argument(
        '--covariance', type=str, default='holi-v3-altmtl',
        help='Covariance mock set (default: holi-v3-altmtl).',
    )
    parser.add_argument(
        '--project', type=str, default='',
        help='Optional measurement project subdirectory under stats_dir.',
    )
    fits_dir = Path(os.getenv('SCRATCH', '.')) / 'fits'
    parser.add_argument(
        '--fits_dir', type=str, default=fits_dir,
        help=f'base directory for fits, default is {fits_dir}'
    )
    cache_dir = Path('.') / '_cache'
    parser.add_argument(
        '--cache_dir', type=str, default=cache_dir,
        help=f'cache directory for emulators and pre-computed covariance, default is {cache_dir}'
    )
    args = parser.parse_args()

    setup_logging()
    options = {'likelihoods': []}
    for tracer in args.tracers:
        likelihood_options = generate_likelihood_options_helper(stats=args.stats, version=args.data, tracer=tracer, region=args.region,
                                                                covariance=args.covariance, project=args.project)
        options['likelihoods'].append(likelihood_options)
    run_fit_from_options(args.actions,
                         get_fits_fn=functools.partial(tools.get_fits_fn, fits_dir=args.fits_dir),
                         cache_dir=args.cache_dir, **options)