from pathlib import Path
import json
import lsstypes as types

import numpy as np
import pytest

from full_shape import tools
from full_shape.run_fit import run_fit_from_options
from full_shape.tools import generate_likelihood_options_helper, str_from_likelihood_options, str_from_options, get_likelihood, fill_fiducial_options, get_stats, setup_logging
from clustering_statistics import tools as clustering_tools
from full_shape.job_scripts.validation_abacus_mocks import (
    KRANGES, LOCAL_SAFE_THREAD_ENV, _apply_local_safe_threads,
    _build_likelihoods_options, _build_run_options, _get_parser,
)


def test_str():
    likelihood_options = generate_likelihood_options_helper(tracer='LRG2')
    for level in [None, 1, 2, 3]:
        s = str_from_likelihood_options(likelihood_options, level=level)
        if level is None:
            assert s == 'LRG2-S2+LRG2-S3'
        elif level == 1:
            assert s == 'LRG2-S2-th-folpsD+LRG2-S3-th-folpsD+cov-mock-holi-v1-altmtl', s
    s = str_from_likelihood_options(likelihood_options, level={'stat': 2})
    assert s == 'LRG2-S2-ell0-k0.02-0.20-0.005-ell2-k0.02-0.20-0.005+LRG2-S3-ell000-k0.02-0.12-0.005-ell202-k0.02-0.08-0.005', s

    likelihood_options = generate_likelihood_options_helper(tracer='LRG3xELG1')
    for level in [None, 1, 2, 3]:
        s = str_from_likelihood_options(likelihood_options, level=level)
        if level is None:
            assert s == 'LRG3xELG1-S2+LRG3xELG1-S3'
        elif level == 1:
            assert s == 'LRG3xELG1-S2-th-folpsD+LRG3xELG1-S3-th-folpsD+cov-mock-holi-v1-altmtl', s
    s = str_from_likelihood_options(likelihood_options, level={'stat': 2})
    assert s == 'LRG3xELG1-S2-ell0-k0.02-0.20-0.005-ell2-k0.02-0.20-0.005+LRG3xELG1-S3-ell000-k0.02-0.12-0.005-ell202-k0.02-0.08-0.005'

    options = {}
    options['likelihoods'] = [likelihood_options]
    options = fill_fiducial_options(options)
    s = str_from_options(options, level=None)
    assert s == 'cosmo-base_ns-fixed_LRG3xELG1-S2+LRG3xELG1-S3', s

    options['cosmology'] = {'model': 'base', 'template': 'direct'}
    options = fill_fiducial_options(options)
    s = str_from_options(options, level=None)
    assert s == 'cosmo-base_LRG3xELG1-S2+LRG3xELG1-S3', s


def time_posterior(posterior):
    import time
    import jax
    n = 10
    key = jax.random.key(42)
    samples = {param.name: param.ref.sample(key, shape=n + 2) for param in posterior.params.select(varied=True)}
    posterior = jax.jit(posterior)
    for i in range(n + 2):
        jax.block_until_ready(posterior(**{name: sample[i] for name, sample in samples.items()}))
        if i == 1:
            t0 = time.time()
    print((time.time() - t0) / n)


def test_likelihood_full_shape(save=False, load=False, theory='comet'):
    from desilike import compile, Posterior, get_params
    options = {}
    tracers = ['LRG2', 'LRG3']
    options['likelihoods'] = [generate_likelihood_options_helper(stats=['mesh2_spectrum', 'mesh3_spectrum'], tracer=tracer) for tracer in tracers]
    save_dir = Path('_save')

    def get_stats_fn(tracer):
        return save_dir / f'likelihood_full_shape_{tracer}.h5'
    load = False

    if save:
        save_dir.mkdir(exist_ok=True)
        options['cosmology'] = {'template': 'direct', 'engine': 'class' if 'folps' in theory else 'eisenstein_hu'}
        options = fill_fiducial_options(options)
        for tracer, likelihood_options in zip(tracers, options['likelihoods']):
            stats = get_stats(observables_options=likelihood_options['observables'], covariance_options=likelihood_options['covariance'])
            stats.write(get_stats_fn(tracer=tracer))

    for template in ['direct', 'shapefit'][:1]:
        options['cosmology'] = {'template': template}
        options = fill_fiducial_options(options)
        if load:
            for tracer, likelihood_options in zip(tracers, options['likelihoods']):
                likelihood_options['stats'] = types.read(get_stats_fn(tracer))

        for likelihood_options in options['likelihoods']:
            for observable_options in likelihood_options['observables']:
                observable_options['theory']['model'] = theory
                if 'comet' in theory:
                    observable_options['emulator'] = {'name': ''}
                else:
                    observable_options['emulator']['order'] = 2
        likelihood = get_likelihood(options['likelihoods'], cosmology_options=options['cosmology'], cache_dir='./_cache')
        #for param in get_params(likelihood).select(solved=True):
        #    param.update(derived=False)
        print(get_params(likelihood).select(solved=True))
        #for param in get_params(likelihood).select(solved=True):
        #    param.update(derived='best')
        posterior = compile(Posterior(likelihood))
        assert np.isfinite(posterior())
        print(posterior.params)
        time_posterior(posterior)
        return

        if template == 'direct':
            assert 'h' in posterior.params.select(varied=True)
        elif template == 'shapefit':
            assert 'df' in posterior.params.select(varied=True)
        from desilike.profilers import Profiler, Minuit
        profiler = Profiler(posterior, kernel=Minuit(), rng=42)
        profiler.maximize()
        profiles = profiler.profiles
        print(profiles.to_stats(tablefmt='pretty'))
        best = profiles.choice(index='argmax', squeeze=True).select(input=True).best
        compile(likelihood)(**best)
        likelihood.likelihoods[0].observables[0].plot(fn='./_tests/plot.png')


    
def test_likelihood_bao(save=False, load=False):
    from desilike import compile, Posterior, get_params

    save_dir = Path('_save')
    stats_dir = Path('/dvs_ro/cfs/cdirs/desi/mocks/cai/LSS/DA2/mocks/desipipe')
    tracers = ['LRG2']

    def get_stats_fn(tracer):
        return save_dir / f'likelihood_bao_{tracer}.h5'

    if save:
        save_dir.mkdir(exist_ok=True)
        options = {'cosmology': {'template': 'bao'}, 'likelihoods': [generate_likelihood_options_helper(stats=['recon_particle2_correlation'], tracer=tracer, version='data-dr2-v1.1', project='', stats_dir=stats_dir) for tracer in tracers]}
        for likelihood_options in options['likelihoods']:
            likelihood_options['covariance'] = {'source': 'rascalc', 'version': 'data-dr2-v1.1', 'stats_dir': stats_dir}
        options = fill_fiducial_options(options)

        for tracer, likelihood_options in zip(tracers, options['likelihoods']):
            get_stats(observables_options=likelihood_options['observables'], covariance_options=likelihood_options['covariance']).write(get_stats_fn(tracer))

    for template in ['bao', 'direct'][:1]:
        options = {'cosmology': {'template': template},
                   'likelihoods': [generate_likelihood_options_helper(stats=['recon_particle2_correlation'], tracer=tracer, version='data-dr2-v1.1', project='', stats_dir=stats_dir, emulator=template == 'direct') for tracer in tracers]}
        for likelihood_options in options['likelihoods']:
            likelihood_options['covariance'] = {'source': 'rascalc', 'version': 'data-dr2-v1.1', 'stats_dir': stats_dir}
        options = fill_fiducial_options(options)
        if load:
            for tracer, likelihood_options in zip(tracers, options['likelihoods']):
                likelihood_options['stats'] = types.read(get_stats_fn(tracer))
        likelihood = get_likelihood(options['likelihoods'], cosmology_options=options['cosmology'], cache_dir='./_cache', cache_mode='w')
        for param in get_params(likelihood).select(solved=True):
            param.update(derived='best')
        posterior = compile(Posterior(likelihood))
        assert np.isfinite(posterior())
        time_posterior(posterior)

        if template == 'direct':
            assert 'h' in posterior.params.select(varied=True)
        else:
            assert 'qpar' in posterior.params.select(varied=True)
            from desilike.profilers import MinuitProfiler
            profiler = MinuitProfiler(posterior, seed=42)
            profiler.maximize()
            profiles = profiler.profiles
            print(profiles.to_stats(tablefmt='pretty'))
            best = profiles.choice(index='argmax', squeeze=True).select(input=True).best
            compile(likelihood)(**best)
            likelihood.likelihoods[0].observables[0].plot(fn='./_tests/plot.png')
            likelihood.likelihoods[0].observables[0].plot_bao(fn='./_tests/plot_bao.png')


def test_covariance():
    options = {}
    options['likelihoods'] = [generate_likelihood_options_helper(stats=['mesh2_spectrum'], tracer=tracer) for tracer in ['LRG3']]
    for likelihood_options in options['likelihoods']:
        likelihood_options['covariance'] = {'source': 'jaxpower', 'version': 'abacus-2ndgen-complete'}
    options = fill_fiducial_options(options)
    likelihood = get_likelihood(options['likelihoods'], cache_dir='./_cache')
    likelihood()


def test_options():
    options = {}
    options['likelihoods'] = [generate_likelihood_options_helper(tracer=tracer) for tracer in ['LRG2', 'LRG3']]
    options = fill_fiducial_options(options)
    options2 = tools._base_type_options(options)
    fn = '_tests/config.yaml'
    tools.write_options(fn, options)
    options3 = tools.read_options(fn)
    assert options3 == options2


if __name__ == '__main__':

    setup_logging()
    #test_likelihood_bao(load=True)
    test_likelihood_full_shape(load=True, save=False, theory='comet')
    #test_covariance()
    #test_str()
    #test_covariance()
    #test_options()
