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


def test_bgs1_abacus_altmtl_uses_any02_file_label():
    assert clustering_tools.get_full_tracer('BGS', version='abacus-2ndgen-dr2-altmtl') == 'BGS_ANY-02'
    assert clustering_tools.get_full_tracer('BGS', version='abacus-hf-dr2-v2-altmtl') == 'BGS_ANY-02'
    assert clustering_tools.get_full_tracer('BGS', version='abacus-2ndgen-dr2-complete') == 'BGS_BRIGHT-21.35'

    likelihood_options = generate_likelihood_options_helper(
        stats=['mesh2_spectrum'],
        tracer='BGS1',
        version='abacus-2ndgen-dr2-altmtl',
        stats_dir=Path('/measurements'),
    )
    catalog = dict(likelihood_options['observables'][0]['catalog'])
    catalog['tracer'] = clustering_tools.get_full_tracer(catalog['tracer'], version=catalog['version'])
    catalog['imock'] = 0
    fn = clustering_tools.get_stats_fn(kind='mesh2_spectrum', **catalog)
    assert 'mesh2_spectrum_poles_BGS_ANY-02_z0.1-0.4_GCcomb_weight-default-FKP' in str(fn)



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
    tracers = ['LRG2', 'LRG3'][:1]
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
        #print(get_params(likelihood).select(solved=True))
        #for param in get_params(likelihood).select(solved=True):
        #    param.update(derived='best')
        posterior = compile(Posterior(likelihood))
        assert np.isfinite(posterior())
        time_posterior(posterior)

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


def test_validation_abacus_mocks_theory_model_options():
    for theory_model in ['folpsD', 'folpsEFT', 'reptvelocileptors']:
        stats = ['mesh2_spectrum']
        likelihoods = _build_likelihoods_options(
            stats=stats,
            tracers=['LRG1'],
            version='abacus-2ndgen-dr2-complete',
            covariance='holi-v3-altmtl',
            stats_dir=Path('/tmp'),
            theory_model=theory_model,
        )
        assert len(likelihoods) == 1
        observables = likelihoods[0]['observables']
        assert [observable['stat']['kind'] for observable in observables] == stats
        assert all(observable['theory']['model'] == theory_model for observable in observables)
        assert all(observable['theory']['prior_basis'] == 'physical_aap' for observable in observables)


def test_validation_abacus_mocks_prior_basis_option():
    likelihoods = _build_likelihoods_options(
        stats=['mesh2_spectrum', 'mesh3_spectrum'],
        tracers=['LRG1'],
        version='abacus-2ndgen-dr2-complete',
        covariance='holi-v3-altmtl',
        stats_dir=Path('/tmp'),
        theory_model='folpsD',
        prior_basis='standard',
    )
    observables = likelihoods[0]['observables']
    assert all(observable['theory']['prior_basis'] == 'standard' for observable in observables)


def test_validation_abacus_mocks_applies_kranges():
    likelihoods = _build_likelihoods_options(
        stats=['mesh2_spectrum', 'mesh3_spectrum'],
        tracers=['LRG1'],
        version='abacus-2ndgen-dr2-complete',
        covariance='holi-v3-altmtl',
        stats_dir=Path('/tmp'),
        theory_model='folpsD',
    )
    observables = {
        observable['stat']['kind']: observable['stat']['select']
        for observable in likelihoods[0]['observables']
    }
    assert observables['mesh2_spectrum'] == KRANGES['mesh2_spectrum']
    assert observables['mesh3_spectrum'] == KRANGES['mesh3_spectrum']


def test_validation_abacus_mocks_copies_kranges():
    likelihoods = _build_likelihoods_options(
        stats=['mesh2_spectrum'],
        tracers=['LRG1'],
        version='abacus-2ndgen-dr2-complete',
        covariance='holi-v3-altmtl',
        stats_dir=Path('/tmp'),
        theory_model='folpsD',
    )
    select = likelihoods[0]['observables'][0]['stat']['select']
    original_kmax = KRANGES['mesh2_spectrum'][0]['k'][1]
    select[0]['k'][1] = 0.30
    assert KRANGES['mesh2_spectrum'][0]['k'][1] == original_kmax


def test_validation_abacus_mocks_nchains_option():
    options = _build_run_options(
        stats=['mesh2_spectrum'],
        tracers=['LRG1'],
        version='abacus-2ndgen-dr2-complete',
        covariance='holi-v3-altmtl',
        stats_dir=Path('/tmp'),
        theory_model='folpsD',
        nchains=4,
    )
    assert options['sampler']['nchains'] == 4


def test_validation_abacus_mocks_run_options_propagate_kranges():
    options = _build_run_options(
        stats=['mesh2_spectrum', 'mesh3_spectrum'],
        tracers=['LRG1'],
        version='abacus-2ndgen-dr2-complete',
        covariance='holi-v3-altmtl',
        stats_dir=Path('/tmp'),
        theory_model='folpsD',
    )
    observables = {
        observable['stat']['kind']: observable['stat']['select']
        for observable in options['likelihoods'][0]['observables']
    }
    assert observables['mesh2_spectrum'] == KRANGES['mesh2_spectrum']
    assert observables['mesh3_spectrum'] == KRANGES['mesh3_spectrum']


def test_validation_abacus_mocks_sampler_defaults_to_emcee():
    options = _build_run_options(
        stats=['mesh2_spectrum'],
        tracers=['LRG1'],
        version='abacus-2ndgen-dr2-complete',
        covariance='holi-v3-altmtl',
        stats_dir=Path('/tmp'),
        theory_model='folpsD',
    )
    assert options['sampler']['sampler'] == 'emcee'


def test_validation_abacus_mocks_sampler_option():
    options = _build_run_options(
        stats=['mesh2_spectrum'],
        tracers=['LRG1'],
        version='abacus-2ndgen-dr2-complete',
        covariance='holi-v3-altmtl',
        stats_dir=Path('/tmp'),
        theory_model='folpsD',
        sampler='mcmc',
    )
    assert options['sampler']['sampler'] == 'mcmc'
    assert options['sampler']['init']['oversample_power'] == 0


def test_validation_abacus_mocks_pocomc_sampler_option():
    options = _build_run_options(
        stats=['mesh2_spectrum'],
        tracers=['LRG1'],
        version='abacus-2ndgen-dr2-complete',
        covariance='holi-v3-altmtl',
        stats_dir=Path('/tmp'),
        theory_model='folpsD',
        sampler='pocomc',
    )
    assert options['sampler']['sampler'] == 'pocomc'
    assert options['sampler']['init'] == {}
    assert 'thin_by' not in options['sampler']['run']
    assert options['sampler']['run']['progress'] is True
    assert options['sampler']['run']['check_every'] == 25


def test_validation_abacus_mocks_pocomc_rejects_thin_by():
    with pytest.raises(ValueError, match='PocoMC.*thin_by'):
        _build_run_options(
            stats=['mesh2_spectrum'],
            tracers=['LRG1'],
            version='abacus-2ndgen-dr2-complete',
            covariance='holi-v3-altmtl',
            stats_dir=Path('/tmp'),
            theory_model='folpsD',
            sampler='pocomc',
            thin_by=2,
        )


def test_validation_abacus_mocks_thin_by_option():
    options = _build_run_options(
        stats=['mesh2_spectrum'],
        tracers=['LRG1'],
        version='abacus-2ndgen-dr2-complete',
        covariance='holi-v3-altmtl',
        stats_dir=Path('/tmp'),
        theory_model='folpsD',
        thin_by=10,
    )
    assert options['sampler']['run']['thin_by'] == 10


def test_validation_abacus_mocks_resume_defaults_to_false():
    options = _build_run_options(
        stats=['mesh2_spectrum'],
        tracers=['LRG1'],
        version='abacus-2ndgen-dr2-complete',
        covariance='holi-v3-altmtl',
        stats_dir=Path('/tmp'),
        theory_model='folpsD',
    )
    assert options['sampler']['resume'] is False


def test_validation_abacus_mocks_resume_option():
    options = _build_run_options(
        stats=['mesh2_spectrum'],
        tracers=['LRG1'],
        version='abacus-2ndgen-dr2-complete',
        covariance='holi-v3-altmtl',
        stats_dir=Path('/tmp'),
        theory_model='folpsD',
        resume=True,
    )
    assert options['sampler']['resume'] is True


def test_propose_fiducial_sampler_options_sets_mcmc_oversample_power_to_zero():
    options = tools.propose_fiducial_sampler_options('mcmc')
    assert options['init']['oversample_power'] == 0


def test_propose_fiducial_sampler_options_leaves_emcee_init_empty():
    options = tools.propose_fiducial_sampler_options('emcee')
    assert options['init'] == {}


def test_validation_abacus_mocks_cosmo_model_option():
    options = _build_run_options(
        stats=['mesh2_spectrum'],
        tracers=['LRG1'],
        version='abacus-2ndgen-dr2-complete',
        covariance='holi-v3-altmtl',
        stats_dir=Path('/tmp'),
        theory_model='folpsD',
        cosmo_model='base',
    )
    assert options['cosmology']['model'] == 'base'

    options = _build_run_options(
        stats=['mesh2_spectrum'],
        tracers=['LRG1'],
        version='abacus-2ndgen-dr2-complete',
        covariance='holi-v3-altmtl',
        stats_dir=Path('/tmp'),
        theory_model='folpsD',
        cosmo_model='base_ns-fixed',
    )
    assert options['cosmology']['model'] == 'base_ns-fixed'

    options = _build_run_options(
        stats=['mesh2_spectrum'],
        tracers=['LRG1'],
        version='abacus-2ndgen-dr2-complete',
        covariance='holi-v3-altmtl',
        stats_dir=Path('/tmp'),
        theory_model='folpsD',
        cosmo_model='fixed',
    )
    assert options['cosmology']['model'] == 'fixed'


def test_validation_abacus_mocks_run_options_propagate_prior_basis():
    options = _build_run_options(
        stats=['mesh2_spectrum', 'mesh3_spectrum'],
        tracers=['LRG1'],
        version='abacus-2ndgen-dr2-complete',
        covariance='holi-v3-altmtl',
        stats_dir=Path('/tmp'),
        theory_model='folpsEFT',
        prior_basis='physical',
    )
    observables = options['likelihoods'][0]['observables']
    assert all(observable['theory']['prior_basis'] == 'physical' for observable in observables)


def test_validation_abacus_mocks_run_options_emulator_default_enabled():
    options = _build_run_options(
        stats=['mesh2_spectrum', 'mesh3_spectrum'],
        tracers=['LRG1'],
        version='abacus-2ndgen-dr2-complete',
        covariance='holi-v3-altmtl',
        stats_dir=Path('/tmp'),
        theory_model='folpsEFT',
    )
    observables = options['likelihoods'][0]['observables']
    assert all(observable['emulator']['name'] == 'taylor' for observable in observables)


def test_validation_abacus_mocks_run_options_can_disable_emulator():
    options = _build_run_options(
        stats=['mesh2_spectrum', 'mesh3_spectrum'],
        tracers=['LRG1'],
        version='abacus-2ndgen-dr2-complete',
        covariance='holi-v3-altmtl',
        stats_dir=Path('/tmp'),
        theory_model='folpsEFT',
        emulator=False,
    )
    observables = options['likelihoods'][0]['observables']
    assert all(observable['emulator']['name'] == '' for observable in observables)


def test_validation_abacus_mocks_parser_accepts_nchains():
    parser = _get_parser()
    args = parser.parse_args(['--todo', 'sample', '--nchains', '4'])
    assert args.todo == ['sample']
    assert args.nchains == 4


def test_validation_abacus_mocks_parser_accepts_sampler():
    parser = _get_parser()
    args = parser.parse_args(['--todo', 'sample', '--sampler', 'mcmc'])
    assert args.todo == ['sample']
    assert args.sampler == 'mcmc'


def test_validation_abacus_mocks_parser_accepts_pocomc_sampler():
    parser = _get_parser()
    args = parser.parse_args(['--todo', 'sample', '--sampler', 'pocomc'])
    assert args.todo == ['sample']
    assert args.sampler == 'pocomc'


def test_validation_abacus_mocks_parser_defaults_sampler_to_emcee():
    parser = _get_parser()
    args = parser.parse_args([])
    assert args.sampler == 'emcee'


def test_validation_abacus_mocks_parser_accepts_thin_by():
    parser = _get_parser()
    args = parser.parse_args(['--todo', 'sample', '--thin_by', '10'])
    assert args.todo == ['sample']
    assert args.thin_by == 10


def test_validation_abacus_mocks_parser_accepts_resume():
    parser = _get_parser()
    args = parser.parse_args(['--todo', 'sample', '--resume'])
    assert args.todo == ['sample']
    assert args.resume is True


def test_validation_abacus_mocks_parser_accepts_local_safe_threads():
    parser = _get_parser()
    args = parser.parse_args(['--local_safe_threads'])
    assert args.local_safe_threads is True


def test_validation_abacus_mocks_parser_accepts_no_emulator():
    parser = _get_parser()
    args = parser.parse_args(['--no_emulator'])
    assert args.no_emulator is True


def test_validation_abacus_mocks_parser_defaults_thin_by_to_one():
    parser = _get_parser()
    args = parser.parse_args([])
    assert args.thin_by == 1


def test_validation_abacus_mocks_parser_defaults_resume_to_false():
    parser = _get_parser()
    args = parser.parse_args([])
    assert args.resume is False


def test_validation_abacus_mocks_parser_defaults_local_safe_threads_to_false():
    parser = _get_parser()
    args = parser.parse_args([])
    assert args.local_safe_threads is False


def test_validation_abacus_mocks_parser_defaults_no_emulator_to_false():
    parser = _get_parser()
    args = parser.parse_args([])
    assert args.no_emulator is False


def test_apply_local_safe_threads_sets_missing_values_only():
    environ = {'OMP_NUM_THREADS': '4'}
    _apply_local_safe_threads(environ)
    assert environ['OMP_NUM_THREADS'] == '4'
    for name, value in LOCAL_SAFE_THREAD_ENV.items():
        assert environ[name] == ('4' if name == 'OMP_NUM_THREADS' else value)


def test_validation_abacus_mocks_parser_accepts_cosmo_params():
    parser = _get_parser()
    args = parser.parse_args(['--cosmo_params', 'base'])
    assert args.cosmo_params == 'base'

    args = parser.parse_args(['--cosmo_params', 'base_ns-fixed'])
    assert args.cosmo_params == 'base_ns-fixed'

    args = parser.parse_args(['--cosmo_params', 'fixed'])
    assert args.cosmo_params == 'fixed'


def test_validation_abacus_mocks_parser_accepts_prior_basis():
    parser = _get_parser()
    args = parser.parse_args(['--prior_basis', 'standard'])
    assert args.prior_basis == 'standard'


def test_validation_abacus_mocks_parser_accepts_stats_and_cache_dirs():
    parser = _get_parser()
    args = parser.parse_args(['--stats_dir', '/tmp/stats', '--cache_dir', '/tmp/cache'])
    assert args.stats_dir == '/tmp/stats'
    assert args.cache_dir == '/tmp/cache'


def test_validation_abacus_mocks_parser_defaults_prior_basis_to_physical_aap():
    parser = _get_parser()
    args = parser.parse_args([])
    assert args.prior_basis == 'physical_aap'


def test_validation_abacus_mocks_parser_defaults_cosmo_params_to_base():
    parser = _get_parser()
    args = parser.parse_args([])
    assert args.cosmo_params == 'base'


def test_validation_abacus_mocks_parser_help_describes_cosmo_params():
    parser = _get_parser()
    help_text = ' '.join(parser.format_help().split())
    assert '--cosmo_params' in help_text
    assert 'base varies h,' in help_text
    assert 'omega_cdm, omega_b, logA, n_s' in help_text
    assert 'base_ns-fixed varies h,' in help_text
    assert 'omega_cdm, omega_b, logA' in help_text
    assert 'fixed varies only nuisance parameters' in help_text
    assert '--prior_basis' in help_text


class _FakeStatsAt:
    def __init__(self, owner):
        self.owner = owner

    def __call__(self, **kwargs):
        return self

    @property
    def observable(self):
        return self

    @property
    def at(self):
        return self

    def replace(self, leaf):
        return self.owner

    def match(self, other):
        self.owner.matched = other
        return self.owner


class _FakeStatsObject:
    def __init__(self, source):
        self.source = source
        self.attrs = {}
        self.at = _FakeStatsAt(self)
        self.matched = None

    def get(self, *args, **kwargs):
        return self

    def labels(self, return_type=None, level=None):
        if return_type == 'keys':
            return ['ells']
        return []

    def edges(self, coord_name):
        return [np.array([[0.0, 0.005]])]

    def select(self, **kwargs):
        return self

    def value(self):
        return np.array([1.0])

    def clone(self, value=None):
        clone = type(self)(self.source)
        clone.attrs = dict(self.attrs)
        return clone


class _FakeGaussianLikelihood:
    def __init__(self, observable, window, covariance):
        self.payload = (observable, window, covariance)


class _FakeMPICOMM:
    rank = 0

    def allgather(self, value):
        return [value]

    def bcast(self, value, root=0):
        return value


def _make_cache_test_observables(stats_dir, kmax):
    return [{
        'stat': {
            'kind': 'mesh2_spectrum',
            'select': [
                {'ells': 0, 'k': [0.02, kmax, 0.005]},
                {'ells': 2, 'k': [0.02, kmax, 0.005]},
            ],
        },
        'catalog': {
            'version': 'abacus-2ndgen-dr2-complete',
            'tracer': 'LRG1',
            'zrange': (0.4, 0.6),
            'region': 'GCcomb',
            'stats_dir': stats_dir,
        },
    }]


def _make_cache_test_covariance_options(stats_dir):
    return {
        'source': 'mock',
        'version': 'holi-v3-altmtl',
        'stats_dir': stats_dir,
        'imock': [0, 1],
        'corrections': [],
    }


def _make_prepared_cache_fn(cache_dir, kind, observables_options, covariance_options, kwargs):
    full_options = {
        'observables': [{name: dict(observable_options[name]) for name in ['stat', 'catalog']} for observable_options in observables_options],
    }
    level = {'stat': 1, 'catalog': 2, 'covariance': 0}
    if kind == 'covariance':
        full_options['covariance'] = covariance_options
        level['covariance'] = 1
    str_from_options = str_from_likelihood_options(full_options, level=level)
    hash_options = tools._hash_options(full_options | kwargs)
    return Path(cache_dir) / 'prepared_stats' / f'{kind}_{str_from_options}-{hash_options}.h5'


def _write_covariance_manifest(cache_dir, observables_options, covariance_options, covariance_cache_fn, imocks):
    prepared_cache_dir = Path(cache_dir) / 'prepared_stats'
    manifest_fn = prepared_cache_dir / 'covariance_manifest.json'
    full_options = tools._get_prepared_cache_options(observables_options, covariance_options, kind='covariance')
    key = tools._hash_options(full_options, length=32)
    manifest = {
        key: {
            'filename': covariance_cache_fn.name,
            'imocks': list(imocks),
        },
    }
    manifest_fn.write_text(json.dumps(manifest))
    return manifest_fn


def test_get_stats_ignores_stale_covariance_cache_prefix(monkeypatch, tmp_path):
    cache_dir = tmp_path / 'cache'
    prepared_cache_dir = cache_dir / 'prepared_stats'
    prepared_cache_dir.mkdir(parents=True)
    observables_options = _make_cache_test_observables(tmp_path, 0.30)
    stale_observables_options = _make_cache_test_observables(tmp_path, 0.20)
    covariance_options = _make_cache_test_covariance_options(tmp_path)

    data_cache_fn = _make_prepared_cache_fn(cache_dir, 'data', observables_options, covariance_options, {'imocks': [None]})
    window_cache_fn = _make_prepared_cache_fn(cache_dir, 'window', observables_options, covariance_options, {'imocks': [None]})
    stale_covariance_cache_fn = _make_prepared_cache_fn(cache_dir, 'covariance', stale_observables_options, covariance_options, {'imocks': [0, 1]})
    for fn in [data_cache_fn, window_cache_fn, stale_covariance_cache_fn]:
        fn.touch()
    for imock in covariance_options['imock']:
        (tmp_path / f'mesh2_spectrum_{imock}.h5').touch()

    def fake_read(path):
        path = Path(path)
        if path == data_cache_fn:
            return _FakeStatsObject('data-cache')
        if path == window_cache_fn:
            return _FakeStatsObject('window-cache')
        if path == stale_covariance_cache_fn:
            raise AssertionError('stale covariance cache should not be read')
        return _FakeStatsObject(f'mock-{path.name}')

    monkeypatch.setattr(tools.types, 'read', fake_read)
    monkeypatch.setattr(tools.types, 'ObservableTree', lambda *args, **labels: args[0])
    monkeypatch.setattr(tools.types, 'cov', lambda mocks: _FakeStatsObject('built-covariance'))
    monkeypatch.setattr(tools.types, 'GaussianLikelihood', _FakeGaussianLikelihood)
    monkeypatch.setattr(tools, 'unpack_stats', lambda likelihood: likelihood.payload)
    monkeypatch.setattr(tools, '_get_covariance_correction_factor', lambda covariance, observables, options: (1., {'corrections': []}))

    _, _, covariance = tools.get_stats(
        observables_options,
        covariance_options=covariance_options,
        unpack=True,
        get_stats_fn=lambda kind, stats_dir, imock=None, **kwargs: Path(stats_dir) / f'{kind}_{imock}.h5',
        cache_dir=cache_dir,
        cache_mode='r',
        mpicomm=_FakeMPICOMM(),
    )

    assert covariance.source == 'built-covariance'


def test_get_stats_reuses_manifest_covariance_cache_without_raw_mocks(monkeypatch, tmp_path):
    cache_dir = tmp_path / 'cache'
    prepared_cache_dir = cache_dir / 'prepared_stats'
    prepared_cache_dir.mkdir(parents=True)
    observables_options = _make_cache_test_observables(tmp_path, 0.30)
    covariance_options = _make_cache_test_covariance_options(tmp_path)

    data_cache_fn = _make_prepared_cache_fn(cache_dir, 'data', observables_options, covariance_options, {'imocks': [None]})
    window_cache_fn = _make_prepared_cache_fn(cache_dir, 'window', observables_options, covariance_options, {'imocks': [None]})
    covariance_cache_fn = _make_prepared_cache_fn(cache_dir, 'covariance', observables_options, covariance_options, {'imocks': [0, 1]})
    for fn in [data_cache_fn, window_cache_fn, covariance_cache_fn]:
        fn.touch()
    _write_covariance_manifest(cache_dir, observables_options, covariance_options, covariance_cache_fn, [0, 1])

    def fake_read(path):
        path = Path(path)
        if path == data_cache_fn:
            return _FakeStatsObject('data-cache')
        if path == window_cache_fn:
            return _FakeStatsObject('window-cache')
        if path == covariance_cache_fn:
            return _FakeStatsObject('manifest-covariance-cache')
        raise AssertionError(f'raw mock path should not be read: {path}')

    monkeypatch.setattr(tools.types, 'read', fake_read)
    monkeypatch.setattr(tools.types, 'GaussianLikelihood', _FakeGaussianLikelihood)
    monkeypatch.setattr(tools, 'unpack_stats', lambda likelihood: likelihood.payload)
    monkeypatch.setattr(tools, '_get_covariance_correction_factor', lambda covariance, observables, options: (1., {'corrections': []}))

    data, window, covariance = tools.get_stats(
        observables_options,
        covariance_options=covariance_options,
        unpack=True,
        get_stats_fn=lambda kind, stats_dir, imock=None, **kwargs: Path(stats_dir) / f'missing_{kind}_{imock}.h5',
        cache_dir=cache_dir,
        cache_mode='r',
        mpicomm=_FakeMPICOMM(),
    )

    assert data.source == 'data-cache'
    assert window.source == 'window-cache'
    assert covariance.source == 'manifest-covariance-cache'


def test_get_stats_missing_covariance_manifest_requires_raw_mocks(monkeypatch, tmp_path):
    cache_dir = tmp_path / 'cache'
    prepared_cache_dir = cache_dir / 'prepared_stats'
    prepared_cache_dir.mkdir(parents=True)
    observables_options = _make_cache_test_observables(tmp_path, 0.30)
    covariance_options = _make_cache_test_covariance_options(tmp_path)

    data_cache_fn = _make_prepared_cache_fn(cache_dir, 'data', observables_options, covariance_options, {'imocks': [None]})
    window_cache_fn = _make_prepared_cache_fn(cache_dir, 'window', observables_options, covariance_options, {'imocks': [None]})
    covariance_cache_fn = _make_prepared_cache_fn(cache_dir, 'covariance', observables_options, covariance_options, {'imocks': [0, 1]})
    for fn in [data_cache_fn, window_cache_fn, covariance_cache_fn]:
        fn.touch()

    def fake_read(path):
        path = Path(path)
        if path == data_cache_fn:
            return _FakeStatsObject('data-cache')
        if path == window_cache_fn:
            return _FakeStatsObject('window-cache')
        if path == covariance_cache_fn:
            raise AssertionError('covariance cache should require a manifest or raw mock discovery')
        return _FakeStatsObject(f'mock-{path.name}')

    monkeypatch.setattr(tools.types, 'read', fake_read)

    with pytest.raises(FileNotFoundError, match='No covariance mock realizations found'):
        tools.get_stats(
            observables_options,
            covariance_options=covariance_options,
            unpack=True,
            get_stats_fn=lambda kind, stats_dir, imock=None, **kwargs: Path(stats_dir) / f'missing_{kind}_{imock}.h5',
            cache_dir=cache_dir,
            cache_mode='r',
            mpicomm=_FakeMPICOMM(),
        )


def test_get_stats_reuses_exact_covariance_cache(monkeypatch, tmp_path):
    cache_dir = tmp_path / 'cache'
    prepared_cache_dir = cache_dir / 'prepared_stats'
    prepared_cache_dir.mkdir(parents=True)
    observables_options = _make_cache_test_observables(tmp_path, 0.30)
    covariance_options = _make_cache_test_covariance_options(tmp_path)

    data_cache_fn = _make_prepared_cache_fn(cache_dir, 'data', observables_options, covariance_options, {'imocks': [None]})
    window_cache_fn = _make_prepared_cache_fn(cache_dir, 'window', observables_options, covariance_options, {'imocks': [None]})
    covariance_cache_fn = _make_prepared_cache_fn(cache_dir, 'covariance', observables_options, covariance_options, {'imocks': [0, 1]})
    for fn in [data_cache_fn, window_cache_fn, covariance_cache_fn]:
        fn.touch()
    for imock in covariance_options['imock']:
        (tmp_path / f'mesh2_spectrum_{imock}.h5').touch()

    def fake_read(path):
        path = Path(path)
        if path == data_cache_fn:
            return _FakeStatsObject('data-cache')
        if path == window_cache_fn:
            return _FakeStatsObject('window-cache')
        if path == covariance_cache_fn:
            return _FakeStatsObject('exact-covariance-cache')
        return _FakeStatsObject(f'mock-{path.name}')

    monkeypatch.setattr(tools.types, 'read', fake_read)
    monkeypatch.setattr(tools.types, 'ObservableTree', lambda *args, **labels: args[0])
    monkeypatch.setattr(tools.types, 'cov', lambda mocks: (_ for _ in ()).throw(AssertionError('exact covariance cache should prevent rebuilding')))
    monkeypatch.setattr(tools.types, 'GaussianLikelihood', _FakeGaussianLikelihood)
    monkeypatch.setattr(tools, 'unpack_stats', lambda likelihood: likelihood.payload)
    monkeypatch.setattr(tools, '_get_covariance_correction_factor', lambda covariance, observables, options: (1., {'corrections': []}))

    _, _, covariance = tools.get_stats(
        observables_options,
        covariance_options=covariance_options,
        unpack=True,
        get_stats_fn=lambda kind, stats_dir, imock=None, **kwargs: Path(stats_dir) / f'{kind}_{imock}.h5',
        cache_dir=cache_dir,
        cache_mode='r',
        mpicomm=_FakeMPICOMM(),
    )

    assert covariance.source == 'exact-covariance-cache'


def test_get_stats_reuses_data_and_window_cache_across_covariance_changes(monkeypatch, tmp_path):
    cache_dir = tmp_path / 'cache'
    prepared_cache_dir = cache_dir / 'prepared_stats'
    prepared_cache_dir.mkdir(parents=True)
    observables_options = _make_cache_test_observables(tmp_path, 0.30)
    covariance_options = _make_cache_test_covariance_options(tmp_path)
    other_covariance_options = covariance_options | {'version': 'other-covariance'}

    data_cache_fn = _make_prepared_cache_fn(cache_dir, 'data', observables_options, covariance_options, {'imocks': [None]})
    window_cache_fn = _make_prepared_cache_fn(cache_dir, 'window', observables_options, covariance_options, {'imocks': [None]})
    covariance_cache_fn = _make_prepared_cache_fn(cache_dir, 'covariance', observables_options, other_covariance_options, {'imocks': [0, 1]})
    for fn in [data_cache_fn, window_cache_fn, covariance_cache_fn]:
        fn.touch()
    for imock in other_covariance_options['imock']:
        (tmp_path / f'mesh2_spectrum_{imock}.h5').touch()

    assert '+cov-' not in data_cache_fn.name
    assert '+cov-' not in window_cache_fn.name
    assert '+cov-' in covariance_cache_fn.name

    def fake_read(path):
        path = Path(path)
        if path == data_cache_fn:
            return _FakeStatsObject('data-cache')
        if path == window_cache_fn:
            return _FakeStatsObject('window-cache')
        if path == covariance_cache_fn:
            return _FakeStatsObject('covariance-cache')
        return _FakeStatsObject(f'mock-{path.name}')

    monkeypatch.setattr(tools.types, 'read', fake_read)
    monkeypatch.setattr(tools.types, 'ObservableTree', lambda *args, **labels: args[0])
    monkeypatch.setattr(tools.types, 'cov', lambda mocks: (_ for _ in ()).throw(AssertionError('exact covariance cache should prevent rebuilding')))
    monkeypatch.setattr(tools.types, 'GaussianLikelihood', _FakeGaussianLikelihood)
    monkeypatch.setattr(tools, 'unpack_stats', lambda likelihood: likelihood.payload)
    monkeypatch.setattr(tools, '_get_covariance_correction_factor', lambda covariance, observables, options: (1., {'corrections': []}))

    data, window, covariance = tools.get_stats(
        observables_options,
        covariance_options=other_covariance_options,
        unpack=True,
        get_stats_fn=lambda kind, stats_dir, imock=None, **kwargs: Path(stats_dir) / f'{kind}_{imock}.h5',
        cache_dir=cache_dir,
        cache_mode='r',
        mpicomm=_FakeMPICOMM(),
    )

    assert data.source == 'data-cache'
    assert window.source == 'window-cache'
    assert covariance.source == 'covariance-cache'



def test_get_sampler_cls_supports_mcmc():
    from desilike.samplers import EmceeSampler, MetropolisHastingsSampler

    assert tools.get_sampler_cls('emcee') is EmceeSampler
    assert tools.get_sampler_cls('mcmc') is MetropolisHastingsSampler


def test_folpsEFT_nuisance_priors_define_refs():
    mesh2_params = tools._get_default_theory_nuisance_priors(
        model='folpsEFT',
        stat='mesh2_spectrum',
        prior_basis='physical_aap',
        b3_coev=True,
        sigma8_fid=0.8,
    )
    for name in ['b1p', 'b2p', 'bsp', 'alpha0p', 'alpha2p', 'alpha4p', 'sn0p', 'sn2p']:
        assert 'ref' in mesh2_params[name], name
        assert mesh2_params[name]['ref']['dist'] == 'norm'
    assert mesh2_params['b3p']['fixed'] is True
    assert 'ref' not in mesh2_params['b3p']
    assert mesh2_params['X_FoG_pp']['fixed'] is True
    assert 'ref' not in mesh2_params['X_FoG_pp']

    mesh3_params = tools._get_default_theory_nuisance_priors(
        model='folpsEFT',
        stat='mesh3_spectrum',
        prior_basis='physical_aap',
        sigma8_fid=0.8,
    )
    for name in ['b1p', 'b2p', 'bsp', 'c1p', 'c2p', 'Pshotp', 'Bshotp']:
        assert 'ref' in mesh3_params[name], name
        assert mesh3_params[name]['ref']['dist'] == 'norm'
    assert mesh3_params['X_FoG_bp']['fixed'] is True
    assert 'ref' not in mesh3_params['X_FoG_bp']


def test_folpsEFT_sampler_start_is_full_rank():
    from desilike.samplers import EmceeSampler

    options = {}
    options['likelihoods'] = _build_likelihoods_options(
        stats=['mesh2_spectrum', 'mesh3_spectrum'],
        tracers=['LRG1'],
        version='abacus-2ndgen-dr2-complete',
        covariance='holi-v3-altmtl',
        stats_dir=Path('/global/cfs/cdirs/desicollab/science/cai/desi-clustering/dr2/summary_statistics/full_shape/base'),
        theory_model='folpsEFT',
    )
    options['cosmology'] = {'template': 'direct'}
    options = fill_fiducial_options(options)
    likelihood = get_likelihood(options['likelihoods'], cosmology_options=options['cosmology'], cache_dir='./_cache')
    likelihood()

    sampler = EmceeSampler(likelihood, seed=42, nwalkers=32)
    start = sampler._get_start()[0]
    assert np.isfinite(start).all()
    assert np.linalg.matrix_rank(start - start.mean(axis=0, keepdims=True)) == start.shape[-1]


def test_validation_abacus_mocks_reptvelocileptors_rejects_mesh3():
    with pytest.raises(ValueError, match='reptvelocileptors.*mesh2_spectrum'):
        _build_likelihoods_options(
            stats=['mesh2_spectrum', 'mesh3_spectrum'],
            tracers=['LRG1'],
            version='abacus-2ndgen-dr2-complete',
            covariance='holi-v3-altmtl',
            stats_dir=Path('/tmp'),
            theory_model='reptvelocileptors',
        )


def test_run_fit_from_options_resume_false_does_not_pass_chains(monkeypatch, tmp_path):
    calls = {}

    class FakeSampler:
        def __init__(self, likelihood, **kwargs):
            calls['init'] = kwargs

        def run(self, **kwargs):
            calls['run'] = kwargs

    monkeypatch.setattr('full_shape.run_fit.get_likelihood', lambda *args, **kwargs: lambda: None)
    monkeypatch.setattr('full_shape.run_fit.tools.write_options', lambda *args, **kwargs: None)
    monkeypatch.setattr('full_shape.run_fit.tools.get_sampler_cls', lambda name: FakeSampler)

    def fake_get_fits_fn(kind='chain', ichain=None, ext='npy', **kwargs):
        filename = kind if ichain is None else f'{kind}_{ichain}'
        return tmp_path / f'{filename}.{ext}'

    run_fit_from_options(
        actions=['sample'],
        get_fits_fn=fake_get_fits_fn,
        likelihoods=[generate_likelihood_options_helper(tracer='LRG1')],
        sampler={'sampler': 'mcmc', 'nchains': 2, 'resume': False, 'run': {'thin_by': 3}, 'init': {}},
    )

    assert calls['init']['save_fn'] == [tmp_path / 'chain_0.npy', tmp_path / 'chain_1.npy']
    assert 'chains' not in calls['init']
    assert calls['run'] == {'thin_by': 3}


def test_run_fit_from_options_resume_true_passes_existing_chains(monkeypatch, tmp_path):
    calls = {}
    chain_paths = [tmp_path / 'chain_0.npy', tmp_path / 'chain_1.npy']
    for path in chain_paths:
        path.touch()

    class FakeSampler:
        def __init__(self, likelihood, **kwargs):
            calls['init'] = kwargs

        def run(self, **kwargs):
            calls['run'] = kwargs

    monkeypatch.setattr('full_shape.run_fit.get_likelihood', lambda *args, **kwargs: lambda: None)
    monkeypatch.setattr('full_shape.run_fit.tools.write_options', lambda *args, **kwargs: None)
    monkeypatch.setattr('full_shape.run_fit.tools.get_sampler_cls', lambda name: FakeSampler)

    def fake_get_fits_fn(kind='chain', ichain=None, ext='npy', **kwargs):
        filename = kind if ichain is None else f'{kind}_{ichain}'
        return tmp_path / f'{filename}.{ext}'

    run_fit_from_options(
        actions=['sample'],
        get_fits_fn=fake_get_fits_fn,
        likelihoods=[generate_likelihood_options_helper(tracer='LRG1')],
        sampler={'sampler': 'emcee', 'nchains': 2, 'resume': True, 'run': {'thin_by': 2}, 'init': {}},
    )

    assert calls['init']['save_fn'] == chain_paths
    assert calls['init']['chains'] == chain_paths
    assert calls['run'] == {'thin_by': 2}


def test_run_fit_from_options_resume_true_requires_existing_chains(monkeypatch, tmp_path):
    monkeypatch.setattr('full_shape.run_fit.get_likelihood', lambda *args, **kwargs: lambda: None)
    monkeypatch.setattr('full_shape.run_fit.tools.write_options', lambda *args, **kwargs: None)
    monkeypatch.setattr('full_shape.run_fit.tools.get_sampler_cls', lambda name: object)

    def fake_get_fits_fn(kind='samples', ichain=None, ext='h5', **kwargs):
        filename = kind if ichain is None else f'{kind}_{ichain}'
        return tmp_path / f'{filename}.{ext}'

    with pytest.raises(FileNotFoundError, match='missing chain file'):
        run_fit_from_options(
            actions=['sample'],
            get_fits_fn=fake_get_fits_fn,
            likelihoods=[generate_likelihood_options_helper(tracer='LRG1')],
            sampler={'sampler': 'mcmc', 'nchains': 2, 'resume': True, 'run': {'thin_by': 1}, 'init': {}},
        )


if __name__ == '__main__':

    setup_logging()
    #test_likelihood_bao(load=True)
    test_likelihood_full_shape(load=False, save=True, theory='comet')
    #test_covariance()
    #test_str()
    #test_covariance()
    #test_options()
