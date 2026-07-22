
import os
from pathlib import Path

from .mapping_likelihoods import get_likelihood_label, get_engine_label


DEFAULT_COSMO_OUTPUT_DIR = Path(os.getenv('SCRATCH', '.')) / 'desi-clustering' / 'cosmo'


def get_desilike_output(model='base', engine=None, likelihoods=None, kind='samples',
                        output_dir=None, run='run1', ext=None, output_label=None):
    """Return the desilike output path for a configuration.

    Parameters
    ----------
    model : str
        Cosmological model, e.g. ``'base'``, ``'base'``, ``'w0wa'``.
    engine : str or dict, optional
        Boltzmann engine, e.g. ``'class'`` or ``'camb'``; ``None`` resolves the
        per-likelihood default and ACE-style engine dicts are labelled ``'ace'``
        (see :func:`~cosmo.desilike.mapping_likelihoods.get_engine_label`).
    likelihoods : str or list of str, optional
        Likelihood name(s) used to build the label.
    kind : str
        Output type: ``'profiles'`` (single ``.h5`` file) or ``'samples'`` (directory).
    output_dir : str or Path, optional
        Base output directory. Defaults to ``DEFAULT_COSMO_OUTPUT_DIR``.
    run : str
        Run identifier, e.g. ``'run1'``.
    ext : str, optional
        File extension override. Only meaningful for ``kind='profiles'``.
    output_label : str, optional
        Override the auto-generated likelihood label.

    Returns
    -------
    Path
        For ``kind='profiles'``: path to the ``.h5`` profiles file.
        For ``kind='samples'``: path to the samples directory.
    """
    if output_dir is None:
        output_dir = DEFAULT_COSMO_OUTPUT_DIR
    label = output_label or get_likelihood_label(likelihoods)
    directory = Path(output_dir) / get_engine_label(engine, likelihoods=likelihoods) / run / model / label
    if kind == 'profiles':
        suffix = f'.{ext.lstrip(".")}' if ext else '.h5'
        return directory / f'profiles{suffix}'
    return directory


def get_posterior(likelihoods, model=None, engine=None, truncate_priors=False, **kwargs):
    """Build and compile a :class:`desilike.base.Posterior` for cosmology inference.

    Parameters
    ----------
    likelihoods : str or list of str
        Likelihood name(s) as registered in ``mapping_likelihoods.get_likelihood``.
    model : str, optional
        Cosmological model string (see :func:`parameters.get_cosmology`).
    engine : str or dict, optional
        Boltzmann solver, e.g. ``'class'`` or ``'camb'``. ``None`` (default) picks
        the per-likelihood default: ``'eisenstein_hu'`` for COMET-only full-shape
        fits, the ACE emulator set for FolpsD-style ones, ``'class'`` otherwise
        (see :func:`~cosmo.desilike.mapping_likelihoods.get_default_engine`).
    truncate_priors : bool, optional
        ACE engines only: intersect the cosmological priors with the emulator
        training ranges (see :func:`parameters.get_cosmology`). Default is ``False``.

    Returns
    -------
    posterior : compiled :class:`desilike.base.Posterior`
    """
    from desilike.base import SumLikelihood, Posterior, compile
    from cosmo.desilike.parameters import get_cosmology, get_prior
    from cosmo.desilike.mapping_likelihoods import get_likelihood, get_parameterization

    if isinstance(likelihoods, str):
        likelihoods = [likelihoods]

    parameterization = get_parameterization(likelihoods=likelihoods)
    cosmo = get_cosmology(model=model, engine=engine, parameterization=parameterization,
                          likelihoods=likelihoods, truncate_priors=truncate_priors)

    all_likes = []
    for name in likelihoods:
        result = get_likelihood(name, cosmo=cosmo, **kwargs)
        if isinstance(result, (list, tuple)):
            all_likes.extend(result)
        else:
            all_likes.append(result)

    likelihood = all_likes[0] if len(all_likes) == 1 else SumLikelihood(all_likes)
    prior = get_prior(likelihood)
    return compile(Posterior(likelihood, prior=prior))


def time_posterior(posterior, nvmap=10, nrepeats=10, rng=42):
    """Time posterior evaluation after JIT compilation, single-point and under vmap.

    Draws evaluation points from the varied parameters' reference distributions,
    then times a jitted single-point call and a jitted ``jax.vmap`` call over
    *nvmap* points in parallel (first call of each excluded as compilation).

    Parameters
    ----------
    posterior : CompiledGraph
        Compiled posterior, as returned by :func:`get_posterior`.
    nvmap : int
        Number of points evaluated in parallel under ``jax.vmap``.
    nrepeats : int
        Number of timed repetitions per configuration.
    rng : int
        Seed for the reference-distribution draws.

    Returns
    -------
    dict
        ``{'jit': ..., 'vmap': ...}``, each with ``'compile'`` (first-call seconds,
        tracing + compilation), ``'mean'`` / ``'best'`` (seconds per call over
        *nrepeats*) and ``'per_point'`` (best seconds divided by the number of
        points in the call).
    """
    import time
    import numpy as np
    import jax

    varied_params = [param for param in posterior.params if not (param.derived or param.fixed)]
    key = jax.random.PRNGKey(rng)
    points = {}
    for param in varied_params:
        key, subkey = jax.random.split(key)
        if param.ref is not None and param.ref.is_proper():
            points[param.name] = np.asarray(param.ref.sample(subkey, shape=(nvmap,) + param.shape))
        else:
            points[param.name] = np.broadcast_to(np.asarray(param.value), (nvmap,) + param.shape).copy()
    single_point = {name: values[0] for name, values in points.items()}

    def logposterior(params):
        return posterior(params)

    # Eager call first: readable traceback on errors, and any one-off lazy setup
    # (data loading, emulator reads, ...) happens outside the timed JIT calls.
    value = jax.block_until_ready(logposterior(single_point))
    if not np.all(np.isfinite(value)):
        print(f'time_posterior: WARNING, non-finite logposterior {value} at the timed point')

    configurations = [('jit', jax.jit(logposterior), single_point, 1),
                      ('vmap', jax.jit(jax.vmap(logposterior)), points, nvmap)]
    timings = {}
    for label, function, arguments, npoints in configurations:
        start = time.perf_counter()
        value = jax.block_until_ready(function(arguments))
        compile_seconds = time.perf_counter() - start
        if not np.all(np.isfinite(value)):
            print(f'time_posterior [{label}]: WARNING, non-finite logposterior {value} at the timed point(s)')
        jax.block_until_ready(function(arguments))  # buffer iteration between compilation and timing
        repeat_seconds = []
        for _ in range(nrepeats):
            start = time.perf_counter()
            jax.block_until_ready(function(arguments))
            repeat_seconds.append(time.perf_counter() - start)
        timings[label] = {'compile': compile_seconds, 'mean': float(np.mean(repeat_seconds)),
                          'best': float(np.min(repeat_seconds)), 'per_point': float(np.min(repeat_seconds)) / npoints}
        print(f'time_posterior [{label}, {npoints} point(s)]: compile {compile_seconds:.2f} s, '
              f'per call {timings[label]["mean"]:.4f} s (best {timings[label]["best"]:.4f} s), '
              f'per point {timings[label]["per_point"]:.4f} s')
    # Graphs going through pure_callback (e.g. CLASS/CAMB engines) are vmapped
    # sequentially, so no speed-up is expected there; pure-JAX graphs (ACE
    # emulators, ...) should approach nvmap.
    print(f'time_posterior: vmap({nvmap}) speed-up over sequential jit calls: '
          f'{timings["jit"]["best"] * nvmap / timings["vmap"]["best"]:.2f}x')
    return timings


def propose_fiducial_sampler_options(sampler=None):
    """Return dictionary of default sampler configuration."""
    if sampler is None:
        sampler = 'emcee'
    init, run = {}, {}
    init['rng'] = 42
    # 'nparallel' (number of independent runs) is left unset: sample_desilike
    # defaults it to one run per MPI rank.
    if sampler in ['emcee', 'zeus', 'mhmcmc', 'nuts', 'numpyro_nuts', 'numpyro_barker']:
        run['min_steps'] = 50
        run['gelman_rubin'] = 1.05
        run['ess'] = 400
    if sampler in ['emcee']:
        init['batch_size'] = 16
        run['thinning'] = 5
    if sampler in ['nuts']:
        init['rescale'] = 'diag'
        init['step_size'] = 0.1
        #run['adaptation'] = dict(initial_step_size=0.01, target_acceptance_rate=0.8, steps=1000, is_mass_matrix_diagonal=False)
        run['adaptation'] = dict(initial_step_size=0.01, target_acceptance_rate=0.8, steps=1000, is_mass_matrix_diagonal=False)
    if sampler in ['numpyro_nuts', 'numpyro_barker']:
        init['rescale'] = 'diag'
        init['step_size'] = 0.1
        run['adaptation'] = dict(steps=500, dense_mass=True)
    if sampler in ['mhmcmc']:
        run['check_every'] = 1000
    if sampler in ['nautilus']:
        init['rescale'] = 'diag'
        init['n_live'] = 1000
        run['n_eff'] = 200
        run['verbose'] = True
    if sampler in ['pocomc']:
        init['batch_size'] = 32
        # Default settings
        #init['n_effective'] = 512
        #init['n_active'] = 256
        #run['n_total'] = 4096  # ESS
        # n_effective *and* n_active must be high enough to get the tails right
        # rescale helps (in case variations of one parameter are much smaller than the others)
        init['rescale'] = 'diag'
        #init['prior'] = 2.
        init['n_effective'] = 1024
        init['n_active'] = 512
        #init['n_effective'] = 2048
        #init['n_active'] = 1024
        init['flow'] = 'nsf6'  # default
        #init['train_config'] = {'epochs': 10000, 'patience': 50, 'batch_size': 512, 'learning_rate': 1e-3}
        run['n_total'] = 2048  # ESS
    fiducial_options = {'kernel': sampler, 'init': init, 'run': run}
    return fiducial_options


def propose_fiducial_profiler_options(profiler=None):
    """Return dictionary of default profiler configuration."""
    if profiler is None:
        profiler = 'minuit'
    fiducial_options = {'kernel': profiler, 'init': {}, 'maximize': {}}
    return fiducial_options


def get_sampler_cls(name):
    """Return sampler class."""
    from desilike.samplers import Emcee, Zeus, MH, PocoMC, Nautilus, BlackjaxNUTS, NumpyroNUTS, NumpyroBarkerMH
    translate = {'emcee': Emcee, 'zeus': Zeus, 'mh': MH, 'pocomc': PocoMC, 'nautilus': Nautilus, 'nuts': BlackjaxNUTS, 'numpyro_nuts': NumpyroNUTS, 'numpyro_barker': NumpyroBarkerMH}
    return translate[name.lower()]


def get_profiler_cls(name):
    """Return profiler class."""
    from desilike.profilers import Minuit
    translate = {'minuit': Minuit}
    return translate[name.lower()]


def profile_desilike(posterior, kernel='minuit', init: dict=None, run: dict=None, output_fn=None):
    from pathlib import Path
    from desilike.profilers import Profiler
    from desilike.conditioning import AffineConditioner
    init = dict(init or {})
    run = dict(run or {})
    if output_fn is not None:
        Path(output_fn).parent.mkdir(parents=True, exist_ok=True)
    cls = get_profiler_cls(kernel)
    kernel_obj = cls(**{name: init.pop(name) for name in list(init) if name not in ['rng', 'rescale', 'covariance']})
    conditioner = AffineConditioner(**{name: init.pop(name, None) for name in ['rescale', 'covariance']})
    profiler = Profiler(posterior, kernel=kernel_obj, output_fn=output_fn, conditioner=conditioner, **init)
    profiler.maximize(**run)
    if profiler.mpicomm.rank == 0:
        print(profiler.profiles.to_stats(tablefmt='pretty'))
    return profiler.profiles


def sample_desilike(posterior, kernel='pocomc', init: dict=None, run: dict=None, output_dir=None, resume=False, profiles_fn=None):
    import logging
    import shutil
    from pathlib import Path
    from desilike.samples import Profiles
    from desilike.samplers import Sampler
    from desilike.conditioning import AffineConditioner
    from desilike.distributed import get_mpicomm
    logger = logging.getLogger('sample_desilike')
    init = dict(init or {})
    run = dict(run or {})
    # One independent run (chain) per MPI rank by default; single-chain runs
    # still get convergence checks through the split-chain Gelman-Rubin.
    init.setdefault('nparallel', get_mpicomm().size)
    if output_dir is not None:
        output_dir = Path(output_dir)
        mpicomm = get_mpicomm()
        if not resume and mpicomm.rank == 0:
            if output_dir.exists():
                for path in output_dir.iterdir():
                    if path.name != 'profiles.h5':
                        path.unlink() if path.is_file() else shutil.rmtree(path)
        mpicomm.Barrier()
        output_dir.mkdir(parents=True, exist_ok=True)
    if profiles_fn is None and output_dir is not None:
        profiles_fn = output_dir / 'profiles.h5'

    if init.get('rescale', False) and init.get('covariance', None) is None:
        profiles = None
        if profiles_fn is not None and Path(profiles_fn).exists():
            profiles = Profiles.read(profiles_fn).choice(index='argmax', squeeze=True)
        if profiles is None:
            logger.warning(f'No profiles found at {profiles_fn}; conditioning will fall back to ref.std().')
        elif profiles.covariance is None:
            logger.warning(f'Covariance is not provided in {profiles_fn}; conditioning will fall back to ref.std().')
        else:
            init['covariance'] = profiles.covariance
            best, error = profiles.best, profiles.error
            if best is not None and error is not None:
                for param in posterior.params.select(varied=True, derived=False):
                    if param.name in error:
                        param.update(ref=dict(dist='norm', loc=best[param.name], scale=error[param.name]))
    cls = get_sampler_cls(kernel)
    _non_kernel = ['rng', 'rescale', 'covariance', 'nparallel', 'prior', 'batch_size']
    kernel_obj = cls(**{name: init.pop(name) for name in list(init) if name not in _non_kernel})
    conditioner = AffineConditioner(**{name: init.pop(name, None) for name in ['rescale', 'covariance']})
    sampler = Sampler(posterior, kernel=kernel_obj, output_dir=output_dir, conditioner=conditioner, **init)
    return sampler.run(**run)