
import os
from pathlib import Path


DEFAULT_COSMO_OUTPUT_DIR = Path(os.getenv('SCRATCH', '.')) / 'desi-clustering' / 'cosmo'


def get_likelihood_label(likelihoods=None):
    """Return a filesystem-friendly label for a likelihood or list of likelihoods."""
    if likelihoods is None:
        return 'none'
    if isinstance(likelihoods, str):
        return likelihoods
    return '_'.join(likelihoods)


def get_desilike_output(model='base', engine='class', likelihoods=None, kind='samples',
                        output_dir=None, run='run1', ext=None, output_label=None):
    """Return the desilike output path for a configuration.

    Parameters
    ----------
    model : str
        Cosmological model, e.g. ``'base'``, ``'base'``, ``'w0wa'``.
    engine : str
        Boltzmann engine, e.g. ``'class'`` or ``'camb'``.
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
    directory = Path(output_dir) / engine / run / model / label
    if kind == 'profiles':
        suffix = f'.{ext.lstrip(".")}' if ext else '.h5'
        return directory / f'profiles{suffix}'
    return directory


def get_posterior(likelihoods, model=None, engine='class', **kwargs):
    """Build and compile a :class:`desilike.base.Posterior` for cosmology inference.

    Parameters
    ----------
    likelihoods : str or list of str
        Likelihood name(s) as registered in ``mapping_likelihoods.get_likelihood``.
    model : str, optional
        Cosmological model string (see :func:`parameters.get_cosmology`).
    engine : str, optional
        Boltzmann solver: ``'class'`` (default) or ``'camb'``.

    Returns
    -------
    posterior : compiled :class:`desilike.base.Posterior`
    """
    from desilike.base import SumLikelihood, Posterior, compile
    from cosmo.desilike.parameters import get_cosmology, get_prior
    from cosmo.desilike.mapping_likelihoods import get_likelihood
    from cosmo.cobaya.mapping_likelihoods import get_parameterization

    if isinstance(likelihoods, str):
        likelihoods = [likelihoods]

    parameterization = get_parameterization(likelihoods=likelihoods)
    cosmo = get_cosmology(model=model, engine=engine, parameterization=parameterization,
                          likelihoods=likelihoods)

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


def propose_fiducial_sampler_options(sampler=None):
    """Return dictionary of default sampler configuration."""
    if sampler is None:
        sampler = 'emcee'
    init, run = {}, {}
    init['rng'] = 42
    if sampler in ['emcee', 'zeus', 'mhmcmc', 'nuts', 'numpyro_nuts', 'numpyro_barker']:
        init['nparallel'] = 4
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
    return profiler.profiles


def sample_desilike(posterior, kernel='pocomc', init: dict=None, run: dict=None, output_dir=None, resume=False):
    import shutil
    from pathlib import Path
    from desilike.samplers import Sampler
    from desilike.conditioning import AffineConditioner
    from desilike.distributed import get_mpicomm
    init = dict(init or {})
    run = dict(run or {})
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
    cls = get_sampler_cls(kernel)
    _non_kernel = ['rng', 'rescale', 'covariance', 'nparallel', 'prior', 'batch_size']
    kernel_obj = cls(**{name: init.pop(name) for name in list(init) if name not in _non_kernel})
    conditioner = AffineConditioner(**{name: init.pop(name, None) for name in ['rescale', 'covariance']})
    sampler = Sampler(posterior, kernel=kernel_obj, output_dir=output_dir, conditioner=conditioner, **init)
    return sampler.run(**run)