#!/usr/bin/env python
"""Launch DESI cosmology inference with desilike."""

from cosmo.desilike.mapping_likelihoods import LIKELIHOOD_COMBINATIONS, normalize_likelihood_combination, install_likelihoods
from cosmo.desilike.run import (get_likelihood_label, get_desilike_output,
                                propose_fiducial_profiler_options, propose_fiducial_sampler_options)


def profile(likelihoods, model='base', engine='class',
            run='run1', output_dir=None, output_label=None, profiler='minuit', timing=False, **kwargs):
    """Build posterior and run profiling for one configuration."""
    import os
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'
    from desilike import distributed
    try: distributed.initialize()
    except RuntimeError: print('Distributed environment already initialized')
    from desilike import setup_logging
    from cosmo.desilike.run import get_posterior, time_posterior, profile_desilike as _profile
    setup_logging()
    posterior = get_posterior(likelihoods, model=model, engine=engine, truncate_priors=True)
    if timing:
        time_posterior(posterior)
        return
    output_fn = get_desilike_output(model=model, engine=engine, likelihoods=likelihoods,
                                    kind='profiles', output_dir=output_dir, run=run, output_label=output_label)
    options = propose_fiducial_profiler_options(profiler)
    _profile(posterior, kernel=profiler, init=options['init'], run=options['maximize'], output_fn=output_fn)


def sample(likelihoods, model='base', engine='class',
           run='run1', output_dir=None, output_label=None, sampler='pocomc', resume=False, **kwargs):
    """Build posterior and run sampling for one configuration."""
    import os
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'
    from desilike import distributed
    try: distributed.initialize()
    except RuntimeError: print('Distributed environment already initialized')
    from desilike import setup_logging
    from cosmo.desilike.run import get_posterior, sample_desilike as _sample
    setup_logging()
    posterior = get_posterior(likelihoods, model=model, engine=engine, truncate_priors=True)
    output_dir_path = get_desilike_output(model=model, engine=engine, likelihoods=likelihoods,
                                          kind='samples', output_dir=output_dir, run=run, output_label=output_label)
    options = propose_fiducial_sampler_options(sampler)
    _sample(posterior, kernel=sampler, init=options['init'], run=options['run'],
                   output_dir=output_dir_path, resume=resume)


def _iter_configs(todo, models, likelihoods, **kwargs):
    models = models or [None]
    for task in todo:
        for model in models:
            for value in likelihoods:
                expanded = normalize_likelihood_combination(value)
                label = get_likelihood_label(expanded)
                yield task, dict(model=model, likelihoods=expanded, output_label=label, **kwargs)


def _setup_task_manager():
    from desipipe import Environment, Queue, TaskManager, setup_logging
    setup_logging()
    queue = Queue('run_desilike')
    queue.clear(kill=False)
    environ = Environment('nersc-cosmodesi')
    environ.update({name: '1' for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']})
    tm = TaskManager(queue=queue, environ=environ)
    provider = dict(provider='nersc', time='04:00:00', mpiprocs_per_worker=4,
                    nodes_per_worker=0.5)
    tm_profile = tm.clone(scheduler=dict(max_workers=20), provider=provider)
    tm_sample = tm.clone(scheduler=dict(max_workers=20), provider=provider)
    return tm_sample, tm_profile


if __name__ == '__main__':

    todo = ['profile', 'sample'][:1]
    models = ['base']
    #likelihoods = [['desi-dr2-bao-all', 'desdovekie'], 'CMB-SP4A']
    #likelihoods = ['desi-dr2-bao-lya-fs']
    #likelihoods = [['abacus-dr2-fs-s2-s3-all-comet']]
    likelihoods = [['abacus-dr2-fs-s2-s3-bgs-folpsD', 'desdovekie']]
    #likelihoods = [['desi-dr2-bao-all', 'planck2018-lowl-TT', 'planck2018-lowl-EE', 'planck2018-highl-plik-TTTEEE']]
    #likelihoods = [['desi-dr2-bao-all', 'desdovekie']]
    engine = 'ace'
    #engine = 'camb'
    #engine = None  # per-likelihood default: eisenstein_hu (comet FS), ACE emulators (folpsD FS), class otherwise
    run = 'run1'
    output_dir = None
    resume = False
    interactive = True

    if False:  # install
        for task, config in _iter_configs(todo, models, likelihoods, engine=engine, run=run, output_dir=output_dir):
            install_likelihoods(config['likelihoods'])

    if not interactive:
        tm_sample, tm_profile = _setup_task_manager()
        profile = tm_profile.python_app(profile)
        sample = tm_sample.python_app(sample)
    
    for task, config in _iter_configs(todo, models, likelihoods, engine=engine, run=run, output_dir=output_dir):
        if task == 'profile':
            profile(**config)
        else:
            sample(resume=resume, sampler='nautilus', **config)
