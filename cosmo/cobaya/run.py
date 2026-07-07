"""Cobaya helpers for DESI cosmology inference.

This module is intentionally small and dictionary-driven. It provides a
``desi-clustering``-style entry point for generating and running Cobaya
configurations from canonical likelihood names registered in ``cosmo.cobaya.mapping_likelihoods``.
"""

import os
from pathlib import Path

from cosmo.cobaya.mapping_likelihoods import make_list
from cosmo.cobaya.mapping_likelihoods import normalize_likelihood_combination
from cosmo.cobaya.mapping_likelihoods import get_cobaya_likelihoods, get_cobaya_priors
from cosmo.cobaya.mapping_likelihoods import get_parameterization as get_likelihood_parameterization, normalize_likelihoods
from cosmo.cobaya.parameters import SUPPORTED_MODELS, SUPPORTED_THEORIES, get_cobaya_params


DEFAULT_COSMO_OUTPUT_DIR = Path(os.getenv('SCRATCH', '.')) / 'desi-clustering' / 'cosmo'


def get_parameterization(likelihoods=None, dataset=None):
    """Return the cosmological parameterization needed for likelihoods."""
    return get_likelihood_parameterization(likelihoods=normalize_likelihood_combination(likelihoods), dataset=dataset)

def get_likelihood_label(likelihoods=None, dataset=None):
    """Return a filesystem-friendly label for likelihoods or a dataset."""
    if isinstance(likelihoods, str) and ',' not in likelihoods:
        return likelihoods
    return '_'.join(normalize_likelihoods(likelihoods=normalize_likelihood_combination(likelihoods), dataset=dataset))


def get_cobaya_output(model='base', theory='camb', likelihoods=None, dataset='desi-dr2-bao-all', sampler='cobaya', output_dir=None,
                      run='run1', suffix=True, ext=None, output_label=None):
    """Return the Cobaya output prefix for a configuration."""
    if output_dir is None:
        output_dir = DEFAULT_COSMO_OUTPUT_DIR
    selected_dataset = None if likelihoods is not None and dataset == 'desi-dr2-bao-all' else dataset
    label = output_label or get_likelihood_label(likelihoods=likelihoods, dataset=selected_dataset)
    output = Path(output_dir) / sampler / theory / run / model / label
    output = output / ('bestfit' if sampler not in {'cobaya', 'mcmc'} else 'chain')
    output = str(output)
    if suffix and sampler not in {'cobaya', 'mcmc'}:
        output += '.bestfit'
    if ext:
        output += f'.{ext.lstrip(".")}'
    return output


def get_cobaya_sampler(sampler='cobaya', likelihoods=None, dataset='desi-dr2-bao-all', seed=None, test=False, temperature=1., **kwargs):
    """Return a Cobaya sampler block."""
    if sampler in {'evaluate', 'eval'}:
        return {'evaluate': None}
    if sampler in {'cobaya', 'mcmc'}:
        config = {
            'drag': False,
            'oversample_power': 0.4,
            'proposal_scale': 1.9,
            'temperature': temperature,
            'Rminus1_stop': 0.01,
            'Rminus1_cl_stop': 0.02,
            'seed': seed,
            'max_tries': 1000,
        }
        config.update(kwargs)
        return {'mcmc': config}
    for minimizer in ['iminuit', 'bobyqa', 'scipy']:
        if minimizer in sampler:
            config = {
                'method': minimizer,
                'ignore_prior': False,
                'max_evals': int(1e6),
                'best_of': 4,
                'confidence_for_unbounded': 0.9999995,
                'seed': seed,
            }
            config.update(kwargs)
            return {'minimize': config}
    raise ValueError(f'Unknown sampler {sampler!r}.')


def get_cobaya_info(model='base', theory='camb', likelihoods=None, dataset='desi-dr2-bao-all', sampler='cobaya', output_dir=None,
                    run='run1', seed=None, test=False, output=True, save_fn=None, python_path=None,
                    likelihood_package=None, likelihood_path=None, output_label=None, **sampler_options):
    """Return a Cobaya info dictionary for DESI cosmology inference."""
    selected_likelihoods = normalize_likelihood_combination(likelihoods)
    selected_dataset = None if selected_likelihoods is not None and dataset == 'desi-dr2-bao-all' else dataset
    cobaya_likelihoods = get_cobaya_likelihoods(
        likelihoods=selected_likelihoods,
        dataset=selected_dataset,
        python_path=python_path,
        likelihood_package=likelihood_package,
        likelihood_path=likelihood_path,
    )
    parameterization = get_parameterization(likelihoods=selected_likelihoods, dataset=selected_dataset)
    params, extra_args = get_cobaya_params(model=model, theory=theory, parameterization=parameterization,
                                           likelihoods=normalize_likelihoods(likelihoods=selected_likelihoods,
                                                                             dataset=selected_dataset))
    sampler_block = get_cobaya_sampler(sampler=sampler, likelihoods=selected_likelihoods, dataset=selected_dataset, seed=seed, test=test, **sampler_options)
    info = {
        'theory': {theory: {'extra_args': extra_args}},
        'likelihood': cobaya_likelihoods,
        'params': params,
        'sampler': sampler_block,
    }
    priors = get_cobaya_priors(likelihoods=selected_likelihoods, dataset=selected_dataset)
    if priors:
        info['prior'] = priors
    if output and 'evaluate' not in sampler_block:
        info['output'] = get_cobaya_output(model=model, theory=theory, likelihoods=selected_likelihoods, dataset=selected_dataset, sampler=sampler,
                                           output_dir=output_dir, run=run, suffix='cobaya' not in sampler,
                                           output_label=output_label)
    if test or 'evaluate' in sampler_block:
        info['stop_at_error'] = True
    if test:
        info['test'] = True
        info['debug'] = True
    if save_fn is not None:
        write_cobaya_yaml(info, save_fn)
    return info


def write_cobaya_yaml(info, filename):
    """Write a Cobaya info dictionary to YAML."""
    import yaml
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    with filename.open('w') as file:
        yaml.dump(info, file, sort_keys=False)
    return filename


def sample_cobaya(resume=False, **kwargs):
    """Run Cobaya sampling/evaluate for one cosmology configuration."""
    import glob
    import traceback
    import yaml
    from mpi4py import MPI
    from cobaya import run as cobaya_run
    from cobaya.log import LoggedError

    mpicomm = MPI.COMM_WORLD
    info = get_cobaya_info(**kwargs)
    if mpicomm.rank == 0:
        print(yaml.dump(info, sort_keys=False))
    output = info.get('output')
    mpicomm.Barrier()
    if output is not None:
        for fn in glob.glob(output + '*.lock*'):
            try:
                os.remove(fn)
            except FileNotFoundError:
                pass
    mpicomm.Barrier()

    success = False
    trace = ''
    try:
        cobaya_run(info, force=not bool(resume), resume=bool(resume))
        success = True
    except LoggedError:
        trace = traceback.format_exc()
    success, traces = all(mpicomm.allgather(success)), mpicomm.allgather(trace)
    if not success:
        raise RuntimeError('Cobaya run failed:\n{}'.format('\n'.join(traces)))
    return success


def profile_cobaya(ignore_prior=False, **kwargs):
    """Run a Cobaya minimizer/profile for one cosmology configuration."""
    kwargs.setdefault('sampler', 'iminuit')
    info = get_cobaya_info(**kwargs)
    sampler = info.setdefault('sampler', {}).setdefault('minimize', {})
    sampler['ignore_prior'] = ignore_prior
    import yaml
    from cobaya.run import run
    print(yaml.dump(info, sort_keys=False))
    run(info, force=True)
    return True


def exists_cobaya_output(*args, **kwargs):
    """Return whether a Cobaya chain appears to exist for a configuration."""
    import glob
    output = get_cobaya_output(*args, **kwargs)
    return len(glob.glob(output + '*.*.txt')) > 0


def yield_configs(models=None, likelihoods=None, datasets=None, theory='camb', sampler='cobaya', **kwargs):
    """Yield configuration dictionaries for desipipe task creation."""
    models = make_list(models) or ['base']
    names = make_list(likelihoods if likelihoods is not None else datasets) or ['desi-dr2-bao-all']
    for model in models:
        for name in names:
            yield {'model': model, 'likelihoods': normalize_likelihood_combination(name), 'dataset': None, 'theory': theory, 'sampler': sampler, **kwargs}
