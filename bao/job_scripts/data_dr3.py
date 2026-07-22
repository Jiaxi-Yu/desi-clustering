"""
Script to run BAO fits.
To create and spawn the tasks on NERSC, use the following commands:
```bash
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
python data_dr2.py
```
"""
import argparse
import os
from pathlib import Path

import numpy as np

from bao import tools, setup_logging


setup_logging()


def run_fit(actions=('profile',), tracer='LRG1', data='data-dr2-v1.1', project='base/bao', covariance='jaxpower', region='GCcomb', recenter=False, stats_dir=tools.base_stats_dir, fits_dir=Path(os.getenv('SCRATCH')) / 'fits'):
    # Everything inside this function will be executed on the compute nodes;
    # This function must be self-contained; and cannot rely on imports from the outer scope.
    import os
    from pathlib import Path
    import functools
    from desilike.distributed import get_mpicomm
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'
    from desilike import distributed
    try: distributed.initialize()
    except RuntimeError: print('Distributed environment already initialized')
    else: print('Initializing distributed environment')
    mpicomm = distributed.get_mpicomm()
    import jax
    from jax import config
    config.update('jax_enable_x64', True)
    from bao import run_fit_from_options, setup_logging
    from bao.tools import generate_likelihood_options_helper, fill_fiducial_options, get_likelihood

    mpicomm = get_mpicomm()
    template = 'bao'
    options = {}
    # use post-reconstruction correlation function
    options['likelihoods'] = [generate_likelihood_options_helper(stats=['recon_particle2_correlation'], tracer=tracer, version=data, stats_dir=stats_dir, project=project, emulator=template == 'direct')]
    for likelihood_options in options['likelihoods']:
        # rascalc = analytical covariance
        for observable_options in likelihood_options['observables']:
            observable_options['catalog']['region'] = region
            observable_options['stat']['jackknife'] = {'nsplits': 60}
            observable_options['stat']['project'] = 'bao/with_desi-clustering'
        cov_stats_dir = tools.base_stats_dir
        if covariance == 'jaxpower':
            likelihood_options['covariance'] = {'source': 'jaxpower', 'version': data, 'project': 'bao/with_desi-clustering', 'stats_dir': stats_dir}
        else:
            likelihood_options['covariance'] = {'source': 'rascalc', 'version': data, 'project': 'bao/rascalc', 'stats_dir': cov_stats_dir}
    options['cosmology'] = {'template': template, 'apmode': 'qisoqap'}
    options = fill_fiducial_options(options)

    options['sampler'] = tools.propose_fiducial_sampler_options(sampler='emcee')
    sampler_kw = {'nparallel': mpicomm.size, 'gelman_rubin': 1.04, 'ess': 600}
    # Distribute arguments
    for section in ['init', 'run']:
        for name, value in options['sampler'][section].items():
            if name in sampler_kw:
                options['sampler'][section][name] = sampler_kw[name]
    options['sampler']['resume'] = False
    options['profiler']['maximize']['niterations'] = 5

    # options contains all possible options; print(options) to look at its content
    get_fits_fn = functools.partial(tools.get_fits_fn, fits_dir=fits_dir, project=project, level={'catalog': 3})
    run_fit_from_options(actions, **options, get_fits_fn=get_fits_fn, cache_dir=None)
    mpicomm.Barrier()
    likelihood = get_likelihood(likelihoods_options=options['likelihoods'],
                                cosmology_options=options['cosmology'], cache_dir=None)
    from desilike import compile
    from desilike.samples import Profiles, MCSamples
    output_fn = get_fits_fn(kind='profiles', **options)
    profiles = Profiles.read(output_fn)
    qnames = ['qiso', 'qap']
    if recenter:
        # Hack to complement DR3 blinding
        for qname in qnames:
            profiles.best[qname][...] = 1.
    if mpicomm.rank == 0:
        profiles.write(output_fn)
        print(profiles.to_stats(tablefmt='pretty'))
    # Evaluate likelihood at dictionary of parameters
    best = profiles.choice(index='argmax', squeeze=True).select(input=True).best
    compile(likelihood)(**best)
    if mpicomm.rank == 0:
        plot_dir = output_fn.parent
        for ilikelihood, sublikelihood in enumerate(likelihood.likelihoods):
            for iobservable, observable in enumerate(sublikelihood.observables):
                observable.plot(fn=plot_dir / f'plot_likelihood{ilikelihood}_observable{iobservable}.pdf')
                observable.plot_bao(fn=plot_dir / f'plot_bao_likelihood{ilikelihood}_observable{iobservable}.pdf')
    output_fns = list(get_fits_fn(kind='samples', **options).parent.glob('samples_*.h5'))
    samples = [MCSamples.read(output_fn) for output_fn in output_fns]
    means = MCSamples.concatenate([sample.remove_burnin(0.3) for sample in samples]).mean(qnames)
    for sample, output_fn in zip(samples, output_fns):
        if recenter:
            for qname, mean in zip(qnames, means):
                sample[qname] = sample[qname] - mean + 1.
        if mpicomm.rank == 0:
            sample.write(output_fn)


if __name__ == '__main__':

    stats_dir = tools.base_stats_dir
    fits_dir = tools.base_fits_dir

    #data = 'data-dr3-matterhorn-v2-v0-bao'
    #recenter = True
    #covariance = 'jaxpower'

    data = 'data-dr2-v1.1'
    recenter = True
    covariance = 'rascalc'
    #covariance = 'jaxpower'

    for tracer in ['BGS1', 'LRG1', 'LRG2', 'LRG3', 'ELG2', 'QSO1']:
        for region in ['GCcomb', 'NGC', 'SGC', 'GCcomb_noDES', 'GCcomb_noN', 'NGCnoN', 'SGCnoDES', 'N', 'S'][:1]:
            run_fit(actions=['profile', 'sample'], data=data, project=f'bao/centered_alpha/{data}' if recenter else f'bao/with_desi-clustering/{data}', tracer=tracer, stats_dir=stats_dir, fits_dir=fits_dir, recenter=recenter, covariance=covariance, region=region)