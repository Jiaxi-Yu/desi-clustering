"""
Script to run BAO fits.
To create and spawn the tasks on NERSC, use the following commands:
```bash
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
python mocks_dr2.py
```
"""
import argparse
import os
from pathlib import Path

import numpy as np

from bao import tools, setup_logging

from desipipe import Queue, Environment, TaskManager, spawn, setup_logging

setup_logging()

queue = Queue('mocks_dr2_bao')
queue.clear(kill=False)

output, error = 'slurm_outputs/mocks_dr2_bao/slurm-%j.out', 'slurm_outputs/mocks_dr2_bao/slurm-%j.err'
kwargs = {}
# environ = Environment('nersc-cosmodesi')
environ = Environment('nersc-cosmodesi')
tm = TaskManager(queue=queue, environ=environ)
tm = tm.clone(scheduler=dict(max_workers=10), provider=dict(provider='nersc', time='02:00:00',
                            mpiprocs_per_worker=1, output=output, error=error, nodes_per_worker=0.05))  # 20 jobs / node


def run_fit(actions=('profile',), tracer='LRG1', data='glam-uchuu-v2-altmtl', project='bao/base', imock=0, stats_dir=tools.base_stats_dir, fits_dir=Path(os.getenv('SCRATCH')) / 'fits'):
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
    options['likelihoods'] = [generate_likelihood_options_helper(stats=['recon_particle2_correlation'], tracer=tracer, version=data, stats_dir=stats_dir, project='bao/base', emulator=False, imock=imock)]
    for likelihood_options in options['likelihoods']:
        # rascalc = analytical covariance
        #for observable_options in likelihood_options['observables']:
            #observable_options['stat']['jackknife'] = {'nsplits': 60}
        cov_stats_dir = tools.base_stats_dir
        likelihood_options['covariance'] = {'source': 'rascalc', 'version': 'data-dr2-v1.1', 'project': 'bao/rascalc', 'imock': None, 'stats_dir': cov_stats_dir}
        #likelihood_options['covariance'] = {'source': 'jaxpower', 'version': 'data-dr2-v1.1', 'stats_dir': stats_dir}
    options['cosmology'] = {'template': 'bao', 'apmode': 'qisoqap'}
    options = fill_fiducial_options(options)

    options['sampler'] = tools.propose_fiducial_sampler_options(sampler='emcee')
    sampler_kw = {'nparallel': 4}
    # Distribute arguments
    for section in ['init', 'run']:
        for name, value in options['sampler'][section].items():
            if name in sampler_kw:
                options['sampler'][section][name] = sampler_kw[name]
    options['profiler']['maximize']['niterations'] = 5

    # options contains all possible options; print(options) to look at its content
    get_fits_fn = functools.partial(tools.get_fits_fn, fits_dir=fits_dir, project=project, imock=imock)
    run_fit_from_options(actions, **options, get_fits_fn=get_fits_fn, cache_dir=None)
    likelihood = get_likelihood(likelihoods_options=options['likelihoods'],
                                cosmology_options=options['cosmology'], cache_dir=None)
    fn = get_fits_fn(kind='profiles', **options)
    from desilike import compile
    from desilike.samples import Profiles
    profiles = Profiles.read(fn)
    # Evaluate likelihood at dictionary of parameters
    best = profiles.choice(index='argmax', squeeze=True).select(input=True).best
    compile(likelihood)(**best)
    if mpicomm.rank == 0:
        plot_dir = fn.parent
        for ilikelihood, sublikelihood in enumerate(likelihood.likelihoods):
            for iobservable, observable in enumerate(sublikelihood.observables):
                observable.plot(fn=plot_dir / f'plot_likelihood{ilikelihood}_observable{iobservable}.pdf')
                observable.plot_bao(fn=plot_dir / f'plot_bao_likelihood{ilikelihood}_observable{iobservable}.pdf')


if __name__ == '__main__':

    stats_dir = tools.base_stats_dir
    fits_dir = tools.base_fits_dir
    imocks = list(range(150, 155))
    mode = 'interactive'

    def get_tm(f):
        return f if mode == 'interactive' else tm.python_app(f)

    for imock in imocks:
        for tracer in ['BGS1', 'LRG1', 'LRG2', 'LRG3', 'ELG2', 'QSO1'][1:2]:
            get_tm(run_fit)(actions=['profile'], data='glam-uchuu-v2-altmtl', project='bao/with_desi-clustering', tracer=tracer, imock=imock, stats_dir=stats_dir, fits_dir=fits_dir)