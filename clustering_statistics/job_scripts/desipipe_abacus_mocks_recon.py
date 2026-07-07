"""
Script to create and spawn desipipe tasks to compute post-recon clustering measurements
(2pcf `recon_particle2_correlation` and power spectrum `recon_mesh2_spectrum`) on AbacusHF mocks.

To create and spawn the tasks on NERSC, use the following commands:
```bash
salloc -N 1 -C "gpu&hbm80g" -t 04:00:00 --gpus 4 --qos interactive --account desi_g
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
export PYTHONPATH=$HOME/cai-dr2-clustering-products/:$PYTHONPATH
python desipipe_abacus_mocks_recon.py         # create the list of tasks
desipipe tasks -q abacus_mocks_recon          # check the list of tasks
desipipe spawn -q abacus_mocks_recon --spawn  # spawn the jobs
desipipe queues -q abacus_mocks_recon         # check the queue
```
"""
import os
from pathlib import Path
import functools

import numpy as np
from desipipe import Queue, Environment, TaskManager, spawn, setup_logging

from clustering_statistics import tools

setup_logging()

queue = Queue('abacus_mocks_recon')
queue.clear(kill=False)

output, error = 'slurm_outputs/abacus_mocks_recon/slurm-%j.out', 'slurm_outputs/abacus_mocks_recon/slurm-%j.err'
kwargs = {}
environ = Environment('nersc-cosmodesi')
tm = TaskManager(queue=queue, environ=environ)
tm_gpu = tm.clone(scheduler=dict(max_workers=10), provider=dict(provider='nersc', time='03:00:00',
                            mpiprocs_per_worker=4, output=output, error=error, stop_after=1, constraint='gpu'))
tm80 = tm.clone(scheduler=dict(max_workers=10), provider=dict(provider='nersc', time='03:00:00',
                            mpiprocs_per_worker=4, output=output, error=error, stop_after=1, constraint='gpu&hbm80g'))

def run_stats(tracer='LRG', project='', version='abacus-hf-dr2-v2-altmtl', onthefly=None, imocks=[0], stats_dir=Path(os.getenv('SCRATCH')) / 'measurements', stats=['recon_mesh2_spectrum', 'recon_particle2_correlation'], analysis='full_shape', ibatch=None, postprocess=None, jackknife=None, **kwargs):
    # Everything inside this function will be executed on the compute nodes;
    # This function must be self-contained; and cannot rely on imports from the outer scope.
    import os
    import sys
    import functools
    from pathlib import Path
    import jax
    from jax import config
    config.update('jax_enable_x64', True)
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'
    try: jax.distributed.initialize()
    except RuntimeError: print('Distributed environment already initialized')
    else: print('Initializing distributed environment')
    from clustering_statistics import tools, setup_logging, compute_stats_from_options, fill_fiducial_options, postprocess_stats_from_options
    setup_logging()

    cache = {}
    zranges = tools.propose_fiducial('zranges', tracer, analysis=analysis)
    for imock in imocks:
        regions = ['NGC', 'SGC']
        for region in regions:
            options = dict(catalog=dict(version=version, tracer=tracer, zrange=zranges, region=region, imock=imock), recon_mesh2_spectrum={}, recon_particle2_correlation={'jackknife': jackknife} if jackknife else {})

            stats_dir_kws = dict(stats_dir=stats_dir, project=project)
            if onthefly == 'complete':
                options['catalog']['complete'] = {}
                get_stats_fn = functools.partial(tools.get_stats_fn, extra='complete', **stats_dir_kws)
            elif onthefly == 'reshuffle':
                options['catalog']['reshuffle'] = {'merged_data_fn': tools.get_catalog_fn(kind='data', **(options['catalog'] | dict(region='ALL')))}
                get_stats_fn = functools.partial(tools.get_stats_fn, extra='reshuffle', **stats_dir_kws)
            else:
                get_stats_fn = functools.partial(tools.get_stats_fn, **stats_dir_kws)

            options = fill_fiducial_options(options)
            for tracer in options['catalog']:
                options['catalog'][tracer]['expand'] = {'parent_randoms_fn': tools.get_catalog_fn(kind='parent_randoms', version='data-dr2-v2', tracer=tracer, nran=options['catalog'][tracer]['nran']), 'from_data': ['Z', 'WEIGHT_SYS', 'FRAC_TLOBS_TILES']}
            compute_stats_from_options(stats, get_stats_fn=get_stats_fn, cache=cache, **options)

    # postprocess
    if postprocess:
        postprocess_options = dict(catalog=dict(version=version, tracer=tracer, zrange=zranges, imock=imocks[0]), imocks=imocks, combine_regions={'stats': stats}, recon_mesh2_spectrum={}, recon_particle2_correlation={})
        postprocess_stats_from_options(postprocess, get_stats_fn=get_stats_fn, **postprocess_options)


def postprocess_stats(tracer='LRG', analysis='full_shape', project='', version='abacus-hf-dr2-v2-altmtl', onthefly=None, imocks=[0], stats_dir=Path(os.getenv('SCRATCH')) / 'measurements', stats=['recon_mesh2_spectrum', 'recon_particle2_correlation'], postprocess=['combine_regions'], **kwargs):
    from clustering_statistics import postprocess_stats_from_options
    zranges = tools.propose_fiducial('zranges', tracer, analysis=analysis)
    options = dict(catalog=dict(version=version, tracer=tracer, zrange=zranges, imock=imocks[0]), imocks=imocks, combine_regions={'stats': stats}, recon_mesh2_spectrum={}, recon_particle2_correlation={})
    stats_dir_kws = dict(stats_dir=stats_dir, project=project)
    if onthefly == 'complete':
        get_stats_fn = functools.partial(tools.get_stats_fn, extra='complete', **stats_dir_kws)
    elif onthefly == 'reshuffle':
        get_stats_fn = functools.partial(tools.get_stats_fn, extra='reshuffle', **stats_dir_kws)
    else:
        get_stats_fn = functools.partial(tools.get_stats_fn, **stats_dir_kws)

    postprocess_stats_from_options(postprocess, get_stats_fn=get_stats_fn, **options)



if __name__ == '__main__':

    mode = 'slurm'
    stats = ['recon_mesh2_spectrum', 'recon_particle2_correlation']
    postprocess = ['combine_regions']
    version = 'abacus-hf-dr2-v2-altmtl'
    imocks2run = np.arange(25)
    stats_dir = tools.base_stats_dir
    analysis = 'bao'
    project = f'{analysis}/base'
    onthefly = None
    jackknife = None

    max_mocks_per_batch_qso    = 25
    max_mocks_per_batch_others = 5

    for tracer in ['LRG', 'ELG_LOPnotqso', 'LRG+ELG_LOPnotqso', 'QSO']:
        max_mocks_per_batch = max_mocks_per_batch_qso if tracer == 'QSO' else max_mocks_per_batch_others
        if True:
            catalog_check_tracer = tracer.split('+')[0] if '+' in tracer else tracer
            exists, missing = tools.checks_if_exists_and_readable(get_fn=functools.partial(tools.get_catalog_fn, tracer=catalog_check_tracer, region='NGC', version=version), test_if_readable=False, imock=imocks2run)[:2]
            imocks = exists[1]['imock']
            rerun = []
            for zrange in tools.propose_fiducial('zranges', tracer, analysis=analysis):
                for kind in ['recon_mesh2_spectrum', 'recon_particle2_correlation']:
                    rexists, missing, unreadable = tools.checks_if_exists_and_readable(get_fn=functools.partial(tools.get_stats_fn, kind=kind, stats_dir=stats_dir, project=project, tracer=tracer, region='NGC', weight='default-FKP', zrange=zrange, version=version), test_if_readable=True, imock=imocks2run)
                    rerun += [imock for imock in imocks if (imock in unreadable[1]['imock']) or (imock not in rexists[1]['imock'])]
            imocks = sorted(set(rerun))

        def get_run_stats():
            _tm = tm80
            if tracer in ['LRG']:
                _tm = tm_gpu
            return run_stats if mode == 'interactive' else _tm.python_app(run_stats)

        run_stats_kws = dict(tracer=tracer, stats_dir=stats_dir, project=project, version=version, stats=stats, analysis=analysis, onthefly=onthefly, postprocess=postprocess, jackknife=jackknife)
        if stats:
            batch_imocks = np.array_split(imocks, max(len(imocks) // max_mocks_per_batch, 1)) if len(imocks) else []
            for _imocks in batch_imocks:
                get_run_stats()(imocks=_imocks, **run_stats_kws)
