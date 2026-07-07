"""
Script to run fits with region splits tests
To create and spawn the tasks on NERSC, use the following commands:
```bash
salloc -N 1 -C gpu -t 04:00:00 --gpus 4 --qos interactive --account desi_g
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
export PYTHONPATH=/global/u1/s/shengyu/Y3/blinded_data_splits/desi-clustering:$PYTHONPATH
srun -n 4 python test_data_splits.py 
```
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from full_shape import tools, setup_logging
setup_logging()

THEORY_MODELS = ['folpsD', 'folpsEFT', 'reptvelocileptors']
COSMO_MODELS = ['base', 'base_ns-fixed', 'fixed']
PRIOR_BASES = ['physical', 'physical_aap', 'tcm_chudaykin_aap', 'standard']
SAMPLERS = ['emcee', 'mcmc']
KRANGES = {
    'mesh2_spectrum': [
        {'ells': 0, 'k': [0.02, 0.20, 0.005]},
        {'ells': 2, 'k': [0.02, 0.20, 0.005]},
        # {'ells': 4, 'k': [0.02, 0.30, 0.005]},
    ],
    'mesh3_spectrum': [
        {'ells': (0, 0, 0), 'k': [0.02, 0.12, 0.005]},
        {'ells': (2, 0, 2), 'k': [0.02, 0.08, 0.005]},
    ],
}

def _validate_theory_model(stats, theory_model):
    if theory_model == 'reptvelocileptors' and 'mesh3_spectrum' in stats:
        raise ValueError('theory model reptvelocileptors is only supported with mesh2_spectrum')

def _apply_kranges(observable_options):
    stat = observable_options['stat']['kind']
    if stat not in KRANGES:
        return
    observable_options['stat']['select'] = [
        {'ells': item['ells'], 'k': list(item['k'])}
        for item in KRANGES[stat]
    ]

def _build_likelihoods_options(stats, tracers, regions, version, covariance, stats_dir, project,
                               theory_model, cov_stats_dir=None, prior_basis='physical_aap',
                               weight='default-FKP', cut=False, auw=False):
    _validate_theory_model(stats, theory_model)
    likelihoods = []
    for tracer in tracers:
        for region in regions:
            likelihood_options = tools.generate_likelihood_options_helper(
                stats=stats,
                tracer=tracer,
                region=region,
                version=version,
                covariance=covariance,
                stats_dir=stats_dir,
                project=project,
            )
            if cov_stats_dir is not None:
                likelihood_options['covariance']['stats_dir'] = cov_stats_dir
                likelihood_options['covariance']['version'] = covariance
            for observable_options in likelihood_options['observables']:
                stat = observable_options['stat']['kind']
                observable_options['catalog']['weight'] = weight
                # Only mesh2_spectrum has dedicated theta-cut / AUW measurement variants.
                observable_options['catalog']['cut'] = bool(cut) if stat == 'mesh2_spectrum' else False
                observable_options['catalog']['auw'] = bool(auw) if stat == 'mesh2_spectrum' else False
                _apply_kranges(observable_options)
                observable_options.setdefault('theory', {})
                observable_options['theory']['model'] = theory_model
                observable_options['theory']['prior_basis'] = prior_basis
            likelihoods.append(likelihood_options)
    return likelihoods

def _build_run_options(stats, tracers, regions, version, covariance, stats_dir, theory_model,
                       project='', cov_stats_dir=None, cosmo_model='base',
                       template='direct', sampler='emcee', nchains=1, thinning=1, resume=False,
                       prior_basis='physical_aap', weight='default-FKP', cut=False, auw=False):
    options = {}
    options['likelihoods'] = _build_likelihoods_options(
        stats=stats,
        tracers=tracers,
        regions=regions,
        version=version,
        covariance=covariance,
        stats_dir=stats_dir,
        project=project,
        cov_stats_dir=cov_stats_dir,
        theory_model=theory_model,
        prior_basis=prior_basis,
        weight=weight,
        cut=cut,
        auw=auw,
    )
    options['cosmology'] = {'template': template, 'model': cosmo_model}
    options['sampler'] = tools.propose_fiducial_sampler_options(sampler=sampler)
    sampler_kw = {'nparallel': nchains, 'thinning': thinning}
    for section in ['init', 'run']:
        for name, value in options['sampler'][section].items():
            if name in sampler_kw:
                options['sampler'][section][name] = sampler_kw[name]
    options['sampler']['resume'] = resume
    return tools.fill_fiducial_options(options)

def run_fit(actions=('profile',), template='direct', stats=['mesh2_spectrum'],
            version=None, tracers=None,  regions=None,
            stats_dir=None, fits_dir=None, project=None,
            covariance='holi-v3-altmtl', cov_stats_dir=None, cache_dir = None, 
            theory_model='folpsD', cosmo_model='base', 
            sampler='emcee', nchains=1, 
            thinning=1, resume=False, prior_basis='physical_aap',
            weight='default-FKP', cut=False, auw=False):
    # Everything inside this function will be executed on the compute nodes;
    # This function must be self-contained; and cannot rely on imports from the outer scope.
    import os
    from pathlib import Path
    import functools
    from mpi4py import MPI
    mpicomm = MPI.COMM_WORLD
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(mpicomm.rank)
    import jax
    from jax import config
    config.update('jax_enable_x64', True)
    from full_shape import run_fit_from_options, setup_logging
    from full_shape.tools import get_likelihood
    from desilike.samples import Profiles
    # You can pass region, version, covariance, ...
    options = _build_run_options(
        stats=stats,
        tracers=tracers,
        regions=regions,
        version=version,
        covariance=covariance,
        stats_dir=stats_dir,
        project=project,
        cov_stats_dir=cov_stats_dir,
        theory_model=theory_model,
        cosmo_model=cosmo_model,
        template=template,
        sampler=sampler,
        nchains=nchains,
        thinning=thinning,
        resume=resume,
        prior_basis=prior_basis,
        weight=weight,
        cut=cut,
        auw=auw,
    )
    get_fits_fn = functools.partial(tools.get_fits_fn, fits_dir=fits_dir, project=project)
    run_fit_from_options(actions, **options, get_fits_fn=get_fits_fn, cache_dir= cache_dir if cache_dir else fits_dir / '_cache')
    if 'profile' in actions:
        likelihood = get_likelihood(likelihoods_options=options['likelihoods'],
                                    cosmology_options=options['cosmology'],
                                    cache_dir=cache_dir if cache_dir else fits_dir / '_cache')
        profiles = Profiles.read(get_fits_fn(kind='profiles', **options))
        # Evaluate likelihood at dictionary of parameters
        best = profiles.choice(index='argmax', squeeze=True).select(input=True).best
        compile(likelihood)(**best)
        if mpicomm.rank == 0:
            plot_dir = get_fits_fn(kind='profiles', **options).parent
            for ilikelihood, sublikelihood in enumerate(likelihood.likelihoods):
                for iobservable, observable in enumerate(sublikelihood.observables):
                    #plot_covariance = sublikelihood.covariance.at.observable.get(observables=observable.name)
                    #plot_covariance = plot_covariance.at.observable.match(observable.data)
                    #observable.covariance = plot_covariance
                    observable.plot(fn=plot_dir / f'plot_likelihood{ilikelihood}_observable{iobservable}.png')

########################################################################################################################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, default='data-dr2-v2', help='Blinded dataset to fit.')
    parser.add_argument('--todo', type=str, nargs='*', default=['profile'], choices=['build', 'profile', 'sample'],
                        help='Run build, profile, and / or sample. Defaults to profile.')
    parser.add_argument('--stats', type=str, nargs='*', default=['mesh2_spectrum', 'mesh3_spectrum'], 
                        choices=['mesh2_spectrum', 'mesh3_spectrum'], help='Statistics to fit. Defaults to mesh2_spectrum.')
    parser.add_argument('--theory_model', type=str, default='folpsD',
                        choices=THEORY_MODELS, help='Theory model to fit. Defaults to folpsD.')
    parser.add_argument('--prior_basis', type=str, default='physical_aap',
                        choices=PRIOR_BASES, help='Nuisance-parameter prior basis. Defaults to physical_aap.')
    parser.add_argument('--cosmo_params', type=str, default='base',
                        choices=COSMO_MODELS,
                        help='Cosmology parameter setup to fit. base varies h, omega_cdm, omega_b, logA, n_s; '
                             'base_ns-fixed varies h, omega_cdm, omega_b, logA; '
                             'fixed varies only nuisance parameters. Defaults to base.')
    parser.add_argument('--sampler', type=str, default='emcee',
                        choices=SAMPLERS, help='desilike sampler backend to use. Defaults to emcee.')
    # parser.add_argument('--tracers', nargs='+', default=None)
    # parser.add_argument('--regions', nargs='+', default=None,
                        # help='Sky regions to include. Defaults to GCcomb.')  
    parser.add_argument('--weight', type=str, default='default-FKP',
                        help='Measurement weight type to read, e.g. default-FKP.')
    parser.add_argument('--thetacut', action='store_true',
                        help='Read theta-cut measurement variants with the _thetacut suffix.')
    parser.add_argument('--auw', action='store_true',
                        help='Read angular-upweighted measurement variants with the _auw suffix.')
    parser.add_argument('--fits_dir', type=str, default=None,
                        help='Base directory for fits results.')
    parser.add_argument('--cache_dir', type=str, default=None,
                        help='Base directory for cached prepared stats and emulators.')
    parser.add_argument('--project', type=str, default='',
                        help='Measurement project subdirectory under stats_dir.')
    parser.add_argument('--covariance', type=str, default='holi-v3-altmtl',
                        help='Covariance mock version. Defaults to holi-v3-altmtl.')
    parser.add_argument('--cov_stats_dir', type=str, default=None,
                        help='Base directory for covariance mocks. Defaults to stats_dir.')
    parser.add_argument('--nchains', type=int, default=4,
                        help='Number of MCMC chains to run with desilike. Defaults to 1.')
    parser.add_argument('--thinning', type=int, default=1,
                        help='Thin samples by this factor while the desilike sampler is running. Defaults to 1.')
    parser.add_argument('--resume', action='store_true',
                        help='Resume sampling from existing chain files in the derived fits directory.')
    args = parser.parse_args()

    stats_dir = Path('/global/cfs/cdirs/desi/science/cai/desi-clustering/dr2/summary_statistics/full_shape/data_splits')
    # fits_dir = Path('/global/cfs/cdirs/desi/science/cai/desi-clustering/dr2/fits/')
    fits_dir = Path(args.fits_dir) if args.fits_dir is not None else Path(os.getenv('SCRATCH', '.')) / 'tests'
    cache_dir = Path(args.cache_dir) if args.cache_dir is not None else fits_dir / '_cache'
    project = args.project or 'blinded_data'

    covariance = args.covariance
    cov_stats_dir = Path(args.cov_stats_dir) if args.cov_stats_dir is not None else stats_dir

    stats = args.stats
    version = args.version

    # thetacut = args.thetacut
    # auw = args.auw
    auw = False
    thetacut= False
    
    fit_tracers = ['LRG1', 'LRG2', 'LRG3', 'QSO1'][:]
    fit_regions = ['GCcomb', 'NS', 'GCcomb_noN', 'GCcomb_noDES'][:]
    fit_regions += ['NGC', 'SGC', 'N', 'NGCnoN', 'S', 'SGCnoDES']

    _validate_theory_model(stats, args.theory_model)

    for tracer in fit_tracers:
        for region in fit_regions:
            run_fit(actions=args.todo, stats=stats, #what to run
                    version=version, tracers=[tracer], regions=[region], # dataset settings
                    stats_dir=stats_dir, fits_dir=fits_dir, project=project, # IO settings
                    covariance=covariance, cov_stats_dir=cov_stats_dir, cache_dir=cache_dir, 
                    theory_model=args.theory_model, # fitting model settings
                    cosmo_model=args.cosmo_params, sampler=args.sampler, nchains=args.nchains, # sampling settings
                    thinning=args.thinning, resume=args.resume, prior_basis=args.prior_basis,
                    weight=args.weight, cut=thetacut, auw=auw)