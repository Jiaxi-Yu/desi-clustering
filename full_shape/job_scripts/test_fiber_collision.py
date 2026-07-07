import argparse
import os
from pathlib import Path

from full_shape import setup_logging

setup_logging()


THEORY_MODELS = ["folpsD", "folpsEFT", "reptvelocileptors"]
COSMO_MODELS = ["base", "base_ns-fixed", "fixed"]
PRIOR_BASES = ["physical", "physical_aap", "tcm_chudaykin_aap", "standard"]
SAMPLERS = ["emcee", "mh", "pocomc"]
KRANGES = {
    "mesh2_spectrum": [{"ells": 0, "k": [0.02, 0.30, 0.01]}, {"ells": 2, "k": [0.02, 0.30, 0.01]}],
    "mesh3_spectrum": [{"ells": (0, 0, 0), "k": [0.02, 0.20, 0.01]}, {"ells": (2, 0, 2), "k": [0.02, 0.10, 0.01]}],
}

TEST_ROOT = Path(os.getenv("SCRATCH", ".")) / "test_fiber_collision"
DEFAULT_STATS_DIR = Path("/global/cfs/cdirs/desicollab/science/cai/desi-clustering/dr2/summary_statistics/full_shape/base")
DEFAULT_FITS_DIR = TEST_ROOT
DEFAULT_CACHE_DIR = TEST_ROOT / "_cache"

def _apply_kranges(observable_options, pkmin=None, pkmax=None):
    stat = observable_options["stat"]["kind"]
    if stat not in KRANGES:
        return

    def get_klim(item):
        klim = list(item["k"])
        if pkmin is not None:
            klim[0] = pkmin
        if pkmax is not None:
            klim[1] = pkmax
        return klim

    observable_options["stat"]["select"] = [{"ells": item["ells"], "k": get_klim(item)} for item in KRANGES[stat]]


def _validate_theory_model(stats, theory_model):
    if theory_model == 'reptvelocileptors' and 'mesh3_spectrum' in stats:
        raise ValueError('theory model reptvelocileptors is only supported with mesh2_spectrum')

def get_abacus_hf_v2_snapshot_redshift(tracer, zrange):
    """
    reference: https://desi.lbl.gov/trac/wiki/CrossAnalysisInfrastructureWG/LSSMocks/DR2HFAbacusMocks#DR2completecutskymocks
    """
    redshifts = {
        ('BGS', (0.1, 0.4)): 0.300,
        ('LRG', (0.4, 0.6)): 0.500,
        ('LRG', (0.6, 0.8)): 0.725,
        ('LRG', (0.8, 1.1)): 0.950,
        ('ELG', (0.8, 1.1)): 0.950,
        ('ELG', (1.1, 1.6)): 1.475,
        ('QSO', (0.8, 2.1)): 1.550,
    }
    return redshifts[tracer, zrange]

def build_likelihoods_options(**kwargs):
    from full_shape import tools

    keys = ('stats', 'tracers', 'version', 'covariance', 'auw', 'thetacut')
    stats, tracers, version, covariance, auw, thetacut = [kwargs.pop(key) for key in keys]
    keys = ('pkmin', 'pkmax', 'theory_model', 'prior_basis')
    pkmin, pkmax, theory_model, prior_basis = [kwargs.pop(key) for key in keys]
    stats_dir = DEFAULT_STATS_DIR
    bispectrum_with_auw = (auw and version == 'abacus-hf-dr2-v2-altmtl' and 'mesh3_spectrum' in stats)
    if (version == 'abacus-hf-dr2-v2-complete') or bispectrum_with_auw:
        stats_dir = Path("/global/cfs/cdirs/desicollab/science/cai/desi-clustering/dr2/summary_statistics/full_shape/fiber_assignment_systematics")
    if kwargs:
        raise ValueError(f'options {kwargs!r} not used')

    _validate_theory_model(stats, theory_model)
    extra = ""
    if version == "abacus-hf-dr2-v2-complete":
        version = "abacus-hf-dr2-v2-altmtl"
        extra = "complete"
    likelihoods = []
    for tracer in tracers:
        if "," in tracer:
            tracer = tracer.split(",")
        likelihood_options = tools.generate_likelihood_options_helper(stats=stats, tracer=tracer, version=version, covariance=covariance, stats_dir=stats_dir)
        likelihood_options["covariance"]["stats_dir"] = DEFAULT_STATS_DIR  # always use default stats_dir for covariance
        if bispectrum_with_auw:
            likelihood_options['covariance']['auw'] = False  # XXX: no measurement
        for observable_options in likelihood_options["observables"]:
            _apply_kranges(observable_options, pkmax=pkmax, pkmin=pkmin)
            observable_options.setdefault("theory", {})
            observable_options["theory"]["model"] = theory_model
            observable_options["theory"]["prior_basis"] = prior_basis
            
            observable_options['theory']['z'] = get_abacus_hf_v2_snapshot_redshift(
                observable_options['catalog']['tracer'], observable_options['catalog']['zrange'])
            observable_options["catalog"]["auw"] = auw
            observable_options["catalog"]["cut"] = thetacut
            if extra:
                observable_options["catalog"]["extra"] = extra
        likelihoods.append(likelihood_options)
    return likelihoods


def build_run_options(**kwargs):
    from full_shape import tools
    template, cosmo_model = kwargs.pop('template', 'direct'), kwargs.pop('cosmo_model', 'base')
    sampler, nchains, resume, thinning, gelman_rubin = kwargs.pop('sampler', 'mh'), kwargs.pop('nchains', 1), kwargs.pop('resume', False), kwargs.pop('thinning', 1), kwargs.pop('gelman_rubin', 0.02)
    options = {}
    options["likelihoods"] = build_likelihoods_options(**kwargs)
    options["cosmology"] = {"template": template, "model": cosmo_model}
    options['sampler'] = tools.propose_fiducial_sampler_options(sampler=sampler)
    sampler_kw = {'nparallel': nchains, "gelman_rubin": gelman_rubin, 'thinning': thinning}
    for section in ['init', 'run']:
        for name, value in options['sampler'][section].items():
            if name in sampler_kw:
                options['sampler'][section][name] = sampler_kw[name]
    options['sampler']['resume'] = resume
    return tools.fill_fiducial_options(options)


def run_fit(
    actions=("profile",), stats=["mesh2_spectrum"], version="abacus-hf-dr2-v2-altmtl", covariance="holi-v3-altmtl",
    tracers=None, auw=False, thetacut=False, pkmin=None, pkmax=None,
    template="direct", cosmo_model='base', theory_model="folpsD", prior_basis="physical_aap",
    sampler="mcmc", nchains=1, thinning=1, resume=False, gelman_rubin=0.02,
    fits_dir=DEFAULT_FITS_DIR, cache_dir=DEFAULT_CACHE_DIR,
):
    # Everything inside this function will be executed on the compute nodes;
    # This function must be self-contained; and cannot rely on imports from the outer scope.
    import os
    from pathlib import Path
    import functools
    from mpi4py import MPI
    from pprint import pprint

    mpicomm = MPI.COMM_WORLD
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(mpicomm.rank)
    import jax
    from jax import config

    config.update("jax_enable_x64", True)
    from full_shape import run_fit_from_options, tools
    from full_shape.tools import get_likelihood
    from desilike.samples import Profiles

    options = build_run_options(
        stats=stats, version=version, covariance=covariance,
        tracers=tracers, auw=auw, thetacut=thetacut, pkmin=pkmin, pkmax=pkmax,
        template=template, theory_model=theory_model, prior_basis=prior_basis, cosmo_model=cosmo_model,
        sampler=sampler, nchains=nchains, thinning=thinning, resume=resume, gelman_rubin=gelman_rubin
    )
    if mpicomm.rank == 0:
        print('fit options:')
        pprint(options)
    get_fits_fn = functools.partial(tools.get_fits_fn, fits_dir=fits_dir)
    cache_dir = Path(cache_dir)
    run_fit_from_options(actions, **options, get_fits_fn=get_fits_fn, cache_dir=cache_dir)
    if "profile" in actions:
        likelihood = get_likelihood(likelihoods_options=options["likelihoods"], cosmology_options=options["cosmology"], cache_dir=cache_dir)
        profiles = Profiles.read(get_fits_fn(kind='profiles', **options))
        # Evaluate likelihood at dictionary of parameters
        best = profiles.choice(index='argmax', squeeze=True).select(input=True).best
        compile(likelihood)(**best)
        if mpicomm.rank == 0:
            plot_dir = get_fits_fn(kind="profiles", **options).parent
            for ilikelihood, sublikelihood in enumerate(likelihood.likelihoods):
                for iobservable, observable in enumerate(sublikelihood.observables):
                    #plot_covariance = sublikelihood.covariance.at.observable.get(observables=observable.name)
                    #plot_covariance = plot_covariance.at.observable.match(observable.data)
                    #observable.covariance = plot_covariance
                    observable.plot(fn=plot_dir / f"plot_likelihood{ilikelihood}_observable{iobservable}.png")


def get_parser():
    datasets = ["abacus-hf-dr2-v2-altmtl", "abacus-hf-dr2-v2-complete"]
    parser = argparse.ArgumentParser(description="Test fiber collision effect")
    group = parser.add_mutually_exclusive_group()
    parser.add_argument("--todo", nargs='+', choices=['sample', 'profile'], default=['profile'])
    parser.add_argument(
        "--stats", nargs="+", default=["mesh2_spectrum"], choices=["mesh2_spectrum", "mesh3_spectrum"],
        help="Statistics to fit. Defaults to %(default)s.")
    parser.add_argument("--dataset", choices=datasets, default="abacus-hf-dr2-v2-altmtl")
    parser.add_argument("--covariance", choices=['holi-v3-altmtl'], default='holi-v3-altmtl')
    parser.add_argument(
        "--tracers", nargs="+", default=['LRG1'],
        help="Tracer(s) to fit. Pass one or more values after --tracers. Defaults to %(default)s. --tracers tracer1 tracer2 assumes no cross correlation between tracer1 and tracer2. Pass --tracers tracer1,tracer2 to account for cross correlation.")
    group.add_argument("--auw", action="store_true", help="fit stats with angular upweights")
    group.add_argument("--thetacut", action="store_true", help="fit stats with thetacut")
    parser.add_argument("--pkmin", type=float, default=KRANGES['mesh2_spectrum'][0]['k'][0], help="kmin cut for mesh2_spectrum fit")
    parser.add_argument("--pkmax", type=float, default=KRANGES['mesh2_spectrum'][0]['k'][1], help="kmax cut for mesh2_spectrum fit")
    parser.add_argument("--theory_model", choices=THEORY_MODELS, default="folpsD")
    parser.add_argument("--prior_basis", choices=PRIOR_BASES, default='physical_aap')
    parser.add_argument(
        "--cosmo_model", type=str, default="base_ns-fixed", choices=COSMO_MODELS,
        help="Cosmology parameter setup to fit. base varies h, omega_cdm, omega_b, logA, n_s; "
             "base_ns-fixed varies h, omega_cdm, omega_b, logA; "
             "fixed varies only nuisance parameters. Defaults to %(default).")
    parser.add_argument(
        "--sampler", type=str, default='mh', choices=SAMPLERS,
        help="desilike sampler backend to use. Defaults to %(default)s.")
    parser.add_argument("--nchains", type=int, default=1, help="Number of MCMC chains to run with desilike. Defaults to 1.")
    parser.add_argument(
        "--thinning", type=int, default=1,
        help="Thin samples by this factor while the desilike sampler is running. Defaults to 1.")
    parser.add_argument("--resume", action="store_true", help="Resume sampling from existing chain files in the derived fits directory.")
    parser.add_argument('--gelman_rubin', type=float, default=0.02, help='default is %(default)s')
    parser.add_argument(
        "--fits_dir", type=Path, default=DEFAULT_FITS_DIR,
        help="Base directory for fits. Defaults to %(default)s.")
    parser.add_argument(
        "--cache_dir", type=Path, default=DEFAULT_CACHE_DIR,
        help="Base directory to store cached files like prepared_stats and emulators. Defaults to %(default)s.")
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    fits_dir = args.fits_dir / args.dataset
    run_fit(
        actions=tuple(args.todo), stats=args.stats, version=args.dataset, covariance=args.covariance,
        tracers=args.tracers, auw=args.auw, thetacut=args.thetacut, pkmin=args.pkmin, pkmax=args.pkmax,
        template="direct", theory_model=args.theory_model, prior_basis=args.prior_basis, cosmo_model=args.cosmo_model,
        sampler=args.sampler, nchains=args.nchains, thinning=args.thinning, resume=args.resume, gelman_rubin=args.gelman_rubin,
        fits_dir=fits_dir, cache_dir=args.cache_dir)