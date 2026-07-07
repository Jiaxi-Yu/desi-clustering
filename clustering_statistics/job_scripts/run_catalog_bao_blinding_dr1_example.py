"""Run the DR1 catalog BAO/AP blinding example as a script.

This is the script equivalent of
``clustering_statistics/nb/example_catalog_bao_blinding.ipynb``. It is meant
for an interactive or batch allocation, not for a login-node notebook kernel.

By default it measures both the Fourier-space power spectrum and the
configuration-space 2PCF. To run only one statistic, pass e.g.::

    srun -n 4 python clustering_statistics/job_scripts/run_catalog_bao_blinding_dr1_example.py --stats particle2_correlation

Example on Perlmutter::

    salloc -N 1 -C "gpu&hbm80g" -t 00:30:00 --gpus 4 --qos interactive --account desi_g
    source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
    module unload desi-clustering || true
    export PYTHONPATH=/global/homes/u/uendert/repos/desi/desiblind:/global/homes/u/uendert/repos/desi/desi-clustering:${PYTHONPATH}
    cd /global/homes/u/uendert/repos/desi/desi-clustering
    srun -n 4 python clustering_statistics/job_scripts/run_catalog_bao_blinding_dr1_example.py
"""

import argparse
import functools
import os
from pathlib import Path
import sys
import warnings

from mpi4py import MPI

mpicomm = MPI.COMM_WORLD

# Keep example output readable under `srun -n N`: nonzero ranks still do the
# computation, but duplicate warnings are suppressed. Rank 0 keeps the warning.
if mpicomm.rank != 0:
    warnings.filterwarnings(
        'ignore',
        message='Input catalogs will be BAO/AP blinded.*',
        category=UserWarning,
        module='clustering_statistics.compute_stats',
    )
    warnings.filterwarnings(
        'ignore',
        message='Statistic/data-vector blinding.*',
        category=UserWarning,
        module='clustering_statistics.compute_stats',
    )


def rank0_print(*args, **kwargs):
    if mpicomm.rank == 0:
        print(*args, **kwargs)


def parse_zrange(text: str) -> tuple[float, float]:
    parts = text.replace(',', ' ').split()
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f'zrange must have two numbers, got {text!r}')
    return (float(parts[0]), float(parts[1]))


def parse_args():
    parser = argparse.ArgumentParser(description='Run the DR1 catalog BAO/AP blinding example.')
    parser.add_argument(
        '--stats', nargs='+', default=['mesh2_spectrum', 'particle2_correlation'],
        choices=['mesh2_spectrum', 'particle2_correlation'],
        help='Statistic(s) to measure. Defaults to both P(k) and xi(s).',
    )
    parser.add_argument('--unblinded', action='store_true',
                        help='Run the matching unblinded reference measurement without catalog_bao_blinding.')
    parser.add_argument('--nran', type=int, default=1,
                        help='Number of random catalogs to use. Defaults to 1 for a quick example; use e.g. 18 for a more realistic DR1 run.')
    parser.add_argument('--parameter-source', choices=['auto', 'explicit', 'desiblind', 'lss'], default='auto',
                        help='BAO/AP parameter source. Default auto uses explicit open demo w0/wa unless a bank is supplied.')
    parser.add_argument('--w0', type=float, default=-0.95, help='Explicit/open demo w0 value.')
    parser.add_argument('--wa', type=float, default=0.10, help='Explicit/open demo wa value.')
    parser.add_argument('--metadata', choices=['open', 'closed'], default='open',
                        help='Whether to write w0/wa in output attrs. Closed hides them from attrs only.')
    parser.add_argument('--output-version-suffix', default='desiblind-bao-blinded',
                        help='Suffix appended to the input catalog version for blinded outputs.')
    parser.add_argument('--parameters-fn', '--parameter-bank', type=Path, default=None, dest='parameters_fn',
                        help='desiblind hashed parameter bank .npy file for parameter-source=desiblind.')
    parser.add_argument('--save-dir', type=Path, default=None,
                        help='Directory containing desiblind catalog_blinding_parameters.npy.')
    parser.add_argument('--bid', type=int, default=None, help='desiblind blinding id for the hashed parameter bank.')
    parser.add_argument('--lss-parameters-fn', '--lss-w0wa-bank', type=Path, default=None, dest='lss_parameters_fn',
                        help='Historical LSS plain-text w0/wa bank.')
    parser.add_argument('--lss-parameter-index', '--parameter-index', type=int, default=None, dest='lss_parameter_index',
                        help='Row index in the LSS w0/wa bank.')
    parser.add_argument('--lss-filerow', '--filerow', type=Path, default=None, dest='lss_filerow',
                        help='File containing row index in the LSS w0/wa bank.')
    parser.add_argument('--no-validate-alpha-shift', action='store_true',
                        help='Disable the Andrade/LSS 3 percent alpha-shift safety check.')
    parser.add_argument('--alpha-zrange', type=parse_zrange, default=(0.4, 2.1),
                        help='Redshift range for alpha-shift validation, e.g. 0.4,2.1.')
    parser.add_argument('--max-alpha-shift', type=float, default=0.03)
    parser.add_argument('--alpha-nz', type=int, default=100)
    parser.add_argument('--diagnostic-plot-dir', type=Path, default=None,
                        help='Optional directory for real-catalog diagnostics showing BAO/AP-blinded data/random matching and weights.')
    parser.add_argument('--diagnostic-plot-prefix', default=None,
                        help='Optional filename prefix for --diagnostic-plot-dir outputs.')
    parser.add_argument('--diagnostic-random-index', type=int, default=0,
                        help='Random catalog index to use in diagnostic plots. Measurement still uses all --nran randoms; default plots random 0 to keep diagnostics lightweight.')
    parser.add_argument('--save-blinded-catalog-dir', type=Path, default=None,
                        help='Optional directory to save the LSS-like post-BAO/post-addnbar blinded data catalog used for the measurement.')
    parser.add_argument('--save-blinded-randoms', choices=['none', 'diagnostic', 'all'], default='diagnostic',
                        help='Which blinded random catalogs to save when --save-blinded-catalog-dir is set. Default diagnostic saves --diagnostic-random-index only; all can be very large for --nran 18.')
    parser.add_argument('--save-blinded-catalog-prefix', default=None,
                        help='Optional filename prefix for saved blinded catalogs. Default includes tracer, region, full nbar z-range, and output suffix.')
    return parser.parse_args()


args = parse_args()

# Prefer local development checkouts over installed packages.
DESI_CLUSTERING_REPO = Path('/global/homes/u/uendert/repos/desi/desi-clustering')
DESIBLIND_REPO = Path('/global/homes/u/uendert/repos/desi/desiblind')
for repo in [DESI_CLUSTERING_REPO, DESIBLIND_REPO]:
    if repo.exists():
        sys.path.insert(0, str(repo))

# JAX configuration should be set before importing JAX-heavy measurement code.
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.90')

import jax
from jax import config

config.update('jax_enable_x64', True)
try:
    jax.distributed.initialize()
except (RuntimeError, ValueError) as exc:
    # Happens if already initialized or when running directly without a JAX coordinator.
    rank0_print(f'jax.distributed.initialize skipped: {exc}')

import lsstypes as types

from clustering_statistics.catalog_blinding import bao as catalog_bao_blinding
from clustering_statistics.catalog_blinding import lss_catalogs as catalog_lss
from clustering_statistics.catalog_blinding.diagnostics import write_diagnostic_plots
from clustering_statistics import compute_stats_from_options, fill_fiducial_options, tools

MEASUREMENT_PRESET = 'full_shape'
STATS = args.stats
DEFAULT_LSS_W0WA_BANK = '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/w0wa_initvalues_zeffcombined_1000realisations.txt'


def parameter_source_for_args(args):
    if args.parameter_source != 'auto':
        return args.parameter_source
    if args.parameters_fn is not None or args.save_dir is not None or args.bid is not None:
        return 'desiblind'
    if args.lss_parameters_fn is not None or args.lss_parameter_index is not None or args.lss_filerow is not None:
        return 'lss'
    return 'explicit'


def make_bao_blinding_options(args):
    if args.unblinded:
        return None
    source = parameter_source_for_args(args)
    options = {
        'parameter_source': source,
        'input_zcol': 'Z',
        'output_zcol': 'Z',
        'metadata': args.metadata,
        'output_version_suffix': args.output_version_suffix,
        'validate_alpha_shift': not args.no_validate_alpha_shift,
        'alpha_zrange': args.alpha_zrange,
        'max_alpha_shift': args.max_alpha_shift,
        'alpha_nz': args.alpha_nz,
    }
    if args.save_blinded_catalog_dir is not None:
        options['save_catalog_dir'] = str(args.save_blinded_catalog_dir)
        options['save_randoms'] = args.save_blinded_randoms
        options['save_random_index'] = args.diagnostic_random_index
        if args.save_blinded_catalog_prefix is not None:
            options['save_catalog_prefix'] = args.save_blinded_catalog_prefix
    if source == 'explicit':
        options['parameters'] = {'w0': args.w0, 'wa': args.wa}
    elif source == 'desiblind':
        if args.parameters_fn is not None:
            options['parameters_fn'] = str(args.parameters_fn)
        if args.save_dir is not None:
            options['save_dir'] = str(args.save_dir)
        if args.bid is not None:
            options['bid'] = args.bid
        # BAO/AP on-the-fly path infers LRG1 from tracer='LRG', zrange=(0.4, 0.6)
    elif source == 'lss':
        options['lss_parameters_fn'] = str(args.lss_parameters_fn or DEFAULT_LSS_W0WA_BANK)
        if args.lss_parameter_index is not None:
            options['lss_parameter_index'] = args.lss_parameter_index
        if args.lss_filerow is not None:
            options['lss_filerow'] = str(args.lss_filerow)
    return options


bao_blinding_options = make_bao_blinding_options(args)

options = dict(
    catalog=dict(
        version='data-dr1-v1.5',
        tracer='LRG',
        zrange=(0.4, 0.6),
        region='NGC',
        weight='default-FKP',
        nran=args.nran,
    ),
    mesh2_spectrum={},
    particle2_correlation={},
)
if bao_blinding_options is not None:
    options['catalog']['catalog_bao_blinding'] = bao_blinding_options

stats_dir = Path(os.getenv('SCRATCH', Path.home())) / 'measurements_bao_blinding_example'
get_stats_fn = functools.partial(tools.get_stats_fn, stats_dir=stats_dir)
cache = {}

rank0_print('DR1 catalog BAO/AP blinding example')
rank0_print('====================================')
rank0_print(f'MPI ranks       : {mpicomm.size}')
rank0_print(f'desi-clustering : {catalog_bao_blinding.__file__}')
rank0_print(f'stats_dir       : {stats_dir}')
rank0_print(f'parameter source: {"unblinded" if args.unblinded else parameter_source_for_args(args)}')
rank0_print(f'stats           : {STATS}')
rank0_print('catalog options :')
for key, value in options['catalog'].items():
    rank0_print(f'  {key:22s}: {value}')
rank0_print('')
rank0_print('Running measurement...')

compute_stats_from_options(
    STATS,
    get_stats_fn=get_stats_fn,
    cache=cache,
    analysis=MEASUREMENT_PRESET,
    **options,
)

mpicomm.Barrier()


def output_catalog_options():
    output_catalog = dict(options['catalog'])
    if bao_blinding_options is not None:
        output_catalog['version'] = catalog_bao_blinding.output_version(
            output_catalog['version'],
            catalog_bao_blinding.resolve_options(
                bao_blinding_options, tracer=output_catalog['tracer'], zrange=output_catalog['zrange']),
        )
    output_catalog.pop('catalog_bao_blinding', None)
    return output_catalog


def write_bao_diagnostics():
    if args.diagnostic_plot_dir is None:
        return None
    if bao_blinding_options is None:
        rank0_print('--diagnostic-plot-dir requested for an unblinded run; no BAO data/random matching diagnostics written.')
        return None

    catalog_options = dict(options['catalog'])
    catalog_options.pop('catalog_bao_blinding', None)
    resolved = catalog_bao_blinding.resolve_options(
        bao_blinding_options, tracer=catalog_options['tracer'], zrange=catalog_options['zrange'])

    rank0_print('')
    rank0_print('Writing data/random diagnostic plots...')
    measurement_zrange = catalog_options.get('zrange', (None, None))
    read_catalog_options = dict(catalog_options)
    nbar_zrange = catalog_lss.fiducial_nbar_zrange(catalog_options.get('tracer'), zrange=measurement_zrange)
    if nbar_zrange is not None:
        read_catalog_options['zrange'] = nbar_zrange
    raw_data_unblinded = tools.read_catalog(kind='data', concatenate=True, keep_columns=True, **read_catalog_options)
    raw_randoms = tools.read_catalog(kind='randoms', concatenate=False, keep_columns=True, **read_catalog_options)
    if isinstance(raw_randoms, list):
        if not (0 <= args.diagnostic_random_index < len(raw_randoms)):
            raise ValueError(f'--diagnostic-random-index must be in [0, {len(raw_randoms) - 1}] for --nran {len(raw_randoms)}')
        raw_random = raw_randoms[args.diagnostic_random_index]
        rank0_print(f'diagnostics use random index {args.diagnostic_random_index}; measurement used --nran {len(raw_randoms)} random catalog(s).')
    else:
        raw_random = raw_randoms

    raw_data_bao = catalog_bao_blinding.apply_to_catalogs(raw_data_unblinded, resolved)
    if resolved.get('apply_nz_reweight', True):
        raw_data_bao, nz_info = catalog_lss.apply_bao_nz_reweight(
            raw_data_unblinded, raw_data_bao,
            zcol_before=resolved['input_zcol'], zcol_after=resolved['output_zcol'],
            zmin=resolved.get('nz_zmin'), zmax=resolved.get('nz_zmax'),
            dz=resolved.get('nz_dz', 0.01), copy=False,
        )
    else:
        nz_info = None
    bao_nz_extra_weight = None if nz_info is None else nz_info.get('correction')
    raw_data_bao = catalog_lss.set_lss_pre_addnbar_weight(raw_data_bao, extra_weight=bao_nz_extra_weight, copy=False)
    split_columns = catalog_lss.split_columns_for_region(
        catalog_options.get('region'), tracer=catalog_options.get('tracer'),
        split_columns=resolved.get('random_split_columns'),
    )
    matched_random = catalog_lss.resample_randoms_from_data(
        raw_random, raw_data_bao,
        columns=resolved.get('random_resample_columns'),
        split_columns=split_columns,
        seed=resolved.get('random_seed', 0),
        compmd=resolved.get('random_compmd', 'ran'),
        copy=True,
    )
    raw_data_bao, matched_random, _ = catalog_lss.add_nbar_fkp(
        raw_data_bao, matched_random,
        zcol=resolved['output_zcol'],
        zmin=nbar_zrange[0] if nbar_zrange is not None else None,
        zmax=nbar_zrange[1] if nbar_zrange is not None else None,
        dz=resolved.get('nbar_dz', resolved.get('nz_dz', 0.01)),
        p0=read_catalog_options.get('FKP_P0', catalog_lss.fiducial_fkp_p0(catalog_options.get('tracer'))),
        compmd=resolved.get('random_compmd', 'ran'),
        randens=resolved.get('randens', 2500.),
        data_extra_weight=bao_nz_extra_weight,
        copy=False,
    )

    zrange = measurement_zrange
    prefix = args.diagnostic_plot_prefix or (
        f"{catalog_options['tracer']}_{catalog_options['region']}_z{zrange[0]}-{zrange[1]}_bao_on_the_fly")
    summary = write_diagnostic_plots(
        args.diagnostic_plot_dir,
        input_data=raw_data_unblinded,
        bao_ap_blinded_data=raw_data_bao,
        final_data=raw_data_bao,
        input_random=raw_random,
        reconstruction_random=matched_random,
        final_random=matched_random,
        modes=('bao',),
        zcol=resolved['output_zcol'],
        zmin=zrange[0], zmax=zrange[1], dz=resolved.get('diagnostic_dz', 0.005),
        prefix=prefix,
        bao_nz_reweight=nz_info,
    )
    if nz_info is not None:
        rank0_print(f"BAO n(z) internal factor ratio min/max: {float(nz_info['ratio'].min()):.6g} / {float(nz_info['ratio'].max()):.6g}")
    rank0_print(f"diagnostics summary: {summary['summary_file']}")
    for key, value in summary['plots'].items():
        rank0_print(f"diagnostics {key}: {value}")
    return summary


def summarize_output(stat, output_file):
    print('')
    print(f'{stat} output')
    print('-' * (len(stat) + 7))
    print(f'file   : {output_file}')
    if not Path(output_file).exists():
        print('status : output file not found')
        raise FileNotFoundError(output_file)
    statistic = types.read(output_file)
    attrs = getattr(statistic, 'attrs', {})
    print('status : file exists and was read successfully')
    print('attrs  :')
    for key in [
        'catalog_bao_blinding',
        'catalog_bao_blinding_mode',
        'catalog_bao_blinding_metadata',
        'catalog_bao_blinding_parameter_source',
        'catalog_bao_blinding_w0',
        'catalog_bao_blinding_wa',
        'catalog_bao_blinding_name',
        'catalog_bao_blinding_bid',
        'catalog_bao_blinding_index',
        'catalog_bao_blinding_max_abs_alpha_parallel_minus_one',
        'catalog_bao_blinding_max_abs_alpha_perp_minus_one',
        'meshsize',
        'boxsize',
        'los',
    ]:
        if key in attrs:
            print(f'  {key:34s}: {attrs[key]}')

    if stat == 'particle2_correlation':
        projected = statistic.project(ells=(0, 2, 4))
        print('projected ells:', projected.ells)


if mpicomm.rank == 0:
    filled = fill_fiducial_options(options, analysis=MEASUREMENT_PRESET)
    output_catalog = output_catalog_options()

    print('')
    print('Measurement finished')
    print('====================')
    for stat in STATS:
        output_file = tools.get_stats_fn(kind=stat, stats_dir=stats_dir, catalog=output_catalog, **filled[stat])
        summarize_output(stat, output_file)
    write_bao_diagnostics()
