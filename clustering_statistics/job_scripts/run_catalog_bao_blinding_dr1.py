"""DR1 catalog-level BAO/AP blinding measurement runner.

This multi-mode runner is the production-like companion to
``clustering_statistics/nb/example_catalog_bao_blinding.ipynb``. It runs the
same desiblind catalog BAO/AP on-the-fly path through ``compute_stats_from_options``, but
loops over DR1 tracers, redshift bins, and regions using the desipipe execution
pattern used elsewhere in this repository.

The default task grid is DR1/Y1 data, all main tracers, all fiducial full-shape
redshift bins, and NGC+SGC regions. The default statistics are both
``mesh2_spectrum`` and ``particle2_correlation``.

Typical usage on Perlmutter
---------------------------

Dry-run task list and input-file checks::

    source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
    module unload desi-clustering || true
    export PYTHONPATH=/global/homes/u/uendert/repos/desi/desiblind:/global/homes/u/uendert/repos/desi/desi-clustering:${PYTHONPATH}
    cd /global/homes/u/uendert/repos/desi/desi-clustering
    python clustering_statistics/job_scripts/run_catalog_bao_blinding_dr1.py --dry-run

Run one task interactively inside an allocation::

    salloc -N 1 -C "gpu&hbm80g" -t 00:30:00 --gpus 4 --qos interactive --account desi_g
    srun -n 4 python clustering_statistics/job_scripts/run_catalog_bao_blinding_dr1.py \
        --mode interactive --tracers LRG --regions NGC --zrange-index 0

Run the selected grid directly in the current allocation, bypassing desipipe
queue overhead::

    salloc -N 1 -C "gpu&hbm80g" -t 02:00:00 --gpus 4 --qos interactive --account desi_g
    srun -n 4 python clustering_statistics/job_scripts/run_catalog_bao_blinding_dr1.py \
        --mode direct-grid

Create desipipe tasks for the full grid::

    python clustering_statistics/job_scripts/run_catalog_bao_blinding_dr1.py --mode desipipe --clear-queue
    desipipe tasks -q catalog_bao_blinding_dr1
    desipipe spawn -q catalog_bao_blinding_dr1 --spawn
    desipipe queues -q catalog_bao_blinding_dr1

Check outputs and make quick plots after jobs finish::

    python clustering_statistics/job_scripts/run_catalog_bao_blinding_dr1.py --check
    python clustering_statistics/job_scripts/run_catalog_bao_blinding_dr1.py --plot

Notes
-----

This is catalog-level BAO/AP blinding only. It does not perform RSD/fNL catalog
blinding, and it does not write blinded FITS catalogs. It writes blinded
clustering statistics under an output-version suffix, e.g.
``data-dr1-v1.5-desiblind-bao-blinded``.

For this public DR1 example the default blinding parameters are intentionally
open and stored in output attrs. The preferred closed workflow is to pass a
private desiblind hashed parameter bank with ``--parameter-source desiblind`` and
``--parameters-fn``/``--bid``; historical LSS ``w0wa`` bank rows can be used with
``--parameter-source lss`` for compatibility. ``metadata='closed'`` only hides
w0/wa from statistic attrs; it is not an unblinding record by itself.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
from pathlib import Path
import sys
import warnings

import numpy as np

# Prefer local development checkouts over installed packages. This matters when
# the cosmodesi environment contains a different desi-clustering main branch.
DESI_CLUSTERING_REPO = Path('/global/homes/u/uendert/repos/desi/desi-clustering')
DESIBLIND_REPO = Path('/global/homes/u/uendert/repos/desi/desiblind')
for repo in [DESI_CLUSTERING_REPO, DESIBLIND_REPO]:
    if repo.exists():
        sys.path.insert(0, str(repo))

from mpi4py import MPI

from clustering_statistics.catalog_blinding import bao as catalog_bao_blinding
from clustering_statistics import fill_fiducial_options, tools

mpicomm = MPI.COMM_WORLD

QUEUE_NAME = 'catalog_bao_blinding_dr1'
VERSION = 'data-dr1-v1.5'
MEASUREMENT_PRESET = 'full_shape'
DEFAULT_TRACERS = ('BGS', 'LRG', 'ELG', 'QSO')
DEFAULT_REGIONS = ('NGC', 'SGC')
DEFAULT_STATS = ('mesh2_spectrum', 'particle2_correlation')
DEFAULT_STATS_DIR = Path(os.getenv('SCRATCH', Path.home())) / 'measurements_catalog_bao_blinding_dr1'
DEFAULT_PLOTS_DIRNAME = 'plots'

DEFAULT_W0 = -0.95
DEFAULT_WA = 0.10
DEFAULT_LSS_W0WA_BANK = '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/w0wa_initvalues_zeffcombined_1000realisations.txt'


def rank0_print(*args, **kwargs):
    if mpicomm.rank == 0:
        print(*args, **kwargs)


def parse_zrange(text: str) -> tuple[float, float]:
    parts = text.replace(',', ' ').split()
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f'zrange must have two numbers, got {text!r}')
    return (float(parts[0]), float(parts[1]))


def parse_args():
    parser = argparse.ArgumentParser(description='Run/check/plot DR1 catalog BAO/AP blinding measurements; desipipe is one optional mode.')
    parser.add_argument('--mode', choices=['dry-run', 'interactive', 'direct-grid', 'desipipe', 'check', 'plot'], default='dry-run',
                        help=('Execution mode. interactive runs one selected task; direct-grid runs the selected grid '
                              'sequentially in the current allocation; desipipe creates queued tasks. '
                              'Convenience flags --dry-run/--check/--plot override this.'))
    parser.add_argument('--dry-run', action='store_true', help='Print task list and input/output paths without creating tasks.')
    parser.add_argument('--check', action='store_true', help='Check expected output files and write a JSON summary.')
    parser.add_argument('--plot', action='store_true', help='Make quick built-in lsstypes plots for existing outputs.')
    parser.add_argument('--clear-queue', action='store_true', help='Clear the desipipe queue before creating tasks.')
    parser.add_argument('--queue', default=QUEUE_NAME, help='desipipe queue name.')
    parser.add_argument('--version', default=VERSION, help='Catalog version.')
    parser.add_argument('--stats-dir', type=Path, default=DEFAULT_STATS_DIR, help='Measurement output directory.')
    parser.add_argument('--output-version-suffix', default='desiblind-bao-blinded',
                        help='Suffix appended to the input catalog version for blinded outputs.')
    parser.add_argument('--plots-dir', type=Path, default=None, help='Plot output directory; default is stats_dir/plots.')
    parser.add_argument('--tracers', nargs='+', default=list(DEFAULT_TRACERS), help='Simple tracer names or full tracer names.')
    parser.add_argument('--regions', nargs='+', default=list(DEFAULT_REGIONS), help='Regions to run.')
    parser.add_argument('--stats', nargs='+', default=list(DEFAULT_STATS), choices=['mesh2_spectrum', 'particle2_correlation'],
                        help='Statistic(s) to measure/check/plot.')
    parser.add_argument('--zrange', type=parse_zrange, default=None,
                        help='Only run this zrange, e.g. --zrange 0.4,0.6. Applied to every selected tracer.')
    parser.add_argument('--zrange-index', type=int, default=None,
                        help='Only run the fiducial zrange at this index for each selected tracer.')
    parser.add_argument('--nran', type=int, default=None, help='Override fiducial random count.')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip tasks for which all requested output statistic files already exist.')
    parser.add_argument('--unblinded', action='store_true',
                        help='Run an unblinded/reference measurement with no catalog BAO/AP blinding.')

    # BAO/AP blinding parameter sources. ``auto`` keeps demos convenient by using
    # explicit --w0/--wa defaults unless a desiblind or LSS bank is supplied.
    parser.add_argument('--parameter-source', choices=['auto', 'explicit', 'desiblind', 'lss'], default='auto',
                        help='BAO/AP parameter source. Preferred closed workflow is desiblind hashed bank.')
    parser.add_argument('--w0', type=float, default=DEFAULT_W0,
                        help='Explicit/open demo w0 value used when parameter-source is explicit/auto.')
    parser.add_argument('--wa', type=float, default=DEFAULT_WA,
                        help='Explicit/open demo wa value used when parameter-source is explicit/auto.')
    parser.add_argument('--metadata', choices=['open', 'closed'], default='open',
                        help='Whether to write w0/wa in output attrs. Closed hides them from attrs only.')
    parser.add_argument('--parameters-fn', '--parameter-bank', type=Path, default=None, dest='parameters_fn',
                        help='desiblind hashed parameter bank .npy file for parameter-source=desiblind.')
    parser.add_argument('--save-dir', type=Path, default=None,
                        help='Directory containing desiblind catalog_blinding_parameters.npy.')
    parser.add_argument('--bid', type=int, default=None, help='desiblind blinding id used with the hashed parameter bank.')
    parser.add_argument('--lss-parameters-fn', '--lss-w0wa-bank', type=Path, default=None, dest='lss_parameters_fn',
                        help='Historical LSS plain-text w0/wa bank. Defaults to the Y1 LSS bank for source=lss.')
    parser.add_argument('--lss-parameter-index', '--parameter-index', type=int, default=None, dest='lss_parameter_index',
                        help='Row index in the LSS w0/wa bank.')
    parser.add_argument('--lss-filerow', '--filerow', type=Path, default=None, dest='lss_filerow',
                        help='File containing row index in the LSS w0/wa bank, matching historical LSS filerow.txt.')
    parser.add_argument('--no-validate-alpha-shift', action='store_true',
                        help='Disable the Andrade/LSS 3 percent alpha-shift safety check. Use only for validation/debugging.')
    parser.add_argument('--alpha-zrange', type=parse_zrange, default=(0.4, 2.1),
                        help='Redshift range for alpha-shift validation, e.g. 0.4,2.1.')
    parser.add_argument('--max-alpha-shift', type=float, default=0.03,
                        help='Maximum allowed |alpha - 1| for alpha-shift validation.')
    parser.add_argument('--alpha-nz', type=int, default=100, help='Number of z samples for alpha-shift validation.')

    parser.add_argument('--max-workers', type=int, default=10, help='desipipe max workers.')
    parser.add_argument('--mpiprocs-per-worker', type=int, default=4, help='MPI processes per desipipe worker.')
    parser.add_argument('--time', default='02:00:00', help='Slurm walltime for desipipe workers.')
    parser.add_argument('--constraint', default='gpu&hbm80g', help='Slurm constraint for desipipe workers.')
    parser.add_argument('--account', default=None, help='Slurm account passed to desipipe provider if supported.')
    args = parser.parse_args()

    if args.dry_run:
        args.mode = 'dry-run'
    if args.check:
        args.mode = 'check'
    if args.plot:
        args.mode = 'plot'
    if args.plots_dir is None:
        args.plots_dir = args.stats_dir / DEFAULT_PLOTS_DIRNAME
    return args


def full_tracer(tracer: str, version: str) -> str:
    """Return desi-clustering full tracer name when a simple name is supplied."""
    try:
        return tools.get_full_tracer(tracer, version=version)
    except NotImplementedError:
        return tracer


def tracer_zranges(tracer: str, args) -> list[tuple[float, float]]:
    if args.zrange is not None:
        zranges = [args.zrange]
    else:
        zranges = list(tools.propose_fiducial('zranges', tracer, analysis=MEASUREMENT_PRESET))
    if args.zrange_index is not None:
        zranges = [zranges[args.zrange_index]]
    return [tuple(map(float, zrange)) for zrange in zranges]


def build_tasks(args) -> list[dict]:
    tasks = []
    for tracer_in in args.tracers:
        tracer = full_tracer(tracer_in, version=args.version)
        for zrange in tracer_zranges(tracer, args):
            for region in args.regions:
                tasks.append({'tracer': tracer, 'zrange': zrange, 'region': region})
    return tasks


def parameter_source_for_args(args) -> str:
    if getattr(args, 'unblinded', False):
        return 'none'
    if args.parameter_source != 'auto':
        return args.parameter_source
    if args.parameters_fn is not None or args.save_dir is not None or args.bid is not None:
        return 'desiblind'
    if args.lss_parameters_fn is not None or args.lss_parameter_index is not None or args.lss_filerow is not None:
        return 'lss'
    return 'explicit'


def bao_blinding_options_for_task(task: dict, args) -> dict | None:
    if getattr(args, 'unblinded', False):
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
    if source == 'explicit':
        options['parameters'] = {'w0': args.w0, 'wa': args.wa}
    elif source == 'desiblind':
        if args.parameters_fn is not None:
            options['parameters_fn'] = str(args.parameters_fn)
        if args.save_dir is not None:
            options['save_dir'] = str(args.save_dir)
        if args.bid is not None:
            options['bid'] = args.bid
        # Let catalog_bao_blinding infer LRG1/LRG2/... from tracer + zrange.
    elif source == 'lss':
        options['lss_parameters_fn'] = str(args.lss_parameters_fn or DEFAULT_LSS_W0WA_BANK)
        if args.lss_parameter_index is not None:
            options['lss_parameter_index'] = args.lss_parameter_index
        if args.lss_filerow is not None:
            options['lss_filerow'] = str(args.lss_filerow)
    else:
        raise ValueError(f'Unknown parameter source {source}')
    return options


def catalog_options_for_task(task: dict, args, *, blinded: bool = True) -> dict:
    options = {
        'version': args.version,
        'tracer': task['tracer'],
        'zrange': task['zrange'],
        'region': task['region'],
        'weight': 'default-FKP',
    }
    if args.nran is not None:
        options['nran'] = args.nran
    if blinded and not getattr(args, 'unblinded', False):
        options['catalog_bao_blinding'] = bao_blinding_options_for_task(task, args)
    return options


def measurement_options_for_task(task: dict, args) -> dict:
    return {
        'catalog': catalog_options_for_task(task, args, blinded=True),
        'mesh2_spectrum': {},
        'particle2_correlation': {},
    }


def output_catalog_for_task(task: dict, args) -> dict:
    options = catalog_options_for_task(task, args, blinded=False)
    if not getattr(args, 'unblinded', False):
        # Output filenames only depend on the output-version suffix. Do not call
        # resolve_options() here: that loads/validates the private parameter bank
        # and can initialize cosmology/JAX just to check whether a file exists.
        resolved_bao_options = {
            'output_version_suffix': bao_blinding_options_for_task(task, args).get(
                'output_version_suffix', 'desiblind-bao-blinded')
        }
        options['version'] = catalog_bao_blinding.output_version(options['version'], resolved_bao_options)
    return options


def expected_output_file(stat: str, task: dict, args) -> Path:
    options = measurement_options_for_task(task, args)
    filled = fill_fiducial_options(options, analysis=MEASUREMENT_PRESET)
    catalog = output_catalog_for_task(task, args)
    return tools.get_stats_fn(kind=stat, stats_dir=args.stats_dir, catalog=catalog, **filled[stat])


def expected_input_files(task: dict, args) -> tuple[Path, list[Path]]:
    catalog = catalog_options_for_task(task, args, blinded=False)
    data_fn = tools.get_catalog_fn(kind='data', **catalog)
    random_fns = tools.get_catalog_fn(kind='randoms', **catalog)
    if isinstance(random_fns, (str, Path)):
        random_fns = [random_fns]
    return Path(data_fn), [Path(fn) for fn in random_fns]


def print_task_table(tasks: list[dict], args):
    print(f'mode           : {args.mode}')
    print(f'version        : {args.version}')
    print(f'stats_dir      : {args.stats_dir}')
    print(f'parameter source: {parameter_source_for_args(args)}')
    print(f'stats          : {args.stats}')
    print(f'n tasks        : {len(tasks)}')
    print('')
    for i, task in enumerate(tasks):
        data_fn, random_fns = expected_input_files(task, args)
        missing_inputs = [fn for fn in [data_fn] + random_fns if not fn.exists()]
        outputs = {stat: expected_output_file(stat, task, args) for stat in args.stats}
        print(f'{i:02d} {task["tracer"]:18s} {task["region"]:3s} z={task["zrange"][0]:.1f}-{task["zrange"][1]:.1f}')
        print(f'    input data   : {data_fn} [{"ok" if data_fn.exists() else "missing"}]')
        print(f'    input random : {len(random_fns)} file(s); missing={len([fn for fn in random_fns if not fn.exists()])}')
        for stat, output in outputs.items():
            print(f'    output {stat:21s}: {output} [{"exists" if output.exists() else "missing"}]')
        if missing_inputs:
            print(f'    WARNING missing inputs: {[str(fn) for fn in missing_inputs[:3]]}')


def run_stats_task(version=VERSION, tracer='LRG', zrange=(0.4, 0.6), region='NGC', stats_dir=DEFAULT_STATS_DIR,
                   stats=DEFAULT_STATS, nran=None, bao_blinding_options=None, measurement_preset='full_shape'):
    """Compute one tracer/region/zrange task on a desipipe worker or MPI allocation."""
    import functools
    import os
    from pathlib import Path
    import sys
    import warnings

    from mpi4py import MPI

    mpicomm = MPI.COMM_WORLD
    if mpicomm.rank != 0:
        warnings.filterwarnings('ignore', message='Input catalogs will be BAO/AP blinded.*', category=UserWarning,
                                module='clustering_statistics.compute_stats')
        warnings.filterwarnings('ignore', message='Statistic/data-vector blinding.*', category=UserWarning,
                                module='clustering_statistics.compute_stats')

    desi_clustering_repo = Path('/global/homes/u/uendert/repos/desi/desi-clustering')
    desiblind_repo = Path('/global/homes/u/uendert/repos/desi/desiblind')
    for repo in [desi_clustering_repo, desiblind_repo]:
        if repo.exists():
            sys.path.insert(0, str(repo))

    os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.90')

    import jax
    from jax import config

    config.update('jax_enable_x64', True)
    try:
        jax.distributed.initialize()
    except RuntimeError:
        pass

    from clustering_statistics import compute_stats_from_options, setup_logging
    from clustering_statistics import tools

    setup_logging()
    if bao_blinding_options is None:
        bao_blinding_options = {'parameter_source': 'explicit', 'parameters': {'w0': DEFAULT_W0, 'wa': DEFAULT_WA}}
    catalog = {
        'version': version,
        'tracer': tracer,
        'zrange': tuple(zrange),
        'region': region,
        'weight': 'default-FKP',
    }
    if bao_blinding_options is not False:
        catalog['catalog_bao_blinding'] = dict(bao_blinding_options)
    if nran is not None:
        catalog['nran'] = nran

    if mpicomm.rank == 0:
        print('DR1 catalog BAO/AP blinding task')
        print('=================================')
        print(f'MPI ranks : {mpicomm.size}')
        print(f'stats     : {list(stats)}')
        for key, value in catalog.items():
            if key == 'catalog_bao_blinding' and isinstance(value, dict) and value.get('metadata') == 'closed':
                value = {k: value[k] for k in ['parameter_source', 'metadata', 'output_version_suffix'] if k in value}
            print(f'{key:22s}: {value}')
        print(f'stats_dir : {stats_dir}')

    get_stats_fn = functools.partial(tools.get_stats_fn, stats_dir=Path(stats_dir))
    compute_stats_from_options(
        list(stats),
        get_stats_fn=get_stats_fn,
        cache={},
        analysis=measurement_preset,
        catalog=catalog,
        mesh2_spectrum={},
        particle2_correlation={},
    )

    mpicomm.Barrier()
    if mpicomm.rank == 0:
        print('task done')


def make_task_manager(args):
    try:
        from desipipe import Environment, Queue, TaskManager
    except ImportError as exc:
        raise ImportError(
            'desipipe is required only for --mode desipipe. Use --mode direct-grid, '
            '--mode dry-run, --mode check, or --mode plot without desipipe; or load an environment with desipipe.'
        ) from exc

    queue = Queue(args.queue)
    if args.clear_queue:
        queue.clear(kill=False)
    output = f'./slurm_outputs/{args.queue}/slurm-%j.out'
    error = f'./slurm_outputs/{args.queue}/slurm-%j.err'
    environ = Environment('nersc-cosmodesi')
    tm = TaskManager(queue=queue, environ=environ)
    provider = dict(provider='nersc', time=args.time, mpiprocs_per_worker=args.mpiprocs_per_worker,
                    output=output, error=error, stop_after=1, constraint=args.constraint)
    if args.account is not None:
        provider['account'] = args.account
    return tm.clone(scheduler=dict(max_workers=args.max_workers), provider=provider)


def create_desipipe_tasks(tasks: list[dict], args):
    tm = make_task_manager(args)
    app = tm.python_app(run_stats_task)
    for task in tasks:
        app(version=args.version, tracer=task['tracer'], zrange=task['zrange'], region=task['region'],
            stats_dir=args.stats_dir, stats=tuple(args.stats), nran=args.nran,
            bao_blinding_options=(False if args.unblinded else bao_blinding_options_for_task(task, args)),
            measurement_preset=MEASUREMENT_PRESET)
    print(f'Created {len(tasks)} desipipe task(s) in queue {args.queue!r}.')
    print(f'Inspect with: desipipe tasks -q {args.queue}')
    print(f'Spawn with  : desipipe spawn -q {args.queue} --spawn')


def task_outputs_exist(task: dict, args) -> bool:
    return all(expected_output_file(stat, task, args).exists() for stat in args.stats)


def filter_existing_tasks(tasks: list[dict], args) -> list[dict]:
    if not args.skip_existing:
        return tasks
    kept, skipped = [], []
    for task in tasks:
        (skipped if task_outputs_exist(task, args) else kept).append(task)
    if skipped:
        print(f'skip-existing: skipped {len(skipped)} task(s) with all requested outputs present')
    return kept


def run_interactive(tasks: list[dict], args):
    if len(tasks) != 1:
        raise ValueError('interactive mode expects exactly one task; use --tracers/--regions/--zrange-index or --zrange to select one')
    task = tasks[0]
    run_stats_task(version=args.version, tracer=task['tracer'], zrange=task['zrange'], region=task['region'],
                   stats_dir=args.stats_dir, stats=tuple(args.stats), nran=args.nran,
                   bao_blinding_options=(False if args.unblinded else bao_blinding_options_for_task(task, args)),
                   measurement_preset=MEASUREMENT_PRESET)


def run_direct_grid(tasks: list[dict], args):
    """Run all selected tasks sequentially in the current allocation."""
    if not tasks:
        rank0_print('direct-grid: no tasks selected')
        return
    for index, task in enumerate(tasks, start=1):
        rank0_print(
            f'direct-grid task {index}/{len(tasks)}: '
            f'{task["tracer"]} {task["region"]} z={task["zrange"][0]:.1f}-{task["zrange"][1]:.1f}'
        )
        run_stats_task(version=args.version, tracer=task['tracer'], zrange=task['zrange'], region=task['region'],
                       stats_dir=args.stats_dir, stats=tuple(args.stats), nran=args.nran,
                       bao_blinding_options=(False if args.unblinded else bao_blinding_options_for_task(task, args)),
                       measurement_preset=MEASUREMENT_PRESET)


def check_outputs(tasks: list[dict], args):
    summary = []
    for task in tasks:
        data_fn, random_fns = expected_input_files(task, args)
        for stat in args.stats:
            output = expected_output_file(stat, task, args)
            entry = {
                'version': args.version,
                'tracer': task['tracer'],
                'region': task['region'],
                'zrange': list(task['zrange']),
                'stat': stat,
                'input_data': str(data_fn),
                'input_data_exists': data_fn.exists(),
                'input_randoms': [str(fn) for fn in random_fns],
                'input_randoms_missing': [str(fn) for fn in random_fns if not fn.exists()],
                'output_file': str(output),
                'output_exists': output.exists(),
                'output_readable': False,
                'attrs': {},
            }
            if output.exists():
                try:
                    import lsstypes as types
                    statistic = types.read(output)
                    attrs = getattr(statistic, 'attrs', {})
                    keys = [key for key in attrs if 'blinding' in key]
                    entry['attrs'] = {key: attrs[key].tolist() if hasattr(attrs[key], 'tolist') else attrs[key] for key in keys}
                    entry['output_readable'] = True
                except Exception as exc:
                    entry['read_error'] = repr(exc)
            summary.append(entry)

    n_outputs = len(summary)
    n_found = sum(entry['output_exists'] for entry in summary)
    n_readable = sum(entry['output_readable'] for entry in summary)
    print(f'outputs found/readable: {n_found}/{n_outputs} found, {n_readable}/{n_outputs} readable')
    for entry in summary:
        status = 'ok' if entry['output_readable'] else ('missing' if not entry['output_exists'] else 'unreadable')
        print(f'{status:10s} {entry["stat"]:21s} {entry["tracer"]:18s} {entry["region"]:3s} z={entry["zrange"][0]:.1f}-{entry["zrange"][1]:.1f}')

    args.stats_dir.mkdir(parents=True, exist_ok=True)
    summary_name = 'summary_catalog_bao_blinding_dr1_unblinded_reference.json' if args.unblinded else 'summary_catalog_bao_blinding_dr1.json'
    summary_fn = args.stats_dir / summary_name
    summary_fn.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f'wrote summary: {summary_fn}')


def plot_outputs(tasks: list[dict], args):
    import matplotlib
    matplotlib.use('Agg')
    import lsstypes as types

    args.plots_dir.mkdir(parents=True, exist_ok=True)
    made = 0
    for task in tasks:
        for stat in args.stats:
            output = expected_output_file(stat, task, args)
            if not output.exists():
                continue
            if stat == 'mesh2_spectrum':
                observable = types.read(output).select(k=slice(0, None, 5))
            elif stat == 'particle2_correlation':
                corr = types.read(output)
                observable = corr.select(s=slice(0, None, 4)).select(s=(20., 180.)).project(ells=[0, 2, 4])
            else:
                continue
            fig = observable.plot(show=False)
            if fig.axes:
                fig.axes[0].set_title(f'{task["tracer"]} {task["region"]} z={task["zrange"][0]}-{task["zrange"][1]} BAO/AP blinded')
            plot_fn = args.plots_dir / f'{stat}_{task["tracer"]}_z{task["zrange"][0]:.1f}-{task["zrange"][1]:.1f}_{task["region"]}.png'
            fig.savefig(plot_fn, dpi=150, bbox_inches='tight')
            made += 1
            print(f'wrote plot: {plot_fn}')
    print(f'made {made} plot(s)')


def main():
    args = parse_args()
    tasks = filter_existing_tasks(build_tasks(args), args)

    if args.mode == 'dry-run':
        print_task_table(tasks, args)
    elif args.mode == 'interactive':
        run_interactive(tasks, args)
    elif args.mode == 'direct-grid':
        run_direct_grid(tasks, args)
    elif args.mode == 'desipipe':
        create_desipipe_tasks(tasks, args)
    elif args.mode == 'check':
        check_outputs(tasks, args)
    elif args.mode == 'plot':
        plot_outputs(tasks, args)
    else:
        raise ValueError(f'unknown mode {args.mode}')


if __name__ == '__main__':
    main()
