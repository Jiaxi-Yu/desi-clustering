#!/usr/bin/env python
"""Launch DESI cosmology inference with Cobaya."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from desipipe import Environment, Queue, TaskManager, setup_logging

from cosmo.cobaya.mapping_likelihoods import LIKELIHOOD_COMBINATIONS, normalize_likelihood_combination
from cosmo.cobaya import get_cobaya_info, get_likelihood_label, profile_cobaya, sample_cobaya, write_cobaya_yaml, yield_configs


DEFAULT_WORKTREE = Path('/global/homes/u/uendert/repos/desi/desi-clustering-cosmo')
QUEUE_NAME = 'desi_clustering_cobaya'
THREAD_ENV = {name: '32' for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']}


def parse_likelihood_combinations(values):
    """Return ``(label, expanded_likelihoods)`` combinations from CLI values."""
    combinations = []
    for value in values or ['desi-dr2-bao-all']:
        label = value if isinstance(value, str) and ',' not in value else get_likelihood_label(likelihoods=normalize_likelihood_combination(value))
        combinations.append((label, normalize_likelihood_combination(value)))
    return combinations


def print_likelihood_combinations():
    """Print available named likelihood-combination presets."""
    for name in sorted(LIKELIHOOD_COMBINATIONS):
        print(f'{name}: {", ".join(LIKELIHOOD_COMBINATIONS[name])}')


def _remove_flag(argv, flag):
    """Return ``argv`` with a boolean CLI flag removed."""
    return [arg for arg in argv if arg != flag]


def run_interactive_node(args, argv):
    """Request an interactive Slurm node and re-run this script in direct mode."""
    child_argv = _remove_flag(list(argv), '--interactive-node')
    if '--interactive' not in child_argv:
        child_argv.append('--interactive')
    command = [
        'srun',
        '-A', args.interactive_account,
        '-C', args.interactive_constraint,
        '-q', args.interactive_qos,
        '-N', str(args.interactive_nodes),
        '-n', str(args.mpiprocs_per_worker),
        '-c', str(args.cpus_per_task),
        '-t', args.time,
        '-J', args.interactive_job_name,
        sys.executable,
        str(Path(__file__).resolve()),
        *child_argv,
    ]
    env = os.environ.copy()
    worktree = str(Path(args.worktree))
    env['PYTHONPATH'] = f'{worktree}:{env.get("PYTHONPATH", "")}' if env.get('PYTHONPATH') else worktree
    for name, value in THREAD_ENV.items():
        env.setdefault(name, value)
    print('Launching interactive node command:')
    print(' '.join(command))
    return subprocess.run(command, env=env).returncode


def setup_task_manager(queue_name=QUEUE_NAME, worktree=DEFAULT_WORKTREE, output_dir='slurm_outputs/cobaya',
                       max_workers=20, time='04:00:00', mpiprocs_per_worker=4, nodes_per_worker=0.5):
    """Return sample/profile task managers for Cobaya jobs."""
    setup_logging()
    queue = Queue(queue_name)
    queue.clear(kill=False)
    command = [
        'module unload desi-clustering || true',
        f'export PYTHONPATH={worktree}:$PYTHONPATH',
    ]
    environ = Environment('nersc-cosmodesi', command=command)
    environ.update(THREAD_ENV)
    output_dir = Path(output_dir)
    output = str(output_dir / 'slurm-%j.out')
    error = str(output_dir / 'slurm-%j.err')
    tm = TaskManager(queue=queue, environ=environ)
    tm_sample = tm.clone(
        scheduler=dict(max_workers=max_workers),
        provider=dict(provider='nersc', time=time, mpiprocs_per_worker=mpiprocs_per_worker,
                      nodes_per_worker=nodes_per_worker, output=output, error=error,
                      killed_at_timeout=True),
    )
    tm_profile = tm.clone(
        scheduler=dict(max_workers=max_workers),
        provider=dict(provider='nersc', time=time, mpiprocs_per_worker=mpiprocs_per_worker,
                      nodes_per_worker=nodes_per_worker, output=output, error=error,
                      killed_at_timeout=True),
    )
    return tm_sample, tm_profile


def run_sample(**kwargs):
    """Self-contained wrapper executed by desipipe workers."""
    from desipipe import setup_logging
    from cosmo.cobaya import sample_cobaya
    setup_logging()
    return sample_cobaya(**kwargs)


def run_profile(**kwargs):
    """Self-contained wrapper executed by desipipe workers."""
    from desipipe import setup_logging
    from cosmo.cobaya import profile_cobaya
    setup_logging()
    return profile_cobaya(**kwargs)


def export_config(config_dir='configs/cobaya', likelihood_label=None, **kwargs):
    """Write one generated Cobaya YAML config and return its filename."""
    info = get_cobaya_info(output=False, **kwargs)
    label = likelihood_label or get_likelihood_label(likelihoods=kwargs.get('likelihoods'), dataset=kwargs.get('dataset'))
    filename = Path(config_dir) / kwargs.get('model', 'base') / f'{label}.yaml'
    return write_cobaya_yaml(info, filename)


def _get_parser():
    parser = argparse.ArgumentParser(description='Launch DESI Cobaya cosmology jobs directly, on an interactive node, or with desipipe.')
    parser.add_argument('--todo', nargs='+', default=['evaluate'], choices=['evaluate', 'sample', 'profile', 'export'],
                        help='Tasks to create/run.')
    parser.add_argument('--models', nargs='+', default=['base'], help='Cosmology models to run.')
    parser.add_argument('--likelihoods', nargs='+', default=['desi-dr2-bao-all'],
                        help='Likelihood combinations or named presets. Each explicit value is comma-separated.')
    parser.add_argument('--list-likelihood-combinations', action='store_true',
                        help='Print available named likelihood-combination presets and exit.')
    parser.add_argument('--theory', default='camb', help='Cobaya theory backend.')
    parser.add_argument('--run', default='run1', help='Run label for output paths.')
    parser.add_argument('--output_dir', default=None, help='Base output directory for Cobaya chains.')
    parser.add_argument('--config-dir', default='configs/cobaya', help='Directory for exported Cobaya YAML configs.')
    parser.add_argument('--queue-name', default=QUEUE_NAME, help='desipipe queue name.')
    parser.add_argument('--worktree', default=str(DEFAULT_WORKTREE), help='Worktree prepended to PYTHONPATH in jobs.')
    parser.add_argument('--interactive', action='store_true', help='Run directly instead of creating desipipe tasks.')
    parser.add_argument('--interactive-node', action='store_true',
                        help='Request an interactive Slurm node with srun, then re-run this script with --interactive.')
    parser.add_argument('--interactive-account', default='desi', help='Slurm account for --interactive-node.')
    parser.add_argument('--interactive-qos', default='interactive', help='Slurm QOS/queue for --interactive-node.')
    parser.add_argument('--interactive-constraint', default='cpu', help='Slurm constraint for --interactive-node.')
    parser.add_argument('--interactive-nodes', type=int, default=1, help='Number of nodes for --interactive-node.')
    parser.add_argument('--interactive-job-name', default='cobaya_interactive', help='Slurm job name for --interactive-node.')
    parser.add_argument('--cpus-per-task', type=int, default=32, help='CPUs per MPI task for --interactive-node.')
    parser.add_argument('--resume', action='store_true', help='Resume Cobaya MCMC chains.')
    parser.add_argument('--test', action='store_true', help='Pass Cobaya test/debug flags.')
    parser.add_argument('--max-workers', type=int, default=20, help='Maximum desipipe workers.')
    parser.add_argument('--time', default='04:00:00', help='NERSC job wall time.')
    parser.add_argument('--mpiprocs-per-worker', type=int, default=4, help='MPI processes per worker.')
    parser.add_argument('--nodes-per-worker', type=float, default=0.5, help='Nodes per worker.')
    return parser


def iter_configs(args):
    likelihoods = parse_likelihood_combinations(args.likelihoods)
    for todo in args.todo:
        sampler = 'evaluate' if todo in {'evaluate', 'export'} else ('iminuit' if todo == 'profile' else 'cobaya')
        for model in args.models:
            for likelihood_label, expanded_likelihoods in likelihoods:
                yield todo, dict(model=model, likelihoods=expanded_likelihoods, likelihood_label=likelihood_label,
                                 dataset=None, theory=args.theory, sampler=sampler, run=args.run,
                                 output_dir=args.output_dir, test=args.test)


def main(args=None):
    argv = sys.argv[1:] if args is None else list(args)
    parser = _get_parser()
    args = parser.parse_args(args=argv)
    if args.list_likelihood_combinations:
        print_likelihood_combinations()
        return

    if args.interactive_node:
        raise SystemExit(run_interactive_node(args, argv))

    if 'export' in args.todo:
        for todo, config in iter_configs(args):
            if todo == 'export':
                config = dict(config)
                likelihood_label = config.pop('likelihood_label', None)
                filename = export_config(config_dir=args.config_dir, likelihood_label=likelihood_label, **config)
                print(filename)
        if set(args.todo) == {'export'}:
            return

    if args.interactive:
        for todo, config in iter_configs(args):
            if todo == 'export':
                continue
            config = dict(config)
            likelihood_label = config.pop('likelihood_label', None)
            if likelihood_label is not None:
                config['output_label'] = likelihood_label
            if todo == 'profile':
                run_profile(**config)
            else:
                run_sample(resume=args.resume, **config)
        return

    tm_sample, tm_profile = setup_task_manager(
        queue_name=args.queue_name,
        worktree=Path(args.worktree),
        max_workers=args.max_workers,
        time=args.time,
        mpiprocs_per_worker=args.mpiprocs_per_worker,
        nodes_per_worker=args.nodes_per_worker,
    )
    sample_app = tm_sample.python_app(run_sample)
    profile_app = tm_profile.python_app(run_profile)

    for todo, config in iter_configs(args):
        if todo == 'export':
            continue
        config = dict(config)
        likelihood_label = config.pop('likelihood_label', None)
        if likelihood_label is not None:
            config['output_label'] = likelihood_label
        if todo == 'profile':
            profile_app(**config)
        else:
            sample_app(resume=args.resume, **config)


if __name__ == '__main__':
    main()
