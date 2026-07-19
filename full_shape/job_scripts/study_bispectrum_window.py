"""Measure bispectrum-window grid convergence without running new MCMC chains.

The script compares window-convolved predictions on identical observable
coordinates.  It can evaluate existing posterior points, validate emulators
against direct FOLPSD calculations, run sparse native-window anchors, estimate
local Fisher shifts, importance-reweight existing chains, and launch cheap
grid-specific profiles.

Example
-------

.. code-block:: bash

    srun -n 1 python study_bispectrum_window.py \
      --observable-dk 0.005 0.01 \
      --theory-dk 0.0025 0.005 0.01 \
      --stages evaluate direct native fisher \
      --chain dk005=/path/to/dk005/chain \
      --chain dk01_dynamic=/path/to/dk01/dynamic/chain \
      --chain dk01_fixed005=/path/to/dk01/fixed/chain \
      --output-dir $PSCRATCH/bk_window_convergence/lrg2

Run ``--stages profile`` separately when Minuit profiles are desired.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

# Prefer the repository containing this script over another editable checkout.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from full_shape import setup_logging, tools  # noqa: E402
from full_shape.bispectrum_window_convergence import (  # noqa: E402
    compare_predictions,
    convergence_order,
    fisher_parameter_bias,
    reweighted_moments,
)
from full_shape.job_scripts import validation_abacus_mocks as validation  # noqa: E402


logger = logging.getLogger('bispectrum_window_study')
DEFAULT_THEORY_DK = (0.0025, 0.005, 0.01)
DEFAULT_OBSERVABLE_DK = (0.005, 0.01)
DEFAULT_COSMO_PARAMS = ('h', 'omega_b', 'omega_cdm', 'logA', 'n_s', 'Omega_m', 'sigma8_m')


def _float_label(value):
    if value == tools.SPECTRUM3_NATIVE_WINDOW_GRID:
        return 'native'
    return f'{float(value):g}'


def _parse_chain(value):
    try:
        label, path = value.split('=', 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError('Chains must be specified as LABEL=PATH.') from exc
    if not label or not path:
        raise argparse.ArgumentTypeError('Chains must be specified as non-empty LABEL=PATH.')
    return label, Path(path).expanduser()


def _json_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f'Cannot serialize {type(value).__name__}.')


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, default=_json_value)


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text('')
        return
    fieldnames = []
    for row in rows:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(name)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _set_observable_dk(options, observable_dk):
    """Set all selected P and B observable bins to a common spacing."""
    observable_dk = float(observable_dk)
    if observable_dk <= 0. or not np.isfinite(observable_dk):
        raise ValueError('observable_dk must be positive and finite.')
    for likelihood_options in options['likelihoods']:
        for observable_options in likelihood_options['observables']:
            for selection in observable_options['stat'].get('select', []):
                selection['k'][2] = observable_dk
                kmin, kmax = selection['k'][:2]
                ratio = (kmax - kmin) / observable_dk
                if not np.isclose(ratio, round(ratio), atol=1e-8):
                    raise ValueError(
                        f'Selection [{kmin}, {kmax}] is not aligned with observable dk={observable_dk}.'
                    )
    return options


def build_options(args, observable_dk, theory_dk, emulator=True):
    """Build one validation configuration while varying only the window grid."""
    numeric_theory_dk = None if theory_dk == tools.SPECTRUM3_NATIVE_WINDOW_GRID else float(theory_dk)
    options = validation._build_run_options(
        stats=['mesh2_spectrum', 'mesh3_spectrum'],
        tracers=[args.tracer],
        version=args.dataset,
        covariance=args.covariance,
        stats_dir=args.stats_dir,
        project=args.project,
        theory_model=args.theory_model,
        cosmo_model=args.cosmo_params,
        prior_basis=args.prior_basis,
        emulator=emulator,
        profile_iterations=args.profile_iterations,
        folpsd_damping=args.folpsd_damping,
        folpsd_damping_method=args.folpsd_damping_method,
        kmax_overrides=validation._parse_kmax_overrides(args.kmax),
        bispectrum_theory_dk=numeric_theory_dk,
    )
    _set_observable_dk(options, observable_dk)
    if theory_dk == tools.SPECTRUM3_NATIVE_WINDOW_GRID:
        for likelihood_options in options['likelihoods']:
            for observable_options in likelihood_options['observables']:
                if 'mesh3_spectrum' in observable_options['stat']['kind']:
                    observable_options.setdefault('window', {})['theory_dk'] = theory_dk
    return options


def _observable_masks(component):
    """Return full-vector masks for all bispectrum bins and individual poles."""
    offset = 0
    masks = {}
    for observable in component.observables:
        size = observable.flatdata.size
        is_bispectrum = observable.name.startswith('mesh3_spectrum')
        if is_bispectrum:
            mask = np.zeros(component.flatdata.size, dtype='?')
            mask[offset:offset + size] = True
            masks['bispectrum'] = mask
            pole_offset = offset
            for label, pole in observable.data.items(level=None):
                pole_mask = np.zeros(component.flatdata.size, dtype='?')
                pole_mask[pole_offset:pole_offset + pole.size] = True
                ell = label.get('ells', 'unknown')
                if isinstance(ell, (tuple, list)):
                    ell = ''.join(str(item) for item in ell)
                masks[f'B{ell}'] = pole_mask
                pole_offset += pole.size
        offset += size
    if 'bispectrum' not in masks:
        raise ValueError('The likelihood does not contain a bispectrum observable.')
    return masks


def build_pipeline(options, cache_dir, cache_mode='rw'):
    """Build a compiled likelihood returning both log likelihood and prediction."""
    from desilike import compile, get_params

    likelihood = tools.get_likelihood(
        likelihoods_options=copy.deepcopy(options['likelihoods']),
        cosmology_options=copy.deepcopy(options['cosmology']),
        cache_dir=cache_dir,
        cache_mode=cache_mode,
    )
    if len(likelihood.likelihoods) != 1:
        raise ValueError('The convergence runner currently expects exactly one tracer likelihood.')
    component = likelihood.likelihoods[0]
    pipeline = compile(
        likelihood,
        output=lambda: (component.logpdf, component.flattheory),
    )
    params = get_params(likelihood).select(input=True, varied=True)
    parameter_defaults = {param.name: float(param.value) for param in params}
    parameter_objects = {param.name: param for param in params}
    covariance = np.asarray(component.covariance.value(), dtype='f8')
    return {
        'likelihood': likelihood,
        'component': component,
        'pipeline': pipeline,
        'parameter_defaults': parameter_defaults,
        'parameter_objects': parameter_objects,
        'data': np.asarray(component.flatdata, dtype='f8'),
        'precision': np.asarray(component.precision, dtype='f8'),
        'covariance': covariance,
        'masks': _observable_masks(component),
    }


def _pipeline_grid_metadata(pipeline_info):
    observable = next(
        observable for observable in pipeline_info['component'].observables
        if observable.name.startswith('mesh3_spectrum')
    )
    encoded = observable.window.attrs.get('spectrum3_window_grid', None)
    if isinstance(encoded, str):
        metadata = json.loads(encoded)
    else:
        metadata = dict(encoded or {})
    metadata.update({
        'ndata': int(observable.flatdata.size),
        'ntheory': int(observable.window.theory.size),
    })
    return metadata


def _evaluate(pipeline_info, points):
    loglikes, predictions, elapsed = [], [], []
    pipeline = pipeline_info['pipeline']
    defaults = pipeline_info['parameter_defaults']
    for point in points:
        params = dict(defaults)
        params.update({name: value for name, value in point['params'].items() if name in defaults})
        start = time.perf_counter()
        loglike, prediction = pipeline(params)
        elapsed.append(time.perf_counter() - start)
        loglikes.append(float(np.asarray(loglike)))
        predictions.append(np.asarray(prediction, dtype='f8'))
    return np.asarray(loglikes), np.asarray(predictions), np.asarray(elapsed)


def _read_chain(path, burnin):
    from desilike.samples import MCSamples

    filenames = sorted(path.glob('samples_*.h5'))
    if not filenames:
        raise FileNotFoundError(f'No samples_*.h5 files found in {path}.')
    return MCSamples.concatenate([
        MCSamples.read(filename).remove_burnin(burnin).ravel()
        for filename in filenames
    ])


def _infer_chain_configuration(path):
    config_path = path / 'config.yaml'
    checks_path = path / 'checks.json'
    checks = None
    if checks_path.exists():
        with checks_path.open() as stream:
            checks = json.load(stream)
    converged = bool(len(checks) >= 2 and checks[-2:] == [True, True]) if checks is not None else None
    if not config_path.exists():
        return {
            'config_path': None, 'observable_dk': None, 'theory_dk': None,
            'checks_path': checks_path if checks_path.exists() else None,
            'converged': converged,
        }
    options = tools.read_options(config_path)
    mesh3 = None
    for likelihood_options in options.get('likelihoods', []):
        for observable_options in likelihood_options.get('observables', []):
            if 'mesh3_spectrum' in observable_options.get('stat', {}).get('kind', ''):
                mesh3 = observable_options
                break
    if mesh3 is None:
        return {
            'config_path': config_path, 'observable_dk': None, 'theory_dk': None,
            'checks_path': checks_path if checks_path.exists() else None,
            'converged': converged,
        }
    spacings = {float(item['k'][2]) for item in mesh3['stat'].get('select', [])}
    observable_dk = spacings.pop() if len(spacings) == 1 else None
    theory_dk = mesh3.get('window', {}).get('theory_dk', None)
    if theory_dk is None:
        theory_dk = observable_dk
    return {
        'config_path': config_path,
        'observable_dk': observable_dk,
        'theory_dk': theory_dk,
        'checks_path': checks_path if checks_path.exists() else None,
        'converged': converged,
    }


def load_chain_points(chain_specs, required_names, observable_dk, nposterior, burnin, seed,
                      extra_names=()):
    """Load deterministic best-fit and posterior subsets matching one data grid."""
    rng = np.random.default_rng(seed)
    points = []
    sources = {}
    for label, path in chain_specs:
        inferred = _infer_chain_configuration(path)
        if inferred['observable_dk'] is not None and not np.isclose(inferred['observable_dk'], observable_dk):
            logger.info(
                'Skipping chain %s for observable dk=%g; its config uses dk=%s.',
                label, observable_dk, inferred['observable_dk'],
            )
            continue
        if inferred['converged'] is False:
            logger.warning('Chain %s does not end in two passing convergence checks.', label)
        chain = _read_chain(path, burnin=burnin)
        missing = [name for name in required_names if name not in chain.names()]
        if missing:
            raise ValueError(f'Chain {path} is missing likelihood inputs: {missing}.')
        stored_names = list(required_names) + [
            name for name in extra_names if name in chain.names() and name not in required_names
        ]
        arrays = {name: np.asarray(chain[name], dtype='f8').reshape(-1) for name in stored_names}
        available = len(next(iter(arrays.values())))
        count = min(int(nposterior), available)
        indices = np.sort(rng.choice(available, size=count, replace=False))
        logposterior = np.asarray(chain['logposterior']).reshape(-1)
        best_index = int(np.nanargmax(logposterior))
        source_points = []
        for kind, index in [('bestfit', best_index)] + [('posterior', int(index)) for index in indices]:
            point = {
                'id': f'{label}:{kind}:{index}',
                'source': label,
                'kind': kind,
                'index': index,
                'params': {name: float(values[index]) for name, values in arrays.items()},
            }
            points.append(point)
            source_points.append(point)
        sources[label] = {
            'path': path,
            'inferred': inferred,
            'chain': chain,
            'points': source_points,
        }
    return points, sources


def _select_direct_points(points, count, seed):
    """Keep all anchors, then draw a deterministic mixed posterior subset."""
    count = min(int(count), len(points))
    anchors = [point for point in points if point['kind'] in ('default', 'bestfit')]
    selected = anchors[:count]
    remaining = count - len(selected)
    if remaining <= 0:
        return selected
    posterior = [point for point in points if point['kind'] == 'posterior']
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(posterior), size=min(remaining, len(posterior)), replace=False))
    selected.extend(posterior[index] for index in indices)
    return selected


def _grid_metrics(reference, candidate, pipeline_info):
    rows = {}
    for block, mask in pipeline_info['masks'].items():
        rows[block] = compare_predictions(
            reference=reference,
            candidate=candidate,
            data=pipeline_info['data'],
            precision=pipeline_info['precision'],
            covariance=pipeline_info['covariance'],
            mask=mask,
        )
    return rows


def _evaluate_grids(args, observable_dk, points, grids, emulator=True, prebuilt=None):
    results = {}
    prebuilt = {} if prebuilt is None else prebuilt
    for grid in grids:
        label = _float_label(grid)
        if label in prebuilt:
            options = prebuilt[label]['options']
            pipeline_info = prebuilt[label]['pipeline_info']
        else:
            logger.info(
                'Building observable dk=%g, theory grid=%s, emulator=%s.',
                observable_dk, label, emulator,
            )
            options = build_options(args, observable_dk, grid, emulator=emulator)
            pipeline_info = build_pipeline(options, cache_dir=args.cache_dir, cache_mode=args.cache_mode)
        loglikes, predictions, elapsed = _evaluate(pipeline_info, points)
        results[label] = {
            'grid': grid,
            'options': options,
            'pipeline_info': pipeline_info,
            'loglikes': loglikes,
            'predictions': predictions,
            'elapsed': elapsed,
        }
    return results


def _summarize_grid_results(points, results, reference_label):
    reference = results[reference_label]
    evaluation_rows, metric_rows = [], []
    for label, result in results.items():
        for index, point in enumerate(points):
            evaluation_rows.append({
                'point_id': point['id'],
                'source': point['source'],
                'kind': point['kind'],
                'grid': label,
                'loglikelihood': result['loglikes'][index],
                'chi2': -2. * result['loglikes'][index],
                'elapsed_seconds': result['elapsed'][index],
            })
            if label == reference_label:
                continue
            for block, metrics in _grid_metrics(
                reference['predictions'][index],
                result['predictions'][index],
                reference['pipeline_info'],
            ).items():
                metric_rows.append({
                    'point_id': point['id'],
                    'source': point['source'],
                    'kind': point['kind'],
                    'grid': label,
                    'reference_grid': reference_label,
                    'block': block,
                    **metrics,
                })
    return evaluation_rows, metric_rows


def _convergence_rows(points, results):
    numeric = sorted(
        [(float(result['grid']), label) for label, result in results.items() if result['grid'] != 'native'],
        reverse=True,
    )
    if len(numeric) < 3:
        return []
    (_, coarse), (_, medium), (_, fine) = numeric[:3]
    masks = results[fine]['pipeline_info']['masks']
    rows = []
    for index, point in enumerate(points):
        for block, mask in masks.items():
            rows.append({
                'point_id': point['id'],
                'source': point['source'],
                'kind': point['kind'],
                'block': block,
                'coarse_grid': coarse,
                'medium_grid': medium,
                'fine_grid': fine,
                **convergence_order(
                    results[coarse]['predictions'][index],
                    results[medium]['predictions'][index],
                    results[fine]['predictions'][index],
                    mask=mask,
                ),
            })
    return rows


def _reweighting_rows(args, points, sources, results):
    rows = []
    point_lookup = {point['id']: index for index, point in enumerate(points)}
    parameters = tuple(args.reweight_params)
    for source, info in sources.items():
        posterior_points = [point for point in info['points'] if point['kind'] == 'posterior']
        if not posterior_points:
            continue
        baseline = info['inferred']['theory_dk']
        if baseline is None:
            logger.warning('Cannot infer baseline theory grid for %s; skipping reweighting.', source)
            continue
        baseline_label = _float_label(baseline)
        if baseline_label not in results:
            logger.warning('Baseline grid %s for %s was not evaluated; skipping reweighting.', baseline_label, source)
            continue
        indices = np.asarray([point_lookup[point['id']] for point in posterior_points])
        baseline_loglike = results[baseline_label]['loglikes'][indices]
        for target, result in results.items():
            if target == baseline_label or result['grid'] == 'native':
                continue
            delta = result['loglikes'][indices] - baseline_loglike
            for parameter in parameters:
                if parameter not in posterior_points[0]['params']:
                    continue
                values = np.asarray([point['params'][parameter] for point in posterior_points])
                rows.append({
                    'source': source,
                    'baseline_grid': baseline_label,
                    'target_grid': target,
                    'parameter': parameter,
                    'npoints': len(values),
                    **reweighted_moments(values, delta),
                })
    return rows


def _parameter_step(parameter, value, relative_step):
    for prior in (getattr(parameter, 'ref', None), getattr(parameter, 'prior', None)):
        scale = getattr(prior, 'scale', None)
        if scale is not None and np.ndim(scale) == 0 and np.isfinite(scale) and scale > 0.:
            return float(relative_step * scale)
    return float(relative_step * max(abs(value), 1.))


def _prior_precision(parameter_names, parameter_objects):
    matrix = np.zeros((len(parameter_names), len(parameter_names)), dtype='f8')
    for index, name in enumerate(parameter_names):
        prior = getattr(parameter_objects[name], 'prior', None)
        scale = getattr(prior, 'scale', None)
        dist = getattr(prior, 'dist', None)
        if dist == 'norm' and scale is not None and np.isfinite(scale) and scale > 0.:
            matrix[index, index] = 1. / float(scale) ** 2
    return matrix


def _fisher_rows(args, points, results, reference_label):
    reference = results[reference_label]
    info = reference['pipeline_info']
    point = next((point for point in points if point['kind'] == 'bestfit'), points[0])
    defaults = dict(info['parameter_defaults'])
    defaults.update(point['params'])
    names = list(info['parameter_defaults'])
    jacobian = []
    for name in names:
        value = defaults[name]
        step = _parameter_step(info['parameter_objects'][name], value, args.fisher_relative_step)
        plus, minus = dict(defaults), dict(defaults)
        plus[name], minus[name] = value + step, value - step
        _, plus_prediction, _ = _evaluate(info, [{'params': plus}])
        _, minus_prediction, _ = _evaluate(info, [{'params': minus}])
        jacobian.append((plus_prediction[0] - minus_prediction[0]) / (2. * step))
    jacobian = np.asarray(jacobian).T
    prior_precision = _prior_precision(names, info['parameter_objects'])
    point_index = points.index(point)
    rows = []
    for label, result in results.items():
        if label == reference_label or result['grid'] == 'native':
            continue
        model_delta = result['predictions'][point_index] - reference['predictions'][point_index]
        model_delta = np.where(info['masks']['bispectrum'], model_delta, 0.)
        bias = fisher_parameter_bias(
            jacobian, info['precision'], model_delta, prior_precision=prior_precision,
        )
        for index, name in enumerate(names):
            rows.append({
                'point_id': point['id'],
                'grid': label,
                'reference_grid': reference_label,
                'parameter': name,
                'shift': bias['shift'][index],
                'sigma': bias['sigma'][index],
                'shift_over_sigma': bias['shift_over_sigma'][index],
            })
    return rows


def _profile_rows(args, observable_dk, grids, output_dir):
    from full_shape import run_fit_from_options
    from desilike.samples import Profiles
    import functools

    rows = []
    fits_dir = output_dir / 'profile_fits'
    get_fits_fn = functools.partial(tools.get_fits_fn, fits_dir=fits_dir)
    for grid in grids:
        if grid == tools.SPECTRUM3_NATIVE_WINDOW_GRID:
            continue
        options = build_options(args, observable_dk, grid, emulator=True)
        run_fit_from_options(
            ['profile'],
            get_fits_fn=get_fits_fn,
            cache_dir=args.cache_dir,
            cache_mode=args.cache_mode,
            **copy.deepcopy(options),
        )
        filename = get_fits_fn(kind='profiles', **options)
        profile = Profiles.read(filename).choice(index='argmax', squeeze=True)
        best = profile.best
        error = profile.error
        for name in best.names():
            rows.append({
                'grid': _float_label(grid),
                'parameter': name,
                'bestfit': float(np.asarray(best[name])),
                'error': float(np.asarray(error[name])) if error is not None and name in error.names() else np.nan,
                'profile_path': str(filename),
            })
    return rows


def _run_one_observable_grid(args, observable_dk, grids, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    provisional_options = build_options(args, observable_dk, grids[0], emulator=True)
    provisional = build_pipeline(provisional_options, args.cache_dir, cache_mode=args.cache_mode)
    defaults = provisional['parameter_defaults']
    points = [{
        'id': 'fiducial:default:0',
        'source': 'fiducial',
        'kind': 'default',
        'index': 0,
        'params': dict(defaults),
    }]
    chain_points, sources = load_chain_points(
        args.chain,
        required_names=list(defaults),
        observable_dk=observable_dk,
        nposterior=args.nposterior,
        burnin=args.burnin,
        seed=args.seed,
        extra_names=args.reweight_params,
    )
    points.extend(chain_points)
    reference_label = _float_label(args.reference_theory_dk)
    manifest = {
        'observable_dk': observable_dk,
        'theory_dk': list(grids),
        'reference_theory_dk': args.reference_theory_dk,
        'dataset': args.dataset,
        'tracer': args.tracer,
        'stages': args.stages,
        'seed': args.seed,
        'nposterior': args.nposterior,
        'burnin': args.burnin,
        'chains': {
            label: {'path': str(info['path']), **info['inferred']}
            for label, info in sources.items()
        },
    }
    _write_json(output_dir / 'manifest.json', manifest)

    results = {}
    if any(stage in args.stages for stage in ('evaluate', 'direct', 'native', 'fisher')):
        first_label = _float_label(grids[0])
        results = _evaluate_grids(
            args, observable_dk, points, grids, emulator=True,
            prebuilt={first_label: {
                'options': provisional_options,
                'pipeline_info': provisional,
            }},
        )
        if reference_label not in results:
            raise ValueError(f'Reference theory grid {reference_label} was not evaluated.')
        evaluations, metrics = _summarize_grid_results(points, results, reference_label)
        _write_csv(output_dir / 'evaluations.csv', evaluations)
        _write_csv(output_dir / 'metrics.csv', metrics)
        _write_csv(output_dir / 'convergence.csv', _convergence_rows(points, results))
        _write_csv(output_dir / 'reweighting.csv', _reweighting_rows(args, points, sources, results))
        predictions = {
            f'{label}__predictions': result['predictions']
            for label, result in results.items()
        }
        predictions['point_ids'] = np.asarray([point['id'] for point in points])
        np.savez_compressed(output_dir / 'predictions.npz', **predictions)
        manifest['grids'] = {
            label: {
                'options_hash': tools._hash_options({
                    'likelihoods': result['options']['likelihoods'],
                    'cosmology': result['options']['cosmology'],
                }),
                'mean_evaluation_seconds': float(np.mean(result['elapsed'])),
                **_pipeline_grid_metadata(result['pipeline_info']),
            }
            for label, result in results.items()
        }
        _write_json(output_dir / 'manifest.json', manifest)

    if 'direct' in args.stages:
        direct_points = _select_direct_points(points, args.ndirect, args.seed)
        direct = _evaluate_grids(args, observable_dk, direct_points, grids, emulator=False)
        direct_rows = []
        for label, result in results.items():
            direct_result = direct[label]
            for index, point in enumerate(direct_points):
                for block, values in _grid_metrics(
                    direct_result['predictions'][index],
                    result['predictions'][index],
                    direct_result['pipeline_info'],
                ).items():
                    direct_rows.append({
                        'point_id': point['id'],
                        'grid': label,
                        'block': block,
                        **values,
                    })
        _write_csv(output_dir / 'emulator_validation.csv', direct_rows)

    if 'native' in args.stages:
        anchor_points = [points[0]] + [point for point in points if point['kind'] == 'bestfit']
        anchor_points = anchor_points[:args.native_anchors]
        compact_reference = _evaluate_grids(
            args, observable_dk, anchor_points, [args.reference_theory_dk], emulator=False,
        )[reference_label]
        native = _evaluate_grids(
            args, observable_dk, anchor_points, [tools.SPECTRUM3_NATIVE_WINDOW_GRID], emulator=False,
        )['native']
        native_rows = []
        for index, point in enumerate(anchor_points):
            for block, values in _grid_metrics(
                native['predictions'][index],
                compact_reference['predictions'][index],
                native['pipeline_info'],
            ).items():
                native_rows.append({
                    'point_id': point['id'],
                    'grid': reference_label,
                    'reference_grid': 'native',
                    'block': block,
                    **values,
                })
        _write_csv(output_dir / 'native_validation.csv', native_rows)

    if 'fisher' in args.stages:
        _write_csv(output_dir / 'fisher_bias.csv', _fisher_rows(args, points, results, reference_label))

    if 'profile' in args.stages:
        _write_csv(output_dir / 'profiles.csv', _profile_rows(args, observable_dk, grids, output_dir))


def _get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='abacus-hf-dr2-v2-altmtl')
    parser.add_argument('--tracer', default='LRG2')
    parser.add_argument('--covariance', default='holi-v3-altmtl')
    parser.add_argument('--stats-dir', type=Path, default=validation.DEFAULT_STATS_DIR)
    parser.add_argument('--cache-dir', type=Path, default=validation.DEFAULT_CACHE_DIR)
    parser.add_argument('--cache-mode', choices=['r', 'rw'], default='rw')
    parser.add_argument('--project', default='full_shape/base')
    parser.add_argument('--theory-model', default='folpsD')
    parser.add_argument('--cosmo-params', default='base')
    parser.add_argument('--prior-basis', default='physical_aap')
    parser.add_argument('--folpsd-damping', default='vdg')
    parser.add_argument('--folpsd-damping-method', default='tree+loop')
    parser.add_argument('--observable-dk', type=float, nargs='+', default=DEFAULT_OBSERVABLE_DK)
    parser.add_argument('--theory-dk', type=float, nargs='+', default=DEFAULT_THEORY_DK)
    parser.add_argument('--reference-theory-dk', type=float, default=0.0025)
    parser.add_argument(
        '--kmax', action='append',
        default=[
            'mesh2_spectrum:0=0.35',
            'mesh2_spectrum:2=0.25',
            'mesh3_spectrum:0,0,0=0.20',
            'mesh3_spectrum:2,0,2=0.03',
        ],
    )
    parser.add_argument('--chain', action='append', type=_parse_chain, default=[])
    parser.add_argument('--burnin', type=float, default=0.3)
    parser.add_argument('--nposterior', type=int, default=2000)
    parser.add_argument('--ndirect', type=int, default=64)
    parser.add_argument('--native-anchors', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--profile-iterations', type=int, default=4)
    parser.add_argument('--fisher-relative-step', type=float, default=0.01)
    parser.add_argument('--reweight-params', nargs='+', default=DEFAULT_COSMO_PARAMS)
    parser.add_argument(
        '--stages', nargs='+',
        choices=['evaluate', 'direct', 'native', 'fisher', 'profile', 'all'],
        default=['evaluate'],
    )
    default_output = Path(os.getenv('PSCRATCH', os.getenv('SCRATCH', '.'))) / 'bk_window_convergence'
    parser.add_argument('--output-dir', type=Path, default=default_output)
    return parser


def main(argv=None):
    args = _get_parser().parse_args(argv)
    setup_logging()
    if 'all' in args.stages:
        args.stages = ['evaluate', 'direct', 'native', 'fisher', 'profile']
    grids = sorted(set(float(value) for value in args.theory_dk))
    if args.reference_theory_dk not in grids:
        grids.append(float(args.reference_theory_dk))
        grids.sort()
    if args.nposterior < 1 or args.ndirect < 1 or args.native_anchors < 1:
        raise ValueError('nposterior, ndirect, and native-anchors must be positive.')
    if not 0. <= args.burnin < 1.:
        raise ValueError('burnin must lie in [0, 1).')
    for observable_dk in args.observable_dk:
        suffix = re.sub(r'[^0-9A-Za-z]+', 'p', f'{observable_dk:g}').strip('p')
        _run_one_observable_grid(
            args,
            observable_dk=float(observable_dk),
            grids=grids,
            output_dir=args.output_dir / f'observable_dk_{suffix}',
        )


if __name__ == '__main__':
    main()
