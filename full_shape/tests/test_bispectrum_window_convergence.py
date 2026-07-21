import numpy as np
import pytest

from full_shape.bispectrum_window_convergence import (
    compare_predictions,
    convergence_order,
    fisher_parameter_bias,
    normalized_importance_weights,
    reweighted_moments,
)
from full_shape.job_scripts import study_bispectrum_window as study


def test_compare_predictions_uses_full_precision_and_selected_delta():
    reference = np.zeros(3)
    candidate = np.array([10., 1., 0.])
    data = np.zeros(3)
    precision = np.array([
        [2., 0.5, 0.],
        [0.5, 3., 0.],
        [0., 0., 4.],
    ])
    covariance = np.eye(3)
    mask = np.array([False, True, False])
    result = compare_predictions(
        reference, candidate, data, precision, covariance=covariance, mask=mask)
    assert result['model_snr2'] == pytest.approx(3.)
    assert result['delta_chi2'] == pytest.approx(3.)
    assert result['max_abs_bin_pull'] == pytest.approx(1.)


def test_convergence_order_recovers_second_order_sequence():
    fine = np.array([1., 2.])
    medium = fine + np.array([1., -1.])
    coarse = medium + 4. * np.array([1., -1.])
    result = convergence_order(coarse, medium, fine)
    assert result['convergence_order'] == pytest.approx(2.)
    assert result['monotonic']


def test_importance_weights_and_reweighted_moments_are_stable():
    delta = np.array([-1000., -1001., -1002.])
    weights, ess = normalized_importance_weights(delta)
    assert weights.sum() == pytest.approx(1.)
    assert 1. <= ess <= 3.
    result = reweighted_moments(np.array([0., 1., 2.]), delta)
    assert result['reweighted_mean'] < result['mean']
    assert result['weight_ess'] == pytest.approx(ess)


def test_fisher_bias_one_parameter():
    jacobian = np.array([[1.], [2.]])
    precision = np.eye(2)
    delta = np.array([0.1, 0.2])
    result = fisher_parameter_bias(jacobian, precision, delta)
    assert result['shift'][0] == pytest.approx(-0.1)
    assert result['sigma'][0] == pytest.approx(1. / np.sqrt(5.))


def test_set_observable_dk_changes_steps_without_changing_limits():
    options = {
        'likelihoods': [{
            'observables': [
                {'stat': {'kind': 'mesh2_spectrum', 'select': [{'k': [0.02, 0.20, 0.01]}]}},
                {'stat': {'kind': 'mesh3_spectrum', 'select': [{'k': [0.02, 0.20, 0.01]}]}},
            ],
        }],
    }
    study._set_observable_dk(options, 0.005)
    selections = [
        observable['stat']['select'][0]['k']
        for observable in options['likelihoods'][0]['observables']
    ]
    assert selections == [[0.02, 0.20, 0.005], [0.02, 0.20, 0.005]]


def test_study_parser_defaults_to_three_grid_convergence():
    args = study._get_parser().parse_args([])
    assert tuple(args.theory_dk) == study.DEFAULT_THEORY_DK
    assert tuple(args.observable_dk) == study.DEFAULT_OBSERVABLE_DK
    assert args.reference_theory_dk == 0.0025


def test_direct_point_selection_preserves_anchors_and_is_deterministic():
    points = [
        {'id': 'fid', 'kind': 'default'},
        {'id': 'best-a', 'kind': 'bestfit'},
        {'id': 'best-b', 'kind': 'bestfit'},
    ] + [{'id': f'sample-{index}', 'kind': 'posterior'} for index in range(10)]
    selected = study._select_direct_points(points, 6, seed=42)
    repeated = study._select_direct_points(points, 6, seed=42)
    assert [point['id'] for point in selected[:3]] == ['fid', 'best-a', 'best-b']
    assert [point['id'] for point in selected] == [point['id'] for point in repeated]
