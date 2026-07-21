"""Numerical diagnostics for bispectrum-window grid convergence.

The helpers in this module are deliberately independent of the DESI likelihood
builder.  They operate on flat predictions and covariance matrices, which
makes the definitions easy to test and lets the batch runner store all inputs
needed to reproduce a reported metric.
"""

from __future__ import annotations

import numpy as np


def _as_vector(value, name):
    value = np.asarray(value, dtype='f8')
    if value.ndim != 1:
        raise ValueError(f'{name} must be one-dimensional; found shape {value.shape}.')
    if value.size == 0:
        raise ValueError(f'{name} cannot be empty.')
    if not np.all(np.isfinite(value)):
        raise ValueError(f'{name} contains non-finite values.')
    return value


def compare_predictions(reference, candidate, data, precision, covariance=None, mask=None):
    """Return covariance-weighted differences between two model predictions.

    Parameters
    ----------
    reference, candidate, data : array
        Flat model and data vectors on identical observable coordinates.
    precision : array
        Precision matrix corresponding to the full data vector.
    covariance : array, optional
        Covariance matrix used for per-bin normalized residuals.  When omitted,
        it is obtained by inverting ``precision``.
    mask : array, optional
        Boolean selection defining the bins attributed to the bispectrum.  The
        model discrepancy is set to zero outside the mask while the full
        precision matrix is retained, preserving cross-covariance effects.
    """
    reference = _as_vector(reference, 'reference')
    candidate = _as_vector(candidate, 'candidate')
    data = _as_vector(data, 'data')
    if reference.shape != candidate.shape or reference.shape != data.shape:
        raise ValueError('reference, candidate, and data must have identical shapes.')
    precision = np.asarray(precision, dtype='f8')
    if precision.shape != (reference.size, reference.size):
        raise ValueError(
            f'precision must have shape {(reference.size, reference.size)}; found {precision.shape}.'
        )
    delta = candidate - reference
    if mask is not None:
        mask = np.asarray(mask, dtype='?')
        if mask.shape != reference.shape:
            raise ValueError(f'mask must have shape {reference.shape}; found {mask.shape}.')
        delta = np.where(mask, delta, 0.)
        candidate = reference + delta
    residual_reference = data - reference
    residual_candidate = data - candidate
    chi2_reference = float(residual_reference @ precision @ residual_reference)
    chi2_candidate = float(residual_candidate @ precision @ residual_candidate)
    model_snr2 = float(delta @ precision @ delta)
    if covariance is None:
        covariance = np.linalg.pinv(precision, hermitian=True)
    covariance = np.asarray(covariance, dtype='f8')
    if covariance.shape != precision.shape:
        raise ValueError(f'covariance must have shape {precision.shape}; found {covariance.shape}.')
    sigma = np.sqrt(np.clip(np.diag(covariance), 0., None))
    valid_sigma = sigma > 0.
    pulls = np.zeros_like(delta)
    pulls[valid_sigma] = delta[valid_sigma] / sigma[valid_sigma]
    scale = np.maximum(np.abs(reference), np.finfo('f8').eps)
    selected = np.ones(reference.size, dtype='?') if mask is None else mask
    return {
        'chi2_reference': chi2_reference,
        'chi2_candidate': chi2_candidate,
        'delta_chi2': chi2_candidate - chi2_reference,
        'model_snr2': model_snr2,
        'model_snr': float(np.sqrt(max(model_snr2, 0.))),
        'max_abs_bin_pull': float(np.max(np.abs(pulls[selected]), initial=0.)),
        'rms_bin_pull': float(np.sqrt(np.mean(pulls[selected] ** 2))) if np.any(selected) else 0.,
        'rms_fractional_difference': (
            float(np.sqrt(np.mean((delta[selected] / scale[selected]) ** 2))) if np.any(selected) else 0.
        ),
        'max_abs_difference': float(np.max(np.abs(delta[selected]), initial=0.)),
    }


def convergence_order(coarse, medium, fine, ratio=2., mask=None):
    """Estimate grid-convergence order from three successively finer predictions.

    The estimate uses Euclidean norms of successive differences:

    ``p = log(||coarse-medium|| / ||medium-fine||) / log(ratio)``.

    A non-positive or non-finite order indicates that the tested points are not
    in a clean monotonic convergence regime.
    """
    coarse = _as_vector(coarse, 'coarse')
    medium = _as_vector(medium, 'medium')
    fine = _as_vector(fine, 'fine')
    if coarse.shape != medium.shape or coarse.shape != fine.shape:
        raise ValueError('coarse, medium, and fine must have identical shapes.')
    ratio = float(ratio)
    if not np.isfinite(ratio) or ratio <= 1.:
        raise ValueError('ratio must be finite and greater than one.')
    if mask is not None:
        mask = np.asarray(mask, dtype='?')
        if mask.shape != coarse.shape:
            raise ValueError(f'mask must have shape {coarse.shape}; found {mask.shape}.')
        coarse, medium, fine = coarse[mask], medium[mask], fine[mask]
    coarse_medium = float(np.linalg.norm(coarse - medium))
    medium_fine = float(np.linalg.norm(medium - fine))
    if coarse_medium == 0. and medium_fine == 0.:
        order = np.inf
    elif coarse_medium <= 0. or medium_fine <= 0.:
        order = np.nan
    else:
        order = float(np.log(coarse_medium / medium_fine) / np.log(ratio))
    return {
        'coarse_medium_norm': coarse_medium,
        'medium_fine_norm': medium_fine,
        'convergence_ratio': coarse_medium / medium_fine if medium_fine else np.inf,
        'convergence_order': order,
        'monotonic': bool(np.isfinite(order) and order > 0.) or bool(np.isinf(order)),
    }


def normalized_importance_weights(delta_loglikelihood):
    """Return stable normalized importance weights and their effective size."""
    delta_loglikelihood = _as_vector(delta_loglikelihood, 'delta_loglikelihood')
    shifted = delta_loglikelihood - np.max(delta_loglikelihood)
    weights = np.exp(shifted)
    total = weights.sum()
    if not np.isfinite(total) or total <= 0.:
        raise ValueError('Importance weights have zero or non-finite normalization.')
    weights /= total
    ess = float(1. / np.sum(weights ** 2))
    return weights, ess


def reweighted_moments(values, delta_loglikelihood):
    """Return original and importance-reweighted moments for one parameter."""
    values = _as_vector(values, 'values')
    if values.shape != np.asarray(delta_loglikelihood).shape:
        raise ValueError('values and delta_loglikelihood must have identical shapes.')
    weights, ess = normalized_importance_weights(delta_loglikelihood)
    mean = float(np.mean(values))
    sigma = float(np.std(values))
    reweighted_mean = float(weights @ values)
    reweighted_sigma = float(np.sqrt(weights @ (values - reweighted_mean) ** 2))
    return {
        'mean': mean,
        'sigma': sigma,
        'reweighted_mean': reweighted_mean,
        'reweighted_sigma': reweighted_sigma,
        'shift': reweighted_mean - mean,
        'shift_over_sigma': (reweighted_mean - mean) / sigma if sigma else np.nan,
        'weight_ess': ess,
        'weight_ess_fraction': ess / values.size,
    }


def fisher_parameter_bias(jacobian, precision, model_delta, prior_precision=None):
    """Project a numerical model error into a local Fisher parameter shift.

    ``model_delta`` is candidate minus reference.  The returned shift has the
    sign required for the candidate model to fit data generated by the
    reference model.
    """
    jacobian = np.asarray(jacobian, dtype='f8')
    precision = np.asarray(precision, dtype='f8')
    model_delta = _as_vector(model_delta, 'model_delta')
    if jacobian.ndim != 2 or jacobian.shape[0] != model_delta.size:
        raise ValueError('jacobian must have shape (number of bins, number of parameters).')
    if precision.shape != (model_delta.size, model_delta.size):
        raise ValueError('precision shape is incompatible with model_delta.')
    fisher = jacobian.T @ precision @ jacobian
    if prior_precision is not None:
        prior_precision = np.asarray(prior_precision, dtype='f8')
        if prior_precision.shape != fisher.shape:
            raise ValueError('prior_precision shape is incompatible with the Fisher matrix.')
        fisher = fisher + prior_precision
    covariance = np.linalg.pinv(fisher, hermitian=True)
    shift = -covariance @ jacobian.T @ precision @ model_delta
    sigma = np.sqrt(np.clip(np.diag(covariance), 0., None))
    return {
        'fisher': fisher,
        'covariance': covariance,
        'shift': shift,
        'sigma': sigma,
        'shift_over_sigma': np.divide(
            shift, sigma, out=np.full_like(shift, np.nan), where=sigma > 0.,
        ),
    }
