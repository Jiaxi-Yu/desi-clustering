"""Diagnostic plots for real catalog-level blinding runs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import lss_catalogs


def _has(catalog, column):
    return catalog is not None and lss_catalogs.has_column(catalog, column)


def _array(catalog, column):
    return np.asarray(catalog[column])


def _weights(catalog, preferred=('WEIGHT',)):
    if catalog is None:
        return None
    for column in preferred:
        if _has(catalog, column):
            return np.asarray(catalog[column], dtype='f8')
    return np.ones(len(_array(catalog, 'Z')), dtype='f8')


def _finite_z(catalog, zcol):
    if catalog is None or not _has(catalog, zcol):
        return np.array([], dtype='f8')
    z = np.asarray(catalog[zcol], dtype='f8')
    return z[np.isfinite(z)]


def _finite_column(catalog, column):
    if catalog is None or not _has(catalog, column):
        return np.array([], dtype='f8')
    values = np.asarray(catalog[column], dtype='f8')
    return values[np.isfinite(values)]


def _edges(catalogs, *, zcol='Z', zmin=None, zmax=None, dz=None, nbins=80):
    z = np.concatenate([_finite_z(catalog, zcol) for catalog in catalogs if catalog is not None])
    if z.size == 0:
        raise ValueError('Cannot make diagnostic plots: no finite redshift values found')
    if zmin is None:
        zmin = float(np.nanmin(z))
    if zmax is None:
        zmax = float(np.nanmax(z))
    if not np.isfinite(zmin) or not np.isfinite(zmax) or zmax <= zmin:
        raise ValueError(f'Invalid diagnostic redshift range: zmin={zmin}, zmax={zmax}')
    if dz is not None and dz > 0 and zmin is not None and zmax is not None:
        nbin = max(1, int(round((zmax - zmin) / dz)))
        return zmin + float(dz) * np.arange(nbin + 1, dtype='f8')
    return np.linspace(zmin, zmax, nbins + 1)


def _hist(catalog, edges, *, zcol='Z', weight_col='WEIGHT'):
    if catalog is None or not _has(catalog, zcol):
        return None
    weights = _weights(catalog, preferred=(weight_col,))
    return np.histogram(np.asarray(catalog[zcol], dtype='f8'), bins=edges, weights=weights)[0].astype('f8')


def _normalized(hist):
    if hist is None:
        return None
    total = np.sum(hist)
    if total <= 0:
        return np.zeros_like(hist)
    return hist / total


def _matching_metric(data, random, edges, *, zcol='Z', weight_col='WEIGHT'):
    hd = _hist(data, edges, zcol=zcol, weight_col=weight_col)
    hr = _hist(random, edges, zcol=zcol, weight_col=weight_col)
    if hd is None or hr is None:
        return None
    nd = _normalized(hd)
    nr = _normalized(hr)
    return {
        'data_weight_sum': float(np.sum(hd)),
        'random_weight_sum': float(np.sum(hr)),
        'random_to_data_weight_sum': None if np.sum(hd) <= 0 else float(np.sum(hr) / np.sum(hd)),
        'max_abs_normalized_hist_delta': float(np.max(np.abs(nd - nr))),
        'mean_abs_normalized_hist_delta': float(np.mean(np.abs(nd - nr))),
    }


def _plot_hist_step(ax, catalog, edges, *, zcol='Z', label, weight_col='WEIGHT', normalize=False, **kwargs):
    hist = _hist(catalog, edges, zcol=zcol, weight_col=weight_col)
    if hist is None:
        return
    if normalize:
        hist = _normalized(hist)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ax.step(centers, hist, where='mid', label=label, **kwargs)


def _value_hist_bins(catalogs, column, *, nbins=60):
    arrays = [_finite_column(catalog, column) for catalog in catalogs if catalog is not None]
    arrays = [array for array in arrays if array.size]
    if not arrays:
        return None
    values = np.concatenate(arrays)
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return None
    if vmax <= vmin:
        pad = max(abs(vmin) * 1e-6, 1e-6)
        vmin -= pad
        vmax += pad
    return np.linspace(vmin, vmax, nbins + 1, dtype='f8')


def _plot_value_hist(ax, catalog, column, bins, *, label):
    values = _finite_column(catalog, column)
    if values.size == 0:
        return
    # Compare shapes, not absolute row counts: random catalogs intentionally have
    # many more rows than data catalogs.
    weights = np.ones(values.size, dtype='f8') / values.size
    ax.hist(values, bins=bins, weights=weights, histtype='step', label=label)


def _summary_stats(catalog, columns):
    out = {}
    if catalog is None:
        return out
    for column in columns:
        if _has(catalog, column):
            values = np.asarray(catalog[column], dtype='f8')
            finite = values[np.isfinite(values)]
            if finite.size:
                out[column] = {
                    'min': float(np.min(finite)),
                    'max': float(np.max(finite)),
                    'mean': float(np.mean(finite)),
                    'std': float(np.std(finite)),
                }
    return out


def _summary_array(values):
    if values is None:
        return None
    values = np.asarray(values, dtype='f8')
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    qs = np.percentile(values, [1., 5., 50., 95., 99.])
    return {
        'min': float(np.min(values)),
        'p01': float(qs[0]),
        'p05': float(qs[1]),
        'median': float(qs[2]),
        'p95': float(qs[3]),
        'p99': float(qs[4]),
        'max': float(np.max(values)),
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
    }


def _mode_flags(modes):
    mode_set = {str(mode).lower() for mode in (modes or [])}
    return {
        'bao': bool(mode_set & {'bao', 'ap', 'bao/ap', 'bao_ap'}),
        'rsd': 'rsd' in mode_set,
        'fnl': 'fnl' in mode_set,
    }


def write_diagnostic_plots(output_dir, *, input_data, final_data, input_random=None,
                           bao_ap_blinded_data=None, reconstruction_random=None,
                           final_random=None, modes=None, zcol='Z', zmin=None,
                           zmax=None, dz=None, prefix='catalog_blinding',
                           bao_nz_reweight=None, fnl_weight_factor=None):
    """Write diagnostic plots for one real catalog-blinding run.

    Parameters are actual catalog states from the current run.  The plots are
    intended to verify the LSS-like matching steps: randoms matched after BAO/AP
    blinding follow the BAO/AP-blinded data used as reconstruction input, and
    final randoms match the final data used for measurements.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on optional runtime
        raise ImportError('Diagnostic plots require matplotlib') from exc

    output_dir = Path(output_dir).expanduser().resolve(strict=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    modes = list(modes or [])
    flags = _mode_flags(modes)
    # In BAO-only runs final data/randoms are the BAO/AP-blinded data/randoms, so
    # plotting both just duplicates curves and invites confusion.  Keep the final
    # step only when another physical effect (RSD/fNL) makes it distinct, or when
    # there is no BAO/AP intermediate at all.
    show_final_step = final_data is not None and (
        bao_ap_blinded_data is None or flags['rsd'] or flags['fnl']
    )
    show_final_random = final_random is not None and (show_final_step or reconstruction_random is None)

    edges = _edges(
        [input_data, bao_ap_blinded_data, final_data, input_random, reconstruction_random, final_random],
        zcol=zcol, zmin=zmin, zmax=zmax, dz=dz,
    )
    paths = {}

    # Data redshift evolution.
    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_hist_step(ax, input_data, edges, zcol=zcol, label='input data', weight_col='WEIGHT', normalize=True)
    if bao_ap_blinded_data is not None:
        _plot_hist_step(ax, bao_ap_blinded_data, edges, zcol=zcol, label='BAO/AP-blinded data', weight_col='WEIGHT', normalize=True)
    if show_final_step:
        final_label = 'final data'
        if flags['rsd'] and flags['bao']:
            final_label = 'final BAO/AP+RSD data'
        elif flags['rsd']:
            final_label = 'final RSD-blinded data'
        _plot_hist_step(ax, final_data, edges, zcol=zcol, label=final_label, weight_col='WEIGHT', normalize=True)
    ax.set_xlabel(zcol)
    ax.set_ylabel('normalized weighted histogram')
    ax.set_title('Data redshift distribution through blinding steps')
    ax.legend(fontsize=8)
    path = output_dir / f'{prefix}_data_redshift_steps.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    paths['data_redshift_steps'] = str(path)

    # Random matching diagnostics.
    random_panels = []
    if reconstruction_random is not None:
        random_panels.append('bao_ap_blinded')
    if show_final_random:
        random_panels.append('final')
    if random_panels:
        nrows = len(random_panels)
        fig, axes = plt.subplots(nrows, 1, figsize=(8, 4 * nrows), squeeze=False)
        for irow, panel in enumerate(random_panels):
            ax = axes[irow, 0]
            if panel == 'bao_ap_blinded':
                reference = bao_ap_blinded_data if bao_ap_blinded_data is not None else final_data
                _plot_hist_step(ax, reference, edges, zcol=zcol,
                                label='BAO/AP-blinded data', weight_col='WEIGHT', normalize=True)
                _plot_hist_step(ax, reconstruction_random, edges, zcol=zcol,
                                label='randoms matched to BAO/AP-blinded data', weight_col='WEIGHT', normalize=True)
                ax.set_title('Random matching for BAO/AP-blinded input')
            else:
                final_label = 'final data'
                random_label = 'final randoms matched from final data'
                if flags['rsd'] and flags['bao']:
                    final_label = 'final BAO/AP+RSD data'
                    random_label = 'final randoms matched from BAO/AP+RSD data'
                _plot_hist_step(ax, final_data, edges, zcol=zcol, label=final_label,
                                weight_col='WEIGHT', normalize=True)
                _plot_hist_step(ax, final_random, edges, zcol=zcol, label=random_label,
                                weight_col='WEIGHT', normalize=True)
                ax.set_title('Final random matching')
            ax.set_xlabel(zcol)
            ax.set_ylabel('normalized weighted histogram')
            ax.legend(fontsize=8)
        path = output_dir / f'{prefix}_random_matching.png'
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        paths['random_matching'] = str(path)

    # Weight diagnostics.
    weight_columns = [col for col in ['WEIGHT_SYS', 'WEIGHT_COMP', 'WEIGHT_ZFAIL', 'WEIGHT', 'WEIGHT_FKP']
                      if _has(final_data, col) or _has(final_random, col) or _has(bao_ap_blinded_data, col)]
    bao_nz_correction = None
    if bao_nz_reweight is not None and bao_nz_reweight.get('correction') is not None:
        bao_nz_correction = np.asarray(bao_nz_reweight['correction'], dtype='f8')
        bao_nz_correction = bao_nz_correction[np.isfinite(bao_nz_correction)]
    fnl_weight_factor = None if fnl_weight_factor is None else np.asarray(fnl_weight_factor, dtype='f8')
    if fnl_weight_factor is not None:
        fnl_weight_factor = fnl_weight_factor[np.isfinite(fnl_weight_factor)]
    n_weight_panels = (len(weight_columns)
                       + (1 if bao_nz_correction is not None and bao_nz_correction.size else 0)
                       + (1 if fnl_weight_factor is not None and fnl_weight_factor.size else 0))
    if n_weight_panels:
        ncols = min(2, n_weight_panels)
        nrows = int(np.ceil(n_weight_panels / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3.5 * nrows), squeeze=False)
        plot_catalogs = [input_data, bao_ap_blinded_data]
        if show_final_step:
            plot_catalogs.append(final_data)
        if show_final_random:
            plot_catalogs.append(final_random)
        elif final_random is not None and not show_final_step:
            # BAO-only: the final random is the BAO/AP-matched random, so show it
            # once with an unambiguous label.
            plot_catalogs.append(final_random)
        for iax, column in enumerate(weight_columns):
            ax = axes.flat[iax]
            bins = _value_hist_bins(plot_catalogs, column, nbins=60)
            if bins is None:
                ax.axis('off')
                continue
            _plot_value_hist(ax, input_data, column, bins, label=f'input data {column}')
            if _has(bao_ap_blinded_data, column):
                _plot_value_hist(ax, bao_ap_blinded_data, column, bins, label=f'BAO/AP-blinded data {column}')
            if show_final_step and _has(final_data, column):
                _plot_value_hist(ax, final_data, column, bins, label=f'final data {column}')
            if final_random is not None and _has(final_random, column):
                random_label = 'final random'
                if not show_final_step:
                    random_label = 'randoms matched to BAO/AP-blinded data'
                _plot_value_hist(ax, final_random, column, bins, label=f'{random_label} {column}')
            ax.set_xlabel(column)
            ax.set_ylabel('fraction of rows')
            ax.legend(fontsize=7)
        next_panel = len(weight_columns)
        if bao_nz_correction is not None and bao_nz_correction.size:
            ax = axes.flat[next_panel]
            bins = np.linspace(float(np.nanmin(bao_nz_correction)), float(np.nanmax(bao_nz_correction)), 61)
            if bins[0] == bins[-1]:
                bins = np.linspace(bins[0] - 0.5, bins[0] + 0.5, 61)
            ax.hist(bao_nz_correction, bins=bins, histtype='step', density=True, label='internal BAO n(z) factor')
            ax.set_xlabel('internal BAO n(z) factor')
            ax.set_ylabel('density')
            ax.legend(fontsize=7)
            next_panel += 1
        if fnl_weight_factor is not None and fnl_weight_factor.size:
            ax = axes.flat[next_panel]
            bins = np.linspace(float(np.nanmin(fnl_weight_factor)), float(np.nanmax(fnl_weight_factor)), 61)
            if bins[0] == bins[-1]:
                bins = np.linspace(bins[0] - 0.5, bins[0] + 0.5, 61)
            ax.hist(fnl_weight_factor, bins=bins, histtype='step', density=True, label='internal fNL weight factor')
            ax.set_xlabel('internal fNL weight factor')
            ax.set_ylabel('density')
            ax.legend(fontsize=7)
            next_panel += 1
        for ax in axes.flat[next_panel:]:
            ax.axis('off')
        path = output_dir / f'{prefix}_weight_diagnostics.png'
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        paths['weight_diagnostics'] = str(path)

    summary = {
        'modes': modes,
        'zcol': zcol,
        'z_edges': [float(edges[0]), float(edges[-1])],
        'n_z_bins': int(len(edges) - 1),
        'plots': paths,
        'plotted_steps': {
            'final_step': bool(show_final_step),
            'final_random': bool(show_final_random),
            'weight_histograms_normalized': True,
            'bao_nz_factor_internal': bool(bao_nz_correction is not None and bao_nz_correction.size),
            'fnl_weight_factor_internal': bool(fnl_weight_factor is not None and fnl_weight_factor.size),
        },
        'row_counts': {
            'input_data': int(len(input_data)) if input_data is not None else None,
            'bao_ap_blinded_data': int(len(bao_ap_blinded_data)) if bao_ap_blinded_data is not None else None,
            'final_data': int(len(final_data)) if final_data is not None else None,
            'input_random': int(len(input_random)) if input_random is not None else None,
            'reconstruction_random': int(len(reconstruction_random)) if reconstruction_random is not None else None,
            'final_random': int(len(final_random)) if final_random is not None else None,
        },
        'matching': {
            'bao_ap_blinded_random_match': _matching_metric(
                bao_ap_blinded_data if bao_ap_blinded_data is not None else final_data,
                reconstruction_random, edges, zcol=zcol,
            ) if reconstruction_random is not None else None,
            'final': _matching_metric(final_data, final_random, edges, zcol=zcol) if show_final_random else None,
        },
        'weights': {
            'input_data': _summary_stats(input_data, weight_columns),
            'bao_ap_blinded_data': _summary_stats(bao_ap_blinded_data, weight_columns),
            'final_data': _summary_stats(final_data, weight_columns),
            'final_random': _summary_stats(final_random, weight_columns),
            'internal_bao_nz_factor': _summary_array(bao_nz_correction),
            'internal_fnl_weight_factor': _summary_array(fnl_weight_factor),
        },
    }
    summary_path = output_dir / f'{prefix}_diagnostics.json'
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    summary['summary_file'] = str(summary_path)
    return summary
