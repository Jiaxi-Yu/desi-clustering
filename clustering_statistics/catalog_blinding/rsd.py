"""RSD catalog-level blinding adapters and realspace reconstruction helpers.

The RSD redshift transform itself lives in ``desiblind.catalog_rsd``. This
module handles the ``desi-clustering`` side: parameter normalization, direct
application to catalog objects, and preparing reconstructed-realspace catalogs
with pyrecon or the JAX-native reconstruction path.
"""

from __future__ import annotations

import numpy as np

def _get_desiblind_rsd_blinder():
    try:
        from desiblind.catalog_rsd import CatalogRSDBlinder
    except ImportError as exc:
        raise ImportError('RSD catalog blinding requires desiblind with desiblind.catalog_rsd') from exc
    return CatalogRSDBlinder


def normalize_bao_parameters(parameters):
    parameters = dict(parameters or {})
    missing = [name for name in ['w0', 'wa'] if parameters.get(name) is None]
    if missing:
        raise ValueError(f'BAO/AP catalog blinding requires parameters: {missing}')
    return {'w0': float(parameters['w0']), 'wa': float(parameters['wa'])}


def normalize_rsd_parameters(parameters):
    """Normalize RSD parameters, deriving fgrowth_blind when needed."""
    # desiblind treats the presence of ``fgrowth_blind`` as an explicit
    # validation override, so do not forward optional CLI keys whose value is
    # None.  This keeps the production BAO+RSD path deriving f_blind from
    # w0/wa/zeff/bias instead of trying to cast None to float.
    parameters = {key: value for key, value in dict(parameters or {}).items() if value is not None}
    rsd = _get_desiblind_rsd_blinder()
    if parameters.get('fgrowth_blind') is not None:
        # desiblind treats explicit fgrowth_blind as a validation/testing or
        # externally prepared override; keep the rest for provenance.
        return rsd._normalize_parameters(parameters)
    required = ['w0', 'wa', 'zeff', 'bias', 'fiducial_f']
    missing = [name for name in required if parameters.get(name) is None]
    if missing:
        raise ValueError(f'RSD catalog blinding requires parameters {missing} or an explicit fgrowth_blind')
    return rsd._normalize_parameters(parameters)






def apply_blinding(tracer_name, catalog, realspace_catalog, *, parameters,
                   zcol='Z', realspace_zcol='Z', output_zcol='Z', copy=True):
    """Apply ``desiblind.catalog_rsd.CatalogRSDBlinder`` to one catalog."""
    rsd = _get_desiblind_rsd_blinder()
    params = normalize_rsd_parameters(parameters)
    out = rsd.apply_blinding(
        tracer_name, catalog, realspace_catalog, parameters=params,
        zcol=zcol, realspace_zcol=realspace_zcol, output_zcol=output_zcol,
        copy=copy,
    )
    return out, params

def _column_names(catalog):
    if hasattr(catalog, 'colnames'):
        return list(catalog.colnames)
    return list(catalog.keys())


def _copy_catalog_columns(catalog, columns=None):
    names = _column_names(catalog) if columns is None else columns
    return {name: np.asarray(catalog[name]) for name in names if name in _column_names(catalog)}


def _is_catalog_list(catalog):
    return isinstance(catalog, (list, tuple))


def to_reconstruction_catalog(catalog, weight_col='WEIGHT'):
    """Return a minimal mockfactory catalog for RSD reconstruction.

    If ``catalog`` is a list/tuple, stack the per-random reconstruction catalogs
    in memory.  This mirrors LSS ``run_reconstruction``, which vstack's the list
    of random FITS files before assigning randoms to pyrecon.
    """
    from cosmoprimo.fiducial import TabulatedDESI
    from mockfactory import Catalog, sky_to_cartesian

    if _is_catalog_list(catalog):
        catalogs = [to_reconstruction_catalog(cat, weight_col=weight_col) for cat in catalog]
        if not catalogs:
            raise ValueError('At least one random catalog is required for reconstruction')
        names = _column_names(catalogs[0])
        stacked = {name: np.concatenate([np.asarray(cat[name]) for cat in catalogs], axis=0) for name in names}
        return Catalog(stacked)

    names = _column_names(catalog)
    required = ['RA', 'DEC', 'Z']
    missing = [name for name in required if name not in names]
    if missing:
        raise ValueError(f'JAX reconstruction requires columns {missing}')
    data = {name: np.asarray(catalog[name]) for name in required}
    out = Catalog(data)
    distance = TabulatedDESI().comoving_radial_distance(out['Z'])
    out['POSITION'] = sky_to_cartesian(distance, out['RA'], out['DEC'], dtype=distance.dtype)
    if weight_col in names:
        out['INDWEIGHT'] = np.asarray(catalog[weight_col], dtype='f8')
    else:
        out['INDWEIGHT'] = np.ones(len(out['Z']), dtype='f8')
    return out


def positions_to_realspace_catalog(template_catalog, positions, zcol='Z'):
    """Return a catalog copy whose RA/DEC/Z come from reconstructed positions."""
    from cosmoprimo.fiducial import TabulatedDESI
    from cosmoprimo.utils import DistanceToRedshift
    from mockfactory import cartesian_to_sky

    positions = np.asarray(positions)
    distance, ra, dec = cartesian_to_sky(positions)
    realspace = template_catalog.copy()
    realspace['RA'] = ra
    realspace['DEC'] = dec
    realspace[zcol] = DistanceToRedshift(TabulatedDESI().comoving_radial_distance)(distance)
    return realspace


def compute_data_boxcenter(catalog, weight_col='WEIGHT'):
    """Return the pyrecon-style data-position bounding-box center."""
    data_rec = to_reconstruction_catalog(catalog, weight_col=weight_col)
    positions = np.asarray(data_rec['POSITION'])
    return (0.5 * (positions.min(axis=0) + positions.max(axis=0))).tolist()


def compute_jaxrecon_realspace_catalog(data_catalog, random_catalog, *, bias, smoothing_radius=15.,
                                       growth_rate=None, threshold_randoms=('mean', 0.01),
                                       mattrs=None, weight_col='WEIGHT', zcol='Z'):
    """Compute an RSD realspace catalog with the JAX-native desi-clustering path."""
    from ..recon_tools import compute_rsd_realspace_positions

    data_rec = to_reconstruction_catalog(data_catalog, weight_col=weight_col)
    random_rec = to_reconstruction_catalog(random_catalog, weight_col=weight_col)
    positions = compute_rsd_realspace_positions(
        lambda: {'data': data_rec, 'randoms': random_rec},
        mattrs=mattrs, bias=bias, smoothing_radius=smoothing_radius,
        growth_rate=growth_rate,
        threshold_randoms=threshold_randoms,
    )
    return positions_to_realspace_catalog(data_catalog, positions, zcol=zcol)


def compute_pyrecon_realspace_catalog(data_catalog, random_catalog, *, bias, growth_rate=0.8,
                                      smoothing_radius=15., boxsize=None, boxcenter=None, nmesh=None,
                                      cellsize=7., threshold_randoms=0.01,
                                      nthreads=64, method='iterative_fft',
                                      weight_col='WEIGHT', zcol='Z'):
    """Compute an RSD realspace catalog with direct pyrecon, without importing LSS."""
    from ..recon_tools import compute_pyrecon_rsd_realspace_positions

    data_rec = to_reconstruction_catalog(data_catalog, weight_col=weight_col)
    random_rec = to_reconstruction_catalog(random_catalog, weight_col=weight_col)
    positions = compute_pyrecon_rsd_realspace_positions(
        data_rec['POSITION'], random_rec['POSITION'],
        data_weights=data_rec['INDWEIGHT'], randoms_weights=random_rec['INDWEIGHT'],
        bias=bias, growth_rate=growth_rate, boxsize=boxsize, boxcenter=boxcenter, nmesh=nmesh,
        cellsize=cellsize, smoothing_radius=smoothing_radius,
        threshold_randoms=threshold_randoms, nthreads=nthreads, method=method,
    )
    return positions_to_realspace_catalog(data_catalog, positions, zcol=zcol)
