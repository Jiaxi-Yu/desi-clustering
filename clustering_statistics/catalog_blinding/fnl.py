"""Catalog-level fNL blinding adapter.

``desiblind`` owns the fNL blinding physics. This module is the
``desi-clustering`` saved-catalog adapter layer: it normalizes CLI parameters,
passes catalog data/randoms to ``desiblind.catalog_fnl.CatalogFNLBlinder``, and
returns the internal fNL weight factor for diagnostics.
"""

from __future__ import annotations

import numpy as np


def _get_desiblind_fnl_blinder():
    try:
        from desiblind.catalog_fnl import CatalogFNLBlinder
    except ImportError as exc:  # pragma: no cover - depends on optional checkout
        raise ImportError(
            'catalog-level fNL blinding requires desiblind with desiblind.catalog_fnl.CatalogFNLBlinder'
        ) from exc
    return CatalogFNLBlinder


def _drop_none(parameters):
    return {key: value for key, value in dict(parameters or {}).items() if value is not None}


def normalize_parameters(parameters, *, tracer_name=None):
    """Normalize fNL parameters with LSS tracer defaults when needed."""
    blinder = _get_desiblind_fnl_blinder()
    return blinder._normalize_parameters(_drop_none(parameters), tracer=tracer_name)


def apply_blinding(tracer_name, catalog, randoms, *, parameters, weight_col='WEIGHT',
                   random_weight_col='WEIGHT', output_weight_col=None,
                   racol='RA', deccol='DEC', zcol='Z',
                   update_weight_comp=True, copy=True, **kwargs):
    """Apply ``desiblind.catalog_fnl.CatalogFNLBlinder`` to one data catalog.

    Returns
    -------
    blinded : catalog-like
        Catalog with fNL folded into ``WEIGHT`` and, by default, ``WEIGHT_COMP``.
    normalized : dict
        Normalized fNL parameter dictionary.
    weight_factor : array
        Internal fNL factor ``new_WEIGHT / old_WEIGHT``. This is intended for
        diagnostics and is not written as a final catalog column.
    """
    blinder = _get_desiblind_fnl_blinder()
    normalized = normalize_parameters(parameters, tracer_name=tracer_name)
    blinded, weight_factor = blinder.apply_blinding(
        tracer_name, catalog, randoms, parameters=normalized,
        weight_col=weight_col, random_weight_col=random_weight_col,
        output_weight_col=output_weight_col,
        racol=racol, deccol=deccol, zcol=zcol,
        update_weight_comp=update_weight_comp, return_weight_factor=True,
        copy=copy, **kwargs,
    )
    return blinded, normalized, np.asarray(weight_factor, dtype='f8')
