"""Small LSS-like catalog helpers for catalog-level blinding.

This module ports the narrow catalog-production behavior needed by the
``desi-clustering`` blinding drivers without importing ``LSS``.  It is inspired
by ``LSS.main.cattools.mkclusran`` / ``clusran_resamp`` and
``LSS.common_tools.mknz`` / ``addnbar``:

* random redshift-dependent columns are sampled from the current data catalog;
* random angular/footprint columns are preserved;
* sampling can be split by columns such as ``PHOTSYS``;
* BAO/AP data can receive an internal ``n(z)_in / n(z)_out`` weight factor;
* final catalogs can get simple ``NZ``/``NX``/``WEIGHT``/``WEIGHT_FKP`` updates.

The blinding transforms themselves remain in ``desiblind``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


DEFAULT_REDSHIFT_COLUMNS = (
    'Z', 'WEIGHT', 'WEIGHT_SYS', 'WEIGHT_COMP', 'WEIGHT_ZFAIL',
    'WEIGHT_FKP', 'TARGETID_DATA', 'WEIGHT_SN',
)
DEFAULT_SPLIT_COLUMNS = ('PHOTSYS',)
LSS_NGC_DEC_SPLIT = 'LSS_NGC_DEC'
LSS_NGC_DEC_THRESHOLD = 32.375
DEFAULT_ANGULAR_COLUMNS = (
    'RA', 'DEC', 'TARGETID', 'TILEID', 'NTILE', 'PHOTSYS', 'FRAC_TLOBS_TILES',
)


def column_names(catalog):
    """Return catalog column names for Astropy tables, mockfactory catalogs, or dicts."""
    if hasattr(catalog, 'colnames'):
        return list(catalog.colnames)
    return list(catalog.keys())


def has_column(catalog, name):
    return name in column_names(catalog)


def copy_catalog(catalog):
    return catalog.copy() if hasattr(catalog, 'copy') else dict(catalog)


def _length(catalog):
    names = column_names(catalog)
    if not names:
        return 0
    return len(catalog[names[0]])


def _source_column(data, column):
    if has_column(data, column):
        return np.asarray(data[column])
    if column == 'TARGETID_DATA' and has_column(data, 'TARGETID'):
        return np.asarray(data['TARGETID'])
    return None


def _assign_column(catalog, column, values):
    catalog[column] = np.asarray(values)


def split_columns_for_region(region, tracer=None, split_columns=DEFAULT_SPLIT_COLUMNS):
    """Return LSS-like random-resampling split keys for a sky region."""
    region = str(region or '').upper()
    # LSS main/cattools.py splits NGC random redshift resampling at DEC=32.375
    # rather than by PHOTSYS.  This preserves the NGC north/south imaging-depth
    # dependence in the joint angular/radial selection.
    if region in {'N', 'NGC', 'NGCNON'}:
        return (LSS_NGC_DEC_SPLIT,)
    return tuple(split_columns or ())


def _valid_split_columns(data, random, split_columns):
    valid = []
    for col in tuple(split_columns or ()):
        if col == LSS_NGC_DEC_SPLIT:
            if has_column(data, 'DEC') and has_column(random, 'DEC'):
                valid.append(col)
        elif has_column(data, col) and has_column(random, col):
            valid.append(col)
    return tuple(valid)


def _split_values(data, random, col):
    if col == LSS_NGC_DEC_SPLIT:
        return (True, False)
    return tuple(np.unique(np.concatenate([np.asarray(data[col]), np.asarray(random[col])])))


def _apply_split_value(catalog, col, value):
    if col == LSS_NGC_DEC_SPLIT:
        mask = np.asarray(catalog['DEC'], dtype='f8') > LSS_NGC_DEC_THRESHOLD
        return mask if value else ~mask
    return np.asarray(catalog[col]) == value


def _group_masks(data, random, split_columns):
    """Yield matched (data_mask, random_mask) pairs for LSS-like split keys."""
    ndata = _length(data)
    nrandom = _length(random)
    if not split_columns:
        yield np.ones(ndata, dtype=bool), np.ones(nrandom, dtype=bool)
        return
    keys = [_split_values(data, random, col) for col in split_columns]
    import itertools
    for values in itertools.product(*keys):
        dmask = np.ones(ndata, dtype=bool)
        rmask = np.ones(nrandom, dtype=bool)
        for col, value in zip(split_columns, values):
            dmask &= _apply_split_value(data, col, value)
            rmask &= _apply_split_value(random, col, value)
        if np.any(rmask):
            if not np.any(dmask):
                raise ValueError(f'Cannot resample randoms for split {dict(zip(split_columns, values))}: no matching data rows')
            yield dmask, rmask


def resample_randoms_from_data(randoms, data, *, columns=DEFAULT_REDSHIFT_COLUMNS,
                               split_columns=DEFAULT_SPLIT_COLUMNS, seed=0,
                               compmd='ran', preserve_weight_normalization=True,
                               copy=True):
    """Return random catalogs whose redshift-dependent columns are sampled from data.

    This mirrors the core LSS ``mkclusran`` / ``clusran_resamp`` behavior used by
    the blinding scripts: random angular/footprint columns are kept, while
    columns such as ``Z`` and weights are sampled from the current data catalog.
    For NGC, pass ``split_columns_for_region('NGC')`` to reproduce the LSS
    DEC=32.375 north/south split; otherwise the default split is by PHOTSYS when
    available.
    """
    if isinstance(randoms, (list, tuple)):
        return [resample_randoms_from_data(
            random, data, columns=columns, split_columns=split_columns,
            seed=int(seed) + iran, compmd=compmd,
            preserve_weight_normalization=preserve_weight_normalization,
            copy=copy,
        ) for iran, random in enumerate(randoms)]

    rng = np.random.default_rng(seed=int(seed))
    out = copy_catalog(randoms) if copy else randoms
    active_columns = [col for col in columns if _source_column(data, col) is not None]
    active_split_columns = _valid_split_columns(data, out, split_columns)

    # Preserve random angular columns by copying the random catalog first, then
    # overwrite only the LSS redshift-dependent columns sampled from data.
    sampled_indices = np.empty(_length(out), dtype='i8')
    sampled_indices.fill(-1)
    split_ratios = []
    split_masks = []
    for dmask, rmask in _group_masks(data, out, active_split_columns):
        nran = int(np.sum(rmask))
        local_data_indices = np.flatnonzero(dmask)
        chosen = rng.choice(local_data_indices, size=nran, replace=True)
        sampled_indices[rmask] = chosen

    if np.any(sampled_indices < 0):  # pragma: no cover - defensive
        raise RuntimeError('internal error: not all random rows received sampled data rows')

    for col in active_columns:
        source = _source_column(data, col)
        _assign_column(out, col, source[sampled_indices])

    if compmd == 'ran' and has_column(out, 'WEIGHT') and has_column(out, 'FRAC_TLOBS_TILES'):
        out['WEIGHT'] = np.asarray(out['WEIGHT'], dtype='f8') * np.asarray(out['FRAC_TLOBS_TILES'], dtype='f8')

    if preserve_weight_normalization and has_column(out, 'WEIGHT') and has_column(data, 'WEIGHT'):
        for dmask, rmask in _group_masks(data, out, active_split_columns):
            dsum = np.sum(np.asarray(data['WEIGHT'], dtype='f8')[dmask])
            rsum = np.sum(np.asarray(out['WEIGHT'], dtype='f8')[rmask])
            if dsum > 0 and rsum > 0:
                split_ratios.append(rsum / dsum)
                split_masks.append(rmask)
        if len(split_ratios) > 1:
            reference = split_ratios[0]
            weights = np.asarray(out['WEIGHT'], dtype='f8').copy()
            for ratio, rmask in zip(split_ratios[1:], split_masks[1:]):
                if ratio > 0:
                    weights[rmask] *= reference / ratio
            out['WEIGHT'] = weights

    if hasattr(out, 'attrs'):
        out.attrs['catalog_blinding_random_resampling'] = 'lss-like-data-resampling'
        out.attrs['catalog_blinding_random_resampling_seed'] = int(seed)
        out.attrs['catalog_blinding_random_resampling_columns'] = ','.join(active_columns)
        out.attrs['catalog_blinding_random_resampling_split_columns'] = ','.join(active_split_columns)
    return out


def _default_weight_columns(catalog):
    cols = column_names(catalog)
    weight_cols = [col for col in ['WEIGHT_COMP', 'WEIGHT_SYS', 'WEIGHT_ZFAIL'] if col in cols]
    if weight_cols:
        weight = np.ones(_length(catalog), dtype='f8')
        for col in weight_cols:
            weight *= np.asarray(catalog[col], dtype='f8')
        return weight
    if 'WEIGHT' in cols:
        return np.asarray(catalog['WEIGHT'], dtype='f8')
    return np.ones(_length(catalog), dtype='f8')


def nz_histogram(catalog, *, zcol='Z', zmin=None, zmax=None, dz=0.01, weights=None):
    """Return LSS-style redshift histogram arrays for a catalog."""
    z = np.asarray(catalog[zcol], dtype='f8')
    if zmin is None:
        zmin = float(np.nanmin(z))
    if zmax is None:
        zmax = float(np.nanmax(z))
    nbin = max(1, int(round((zmax - zmin) / dz)))
    edges = float(zmin) + float(dz) * np.arange(nbin + 1, dtype='f8')
    if weights is None:
        weights = _default_weight_columns(catalog)
    hist, edges = np.histogram(z, bins=edges, weights=weights)
    return hist.astype('f8'), edges


def apply_bao_nz_reweight(data_before, data_after, *, zcol_before='Z', zcol_after='Z',
                          zmin=None, zmax=None, dz=0.01, copy=True):
    """Return the BAO/AP ``n(z)_in / n(z)_out`` correction as internal state.

    Older LSS BAO-blinding scripts folded this correction into ``WEIGHT_SYS``.
    For DR3-style workflows we keep ``WEIGHT_SYS`` as the imaging/systematics
    weight so it can be restored/replaced later by ``TARGETID`` matching.  The
    returned per-row ``correction`` is folded into the final clustering
    ``WEIGHT`` internally by :func:`set_lss_pre_addnbar_weight` and
    :func:`add_nbar_fkp`, but no extra blinding-specific catalog column is
    created or written. Rows outside the configured redshift range keep unit
    correction.
    """
    out = copy_catalog(data_after) if copy else data_after
    nz_in, edges = nz_histogram(data_before, zcol=zcol_before, zmin=zmin, zmax=zmax, dz=dz)
    nz_out, _ = nz_histogram(data_after, zcol=zcol_after, zmin=edges[0], zmax=edges[-1], dz=edges[1] - edges[0])
    ratio = np.ones_like(nz_out, dtype='f8')
    valid = nz_out > 0
    ratio[valid] = nz_in[valid] / nz_out[valid]
    z = np.asarray(out[zcol_after], dtype='f8')
    idx = np.floor((z - edges[0]) / (edges[1] - edges[0])).astype('i8')
    in_range = (idx >= 0) & (idx < len(ratio))
    correction = np.ones(len(z), dtype='f8')
    correction[in_range] = ratio[idx[in_range]]

    if hasattr(out, 'attrs'):
        out.attrs['catalog_blinding_bao_nz_reweight'] = True
        out.attrs['catalog_blinding_bao_nz_reweight_internal'] = True
        out.attrs['catalog_blinding_bao_nz_reweight_dz'] = float(edges[1] - edges[0])
    return out, {'nz_in': nz_in, 'nz_out': nz_out, 'ratio': ratio, 'correction': correction, 'edges': edges}


def _simple_tracer_name(tracer):
    text = str(tracer)
    if 'BGS' in text:
        return 'BGS'
    if 'LRG' in text:
        return 'LRG'
    if 'LGE' in text:
        return 'LGE'
    if 'ELG' in text:
        return 'ELG'
    if 'QSO' in text:
        return 'QSO'
    return text


def fiducial_nbar_zrange(tracer, zrange=None):
    """Return the LSS tracer-wide z-range used for nbar/FKP construction."""
    simple = _simple_tracer_name(tracer)
    if simple == 'BGS':
        return (0.1, 0.4)
    if simple in {'LRG', 'LGE'}:
        return (0.4, 1.1)
    if simple == 'ELG':
        return (0.8, 1.6)
    if simple == 'QSO':
        return (0.8, 3.5)
    if zrange is not None:
        return tuple(map(float, zrange))
    return None


def fiducial_fkp_p0(tracer, default=10000.):
    """Return the LSS FKP P0 used by tracer."""
    simple = _simple_tracer_name(tracer)
    return {'BGS': 7e3, 'LRG': 1e4, 'LGE': 1e4, 'ELG': 4e3, 'QSO': 6e3}.get(simple, float(default))


def _ntile_index(catalog):
    ntile = np.asarray(catalog['NTILE'], dtype='i8') if has_column(catalog, 'NTILE') else np.ones(_length(catalog), dtype='i8')
    # LSS NTILE is positive and indexed as NTILE-1.  Be defensive for toy tests.
    idx = ntile - 1 if ntile.size == 0 or np.nanmin(ntile) >= 1 else ntile
    return np.clip(idx.astype('i8'), 0, None)


def _ntile_mean(ntile_index, values):
    ntile_index = np.asarray(ntile_index, dtype='i8')
    values = np.asarray(values, dtype='f8')
    minlength = int(np.max(ntile_index)) + 1 if ntile_index.size else 1
    denom = np.bincount(ntile_index, minlength=minlength)
    num = np.bincount(ntile_index, weights=values, minlength=minlength)
    out = np.ones(minlength, dtype='f8')
    ok = denom > 0
    out[ok] = num[ok] / denom[ok]
    return out


def _component_weight(catalog, extra_weight=None):
    if all(has_column(catalog, col) for col in ['WEIGHT_COMP', 'WEIGHT_SYS', 'WEIGHT_ZFAIL']):
        weight = (np.asarray(catalog['WEIGHT_COMP'], dtype='f8') *
                  np.asarray(catalog['WEIGHT_SYS'], dtype='f8') *
                  np.asarray(catalog['WEIGHT_ZFAIL'], dtype='f8'))
        if extra_weight is not None:
            weight = weight * np.asarray(extra_weight, dtype='f8')
        return weight
    if has_column(catalog, 'WEIGHT'):
        return np.asarray(catalog['WEIGHT'], dtype='f8').copy()
    return np.ones(_length(catalog), dtype='f8')


def set_lss_pre_addnbar_weight(catalog, *, extra_weight=None, copy=True):
    """Set pre-addnbar clustering ``WEIGHT`` as LSS ``mkclusdat`` does.

    Existing DR1 clustering catalogs already contain post-addnbar ``WEIGHT``
    values divided by per-NTILE completeness.  For a faithful on-the-fly BAO
    workflow we need the intermediate LSS state used by ``mkclusran`` before
    ``addnbar``: ``WEIGHT_COMP * WEIGHT_SYS * WEIGHT_ZFAIL`` times
    any explicit internal blinding weight factors and optional ``WEIGHT_BLIND``,
    with no per-NTILE division.  Randoms sample this temporary
    value, and ``add_nbar_fkp`` applies the final ntile division exactly once.
    """
    out = copy_catalog(catalog) if copy else catalog
    wt = _component_weight(out, extra_weight=extra_weight)
    if has_column(out, 'WEIGHT_BLIND'):
        wt = wt * np.asarray(out['WEIGHT_BLIND'], dtype='f8')
    out['WEIGHT'] = wt
    return out


def _first_random(randoms):
    if isinstance(randoms, (list, tuple)):
        return randoms[0] if randoms else None
    return randoms


def _survey_area_from_random(random, *, randens=2500., compmd='ran'):
    if random is None:
        raise ValueError('A random catalog is required to compute LSS-like nbar area')
    if compmd == 'ran' and has_column(random, 'FRAC_TLOBS_TILES'):
        return float(np.sum(np.asarray(random['FRAC_TLOBS_TILES'], dtype='f8')) / float(randens))
    return float(_length(random) / float(randens))


def _nbar_from_data_and_random_area(data, random, *, zcol='Z', zmin=None, zmax=None,
                                    dz=0.01, randens=2500., compmd='ran', data_extra_weight=None):
    """LSS ``mknz`` equivalent: weighted counts divided by shell volume."""
    if zmin is None or zmax is None:
        z = np.asarray(data[zcol], dtype='f8')
        finite = z[np.isfinite(z)]
        if finite.size == 0:
            raise ValueError('Cannot compute nbar: no finite data redshifts')
        if zmin is None:
            zmin = float(np.min(finite))
        if zmax is None:
            zmax = float(np.max(finite))
    zmin, zmax, dz = float(zmin), float(zmax), float(dz)
    nbin = int((zmax - zmin) * (1. + dz / 10.) / dz)
    nbin = max(nbin, 1)
    area = _survey_area_from_random(random, randens=randens, compmd=compmd)
    wts = _component_weight(data, extra_weight=data_extra_weight)
    hist, edges = np.histogram(np.asarray(data[zcol], dtype='f8'), bins=nbin, range=(zmin, zmax), weights=wts)
    from cosmoprimo.fiducial import TabulatedDESI
    cosmo = TabulatedDESI()
    distance = cosmo.comoving_radial_distance(edges)
    volume = area / (360. * 360. / np.pi) * 4. * np.pi / 3. * np.diff(distance ** 3.)
    nbar = np.zeros_like(hist, dtype='f8')
    ok = volume > 0
    nbar[ok] = hist[ok] / volume[ok]
    return nbar, edges, {'area': area, 'volume': volume, 'weighted_counts': hist}


def add_nbar_fkp(data, randoms=None, *, zcol='Z', zmin=None, zmax=None, dz=0.01,
                 p0=10000., compmd='ran', randens=2500., data_extra_weight=None, copy=True):
    """Add LSS-like ``NZ``/``NX``/``WEIGHT``/``WEIGHT_FKP`` columns.

    This ports the core behavior of LSS ``mknz`` + ``addnbar`` for clustering
    catalogs.  ``NZ`` is a true number density estimated from weighted data
    counts divided by comoving shell volume using the effective random area.
    Data ``WEIGHT`` is rebuilt from component weights, any internal
    blinding-weight factor such as BAO n(z), and per-NTILE completeness.  Random ``WEIGHT`` preserves the relative normalization already
    introduced by ``mkclusran``/``clusran_resamp`` via ``wtfac = old_WEIGHT / wt``
    before the final per-NTILE division, exactly as in LSS ``addnbar``.
    """
    first_random = _first_random(randoms)
    out_data = copy_catalog(data) if copy else data
    nbar, edges, nbar_info = _nbar_from_data_and_random_area(
        out_data, first_random, zcol=zcol, zmin=zmin, zmax=zmax, dz=dz,
        randens=randens, compmd=compmd,
    )
    dz_eff = edges[1] - edges[0]

    data_ntile = _ntile_index(out_data)
    if has_column(out_data, 'WEIGHT_COMP'):
        weight_ntile = _ntile_mean(data_ntile, np.asarray(out_data['WEIGHT_COMP'], dtype='f8'))
    else:
        weight_ntile = np.ones(int(np.max(data_ntile)) + 1 if data_ntile.size else 1, dtype='f8')
    comp_ntile = np.ones_like(weight_ntile)
    ok = weight_ntile > 0
    comp_ntile[ok] = 1. / weight_ntile[ok]

    if compmd == 'ran' and first_random is not None and has_column(first_random, 'NTILE') and has_column(first_random, 'FRAC_TLOBS_TILES'):
        random_ntile = _ntile_index(first_random)
        ftile_ntile = _ntile_mean(random_ntile, np.asarray(first_random['FRAC_TLOBS_TILES'], dtype='f8'))
        if len(ftile_ntile) > len(comp_ntile):
            comp_ntile = np.pad(comp_ntile, (0, len(ftile_ntile) - len(comp_ntile)), constant_values=1.)
            weight_ntile = np.pad(weight_ntile, (0, len(ftile_ntile) - len(weight_ntile)), constant_values=1.)
        comp_ntile[:len(ftile_ntile)] *= ftile_ntile
    else:
        ftile_ntile = np.ones_like(comp_ntile)

    def _nbar_at_z(cat):
        z = np.asarray(cat[zcol], dtype='f8')
        idx = ((z - edges[0]) / dz_eff).astype('i8')
        valid = (z > edges[0]) & (z < edges[-1]) & (idx >= 0) & (idx < len(nbar))
        values = np.zeros(len(z), dtype='f8')
        values[valid] = nbar[idx[valid]]
        return values

    def _ensure_ntile_arrays(idx):
        nonlocal comp_ntile, weight_ntile
        needed = int(np.max(idx)) + 1 if idx.size else 1
        if needed > len(comp_ntile):
            comp_ntile = np.pad(comp_ntile, (0, needed - len(comp_ntile)), constant_values=1.)
        if needed > len(weight_ntile):
            weight_ntile = np.pad(weight_ntile, (0, needed - len(weight_ntile)), constant_values=1.)

    def _apply_data(cat):
        idx = _ntile_index(cat)
        _ensure_ntile_arrays(idx)
        nz = _nbar_at_z(cat)
        cat['NZ'] = nz
        cat['NX'] = nz * comp_ntile[idx]
        wt = _component_weight(cat, extra_weight=data_extra_weight)
        if has_column(cat, 'WEIGHT_BLIND'):
            wt = wt * np.asarray(cat['WEIGHT_BLIND'], dtype='f8')
        cat['WEIGHT'] = wt / weight_ntile[idx]
        cat['WEIGHT_FKP'] = 1. / (1. + np.asarray(cat['NX'], dtype='f8') * float(p0))
        return cat

    def _apply_random(cat):
        old_weight = np.asarray(cat['WEIGHT'], dtype='f8').copy() if has_column(cat, 'WEIGHT') else np.ones(_length(cat), dtype='f8')
        idx = _ntile_index(cat)
        _ensure_ntile_arrays(idx)
        nz = _nbar_at_z(cat)
        cat['NZ'] = nz
        cat['NX'] = nz * comp_ntile[idx]
        wt = _component_weight(cat)
        if compmd == 'ran' and has_column(cat, 'FRAC_TLOBS_TILES'):
            wt = wt * np.asarray(cat['FRAC_TLOBS_TILES'], dtype='f8')
        if has_column(cat, 'WEIGHT_BLIND'):
            wt = wt * np.asarray(cat['WEIGHT_BLIND'], dtype='f8')
        wtfac = np.ones(_length(cat), dtype='f8')
        sel = wt > 0
        wtfac[sel] = old_weight[sel] / wt[sel]
        cat['WEIGHT'] = wtfac * wt / weight_ntile[idx]
        cat['WEIGHT_FKP'] = 1. / (1. + np.asarray(cat['NX'], dtype='f8') * float(p0))
        return cat

    out_data = _apply_data(out_data)
    info = {
        'nz': nbar,
        'edges': edges,
        'area': nbar_info['area'],
        'volume': nbar_info['volume'],
        'weighted_counts': nbar_info['weighted_counts'],
        'weight_ntile': weight_ntile,
        'comp_ntile': comp_ntile,
        'ftile_ntile': ftile_ntile,
        'p0': float(p0),
    }
    if randoms is None:
        return out_data, None, info
    if isinstance(randoms, (list, tuple)):
        out_randoms = [_apply_random(copy_catalog(random) if copy else random) for random in randoms]
    else:
        out_randoms = _apply_random(copy_catalog(randoms) if copy else randoms)
    return out_data, out_randoms, info


def read_fits_catalog(filename, ext='LSS'):
    """Read a FITS catalog as an Astropy table."""
    import fitsio
    from astropy.table import Table

    filename = Path(filename).expanduser().resolve(strict=True)
    try:
        data = fitsio.read(str(filename), ext=ext)
    except Exception:
        data = fitsio.read(str(filename))
    return Table(data)


def write_fits_catalog(catalog, filename, ext='LSS', clobber=False):
    """Write a catalog-like object to a FITS file."""
    import fitsio

    filename = Path(filename).expanduser().resolve(strict=False)
    filename.parent.mkdir(parents=True, exist_ok=True)
    if filename.exists() and not clobber:
        raise FileExistsError(f'{filename} exists; pass --clobber to overwrite')
    fitsio.write(str(filename), np.asarray(catalog), extname=ext, clobber=True)
    return filename
