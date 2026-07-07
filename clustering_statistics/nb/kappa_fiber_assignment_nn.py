"""
copied from https://github.com/cosmodesi/desi-clustering/blob/desilike-jax/clustering_statistics/nb/kappa_fiber_assignment.py
Run on the GPU, with
```
srun -n 1 python kappa_fiber_assignment.py
```
(single process only!)
"""
from pathlib import Path
import os
import numpy as np
import healpy as hp
import jax
from jax import config
from scipy.spatial import cKDTree

from lsstypes import ObservableTree

from mpytools import Catalog
from clustering_statistics import tools
from clustering_statistics.tools import setup_logging, _compute_binned_weight
from clustering_statistics.correlation2_tools import prepare_cucount_particles


dirname = Path('./kappa_fiber_assignment/')
dirname.mkdir(exist_ok=True)


def get_output_fn(basename, imock):
    return dirname / f'{basename}_{imock:04d}.h5'


def get_nearest_neighbor_weight(ra, dec, mask_assigned):
    """
    Compute nearest-neighbor upweights.

    Parameters
    ----------
    ra, dec : array
        Sky coordinates in degrees.
    mask_assigned : bool array
        True for galaxies assigned a fiber.

    Returns
    -------
    weight : array
        Nearest-neighbor weights. Starts at one; each unassigned galaxy
        increments the weight of its nearest assigned neighbor by one.
    """
    weight = np.ones(len(ra), dtype=float)

    if np.all(mask_assigned):
        return weight

    assigned = np.flatnonzero(mask_assigned)
    unassigned = np.flatnonzero(~mask_assigned)

    ra_rad = np.deg2rad(ra)
    dec_rad = np.deg2rad(dec)

    xyz = np.column_stack([np.cos(dec_rad) * np.cos(ra_rad), np.cos(dec_rad) * np.sin(ra_rad), np.sin(dec_rad)])

    tree = cKDTree(xyz[assigned])
    _, index = tree.query(xyz[unassigned], k=1)

    np.add.at(weight, assigned[index], 1.)

    return weight



def compute_auw(imock, FKP_P0=4e3, zrange=(1.1, 1.6)):
    mock_dir = Path(
        '/dvs_ro/cfs/cdirs/desi/mocks/cai/GLAM-Uchuu/lightcones/lensing/')
    fn = mock_dir / f'{imock:04d}/maps/kappa_CMB_Born.fits'
    tracer = 'ELG_LOPnotqso'
    weight = 'default-FKP'
    kappamap = Catalog({'INDWEIGHT': 1 + hp.read_map(fn)})
    nside = hp.npix2nside(kappamap.csize)
    pix = np.arange(kappamap.csize)
    theta, phi = hp.pix2ang(nside, pix, nest=False)
    kappamap['RA'] = np.degrees(phi)
    kappamap['DEC'] = 90.0 - np.degrees(theta)

    kw_catalog = dict(version='glam-uchuu-v2-altmtl', tracer=tracer, weight=weight,
                      region='NGC', nran=2, keep_columns=True, imock=imock, FKP_P0=FKP_P0)
    expand = {'parent_randoms_fn': tools.get_catalog_fn(
        kind='parent_randoms', version='data-dr2-v2', tracer=kw_catalog['tracer'], nran=kw_catalog['nran'])}
    data = tools.prepare_catalog(tools.read_catalog(
        kind='data', **kw_catalog), kind='data', zrange=zrange, **kw_catalog)
    randoms = tools.prepare_catalog(tools.read_catalog(
        kind='randoms', expand=expand, **kw_catalog), kind='randoms', zrange=zrange, **kw_catalog)

    # going back to full data for ELG and LRG, to get the full set of TILELOCIDs for computing new FRACZ_TILELOCID
    raw_full_data = tools.read_catalog(kind='full_data', **kw_catalog)
    # not used
    binned_weight = {}
    binned_weight['weight_ntile'] = {column: _compute_binned_weight(data[column], data['INDWEIGHT'] / data['WEIGHT_COMP'], mpicomm=data.mpicomm) for column in ['NTILE']}
    fibered_data = tools.prepare_catalog(raw_full_data, kind='fibered_data', **kw_catalog, binned_weight=binned_weight)
    parent_data = tools.prepare_catalog(raw_full_data, kind='parent_data', **kw_catalog, binned_weight=binned_weight)
    raw_randoms = tools.read_catalog(kind='randoms', expand=expand, **kw_catalog)
    fibered_randoms = tools.prepare_catalog(raw_randoms, kind='fibered_randoms', **kw_catalog, binned_weight=binned_weight)
    parent_randoms = tools.prepare_catalog(raw_randoms, kind='parent_randoms', **kw_catalog, binned_weight=binned_weight)

    complete, reshuffle = {}, {}
    complete_data = tools.prepare_catalog(tools.read_catalog(
        kind='data', complete=complete, **kw_catalog), kind='data', zrange=zrange, **kw_catalog)
    complete_randoms = tools.prepare_catalog(tools.read_catalog(
        kind='randoms', expand=expand, complete=complete, reshuffle=reshuffle, **kw_catalog), kind='randoms', zrange=zrange, **kw_catalog)

    weight = get_nearest_neighbor_weight(raw_full_data['RA'], raw_full_data['DEC'], raw_full_data['ZWARN'] != 999999)

    _, index, index_full = np.intersect1d(fibered_data['TARGETID'], raw_full_data['TARGETID'], return_indices=True)
    fibered_data['INDWEIGHT'][index] = weight[index_full]
    parent_data['INDWEIGHT'][:] = 1.
    
    _, index, index_full = np.intersect1d(data['TARGETID'], raw_full_data['TARGETID'], return_indices=True)
    data['INDWEIGHT'][index] = weight[index_full]
    randoms['INDWEIGHT'] = np.ones(len(randoms))
    complete_data['INDWEIGHT'] = np.ones(len(complete_data))
    complete_randoms['INDWEIGHT'] = np.ones(len(complete_randoms))
    # tools.renormalize_randoms_over_data(
    #    fibered_randoms, fibered_data, tracer=tracer)
    # tools.renormalize_randoms_over_data(
    #    parent_randoms, parent_data, tracer=tracer)
    # tools.renormalize_randoms_over_data(randoms, data, tracer=tracer)
    # tools.renormalize_randoms_over_data(
    #    complete_randoms, complete_data, tracer=tracer)

    def copy(catalog):
        catalog = catalog[['RA', 'DEC', 'INDWEIGHT']]
        for name in catalog:
            catalog[name] = np.array(catalog[name], dtype='f8')
        return catalog

    from cucount.jax import BinAttrs, SelectionAttrs, WeightAttrs, get_sharding_mesh
    from cucount.types import count2

    def get_counts(*get_data, battrs=None, norm=None):
        all_particles = prepare_cucount_particles(
            *get_data, positions_type='rd')
        all_particles = [particles['data'] for particles in all_particles]
        if battrs is None:
            battrs = {'theta': np.arange(0.001, 0.5, 0.005)}
        battrs = BinAttrs(**battrs)
        return count2(*all_particles, battrs=battrs, norm=norm)['weight']

    result = {}
    # fibered_data, parent_data, complete_data, data, complete_randoms, randoms, kappamap = [copy(
    #    catalog) for catalog in [fibered_data, parent_data, complete_data, data, complete_randoms, randoms, kappamap]]
    complete_data, data, complete_randoms, randoms, kappamap = [copy(
        catalog) for catalog in [complete_data, data, complete_randoms, randoms, kappamap]]

    # result['GfGf'] = get_counts(lambda: {'data': fibered_data})
    result['KK'] = get_counts(lambda: {'data': kappamap})
    result['GpGp'] = get_counts(lambda: {'data': parent_data})
    result['GpK'] = get_counts(
        lambda: {'data': parent_data}, lambda: {'data': kappamap})
    result['GfK'] = get_counts(
        lambda: {'data': fibered_data}, lambda: {'data': kappamap})
    result['GcK'] = get_counts(
        lambda: {'data': complete_data}, lambda: {'data': kappamap})
    result['GK'] = get_counts(
        lambda: {'data': data}, lambda: {'data': kappamap})
    result['GcGc'] = get_counts(lambda: {'data': complete_data}, lambda: {
                                'data': complete_data})
    result['GG'] = get_counts(lambda: {'data': data}, lambda: {'data': data})
    result['RcRc'] = get_counts(lambda: {'data': complete_randoms}, lambda: {
                                'data': complete_randoms})
    result['RR'] = get_counts(
        lambda: {'data': randoms}, lambda: {'data': randoms})
    result['RcK'] = get_counts(
        lambda: {'data': complete_randoms}, lambda: {'data': kappamap})
    result['RK'] = get_counts(
        lambda: {'data': randoms}, lambda: {'data': kappamap})
    result['RcK1'] = get_counts(lambda: {'data': complete_randoms}, lambda: {'data': kappamap.clone(INDWEIGHT=kappamap.ones())})
    result['RK1'] = get_counts(lambda: {'data': randoms}, lambda: {'data': kappamap.clone(INDWEIGHT=kappamap.ones())})
    result['K1K1'] = get_counts(lambda: {'data': kappamap.clone(INDWEIGHT=kappamap.ones())})
    result = ObservableTree(list(result.values()), pairs=list(result.keys()))
    result.write(get_output_fn('all_counts_nn', imock=imock))


if __name__ == '__main__':

    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'
    config.update('jax_enable_x64', True)

    try:
        jax.distributed.initialize()
    except RuntimeError:
        print('Distributed environment already initialized')
    else:
        print('Initializing distributed environment')

    setup_logging()

    imock = 150
    # compute_auw(imock, zrange=(0.8, 1.1))
    compute_auw(imock)
    # compute_auw(imock, weight='default')
