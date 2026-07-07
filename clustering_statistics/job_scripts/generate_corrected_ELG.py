"""
salloc -N 1 -C cpu -t 02:00:00 --qos interactive
srun -n 10 python generate_corrected_ELG.py
"""

from pathlib import Path
import os
import numpy as np
import healpy as hp
from scipy.spatial import cKDTree

from lsstypes import ObservableTree

from mpytools import Catalog
from clustering_statistics import tools
from clustering_statistics.tools import setup_logging, _compute_binned_weight, select_region


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


def get_fracz_tilelocid(tilelocid, mask_assigned):
    full_locid, inv, full_nlocid = np.unique(tilelocid, return_inverse=True, return_counts=True)
    fibered_locid, fibered_nlocid = np.unique(tilelocid[mask_assigned], return_counts=True)
    idx = np.searchsorted(full_locid, fibered_locid)
    full_nfibered = np.zeros_like(full_nlocid, dtype=float)
    full_nfibered[idx] = fibered_nlocid
    frac_per_locid = full_nfibered / full_nlocid
    return frac_per_locid[inv]


def generate_ELG_catalogs(weight='tilelocid-LRG1', out_dir=None, **catalog):
    from mpi4py import MPI

    get_catalog_fn = catalog.pop('get_catalog_fn', tools.get_catalog_fn)

    kw_catalog = dict(catalog)
    kw_catalog['region'] = 'ALL'
    kw_catalog['tracer'] = 'ELG_LOPnotqso'
    kw_catalog['keep_columns'] = True
    nran = kw_catalog.pop('nran')
    all_iran = range(nran)

    catalog_regions = ['NGC', 'SGC']
    renormalization_regions = ['N', 'S']
    with_weight_ntile = False
    with_ftile_ntile = False
    with_frac_tlobs_tiles = False

    expand = {'parent_randoms_fn': tools.get_catalog_fn(kind='parent_randoms', version='data-dr2-v2', tracer=kw_catalog['tracer'], nran=nran), 'from_data': ['Z']}

    mpicomm = MPI.COMM_SELF
    data = tools.read_catalog(kind='data', get_catalog_fn=get_catalog_fn, mpicomm=mpicomm, **kw_catalog)
    assert data.mpicomm.size == 1
    # this is for randoms
    data_wtotp = data['WEIGHT_COMP'] * data['WEIGHT_SYS'] * data['WEIGHT_ZFAIL']
    data_wcomp_ntile = {}
    for region in catalog_regions:
        mask_region_data = select_region(data['RA'], data['DEC'], region)
        data_wcomp_ntile[region] = _compute_binned_weight(data['NTILE'][mask_region_data], data_wtotp[mask_region_data] / data['WEIGHT'][mask_region_data])

    raw_full_data_ELG = tools.read_catalog(kind='full_data', get_catalog_fn=get_catalog_fn, mpicomm=mpicomm, **kw_catalog)
    full_data = raw_full_data_ELG

    fibered_data_ELG = raw_full_data_ELG['ZWARN'] != 999999

    if 'tilelocid-LRG' in weight:
        raw_full_data_LRG = tools.read_catalog(kind='full_data', get_catalog_fn=get_catalog_fn, mpicomm=mpicomm, **(kw_catalog | dict(tracer='LRG')))
        notfibered_data_LRG = raw_full_data_LRG['ZWARN'] == 999999
    
        mask_LRG_around_ELG = np.isin(raw_full_data_LRG['TILELOCID'][notfibered_data_LRG], raw_full_data_ELG['TILELOCID'][fibered_data_ELG])
    
        columns = ['TARGETID', 'ZWARN', 'FRACZ_TILELOCID', 'TILELOCID']
        LRG_around_ELG = raw_full_data_LRG[columns][notfibered_data_LRG][mask_LRG_around_ELG]
        nrepeats = 1
        if 'LRG19' in weight:
            nrepeats = 19
        if 'LRG0' in weight:
            nrepeats = 0
        ELG_with_LRG = Catalog.concatenate([raw_full_data_ELG[columns]] + [LRG_around_ELG] * nrepeats)
        ELG_with_LRG['FRACZ_TILELOCID'] = get_fracz_tilelocid(ELG_with_LRG['TILELOCID'], ELG_with_LRG['ZWARN'] != 999999)
    
        _, index, index_ELG_with_LRG = np.intersect1d(full_data['TARGETID'], ELG_with_LRG['TARGETID'], return_indices=True)
        full_data['FRACZ_TILELOCID'][index] = ELG_with_LRG['FRACZ_TILELOCID'][index_ELG_with_LRG]
    
        _, index, index_ELG_with_LRG = np.intersect1d(data['TARGETID'], ELG_with_LRG['TARGETID'], return_indices=True)
        data['FRACZ_TILELOCID'] = data.zeros()
        data['FRACZ_TILELOCID'][index] = ELG_with_LRG['FRACZ_TILELOCID'][index_ELG_with_LRG]
        data['WEIGHT_COMP'] = 1. / data['FRACZ_TILELOCID']
        if not with_frac_tlobs_tiles:
            data['WEIGHT_COMP'] /= data['FRAC_TLOBS_TILES']
        else:
            data['FRAC_TLOBS_TILES'][:] = 1.
    elif 'nn' in weight:
        invweight = 1. / get_nearest_neighbor_weight(full_data['RA'], full_data['DEC'], full_data['ZWARN'] != 999999)
        full_data['FRACZ_TILELOCID'] = invweight
        full_data['FRAC_TLOBS_TILES'][:] = 1.
        _, index, index_full = np.intersect1d(data['TARGETID'], full_data['TARGETID'], return_indices=True)
        data['FRACZ_TILELOCID'] = data.zeros()
        data['FRACZ_TILELOCID'][index] = invweight[index_full]
        data['WEIGHT_COMP'] = 1. / data['FRACZ_TILELOCID']
        data['FRAC_TLOBS_TILES'][:] = 1.

    data['WEIGHT'] = data['WEIGHT_COMP'] * data['WEIGHT_SYS'] * data['WEIGHT_ZFAIL']

    sorted_index = np.argsort(data['TARGETID'])

    def read_randoms(iran):
        return tools.read_catalog(kind='randoms', expand=expand, get_catalog_fn=get_catalog_fn, mpicomm=mpicomm, **kw_catalog, concatenate=False, nran=[iran])[0]

    def get_randoms_index(randoms):
        index_in_sorted = np.searchsorted(data['TARGETID'], randoms['TARGETID_DATA'], sorter=sorted_index)
        index = sorted_index[index_in_sorted]
        return index

    def add_randoms_base_weights(randoms):
        index = get_randoms_index(randoms)
        randoms['FRAC_TLOBS_TILES'] = randoms.ones()

        if with_frac_tlobs_tiles:
            for region in catalog_regions:
                mask_region_data = select_region(data['RA'], data['DEC'], region)
                mask_region_randoms = select_region(randoms['RA'], randoms['DEC'], region)
                randoms['FRAC_TLOBS_TILES'][mask_region_randoms] = randoms['WEIGHT'][mask_region_randoms] / data_wtotp[index[mask_region_randoms]] * data_wcomp_ntile[region][randoms['NTILE'][mask_region_randoms]]
                #randoms['FRAC_TLOBS_TILES'][mask_region_randoms] *= data['FRAC_TLOBS_TILES'][mask_region_data].mean() /randoms['FRAC_TLOBS_TILES'][mask_region_randoms].mean()  # renormalize, as above estimate is up to a renormalization

        # Transfer data weight to randoms
        for name in ['WEIGHT_SYS', 'WEIGHT_COMP', 'WEIGHT_ZFAIL']:
            randoms[name] = data[name][index]

        randoms['WEIGHT'] = randoms['FRAC_TLOBS_TILES'] * randoms['WEIGHT_COMP'] * randoms['WEIGHT_SYS'] * randoms['WEIGHT_ZFAIL']
        return randoms

    P0 = 4e3
    nz = {region: np.loadtxt(get_catalog_fn(kind='nz', **(kw_catalog | dict(region=region))), unpack=True) for region in catalog_regions}

    for name in ['NZ', 'NX']:
        data[name] = data.zeros()

    weight_ntile, comp_ntile = {}, {}

    randoms0 = add_randoms_base_weights(read_randoms(0))

    # Loop on region NGC, SGC
    for region in nz:
        nzregion = nz[region]
        zedges = np.insert(nzregion[2], 0, nzregion[1][0])

        def get_nz(z):
            idx = np.digitize(z, zedges, right=False) - 1
            mask = (idx >= 0) & (idx < nzregion[3].size)
            tmpnz = np.zeros_like(z, dtype=data['WEIGHT_COMP'].dtype)
            tmpnz[mask] = nzregion[3][idx[mask]]
            return tmpnz

        mask_region_data = select_region(data['RA'], data['DEC'], region)
        data_ntile = data['NTILE'][mask_region_data]
        weight_ntile[region] = _compute_binned_weight(data_ntile, data['WEIGHT_COMP'][mask_region_data])
        if not with_weight_ntile:
            weight_ntile[region][:] = 1.

        mask_region_randoms0 = select_region(randoms0['RA'], randoms0['DEC'], region)
        randoms0_ntile = randoms0['NTILE'][mask_region_randoms0]
        ftile_ntile = _compute_binned_weight(randoms0_ntile, randoms0['FRAC_TLOBS_TILES'][mask_region_randoms0])
        comp_ntile[region] = ftile_ntile / weight_ntile[region]
        if not with_ftile_ntile:
            comp_ntile[region][:] = 1.

        tmpnz = get_nz(data['Z'][mask_region_data])
        data['NZ'][mask_region_data] = tmpnz
        data['NX'][mask_region_data] = comp_ntile[region][data_ntile] * tmpnz
        data['WEIGHT'][mask_region_data] /= weight_ntile[region][data_ntile]

    del randoms0

    data['WEIGHT_FKP'] = 1. / (1. + P0 * data['NX'])

    sum_data_weights = []
    for region in renormalization_regions:
        mask_data = select_region(data['RA'], data['DEC'], region=region)
        sum_data_weights.append(data['WEIGHT'][mask_data].csum())
    sum_data_weights = np.array(sum_data_weights)

    fn = {region: get_catalog_fn(cat_dir=out_dir, kind='data', **(kw_catalog | dict(region=region))) for region in catalog_regions}
    for region in catalog_regions:
        mask_data = select_region(data['RA'], data['DEC'], region=region)
        data[mask_data].write(fn[region])

    fn = {region: get_catalog_fn(cat_dir=out_dir, kind='full_data', **(kw_catalog | dict(region=region))) for region in catalog_regions}
    for region in catalog_regions:
        mask_data = select_region(full_data['RA'], full_data['DEC'], region=region)
        full_data[mask_data].write(fn[region])

    # Iterate on randoms; reassign WEIGHT* from data Z, NZ, NX, renormalize weights
    for iran in all_iran:
        randoms = add_randoms_base_weights(read_randoms(iran))

        for name in ['NZ', 'NX']:
            randoms[name] = randoms.zeros()

        for region in nz:
            nzregion = nz[region]
            zedges = np.insert(nzregion[2], 0, nzregion[1][0])

            def get_nz(z):
                idx = np.digitize(z, zedges, right=False) - 1
                mask = (idx >= 0) & (idx < nzregion[3].size)
                tmpnz = np.zeros_like(z, dtype=data['WEIGHT_COMP'].dtype)
                tmpnz[mask] = nzregion[3][idx[mask]]
                return tmpnz

            mask_region_randoms = select_region(randoms['RA'], randoms['DEC'], region)
            randoms_ntile = randoms['NTILE'][mask_region_randoms]
            tmpnz = get_nz(randoms['Z'][mask_region_randoms])
            randoms['NZ'][mask_region_randoms] = tmpnz
            randoms['NX'][mask_region_randoms] = comp_ntile[region][randoms_ntile] * tmpnz
            randoms['WEIGHT'][mask_region_randoms] /= weight_ntile[region][randoms_ntile]

        randoms['WEIGHT_FKP'] = 1. / (1. + P0 * randoms['NX'])

        sum_randoms_weights = []
        for region in renormalization_regions:
            mask_randoms = select_region(randoms['RA'], randoms['DEC'], region=region)
            sum_randoms_weights.append(randoms['WEIGHT'][mask_randoms].csum())
        sum_randoms_weights = np.array(sum_randoms_weights)

        global_alpha = sum(sum_data_weights) / sum(sum_randoms_weights)
        alphas = sum_data_weights / sum_randoms_weights / global_alpha

        for region, alpha in zip(renormalization_regions, alphas):
            mask_randoms = select_region(randoms['RA'], randoms['DEC'], region=region)
            randoms['WEIGHT'][mask_randoms] *= alpha

        for region in catalog_regions:
            mask_randoms = select_region(randoms['RA'], randoms['DEC'], region=region)
            fn = get_catalog_fn(cat_dir=out_dir, kind='randoms', **(kw_catalog | dict(region=region, nran=[iran])))[0]
            randoms[mask_randoms].write(fn)

        del randoms


if __name__ == '__main__':

    setup_logging()
    from mpi4py import MPI

    mpicomm = MPI.COMM_WORLD

    #version = 'abacus-hf-dr2-v2-altmtl'
    #imocks = list(range(2))
    version = 'glam-uchuu-v2-altmtl'
    imocks = list(range(150, 152))

    weight = 'nn'
    #weight = 'tilelocid-LRG0'
    #weight = 'tilelocid-LRG1'
    #weight = 'tilelocid-LRG19'
    out_dir = tools.base_stats_dir / f'auxiliary_data/fiber_assignment_systematics_ELG_{weight}' / version

    local_imocks = imocks[mpicomm.rank::mpicomm.size]
    for imock in local_imocks:
        generate_ELG_catalogs(version=version, weight=weight, out_dir=out_dir / f'mock{imock:d}', imock=imock, nran=5, get_catalog_fn=tools.get_catalog_fn)

    mpicomm.Barrier()