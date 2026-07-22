#!/usr/bin/env python
"""
Test the impact on the power spectrum shot noise of angular (imaging systematic) weights
applied to both data and randoms.

First step: compute the power spectrum of the DA2 fNL ELG catalogs with imaging systematic
weights WEIGHT_IMLIN and WEIGHT_IMLIN_DES applied to the data; WEIGHT_IMLIN_DES is also
reassigned to the randoms via TARGETID_DATA match with the data, and folded into the
randoms WEIGHT.

Run with e.g.:
```bash
salloc -N 1 -C "gpu&hbm80g" -t 04:00:00 --gpus 4 --qos interactive --account desi_g
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
srun -n 4 python test_angular_shotnoise.py
```
"""
import os
import logging
from pathlib import Path

import numpy as np
from mpi4py import MPI

from clustering_statistics import tools
from clustering_statistics.tools import renormalize_randoms_over_data, Catalog, desi_dir, select_region, default_mpicomm, setup_logging
from clustering_statistics.spectrum2_tools import compute_mesh2_spectrum
from clustering_statistics.spectrum3_tools import compute_mesh3_spectrum


logger = logging.getLogger('angular_shotnoise')


cat_dir = desi_dir / 'survey/catalogs/DA2/LSS/loa-v1/LSScats/v2/fNL'
stats_dir = Path(os.getenv('SCRATCH', '.')) / 'measurements' / 'angular_shotnoise3'


def get_catalog_fn(kind='data', tracer='ELG_LOPnotqso', region='NGC', iran=0):
    """Return fNL clustering catalog filename."""
    if kind == 'data':
        return cat_dir / f'{tracer}_{region}_clustering.dat.fits'
    return cat_dir / f'{tracer}_{region}_{iran:d}_clustering.ran.fits'


@default_mpicomm
def compute_spectrum_imlin(nran=4, zrange=(1.1, 1.6), tracer='ELG_LOPnotqso', region=None,
                           imweights=['WEIGHT_IMLIN'], refweight='WEIGHT_SYS', renorm_imweight=False, norm_regions=('N', 'S'), renorm_randoms=True,
                           randomize=None, mattrs=None, mpicomm=None):
    r"""
    Compute the power spectrum of the fNL ELG catalogs with imaging systematic weights.

    Data individual weight is WEIGHT times the ``data_weights`` columns.
    The ``randoms_weights`` columns are reassigned from the data to the randoms via
    TARGETID_DATA match, and folded into the randoms WEIGHT.

    If ``randomize``, the ratio data[imweight] / data['WEIGHT_SYS'] is randomized
    (breaking its correlation with the imaging systematics) before entering data
    INDWEIGHT, and propagated to the randoms INDWEIGHT through the TARGETID_DATA match.
    Randomization is performed in each region (NGC / SGC) separately; the same random
    draws (seed 42) are used for all ``imweights``. Options are:

    - 'isotropic': reshuffle the ratio among the data
    - 'angular': bin the ratio in healpix pixels (nside=256, nest=True, mean per pixel),
      reshuffle the values of the non-empty pixels, and use the (reshuffled) pixel value
      instead of the per-object ratio
    - 'isotropic_norm': same as 'isotropic', but draw per-object values from a Gaussian
      with the mean and std of the ratio
    - 'angular_norm': same as 'angular', but draw pixel values from a Gaussian with the
      mean and std of the non-empty pixel values

    Returns
    -------
    spectra : list of dict
        One dict per ``imweight``, with keys 'mesh2_spectrum' (:class:`Mesh2SpectrumPoles`)
        and 'mesh3_spectrum' (:class:`Mesh3SpectrumPoles`).
    """
    if mattrs is None: mattrs = dict(meshsize=960, cellsize=7.5)

    data = randoms = None
    regions = ['NGC', 'SGC']
    renorm_imweight_regions = ['N', 'SnoDES', 'DES']

    def match_data_column(randoms, data):
        sorted_index = np.argsort(data['TARGETID'])
        index_in_sorted = np.searchsorted(data['TARGETID'][sorted_index], randoms['TARGETID_DATA'])
        index = sorted_index[np.clip(index_in_sorted, 0, len(sorted_index) - 1)]
        return index

    if mpicomm.rank == 0:  # faster to read and process catalogs from one rank
        data = tools._read_catalog([get_catalog_fn(kind='data', tracer=tracer, region=_region) for _region in regions], mpicomm=MPI.COMM_SELF)
        randoms = tools._read_catalog([get_catalog_fn(kind='randoms', tracer=tracer, region=_region, iran=iran) for _region in regions for iran in range(nran)], mpicomm=MPI.COMM_SELF)
        for catalog in [data, randoms]:
            catalog['WEIGHT_ONE'] = catalog.ones()

        randoms['INDEX'] = match_data_column(randoms, data)

        if randomize:
            choices = ['constant', 'isotropic', 'angular', 'isotropic_norm', 'angular_norm']
            if randomize not in choices:
                raise ValueError(f"randomize must be one of {choices}, got {randomize!r}")
            region_masks = [tools.select_region(data['RA'], data['DEC'], region=_region) for _region in regions]
            if 'angular' in randomize:
                import healpy as hp
                data_pix = hp.ang2pix(256, data['RA'], data['DEC'], nest=True, lonlat=True)
            # Per-region draws, generated once so all imweights share the same randomization
            rng = np.random.default_rng(seed=42)
            draws = []
            for mask in region_masks:
                size = np.unique(data_pix[mask]).size if 'angular' in randomize else int(mask.sum())
                draws.append(rng.standard_normal(size) if 'norm' in randomize else rng.permutation(size))

            def randomize_ratio(ratio):
                randomized = np.empty_like(ratio)
                if randomize == 'constant':
                    for renorm_imweight_region in renorm_imweight_regions:
                        mask = select_region(data['RA'], data['DEC'], region=renorm_imweight_region)
                        randomized[mask] = ratio[mask].mean()
                    return randomized
                for mask, draw in zip(region_masks, draws):
                    values = ratio[mask]
                    if 'angular' in randomize:
                        # mean ratio in each non-empty healpix pixel
                        _, inverse = np.unique(data_pix[mask], return_inverse=True)
                        values = np.bincount(inverse, weights=values) / np.bincount(inverse)
                    if 'norm' in randomize:
                        values = values.mean() + values.std() * draw
                    else:
                        values = values[draw]
                    if 'angular' in randomize:
                        values = values[inverse]
                    randomized[mask] = values
                return randomized

    spectra = []
    for imweight in imweights:
        if mpicomm.rank == 0:
            ratio = data[imweight] / data[refweight]
            print(imweight, refweight, ratio, renorm_imweight)
            if randomize:
                ratio = randomize_ratio(ratio)
            if renorm_imweight:
                mean_ratios = []
                for renorm_imweight_region in renorm_imweight_regions:
                    mask = select_region(data['RA'], data['DEC'], region=renorm_imweight_region)
                    #mask &= (data['Z'] > 0.8) & (data['Z'] < 1.6)
                    mean_ratios.append(ratio[mask].mean())
                mean_ratios = np.array(mean_ratios) / np.mean(mean_ratios)
                print(f'Mean imweight ratios {mean_ratios}')
                for renorm_imweight_region, mean_ratio in zip(renorm_imweight_regions, mean_ratios):
                    mask = select_region(data['RA'], data['DEC'], region=renorm_imweight_region)
                    ratio[mask] /= mean_ratio
            data['INDWEIGHT'] = data['WEIGHT'] * ratio * data['WEIGHT_FKP']
            randoms['INDWEIGHT'] = randoms['WEIGHT'] * ratio[randoms['INDEX']] * randoms['WEIGHT_FKP']

        catalogs = {'data': data.deepcopy() if data is not None else None,
                    'randoms': randoms.deepcopy() if randoms is not None else None}
        if mpicomm.size > 1:
            catalogs['data'] = Catalog.scatter(data, mpicomm=mpicomm, mpiroot=0)
            catalogs['randoms'] = Catalog.scatter(randoms, mpicomm=mpicomm, mpiroot=0)
        for kind, catalog in catalogs.items():
            catalogs[kind] = tools.mask_catalog(catalog, kind, zrange=zrange)
        #if renorm_randoms:
        #    renormalize_randoms_over_data(catalogs['randoms'], catalogs['data'], regions=norm_regions)
        for kind, catalog in catalogs.items():
            catalogs[kind] = tools.mask_catalog(catalog, kind, zrange=zrange, region=region)
        for kind, catalog in catalogs.items():
            catalog = tools.set_positions_from_rdz(catalog)
            catalogs[kind] = catalog[['POSITION', 'RA', 'DEC', 'INDWEIGHT', 'TARGETID']]
            for column in catalogs[kind]:
                catalogs[kind][column] = catalogs[kind][column].astype(int if column == 'TARGETID' else float)

        if True:
            sum_data_weights, sum_randoms_weights = [], []
            for _region in norm_regions:
                mask_data = select_region(catalogs['data']['RA'], catalogs['data']['DEC'], region=_region)
                mask_randoms = select_region(catalogs['randoms']['RA'], catalogs['randoms']['DEC'], region=_region)
                sum_data_weights.append(catalogs['data']['INDWEIGHT'][mask_data].csum())
                sum_randoms_weights.append(catalogs['randoms']['INDWEIGHT'][mask_randoms].csum())
            sum_data_weights, sum_randoms_weights = np.array(sum_data_weights), np.array(sum_randoms_weights)
            alphas = sum_data_weights / sum_randoms_weights / (sum(sum_data_weights) / sum(sum_randoms_weights))
            print('ALPHAS', norm_regions, alphas, catalogs['data'].csize, catalogs['randoms'].csize)

        spectra.append({'mesh2_spectrum': compute_mesh2_spectrum(lambda: catalogs, mattrs=mattrs)['raw'],
                        'mesh3_spectrum': compute_mesh3_spectrum(lambda: catalogs, mattrs=mattrs)['raw']})

    return spectra


if __name__ == '__main__':
    os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.9')
    import jax
    jax.config.update('jax_enable_x64', True)
    try: jax.distributed.initialize()
    except RuntimeError: pass
    setup_logging()

    mpicomm = MPI.COMM_WORLD
    tracer = 'ELG_LOPnotqso'
    #tracer = 'QSO'
    imweights = ['WEIGHT_SYS', 'WEIGHT_IMLIN', 'WEIGHT_IMLIN_DES', 'WEIGHT_IMLIN_DES_NORM_GLOBAL', 'WEIGHT_ONE'][:3]
    renorm_imweight = True #True
    renorm_randoms = False #False
    for randomize in [None, 'constant', 'isotropic', 'angular', 'isotropic_norm', 'angular_norm'][:1]:
        for region in ['NGC', 'SGC']:
            for norm_regions in [['N', 'S'], ['N', 'SnoDES', 'DES']][1:]:
                spectra = compute_spectrum_imlin(imweights=imweights, region=region, tracer=tracer, norm_regions=norm_regions, randomize=randomize, renorm_imweight=renorm_imweight, renorm_randoms=renorm_randoms, mpicomm=mpicomm)
                norm_regions = ''.join(norm_regions)
                randomize_label = f'_randomize-{randomize}' if randomize else ''
                renorm_imweight_label = '_renorm-imweight' if renorm_imweight else ''
                renorm_randoms_label = '' if renorm_randoms else '_no-renorm-ran'
                for spectrum, imweight in zip(spectra, imweights):
                    for stat, value in spectrum.items():
                        print(stat, value.value().sum())
                        tools.write_stats(stats_dir / f'{stat}_{tracer}_{region}_{imweight}_norm{norm_regions}{randomize_label}{renorm_imweight_label}{renorm_randoms_label}.h5', value, mpicomm=mpicomm)
