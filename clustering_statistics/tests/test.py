"""
salloc -N 1 -C "gpu&hbm80g" -t 02:00:00 --gpus 4 --qos interactive --account desi_g
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
srun -n 4 python test.py
"""
import os
import sys
import copy
import functools
import logging
from pathlib import Path

import jax
import numpy as np
import lsstypes as types

from clustering_statistics import tools, setup_logging, compute_stats_from_options, postprocess_stats_from_options


def test_stats_fn(stats=['mesh2_spectrum']):
    catalog_options = {'imock': '*', 'weight': 'default-FKP', 'version': 'abacus-2ndgen-complete', 'tracer': ('LRG', 'ELG_LOP'), 'zrange': ((0.8, 1.1), (0.8, 1.1)), 'region': 'GCcomb', 'stats_dir': '/dvs_ro/cfs/cdirs/desi/mocks/cai/LSS/DA2/mocks/desipipe'}
    fn = tools.get_stats_fn(kind='mesh2_spectrum', **catalog_options)

    catalog_options = dict(version='holi-v1-altmtl', tracer='LRG', zrange=(0.4, 0.5), region='NGC', weight='default-FKP', imock=451)
    for stat in stats:
        for kw in [{'auw': True}, {'cut': True}]:
            fn1 = tools.get_stats_fn(kind=stat, **catalog_options, **kw)
            fn2 = tools.get_stats_fn(kind=stat, catalog=catalog_options, **kw)
            assert fn2 == fn1, f'{fn2} != {fn1}'

    catalog_options = dict(version='holi-v1-altmtl', tracer=('LRG', 'ELG'), zrange=((0.4, 0.5), (0.4, 0.5)), region='NGC', weight='default-FKP', imock=451)
    for stat in stats:
        for kw in [{'auw': True}, {'cut': True}]:
            fn1 = tools.get_stats_fn(kind=stat, **catalog_options, **kw)
            fn2 = tools.get_stats_fn(kind=stat, catalog=catalog_options, **kw)
            assert fn2 == fn1, f'{fn2} != {fn1}'
            _catalog_options = dict(catalog_options)
            _catalog_options.pop('tracer')
            zrange = _catalog_options.pop('zrange')
            fn2 = tools.get_stats_fn(kind=stat, catalog={'LRG': _catalog_options | dict(zrange=zrange[0]), 'ELG': _catalog_options | dict(zrange=zrange[1])}, **kw)
            assert 'mesh2_spectrum_poles_LRGxELG_z0.4-0.5_NGC_weight-default-FKP' in str(fn1)
            assert fn2 == fn1, f'{fn2} != {fn1}'


def test_auw2(stats=['mesh2_spectrum', 'particle2_correlation'][:1]):
    stats_dir = Path(os.getenv('SCRATCH')) / 'clustering-measurements-checks'
    for tracer in ['LRG']:
        zranges = tools.propose_fiducial('zranges', tracer)[:1]
        for region in ['NGC', 'SGC']:
            catalog_options = dict(version='holi-v1-altmtl', tracer=tracer, zrange=zranges, region=region, imock=451)
            #catalog_options = dict(version='data-dr1-v1.5', tracer=tracer, zrange=zranges, region=region, weight='default-FKP', nran=1)
            compute_stats_from_options(stats, catalog=catalog_options, get_stats_fn=functools.partial(tools.get_stats_fn, stats_dir=stats_dir), mesh2_spectrum={'cut': True, 'auw': True}, particle2_correlation={'auw': True})


def test_blinding(stats=['mesh2_spectrum', 'mesh3_spectrum']):
    stats_dir = Path(os.getenv('SCRATCH')) / 'clustering-measurements-checks'
    for tracer in ['LRG']:
        zrange = tools.propose_fiducial('zranges', tracer)[0]
        for region in ['NGC', 'SGC'][:1]:
            options = dict(catalog=dict(version='data-dr2-v2', tracer=tracer, zrange=zrange, region=region, nran=2))
            blinded_options = tools.fill_fiducial_options(options)
            compute_stats_from_options(stats, get_stats_fn=functools.partial(tools.get_stats_fn, stats_dir=stats_dir), **blinded_options)
            options2 = copy.deepcopy(options)
            options2['catalog'].update(version=None, cat_dir=tools.desi_dir / f'survey/catalogs/DA2/LSS/loa-v1/LSScats/v2/nonKP', ext=None)
            unversioned_options = tools.fill_fiducial_options(options2)
            compute_stats_from_options(stats, get_stats_fn=functools.partial(tools.get_stats_fn, stats_dir=stats_dir), **unversioned_options)
            analysis = 'full_shape_protected'
            protected_options = tools.fill_fiducial_options(options, analysis=analysis)
            compute_stats_from_options(stats, get_stats_fn=functools.partial(tools.get_stats_fn, stats_dir=stats_dir / 'protected'), analysis=analysis, **protected_options)
            for stat in stats:
                stat_kwargs = {}
                if stat == 'mesh3_spectrum':
                    stat_kwargs['basis'] = blinded_options[stat].get('basis')
                blinded_fn = tools.get_stats_fn(kind=stat, stats_dir=stats_dir, catalog=blinded_options['catalog'], **stat_kwargs)
                blinded_fn2 = tools.get_stats_fn(kind=stat, stats_dir=stats_dir, catalog=blinded_options['catalog'], version=None, **stat_kwargs)
                assert blinded_fn2 != blinded_fn
                protected_fn = tools.get_stats_fn(kind=stat, stats_dir=stats_dir / 'protected', catalog=protected_options['catalog'], **stat_kwargs)
                blinded = types.read(blinded_fn)
                assert len(blinded.ells) > 1
                blinded2 = types.read(blinded_fn2)
                assert np.allclose(blinded2.value(), blinded.value())
                protected = types.read(protected_fn)
                assert len(protected.ells) == 1
                blinded = blinded.match(protected)
                assert not np.allclose(protected.value(), blinded.value())


def test_bitwise(stats=['mesh2_spectrum', 'particle2_correlation']):
    stats_dir = Path(os.getenv('SCRATCH')) / 'clustering-measurements-checks'
    for tracer in ['LRG']:
        zranges = tools.propose_fiducial('zranges', tracer)
        for region in ['NGC', 'SGC']:
            #catalog_options = dict(version='holi-v1-altmtl', tracer=tracer, zrange=zranges, region=region, imock=451)
            catalog_options = dict(version='data-dr1-v1.5', tracer=tracer, zrange=zranges, region=region, weight='default-bitwise-FKP', nran=1)
            compute_stats_from_options(stats, catalog=catalog_options, get_stats_fn=functools.partial(tools.get_stats_fn, stats_dir=stats_dir), mesh2_spectrum={'cut': True, 'auw': True, 'mattrs': {'meshsize': 512}}, particle2_correlation={'auw': True})
            #compute_stats_from_options(stats, catalog=catalog_options, get_stats_fn=functools.partial(tools.get_stats_fn, stats_dir=stats_dir), mesh2_spectrum={'mattrs': {'meshsize': 512}})


def test_spectrum3(stats=['mesh3_spectrum']):
    stats_dir = Path(os.getenv('SCRATCH')) / 'clustering-measurements-checks'
    for tracer in ['LRG']:
        zranges = tools.propose_fiducial('zranges', tracer)
        for region in ['NGC', 'SGC']:
            #catalog_options = dict(version='holi-v1-altmtl', tracer=tracer, zrange=zranges, region=region, imock=451)
            catalog_options = dict(version='data-dr1-v1.5', tracer=tracer, zrange=zranges, region=region, weight='default-FKP', nran=1)
            compute_stats_from_options(stats, catalog=catalog_options, get_stats_fn=functools.partial(tools.get_stats_fn, stats_dir=stats_dir), mesh3_spectrum={'basis': 'scoccimarro', 'ells': [0, 2]})


def test_recon(stat='recon_particle2_correlation'):
    stats_dir = Path(os.getenv('SCRATCH')) / 'clustering-measurements-checks'
    for tracer in ['LRG']:
        zrange = tools.propose_fiducial('zranges', tracer)[0]
        for region in ['NGC', 'SGC'][:1]:
            catalog_options = dict(version='holi-v1-altmtl', tracer=tracer, zrange=zrange, region=region, weight='default-FKP', imock=451, nran=2)
            catalog_options.update(expand={'parent_randoms_fn': tools.get_catalog_fn(kind='parent_randoms', version='data-dr2-v2', tracer=tracer, nran=catalog_options['nran'])})
            compute_stats_from_options(stat, catalog=catalog_options, get_stats_fn=functools.partial(tools.get_stats_fn, stats_dir=stats_dir), mesh2_spectrum={}, particle2_correlation={})
            fn = tools.get_stats_fn(stats_dir=stats_dir, kind='recon_particle2_correlation', catalog=catalog_options)
            assert np.allclose(types.read(fn).attrs['zeff'], 0.5095347497074821)


def test_correlation2():
    stats_dir = Path(os.getenv('SCRATCH')) / 'clustering-measurements-checks'
    stats = ['particle2_correlation']

    for jackknife in [{}, {'nsplits': 60}][:1]:
        for tracer in ['LRG']:
            zrange = tools.propose_fiducial('zranges', tracer)[0]
            for region in ['NGC', 'SGC']:
                catalog_options = dict(version='holi-v1-altmtl', tracer=tracer, zrange=zrange, region=region, weight='default-FKP', imock=451, nran=2)
                catalog_options.update(expand={'parent_randoms_fn': tools.get_catalog_fn(kind='parent_randoms', version='data-dr2-v2', tracer=tracer, nran=catalog_options['nran'])})
                compute_stats_from_options(stats, catalog=catalog_options, get_stats_fn=functools.partial(tools.get_stats_fn, stats_dir=stats_dir), particle2_correlation={'jackknife': jackknife})
                exit()

        options = dict(catalog=catalog_options, combine_regions={'stats': ['particle2_correlation']}, particle2_correlation={'jackknife': jackknife})
        postprocess_stats_from_options(['combine_regions'], get_stats_fn=functools.partial(tools.get_stats_fn, stats_dir=stats_dir), **options)

    fn = tools.get_stats_fn(stats_dir=stats_dir, kind='particle2_correlation', catalog=catalog_options)
    fn_jack = tools.get_stats_fn(stats_dir=stats_dir, kind='particle2_correlation', catalog=catalog_options, jackknife={'nsplits': 60})
    correlation = types.read(fn)
    correlation_jack = types.read(fn_jack)
    for name in ['DD', 'DS', 'SD', 'SS', 'RR']:
        assert np.allclose(correlation_jack.get(name).value(), correlation.get(name).value(), rtol=1e-8)
    for c in [correlation_jack, correlation]:
        assert np.allclose(c.value(), (c.get('DD').value() - c.get('DS').value() - c.get('SD').value() + c.get('SS').value()) / c.get('SS').value())


def test_complete_catalog():
    tracer = 'LRG'
    for version, imock in zip(['holi-v1-altmtl', 'glam-uchuu-v1-altmtl'], [460, 128]):
        catalog_options = dict(version=version, tracer=tracer, zrange=(0.8, 1.1), region='NGC', weight='default', imock=imock, nran=2)
        for kind in ['forfa_data', 'full_data', 'nz']:
            fn = tools.get_catalog_fn(kind=kind, **catalog_options)
            assert fn.exists(), fn
    complete, reshuffle = {}, {}
    data = tools.read_clustering_catalog(kind='data', complete=complete, reshuffle=reshuffle, **catalog_options)
    expand = {'parent_randoms_fn': tools.get_catalog_fn(kind='parent_randoms', version='data-dr2-v2', tracer=tracer, nran=catalog_options['nran'])}
    randoms = tools.read_clustering_catalog(kind='randoms', complete=complete, reshuffle=reshuffle, expand=expand, **catalog_options)


def test_complete_stats():
    stats_dir = Path(os.getenv('SCRATCH')) / 'clustering-measurements-checks'
    stat = 'mesh2_spectrum'
    for tracer in ['LRG']:
        zrange = tools.propose_fiducial('zranges', tracer)[0]
        for region in ['NGC', 'SGC'][:1]:
            catalog_options = dict(version='holi-v1-altmtl', tracer=tracer, zrange=zrange, region=region, weight='default', imock=451, nran=2)
            catalog_options.update(complete={}, expand={'parent_randoms_fn': tools.get_catalog_fn(kind='parent_randoms', version='data-dr2-v2', tracer=tracer, nran=catalog_options['nran'])})
            #catalog_options.update(expand={'parent_randoms_fn': tools.get_catalog_fn(kind='randoms', version='holi-v1-altmtl', tracer=tracer, region=region, nran=catalog_options['nran'], imock=catalog_options['imock'])})
            compute_stats_from_options(stat, catalog=catalog_options, get_stats_fn=functools.partial(tools.get_stats_fn, stats_dir=stats_dir), mesh2_spectrum={}, particle2_correlation={})
            fn = tools.get_stats_fn(kind=stat, stats_dir=stats_dir, **catalog_options)
            if jax.process_index() == 0:
                spectrum = types.read(fn)
                assert np.allclose(np.mean(spectrum.value()), 4839.647701697041)


def test_expand_randoms_catalog():
    stats_dir = Path(os.getenv('SCRATCH')) / 'clustering-measurements-checks'
    for tracer in ['LRG', 'ELG_LOPnotqso', 'QSO']:
        for zrange in tools.propose_fiducial('zranges', tracer):
            for region in ['NGC', 'SGC']:
                version = 'data-dr2-v2'
                catalog_options = dict(version=version, tracer=tracer, zrange=zrange, region=region, weight='default-FKP', nran=1)
                data = tools.read_clustering_catalog(kind='data', keep_columns=True, **catalog_options)
                expand = {'parent_randoms_fn': tools.get_catalog_fn(kind='parent_randoms', version=version, tracer=tracer, nran=catalog_options['nran']),
                         'from_data': ['Z', 'WEIGHT_SYS', 'FRAC_TLOBS_TILES']}
                randoms_ref = tools.read_clustering_catalog(kind='randoms', keep_columns=True, **catalog_options)
                randoms = tools.read_clustering_catalog(kind='randoms', expand=expand, keep_columns=True, **catalog_options)
                assert not np.allclose(randoms['FRAC_TLOBS_TILES'], 1.)
                regions = ['N', 'S']
                if tracer == 'QSO':
                    regions = ['N', 'SnoDES', 'DES']
                for region in regions:
                    mask_region = tools.select_region(randoms['RA'], randoms['DEC'], region)
                    if not mask_region.any(): continue
                    ratio = randoms['FRAC_TLOBS_TILES'][mask_region] / randoms_ref['FRAC_TLOBS_TILES'][mask_region]
                    alpha = np.nanmean(ratio)
                    assert np.allclose(randoms['FRAC_TLOBS_TILES'][mask_region], randoms_ref['FRAC_TLOBS_TILES'][mask_region] * alpha, equal_nan=True, rtol=1e-4)


def test_expand_randoms_stats(stat='mesh2_spectrum'):
    stats_dir = Path(os.getenv('SCRATCH')) / 'clustering-measurements-checks'
    stat = 'mesh2_spectrum'
    for tracer in ['LRG']:
        zrange = tools.propose_fiducial('zranges', tracer)[0]
        for region in ['NGC', 'SGC'][:1]:
            catalog_options = dict(version='holi-v1-altmtl', tracer=tracer, zrange=zrange, region=region, weight='default', imock=451, nran=2)
            catalog_options.update(expand={'parent_randoms_fn': tools.get_catalog_fn(kind='parent_randoms', version='data-dr2-v2', tracer=tracer, nran=catalog_options['nran'])})
            #catalog_options.update(expand={'parent_randoms_fn': tools.get_catalog_fn(kind='randoms', version='holi-v1-altmtl', tracer=tracer, region=region, nran=catalog_options['nran'], imock=catalog_options['imock'])})
            compute_stats_from_options(stat, catalog=catalog_options, get_stats_fn=functools.partial(tools.get_stats_fn, stats_dir=stats_dir), mesh2_spectrum={}, particle2_correlation={})
            fn = tools.get_stats_fn(kind=stat, stats_dir=stats_dir, **catalog_options)
            if jax.process_index() == 0:
                spectrum = types.read(fn)
                assert np.allclose(np.mean(spectrum.value()), 4761.469528749514), np.mean(spectrum.value())


def test_optimal_weights(stats=['mesh2_spectrum']):
    stats_dir = Path(os.getenv('SCRATCH')) / 'clustering-measurements-checks'
    for tracer in [('LRG', 'ELG_LOPnotqso')]:
        zranges = (0.8, 1.1)
        for region in ['NGC', 'SGC'][:1]:
            #catalog_options = dict(version='holi-v1-altmtl', tracer=tracer, zrange=zranges, region=region, imock=451)
            catalog_options = dict(version='data-dr1-v1.5', tracer=tracer, zrange=zranges, region=region, weight='default-FKP', nran=1)
            #compute_stats_from_options(stats, catalog=catalog_options, get_stats_fn=functools.partial(tools.get_stats_fn, stats_dir=stats_dir), mesh2_spectrum={'auw': True, 'cut': True}, analysis='png_local')
            catalog_options_oqe = dict(version='data-dr1-v1.5', tracer=tracer, zrange=zranges, region=region, weight='default-FKP-oqe', nran=1)
            compute_stats_from_options(stats, catalog=catalog_options_oqe, get_stats_fn=functools.partial(tools.get_stats_fn, stats_dir=stats_dir), mesh2_spectrum={'auw': True, 'cut': True}, analysis='png_local')
            for stat in stats:
                fn = tools.get_stats_fn(kind=stat, stats_dir=stats_dir, **catalog_options)
                fn_oqe = tools.get_stats_fn(kind=stat, stats_dir=stats_dir, **catalog_options_oqe)
                if jax.process_index() == 0:
                    spectrum = types.read(fn)
                    spectrum_oqe = types.read(fn_oqe)
                    assert not np.allclose(spectrum_oqe.value(), spectrum.value())


def test_cross(stats=['mesh2_spectrum']):
    stats_dir = Path(os.getenv('SCRATCH')) / 'clustering-measurements-checks'
    for tracer in [('LRG', 'ELG_LOPnotqso')]:
        zranges = [(0.8, 1.1)]
        for region in ['NGC', 'SGC'][:1]:
            catalog_options = dict(version='data-dr2-v2', tracer=tracer, zrange=zranges, region=region, weight='default-FKP', nran=1)
            compute_stats_from_options(stats, catalog=catalog_options, get_stats_fn=functools.partial(tools.get_stats_fn, stats_dir=stats_dir), mesh2_spectrum={'auw': True, 'cut': True}, particle2_correlation={}, analysis='png_local')


def test_window2(stats=['mesh2_spectrum']):
    from jaxpower import get_mesh_attrs
    from clustering_statistics.tools import propose_fiducial
    from clustering_statistics.spectrum3_tools import _get_window_edges
    #mattrs = get_mesh_attrs(boxcenter=0., **propose_fiducial(kind='mesh2_spectrum', tracer='QSO')['mattrs'])
    #for edges in _get_window_edges(mattrs, scales=(1, 4)):
    #    print(edges, len(edges))
    stats_dir = Path(os.getenv('SCRATCH')) / 'clustering-measurements-checks'
    for stat in stats:
        for tracer in ['LRG']:
            zranges = [(0.8, 1.1)]
            for region in ['NGC', 'SGC'][:1]:
                for method in ['smooth_mesh', 'smooth_particle', 'exact'][1:2]:

                    def get_stats_fn(*args, extra='', **kwargs):
                        extra = f'{extra}_{method}' if extra else method
                        return tools.get_stats_fn(*args, stats_dir=stats_dir, extra=extra, **kwargs)

                    catalog_options = dict(version='holi-v1-altmtl', tracer=tracer, zrange=zranges, region=region, imock=451)
                    #catalog_options = dict(version='data-dr1-v1.5', tracer=tracer, zrange=zranges, region=region, weight='default-FKP', nran=1)
                    options = {}
                    options['mesh2_spectrum'] = {'mattrs': {'meshsize': 250, 'boxsize': 6000.}}
                    options['window_mesh2_spectrum'] = {'cut': True, 'method': method, 'split_randoms': (50, 10)}
                    compute_stats_from_options([stat, f'window_{stat}'][1:], catalog=catalog_options, get_stats_fn=get_stats_fn, **options)
        if 'mesh3' in stat: continue
        '''
        for tracer in ['LRG', ('LRG', 'ELG_LOPnotqso')][1:]:
            zranges = [(0.8, 1.1)]
            for region in ['NGC', 'SGC'][:1]:
                catalog_options = dict(version='holi-v1-altmtl', tracer=tracer, zrange=zranges, region=region, imock=451, nran=1)
                #catalog_options = dict(version='data-dr1-v1.5', tracer=tracer, zrange=zranges, region=region, weight='default-FKP', nran=1)
                compute_stats_from_options([stat, f'window_{stat}'][1:], catalog=catalog_options, get_stats_fn=functools.partial(tools.get_stats_fn, stats_dir=stats_dir), mattrs={'meshsize': 400}, mesh2_spectrum={}, window_mesh2_spectrum={'cut': True}, analysis='png_local')
        '''


def test_window3(stats=['mesh3_spectrum']):
    from jaxpower import get_mesh_attrs
    from clustering_statistics.tools import propose_fiducial
    from clustering_statistics.spectrum3_tools import _get_window_edges
    stats_dir = Path(os.getenv('SCRATCH')) / 'clustering-measurements-checks'
    for stat in stats:
        for tracer in ['LRG']:
            zranges = [(0.8, 1.1)]
            for region in ['NGC', 'SGC'][:1]:
                catalog_options = dict(version='holi-v1-altmtl', tracer=tracer, zrange=zranges, region=region, nran=1, imock=451)
                #catalog_options = dict(version='data-dr1-v1.5', tracer=tracer, zrange=zranges, region=region, weight='default-FKP', nran=1)
                options = {}
                options['mesh3_spectrum'] = {'mattrs': {'meshsize': 250}}
                for method in ['smooth_mesh', 'smooth_particle'][1:]:

                    def get_stats_fn(*args, extra='', **kwargs):
                        extra = f'{extra}_{method}' if extra else method
                        return tools.get_stats_fn(*args, stats_dir=stats_dir, extra=extra, **kwargs)

                    options['window_mesh3_spectrum'] = {'buffer_size': 40, 'method': method, 'split_randoms': (5, 1)}
                    compute_stats_from_options([stat, f'window_{stat}'][1:], catalog=catalog_options, **options, get_stats_fn=get_stats_fn)


def test_covariance():
    stats_dir = Path(os.getenv('SCRATCH')) / 'clustering-measurements-checks'
    stats = ['mesh2_spectrum', 'window_mesh2_spectrum', 'covariance_mesh2_spectrum'][-1:]
    zranges = [(0.8, 1.1)]

    for tracer in ['LRG', 'ELG_LOPnotqso']:
        for region in ['NGC', 'SGC'][:1]:
            catalog_options = dict(version='data-dr1-v1.5', tracer=tracer, zrange=zranges, region=region, weight='default-FKP', nran=1)
            compute_stats_from_options(stats, catalog=catalog_options, get_stats_fn=functools.partial(tools.get_stats_fn, stats_dir=stats_dir), analysis='png_local')

    for tracer in [('LRG', 'ELG_LOPnotqso')]:
        for region in ['NGC', 'SGC'][:1]:
            catalog_options = dict(version='data-dr1-v1.5', tracer=tracer, zrange=zranges, region=region, weight='default-FKP', nran=1)
            compute_stats_from_options(stats, catalog=catalog_options, get_stats_fn=functools.partial(tools.get_stats_fn, stats_dir=stats_dir), analysis='png_local')


def test_rotation():
    stats_dir = Path(os.getenv('SCRATCH')) / 'clustering-measurements-checks'
    stats = ['mesh2_spectrum', 'window_mesh2_spectrum', 'covariance_mesh2_spectrum'][1:2]
    postprocess = ['rotation_mesh2_spectrum']
    for tracer in ['LRG', 'ELG_LOPnotqso'][:0]:
        zranges = [(0.8, 1.1)]
        for region in ['NGC', 'SGC'][:1]:
            catalog_options = dict(version='data-dr1-v1.5', tracer=tracer, zrange=zranges, region=region, weight='default-FKP', nran=1)
            #compute_stats_from_options(stats, catalog=catalog_options, get_stats_fn=functools.partial(tools.get_stats_fn, stats_dir=stats_dir), mesh2_spectrum={'cut': True}, window_mesh2_spectrum={'cut': True})
            postprocess_stats_from_options(postprocess, catalog=catalog_options, get_stats_fn=functools.partial(tools.get_stats_fn, stats_dir=stats_dir), rotation_mesh2_spectrum={'data': dict(version='data-dr1-v1.5'), 'theory': dict(version='data-dr1-v1.5')})
    for tracer in [('LRG', 'ELG_LOPnotqso')]:
        zranges = [(0.8, 1.1)]
        for region in ['NGC', 'SGC'][:1]:
            catalog_options = dict(version='data-dr1-v1.5', tracer=tracer, zrange=zranges, region=region, weight='default-FKP', nran=1)
            #compute_stats_from_options(stats, catalog=catalog_options, get_stats_fn=functools.partial(tools.get_stats_fn, stats_dir=stats_dir), mesh2_spectrum={'auw': True, 'cut': True}, window_mesh2_spectrum={'cut': True})
            postprocess_stats_from_options(postprocess, catalog=catalog_options, get_stats_fn=functools.partial(tools.get_stats_fn, stats_dir=stats_dir), rotation_mesh2_spectrum={'data': dict(version='data-dr1-v1.5'), 'theory': dict(version='data-dr1-v1.5')})


def test_norm():
    stats_dir = Path(os.getenv('SCRATCH')) / 'clustering-measurements-checks'
    stat = 'mesh3_spectrum'
    for tracer in ['BGS_BRIGHT-21.5']:
        zrange = tools.propose_fiducial('zranges', tracer)[0]
        for region in ['NGC']:
            catalog_options = dict(version='data-dr1-v1.5', tracer=tracer, zrange=zrange, region=region, weight='default-FKP', nran=2)
            compute_stats_from_options(stat, catalog=catalog_options, get_stats_fn=functools.partial(tools.get_stats_fn, stats_dir=stats_dir))
            fn = tools.get_stats_fn(kind=stat, stats_dir=stats_dir, basis='sugiyama-diagonal', **catalog_options)
            if jax.process_index() == 0:
                spectrum = types.read(fn)
                #print(spectrum.get((0, 0, 0)).values('norm').mean())
                assert np.allclose(spectrum.get((0, 0, 0)).values('norm').mean(), 1.28543918)


def test_window_fm(tracer='QSO'):
    # FIXME:
    # - zrange should be None when reading the catalog, then compute_stats loops over another independent list of zranges can be provided for the measurements
    # - region may be ALL when reading the catalog, then compute_stats loops over regions
    stats_dir = Path(os.getenv('SCRATCH')) / 'clustering-measurements-checks'
    catalog_options = {
        'version': 'holi-v1-altmtl',
        'tracer': tracer,
        'zrange': {'QSO': (0.8, 3.5), 'LRG': (0.4, 1.1)}[tracer],
        'region': 'ALL',
        'imock': 451,
        'nran': 1,
        'keep_columns': True,
        'weight': 'default-FKP',
    }
    analysis = 'png_local'
    mattrs = {'cellsize': 80.0}
    extra = f"mytest_tracer_{tracer}"
    options = {
        'catalog': catalog_options,
        'mattrs': mattrs,
        'mesh2_spectrum': {'optimal_weights': functools.partial(tools.compute_fiducial_png_weights, tracer=tracer)},
        'window_mesh2_spectrum': {'method': 'exact'},
        'combine_window_mesh2_spectrum': {'effect': 'RIC+AMR'},
        'window_mesh2_spectrum_fm': {'theory': None,
                                     'n_realizations': 2, 'seeds': [42, 84], 'theory_rebin': 5},
    }

    get_stats_fn = functools.partial(tools.get_stats_fn, stats_dir=stats_dir, extra=extra)
    for region in ['NGC', 'SGC']:
        compute_stats_from_options(['mesh2_spectrum', 'window_mesh2_spectrum'], get_stats_fn=get_stats_fn, **(options | {'catalog': catalog_options | dict(region=region)}), analysis=analysis)

    for region in ['NGC', 'SGC']:
        postprocess_stats_from_options(['combine_window_mesh2_spectrum'], get_stats_fn=get_stats_fn, **(options | {'catalog': catalog_options | dict(region=region)}), analysis=analysis)


def test_count3close():
    for tracer in ['BGS_BRIGHT-21.35', 'LRG', 'ELG_LOPnotqso', 'QSO'][1:2]:
        for zrange in tools.propose_fiducial('zranges', tracer):
            for region in ['NGC', 'SGC'][:1]:
                version = 'data-dr2-v2'
                catalog_options = dict(version=version, tracer=tracer, zrange=zrange, region=region, weight='default-FKP', nran=1)
                data = tools.read_clustering_catalog(kind='data', keep_columns=True, **catalog_options)
                from cucount.numpy import count2, count3close, Particles, BinAttrs, SelectionAttrs, MeshAttrs, setup_logging
                setup_logging()
                data = Particles(data['POSITION'], data['INDWEIGHT'])
                battrs = BinAttrs(theta=np.linspace(0., 1., 100))
                import time
                t0 = time.time()
                counts = count2(data, data, battrs=battrs)['weight']
                print(f'count2 {time.time() - t0:.2f}')
                """
                t0 = time.time()
                sattrs = SelectionAttrs(theta=(0., 0.05))
                counts = count3close(data, data, data, battrs12=battrs, sattrs12=sattrs, battrs13=battrs)['weight']
                print(f'count3 {time.time() - t0:.2f}')
                """
                t0 = time.time()
                battrs = BinAttrs(s=np.linspace(0., 100., 100))
                counts = count2(data, data, battrs=battrs)['weight']
                print(f'count2 {time.time() - t0:.2f}')
                t0 = time.time()
                sattrs = SelectionAttrs(theta=(0., 0.05))
                battrs = BinAttrs(s=np.linspace(0., 100., 100))
                counts = count3close(data, data, data, battrs12=battrs, sattrs12=sattrs, battrs13=battrs)['weight']
                print(f'count3 {time.time() - t0:.2f}')



def test_auw3(stats=['mesh3_spectrum']):
    stats_dir = Path(os.getenv('SCRATCH')) / 'clustering-measurements-checks'
    for tracer in ['LRG', 'ELG_LOPnotqso'][1:]:
        zranges = tools.propose_fiducial('zranges', tracer)[-1:]
        for region in ['NGC', 'SGC'][:1]:
            catalog_options = dict(version='abacus-hf-dr2-v2-altmtl', tracer=tracer, zrange=zranges, region=region, imock=1, nran=1)
            #catalog_options = dict(version='data-dr1-v1.5', tracer=tracer, zrange=zranges, region=region, weight='default-FKP', nran=1)
            catalog_options['expand'] = {'parent_randoms_fn': tools.get_catalog_fn(kind='parent_randoms', version='data-dr2-v2', tracer=tracer, nran=catalog_options['nran'])}
            compute_stats_from_options(stats, catalog=catalog_options, get_stats_fn=functools.partial(tools.get_stats_fn, stats_dir=stats_dir), mesh3_spectrum={'ells': [(0, 0, 0), (2, 0, 2)], 'auw': True})



def test_correlation3():
    stats_dir = Path(os.getenv('SCRATCH')) / 'clustering-measurements-checks'
    stats = ['particle3_correlation']

    for tracer in ['LRG']:
        zrange = tools.propose_fiducial('zranges', tracer)[0]
        zrange = (0.4, 1.1)
        for region in ['NGC', 'SGC'][:1]:
            catalog_options = dict(version='holi-v1-altmtl', tracer=tracer, zrange=zrange, region=region, weight='default-FKP', imock=451, nran=5)
            catalog_options.update(expand={'parent_randoms_fn': tools.get_catalog_fn(kind='parent_randoms', version='data-dr2-v2', tracer=tracer, nran=catalog_options['nran'])})
            #particle3_correlation = {'battrs': dict(s=np.linspace(0., 180., 181), pole=(list(range(2)), 'firstpoint'))}
            particle3_correlation = {'split_randoms': (1.5, 7), 'battrs': dict(s=np.linspace(0., 160., 21), pole=(list(range(6)), 'firstpoint'))}
            compute_stats_from_options(stats, catalog=catalog_options, particle3_correlation=particle3_correlation, get_stats_fn=functools.partial(tools.get_stats_fn, stats_dir=stats_dir))

    #options = dict(catalog=catalog_options, combine_regions={'stats': ['particle3_correlation']})
    #postprocess_stats_from_options(['combine_regions'], get_stats_fn=functools.partial(tools.get_stats_fn, stats_dir=stats_dir), **options)


def test_close_pair_correction():
    stats_dir = Path(os.getenv('SCRATCH')) / 'clustering-measurements-close-pairs'
    stats = ['mesh2_spectrum', 'particle2_correlation', 'mesh3_spectrum', 'particle3_correlation']

    for tracer in ['LRG']:
        zrange = tools.propose_fiducial('zranges', tracer)[-1]
        for region in ['NGC', 'SGC'][:1]:
            catalog_options = dict(version='abacus-hf-dr2-v2-altmtl', tracer=tracer, zrange=zrange, region=region, imock=1, nran=1)
            #catalog_options = dict(version='data-dr1-v1.5', tracer=tracer, zrange=zrange, region=region, weight='default-FKP', nran=1)
            catalog_options['expand'] = {'parent_randoms_fn': tools.get_catalog_fn(kind='parent_randoms', version='data-dr2-v2', tracer=tracer, nran=catalog_options['nran'])}
            get_stats_fn = functools.partial(tools.get_stats_fn, stats_dir=stats_dir)
            kw = dict(mesh2_spectrum={}, mesh3_spectrum={},
                      particle2_correlation={'battrs': dict(s=np.linspace(0., 80., 81), mu=(np.linspace(-1., 1., 51), 'midpoint'))},
                      particle3_correlation={'battrs': dict(s=np.linspace(0., 40., 41), pole=(list(range(2)), 'firstpoint'))})
            compute_stats_from_options(stats, catalog=catalog_options, **kw, get_stats_fn=get_stats_fn)

            for name in kw:
                kw[name]['auw'] = True
                if '3' not in name:
                    kw[name]['cut'] = True
            compute_stats_from_options(stats + ['close_pair_correction'], catalog=catalog_options, **kw, get_stats_fn=get_stats_fn)


def test_particle_vs_fft():
    from jaxpower import ParticleField, split_particles, MeshAttrs, BinMesh3CorrelationPoles, compute_mesh3_correlation, get_smooth3_window_bin_attrs
    from jaxpower.particle3 import convert_particles
    from cucount.jax import BinAttrs, WeightAttrs
    from cucount.types import count3, count3_analytic
    from lsstypes.types import convert_ells

    catalog_options = dict(version='data-dr2-v2', tracer='LRG', zrange=(0.4, 0.6), region='NGC', weight='default-FKP', imock=1, nran=1)
    catalog = tools.read_clustering_catalog(kind='randoms', **catalog_options)

    mattrs = MeshAttrs(meshsize=256, boxsize=4000.)
    particles = ParticleField(catalog['POSITION'], catalog['INDWEIGHT'], attrs=mattrs,  exchange=True, backend='mpi')
    #particles = split_particles([particles, None, None], seed=42)
    particles = [particles] * 3

    los = 'local'

    ells = [(0, 0, 0), (2, 0, 2)]
    # Get window basis attributes (e.g., which multipoles to compute)
    kw, ellsin = get_smooth3_window_bin_attrs(ells, ellsin=2, return_ellsin=True)
    # Filter to low multipoles only (reduce computational cost)
    ells = kw['ells'] = [ell for ell in kw['ells'] if all(ell <= 2 for ell in ell)]
    edges = np.linspace(0., 200., 20)

    sbin = BinMesh3CorrelationPoles(mattrs, edges=edges, **kw, buffer_size=20)
    kw_paint = dict(resampler='tsc', interlacing=3, compensate=True)
    meshes = [particle.paint(**kw_paint) for particle in particles]
    correlation_mesh = compute_mesh3_correlation(meshes, bin=sbin, los=los)

    _ells = convert_ells(ells, 'sugiyama', 'slepian')
    ells12, ells13 = [tuple(np.unique([ell[idim] for ell in _ells])) for idim in range(2)]
    battrs12, battrs13 = [BinAttrs(s=edges, pole=(ell, 'firstpoint')) for ell in [ells12, ells13]]
    RRR0 = count3_analytic(mattrs=1., battrs12=battrs12, battrs13=battrs13)

    particles = [convert_particles(particle) for particle in particles]
    wattrs = WeightAttrs()
    counts = count3(*particles, battrs12=battrs12, battrs13=battrs13, wattrs=wattrs)['weight']
    correlation_particle = counts.to_basis('sugiyama', ells=ells)
    norm = next(iter(correlation_mesh)).values('norm').mean()

    def renormalize(pole):
        return pole.clone(counts=pole.values('counts'), norm=norm * RRR0.get((0, 0, 0)).value())

    correlation_particle = correlation_particle.map(renormalize)

    if jax.process_index() == 0:
        import matplotlib as mpl
        from matplotlib import pyplot as plt
        fig, lax = plt.subplots(len(ells) + 1, figsize=(8, 14))
        for ill, ell in enumerate(ells):
            color = f'C{ill:d}'
            pole = correlation_mesh.get(ell)
            s = pole.coords('s')
            mask = (s[..., 1] > s[..., 0])
            idx = np.arange(mask.sum())
            lax[ill].plot(idx, s.prod(axis=-1)[mask] * pole.value()[mask], color=color, label=str(ell))
            pole_particle = correlation_particle.get(ell).ravel()
            lax[ill].plot(idx, s.prod(axis=-1)[mask] * pole_particle.value()[mask], color=color, linestyle='--')
            lax[ill].legend(frameon=False)
        for idim in range(s.shape[1]):
            lax[-1].plot(idx, s[mask, idim], color=f'C{idim:d}', label=f'$s_{idim + 1:d}$')
        lax[-1].set_ylabel(r'$s$ [$\mathrm{Mpc}/h$]')
        lax[-1].set_xlabel(r'bin index')
        lax[-1].legend(frameon=False, ncol=s.shape[1])
        plt.savefig('test_particle_vs_fft_auto.png')
        plt.close(plt.gcf())


        s1_values = [40., 80., 150.]
        ells = [(0, 0, 0), (1, 1, 0), (0, 2, 2)]
        fig, lax = plt.subplots(len(ells), sharex=True, squeeze=False)
        lax = lax.ravel()
        # Colormap for s1
        cmap = plt.get_cmap('viridis')
        norm = mpl.colors.Normalize(vmin=min(s1_values), vmax=max(s1_values))

        select = {'s1': (20., 400.), 's2': (20., 400.)}
        for ill, ell in enumerate(ells):
            label = {'ells': ell}
            pole = correlation_mesh.get(**label).unravel().select(**select)
            pole_particle = correlation_particle.get(**label).select(**select)
            ax = lax[ill]
            for s1 in s1_values:
                is1 = np.argmin(np.abs(pole.coords('s1') - s1))
                s1 = pole.coords('s1')[is1]
                color = cmap(norm(s1))
                ax.plot(s:=pole.coords('s2'), pole.value()[is1, :], color=color, linestyle='-', label=fr'$s_1 = {s1:.0f}$')
                ax.plot(s:=pole_particle.coords('s2'), pole_particle.value()[is1, :], color=color, linestyle='--')
            ax.grid(True)
            ax.set_ylabel(rf'$Q_{{{ell[0]:d}{ell[1]:d}{ell[2]:d}}}$')
        #lax[0].set_title(f'{tracer} {region} window correlation in {zrange[0]:.1f} < z < {zrange[1]:.1f}')
        lax[0].legend(frameon=False)
        lax[-1].set_xscale('log')
        lax[-1].set_xlabel(r'$s_2$ [$\mathrm{Mpc}/h$]')
        plt.savefig('test_particle_vs_fft_auto2.png')
        plt.close(plt.gcf())


if __name__ == '__main__':

    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.85'
    from jax import config
    config.update('jax_enable_x64', True)
    # jax.config.update('jax_debug_nans', True)
    #config.update('jax_num_cpu_devices', 4)
    #config.update('jax_platform_name', 'cpu')

    setup_logging()

    jax.distributed.initialize()

    test_particle_vs_fft()
    # test_window_fm('LRG')
    # test_close_pair_correction()
    # test_auw3()
    # test_window_fm('LRG')
    # test_correlation2()
    # test_correlation3()
    # test_covariance()
    # test_stats_fn()
    # test_complete_catalog()
    # test_expand_randoms_catalog()
    # test_complete_stats()
    # test_expand_randoms_stats()
    # test_blinding()
    # test_covariance()
    # test_rotation()
    # test_stats_fn()
    # test_auw2()
    # test_bitwise()
    # test_expand_randoms_stats()
    # test_optimal_weights()
    # test_cross()
    # test_window2()
    # test_window3()
    # test_spectrum3()
    # test_norm()
    # test_recon()
    # test_covariance()
    # jax.distributed.shutdown()
