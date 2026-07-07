import numpy as np
from mockfactory import Catalog

from clustering_statistics.catalog_blinding import lss_catalogs


def make_data():
    return Catalog({
        'TARGETID': np.arange(6, dtype='i8'),
        'RA': np.arange(6., dtype='f8'),
        'DEC': np.arange(6., dtype='f8'),
        'Z': np.array([0.41, 0.42, 0.43, 0.91, 0.92, 0.93]),
        'PHOTSYS': np.array(['N', 'N', 'N', 'S', 'S', 'S']),
        'WEIGHT': np.array([1., 2., 3., 4., 5., 6.]),
        'WEIGHT_SYS': np.ones(6),
        'WEIGHT_COMP': np.ones(6),
        'WEIGHT_ZFAIL': np.ones(6),
    })


def make_random():
    return Catalog({
        'TARGETID': np.arange(100, 108, dtype='i8'),
        'RA': np.linspace(10., 17., 8),
        'DEC': np.linspace(-1., 1., 8),
        'Z': np.zeros(8),
        'PHOTSYS': np.array(['N', 'N', 'N', 'N', 'S', 'S', 'S', 'S']),
        'WEIGHT': np.ones(8),
        'WEIGHT_SYS': np.ones(8),
        'WEIGHT_COMP': np.ones(8),
        'WEIGHT_ZFAIL': np.ones(8),
        'FRAC_TLOBS_TILES': np.array([0.5, 1.0, 1.5, 2.0, 0.7, 1.1, 1.3, 1.9]),
    })


def test_resample_randoms_from_data_preserves_angles_and_samples_by_photsys():
    data = make_data()
    random = make_random()
    out = lss_catalogs.resample_randoms_from_data(random, data, seed=3)

    # Angular footprint comes from the original randoms.
    np.testing.assert_array_equal(out['RA'], random['RA'])
    np.testing.assert_array_equal(out['DEC'], random['DEC'])
    np.testing.assert_array_equal(out['TARGETID'], random['TARGETID'])

    # Redshift-dependent columns come from the matching PHOTSYS data subset.
    assert set(np.asarray(out['Z'])[:4]).issubset(set(np.asarray(data['Z'])[:3]))
    assert set(np.asarray(out['Z'])[4:]).issubset(set(np.asarray(data['Z'])[3:]))
    assert 'TARGETID_DATA' in out
    assert set(np.asarray(out['TARGETID_DATA'])[:4]).issubset(set(np.asarray(data['TARGETID'])[:3]))
    assert set(np.asarray(out['TARGETID_DATA'])[4:]).issubset(set(np.asarray(data['TARGETID'])[3:]))



def test_resample_randoms_from_data_supports_lss_ngc_dec_split():
    data = Catalog({
        'TARGETID': np.arange(4, dtype='i8'),
        'RA': np.zeros(4),
        'DEC': np.array([40., 41., 20., 21.]),
        'Z': np.array([0.51, 0.52, 0.81, 0.82]),
        'WEIGHT': np.ones(4),
        'FRAC_TLOBS_TILES': np.ones(4),
    })
    random = Catalog({
        'TARGETID': np.arange(10, 16, dtype='i8'),
        'RA': np.zeros(6),
        'DEC': np.array([40., 42., 43., 20., 22., 23.]),
        'Z': np.zeros(6),
        'WEIGHT': np.ones(6),
        'FRAC_TLOBS_TILES': np.ones(6),
    })
    out = lss_catalogs.resample_randoms_from_data(
        random, data, seed=1,
        split_columns=lss_catalogs.split_columns_for_region('NGC'),
    )
    north = np.asarray(out['DEC']) > lss_catalogs.LSS_NGC_DEC_THRESHOLD
    assert set(np.asarray(out['Z'])[north]).issubset({0.51, 0.52})
    assert set(np.asarray(out['Z'])[~north]).issubset({0.81, 0.82})


def test_resample_randoms_from_data_is_deterministic_and_scales_weight_by_frac_tlobs():
    data = make_data()
    random = make_random()
    out1 = lss_catalogs.resample_randoms_from_data(random, data, seed=11, preserve_weight_normalization=False)
    out2 = lss_catalogs.resample_randoms_from_data(random, data, seed=11, preserve_weight_normalization=False)
    np.testing.assert_array_equal(out1['Z'], out2['Z'])

    # With preserve_weight_normalization disabled this is exactly the LSS first
    # step: sampled data WEIGHT times random FRAC_TLOBS_TILES.
    sampled_weights = []
    for z in out1['Z']:
        sampled_weights.append(np.asarray(data['WEIGHT'])[np.asarray(data['Z']) == z][0])
    np.testing.assert_allclose(out1['WEIGHT'], np.asarray(sampled_weights) * random['FRAC_TLOBS_TILES'])


def test_apply_bao_nz_reweight_returns_internal_factor_without_catalog_column():
    before = Catalog({'Z': np.array([0.1, 0.2, 0.8, 0.9]), 'WEIGHT_SYS': np.ones(4), 'WEIGHT': np.ones(4)})
    after = Catalog({'Z': np.array([0.1, 0.2, 0.2, 0.9]), 'WEIGHT_SYS': np.ones(4), 'WEIGHT': np.ones(4)})
    out, info = lss_catalogs.apply_bao_nz_reweight(before, after, zmin=0., zmax=1., dz=0.5)
    # First bin: before has 2, after has 3 -> internal factor in first bin is 2/3.
    # Second bin: before has 2, after has 1 -> internal factor in second bin is 2.
    expected = np.array([2/3, 2/3, 2/3, 2.])
    assert 'WEIGHT_BAO_NZ' not in out
    # Keep imaging/systematics WEIGHT_SYS unchanged; the BAO n(z) correction is
    # returned as internal state and included later when rebuilding total WEIGHT.
    np.testing.assert_allclose(out['WEIGHT_SYS'], np.ones(4))
    np.testing.assert_allclose(out['WEIGHT'], np.ones(4))
    np.testing.assert_allclose(info['ratio'], [2/3, 2.])
    np.testing.assert_allclose(info['correction'], expected)


def test_set_lss_pre_addnbar_weight_rebuilds_weight_from_components():
    catalog = Catalog({
        'Z': np.array([0.1, 0.2]),
        'WEIGHT': np.array([10., 20.]),
        'WEIGHT_SYS': np.array([2., 3.]),
        'WEIGHT_COMP': np.array([4., 5.]),
        'WEIGHT_ZFAIL': np.array([6., 7.]),
    })
    out = lss_catalogs.set_lss_pre_addnbar_weight(catalog, extra_weight=np.array([0.5, 2.]))
    np.testing.assert_allclose(out['WEIGHT'], [24., 210.])
    np.testing.assert_allclose(out['WEIGHT_SYS'], [2., 3.])
    np.testing.assert_allclose(catalog['WEIGHT'], [10., 20.])


def test_add_nbar_fkp_includes_internal_bao_nz_factor_in_final_weight():
    data = Catalog({
        'Z': np.array([0.45, 0.55]),
        'RA': np.array([0., 1.]),
        'DEC': np.array([0., 1.]),
        'NTILE': np.array([1, 1]),
        'WEIGHT_COMP': np.array([2., 2.]),
        'WEIGHT_SYS': np.array([3., 3.]),
        'WEIGHT_ZFAIL': np.array([5., 5.]),
        'FRAC_TLOBS_TILES': np.ones(2),
    })
    random = Catalog({
        'Z': np.array([0.45, 0.55]),
        'RA': np.array([0., 1.]),
        'DEC': np.array([0., 1.]),
        'NTILE': np.array([1, 1]),
        'WEIGHT_COMP': np.ones(2),
        'WEIGHT_SYS': np.ones(2),
        'WEIGHT_ZFAIL': np.ones(2),
        'WEIGHT': np.ones(2),
        'FRAC_TLOBS_TILES': np.ones(2),
    })
    out_data, _, _ = lss_catalogs.add_nbar_fkp(
        data, random, zmin=0.4, zmax=0.7, dz=0.1, p0=100.,
        data_extra_weight=np.array([0.5, 2.0]),
    )
    # NTILE mean WEIGHT_COMP is 2, so the final LSS weight is
    # WEIGHT_COMP * WEIGHT_SYS * WEIGHT_ZFAIL * internal_bao_nz_factor / 2.
    np.testing.assert_allclose(out_data['WEIGHT'], [7.5, 30.])
    np.testing.assert_allclose(out_data['WEIGHT_SYS'], [3., 3.])
    assert 'WEIGHT_BAO_NZ' not in out_data


def test_add_nbar_fkp_adds_final_density_columns_to_data_and_randoms():
    data = make_data()
    random = make_random()
    out_data, out_random, info = lss_catalogs.add_nbar_fkp(data, random, zmin=0.4, zmax=1.0, dz=0.3, p0=100.)
    for catalog in [out_data, out_random]:
        for col in ['NZ', 'NX', 'WEIGHT', 'WEIGHT_FKP']:
            assert col in catalog
        assert np.all(np.asarray(catalog['WEIGHT_FKP']) > 0.)
        assert np.all(np.asarray(catalog['WEIGHT_FKP']) <= 1.)
    assert 'edges' in info and 'nz' in info
