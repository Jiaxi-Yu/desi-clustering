import argparse

import fitsio
import numpy as np
import pytest
from mockfactory import Catalog

from clustering_statistics.catalog_blinding import bao, cli as driver, fnl, rsd

try:
    from desiblind.catalog_bao import CatalogBAOBlinder
    from desiblind.catalog_rsd import CatalogRSDBlinder
    from desiblind.catalog_fnl import CatalogFNLBlinder
except ImportError:  # pragma: no cover - depends on optional checkout
    CatalogBAOBlinder = None
    CatalogRSDBlinder = None
    CatalogFNLBlinder = None


pytestmark = pytest.mark.skipif(
    CatalogBAOBlinder is None or CatalogRSDBlinder is None or CatalogFNLBlinder is None,
    reason='desiblind catalog_bao/catalog_rsd/catalog_fnl modules are not importable',
)


def make_catalog(z):
    z = np.asarray(z, dtype='f8')
    return Catalog({
        'TARGETID': np.arange(len(z), dtype='i8'),
        'RA': np.linspace(10., 20., len(z)),
        'DEC': np.linspace(-5., 5., len(z)),
        'Z': z,
        'PHOTSYS': np.array(['N'] * len(z)),
        'WEIGHT': np.ones(len(z)),
        'WEIGHT_SYS': np.ones(len(z)),
        'WEIGHT_COMP': np.ones(len(z)),
        'WEIGHT_ZFAIL': np.ones(len(z)),
        'FRAC_TLOBS_TILES': np.ones(len(z)),
    })


def base_args(input_fn, output_fn, **overrides):
    args = dict(
        input_catalog=str(input_fn), output_catalog=str(output_fn), output_random_catalog=None,
        random_catalog=None, save_reconstruction_random_catalog=None,
        realspace_catalog=None, run_pyrecon=False, run_jaxrecon=False, save_realspace_catalog=None,
        fits_ext='LSS', modes=['rsd'], tracer_name='LRG3', input_zcol='Z', output_zcol='Z',
        realspace_zcol='Z', w0=None, wa=None, zeff=None, bias=None, fnl=None, fiducial_f=0.8,
        random_seed=0, random_resample_columns=['Z', 'WEIGHT', 'WEIGHT_SYS', 'WEIGHT_COMP', 'WEIGHT_ZFAIL', 'TARGETID_DATA'],
        random_split_columns=['PHOTSYS'], skip_bao_nz_reweight=True, skip_final_random_resample=False,
        skip_final_nbar=True, nz_zmin=None, nz_zmax=None, nz_dz=0.01, p0=10000., compmd='ran',
        recon_bias=None, recon_method='iterative_fft', recon_smoothing_radius=15.,
        recon_threshold_randoms=0.01, recon_threshold_randoms_method='mean', recon_growth_rate=None,
        recon_cellsize=None, recon_meshsize=None, recon_boxsize=None, recon_boxcenter=None,
        recon_nthreads=64, recon_weight_col='WEIGHT', fgrowth_blind=0.88, max_df_fraction=0.1,
        fnl_smoothing_radius=30., fnl_recon='IterativeFFTReconstruction', fnl_cellsize=None,
        summary_file=None, clobber=False,
    )
    args.update(overrides)
    return argparse.Namespace(**args)


def test_normalize_modes_orders_bao_before_rsd():
    assert driver.normalize_modes(['rsd', 'bao']) == ('bao', 'rsd')
    assert driver.normalize_modes('bao,rsd') == ('bao', 'rsd')
    assert driver.normalize_modes(['ap']) == ('bao',)
    assert driver.normalize_modes(['fnl']) == ('fnl',)
    assert driver.normalize_modes(['fnl', 'rsd', 'bao']) == ('bao', 'rsd', 'fnl')


def test_bao_adapter_delegates_to_desiblind():
    catalog = make_catalog([0.5, 0.7, 0.9])
    params = {'w0': -0.95, 'wa': 0.1}
    blinded = bao.apply_blinding('LRG3', catalog, parameters=params)
    expected = CatalogBAOBlinder.apply_blinding('LRG3', catalog, parameters=params)
    np.testing.assert_allclose(blinded['Z'], expected['Z'], rtol=0, atol=1e-12)
    assert blinded is not catalog


def test_rsd_adapter_delegates_to_desiblind():
    catalog = make_catalog([0.5, 0.7, 0.9])
    realspace = make_catalog([0.49, 0.69, 0.89])
    params = {'w0': -0.95, 'wa': 0.1, 'zeff': 0.8, 'bias': 2.0, 'fiducial_f': 0.8}
    blinded, applied = rsd.apply_blinding('LRG3', catalog, realspace, parameters=params)
    normalized = CatalogRSDBlinder._normalize_parameters(params)
    expected = CatalogRSDBlinder.apply_blinding('LRG3', catalog, realspace, parameters=normalized)
    np.testing.assert_allclose(blinded['Z'], expected['Z'], rtol=0, atol=1e-12)
    assert applied['fgrowth_blind'] == normalized['fgrowth_blind']


def test_fnl_adapter_delegates_to_desiblind(monkeypatch):
    catalog = make_catalog([0.5, 0.7, 0.9])
    random = make_catalog([0.51, 0.71, 0.91, 1.0])
    expected_factor = np.array([1.1, 0.9, 1.05])

    def fake_apply_blinding(name, data, randoms, parameters=None, **kwargs):
        assert name == 'LRG3'
        assert parameters['fnl'] == 5.0
        assert parameters['zeff'] == 0.8
        assert parameters['bias'] == 2.0
        assert kwargs['return_weight_factor'] is True
        out = data.copy()
        out['WEIGHT'] = np.asarray(out['WEIGHT']) * expected_factor
        out['WEIGHT_COMP'] = np.asarray(out['WEIGHT_COMP']) * expected_factor
        return out, expected_factor

    monkeypatch.setattr(CatalogFNLBlinder, 'apply_blinding', fake_apply_blinding)
    blinded, applied, factor = fnl.apply_blinding(
        'LRG3', catalog, random, parameters={'fnl': 5.0, 'zeff': 0.8, 'bias': 2.0}
    )
    np.testing.assert_allclose(factor, expected_factor)
    np.testing.assert_allclose(blinded['WEIGHT'], np.asarray(catalog['WEIGHT']) * expected_factor)
    np.testing.assert_allclose(blinded['WEIGHT_COMP'], np.asarray(catalog['WEIGHT_COMP']) * expected_factor)
    assert applied['fnl'] == 5.0


def test_rsd_normalization_ignores_none_fgrowth_override():
    params = {'w0': -0.95, 'wa': 0.1, 'zeff': 0.8, 'bias': 2.0, 'fiducial_f': 0.8, 'fgrowth_blind': None}
    normalized = rsd.normalize_rsd_parameters(params)
    assert normalized['fgrowth_blind'] == pytest.approx(0.88)
    assert normalized['max_df_fraction'] == 0.1


def test_fits_roundtrip_rsd_only_driver(tmp_path):
    input_fn = tmp_path / 'input.fits'
    realspace_fn = tmp_path / 'realspace.fits'
    output_fn = tmp_path / 'output.fits'
    summary_fn = tmp_path / 'summary.json'
    data = np.array([(1, 10., 0., 0.7), (2, 20., 1., 0.9)],
                    dtype=[('TARGETID', 'i8'), ('RA', 'f8'), ('DEC', 'f8'), ('Z', 'f8')])
    realspace = data.copy()
    realspace['Z'] -= 0.01
    fitsio.write(input_fn, data, extname='LSS', clobber=True)
    fitsio.write(realspace_fn, realspace, extname='LSS', clobber=True)

    args = base_args(input_fn, output_fn, realspace_catalog=str(realspace_fn), summary_file=str(summary_fn))
    summary = driver.run_from_args(args)
    out = fitsio.read(output_fn, ext='LSS')
    expected_z = CatalogRSDBlinder.transform_redshift(data['Z'], realspace['Z'], fgrowth_fid=0.8, fgrowth_blind=0.88)
    np.testing.assert_allclose(out['Z'], expected_z, rtol=0, atol=1e-12)
    assert summary['modes'] == ['rsd']
    assert summary['rows'] == 2
    assert summary_fn.exists()


def test_recon_tools_rsd_helper_delegates_to_compute_reconstruction(monkeypatch):
    from clustering_statistics import recon_tools

    expected = np.arange(6, dtype='f8').reshape(2, 3)
    calls = []

    def fake_compute_reconstruction(get_data_randoms, **kwargs):
        calls.append(kwargs)
        assert get_data_randoms() == {'sentinel': True}
        return expected, None

    monkeypatch.setattr(recon_tools, 'compute_reconstruction', fake_compute_reconstruction)
    actual = recon_tools.compute_rsd_realspace_positions(
        lambda: {'sentinel': True}, mattrs={'cellsize': 7.}, bias=2.0, smoothing_radius=15.,
    )
    np.testing.assert_array_equal(actual, expected)
    assert calls == [{'mattrs': {'cellsize': 7.}, 'mode': 'rsd', 'bias': 2.0, 'smoothing_radius': 15., 'growth_rate': None, 'threshold_randoms': ('mean', 0.01)}]


def test_jaxrecon_driver_uses_lss_like_bao_random_for_reconstruction_and_final_resample(monkeypatch, tmp_path):
    input_fn = tmp_path / 'input.fits'
    random_fn = tmp_path / 'random.fits'
    output_fn = tmp_path / 'output.fits'
    output_random_fn = tmp_path / 'output_random.fits'
    realspace_fn = tmp_path / 'realspace.fits'
    summary_fn = tmp_path / 'summary.json'
    data = np.array([(1, 10., 0., 0.7, 'N', 1.0, 1.0, 1.0, 1.0),
                     (2, 20., 1., 0.9, 'N', 2.0, 1.0, 1.0, 1.0)],
                    dtype=[('TARGETID', 'i8'), ('RA', 'f8'), ('DEC', 'f8'), ('Z', 'f8'), ('PHOTSYS', 'U1'),
                           ('WEIGHT', 'f8'), ('WEIGHT_SYS', 'f8'), ('WEIGHT_COMP', 'f8'), ('WEIGHT_ZFAIL', 'f8')])
    random = np.array([(11, 11., 0.1, 0.71, 'N', 1.0, 1.0, 1.0, 1.0, 0.5),
                       (12, 21., 1.1, 0.91, 'N', 1.0, 1.0, 1.0, 1.0, 1.5)],
                      dtype=data.dtype.descr + [('FRAC_TLOBS_TILES', 'f8')])
    fitsio.write(input_fn, data, extname='LSS', clobber=True)
    fitsio.write(random_fn, random, extname='LSS', clobber=True)

    calls = []

    def fake_jaxrecon_realspace(data_catalog, random_catalog, **kwargs):
        calls.append({
            'kwargs': kwargs,
            'data_z': np.asarray(data_catalog['Z']).copy(),
            'random_z': np.asarray(random_catalog['Z']).copy(),
            'random_ra': np.asarray(random_catalog['RA']).copy(),
        })
        realspace = data_catalog.copy()
        realspace['Z'] = np.asarray(data_catalog['Z']) - 0.01
        return realspace

    monkeypatch.setattr(rsd, 'compute_jaxrecon_realspace_catalog', fake_jaxrecon_realspace)
    args = base_args(
        input_fn, output_fn, output_random_catalog=str(output_random_fn), random_catalog=str(random_fn),
        run_jaxrecon=True, modes=['bao', 'rsd'], w0=-0.95, wa=0.10, bias=2.0,
        recon_meshsize=16, recon_boxsize=4000., recon_boxcenter=[1., 2., 3.],
        save_realspace_catalog=str(realspace_fn), summary_file=str(summary_fn), random_seed=5,
    )
    summary = driver.run_from_args(args)
    out = fitsio.read(output_fn, ext='LSS')
    outr = fitsio.read(output_random_fn, ext='LSS')

    expected_bao_z = CatalogBAOBlinder.transform_redshift(data['Z'], w0=-0.95, wa=0.10)
    expected_z = CatalogRSDBlinder.transform_redshift(expected_bao_z, expected_bao_z - 0.01, fgrowth_fid=0.8, fgrowth_blind=0.88)
    np.testing.assert_allclose(out['Z'], expected_z, rtol=0, atol=1e-12)

    assert calls
    np.testing.assert_allclose(calls[0]['data_z'], expected_bao_z, rtol=0, atol=1e-12)
    assert set(calls[0]['random_z']).issubset(set(np.asarray(expected_bao_z)))
    np.testing.assert_allclose(calls[0]['random_ra'], random['RA'])
    assert set(outr['Z']).issubset(set(out['Z']))
    np.testing.assert_allclose(outr['RA'], random['RA'])
    assert summary['lss_like_catalogs']['reconstruction_random_resampled'] is True
    assert summary['lss_like_catalogs']['final_random_resampled'] is True


def test_jaxrecon_driver_requires_random_catalog(tmp_path):
    input_fn = tmp_path / 'input.fits'
    output_fn = tmp_path / 'output.fits'
    data = np.array([(1, 10., 0., 0.7)],
                    dtype=[('TARGETID', 'i8'), ('RA', 'f8'), ('DEC', 'f8'), ('Z', 'f8')])
    fitsio.write(input_fn, data, extname='LSS', clobber=True)
    args = base_args(input_fn, output_fn, run_jaxrecon=True)
    with pytest.raises(ValueError, match='--random-catalog'):
        driver.run_from_args(args)


def test_pyrecon_driver_path_uses_direct_pyrecon(monkeypatch, tmp_path):
    input_fn = tmp_path / 'input.fits'
    random_fn = tmp_path / 'random.fits'
    output_fn = tmp_path / 'output.fits'
    realspace_fn = tmp_path / 'realspace.fits'
    data = np.array([(1, 10., 0., 0.7, 1.0), (2, 20., 1., 0.9, 2.0)],
                    dtype=[('TARGETID', 'i8'), ('RA', 'f8'), ('DEC', 'f8'), ('Z', 'f8'), ('WEIGHT', 'f8')])
    random = np.array([(11, 11., 0.1, 0.71, 1.0), (12, 21., 1.1, 0.91, 1.0)], dtype=data.dtype)
    fitsio.write(input_fn, data, extname='LSS', clobber=True)
    fitsio.write(random_fn, random, extname='LSS', clobber=True)

    calls = []

    def fake_pyrecon_realspace(data_catalog, random_catalog, **kwargs):
        calls.append(kwargs)
        realspace = data_catalog.copy()
        realspace['Z'] = np.asarray(data_catalog['Z']) - 0.02
        return realspace

    monkeypatch.setattr(rsd, 'compute_pyrecon_realspace_catalog', fake_pyrecon_realspace)
    args = base_args(
        input_fn, output_fn, random_catalog=str(random_fn), run_pyrecon=True,
        recon_meshsize=64, recon_boxsize=6000., recon_nthreads=8, bias=2.0,
    )
    summary = driver.run_from_args(args)
    out = fitsio.read(output_fn, ext='LSS')
    expected_z = CatalogRSDBlinder.transform_redshift(data['Z'], data['Z'] - 0.02, fgrowth_fid=0.8, fgrowth_blind=0.88)
    np.testing.assert_allclose(out['Z'], expected_z, rtol=0, atol=1e-12)
    assert summary['realspace_source'] == 'pyrecon'
    assert summary['reconstruction']['backend'] == 'pyrecon'
    assert realspace_fn.exists() is False
    assert calls == [{
        'bias': 2.0,
        'smoothing_radius': 15.,
        'growth_rate': 0.8,
        'boxsize': 6000.,
        'boxcenter': None,
        'nmesh': 64,
        'cellsize': 7.,
        'threshold_randoms': 0.01,
        'nthreads': 8,
        'method': 'iterative_fft',
        'weight_col': 'WEIGHT',
        'zcol': 'Z',
    }]


def test_driver_writes_real_run_diagnostic_plots(tmp_path):
    pytest.importorskip('matplotlib')
    input_fn = tmp_path / 'input.fits'
    random_fn = tmp_path / 'random.fits'
    output_fn = tmp_path / 'output.fits'
    output_random_fn = tmp_path / 'output_random.fits'
    plot_dir = tmp_path / 'diagnostics'
    data = np.array([(1, 10., 0., 0.7, 'N', 1.0, 1.0, 1.0, 1.0),
                     (2, 20., 1., 0.9, 'N', 2.0, 1.0, 1.0, 1.0),
                     (3, 30., 2., 1.0, 'N', 1.5, 1.0, 1.0, 1.0)],
                    dtype=[('TARGETID', 'i8'), ('RA', 'f8'), ('DEC', 'f8'), ('Z', 'f8'), ('PHOTSYS', 'U1'),
                           ('WEIGHT', 'f8'), ('WEIGHT_SYS', 'f8'), ('WEIGHT_COMP', 'f8'), ('WEIGHT_ZFAIL', 'f8')])
    random = np.array([(11, 11., 0.1, 0.71, 'N', 1.0, 1.0, 1.0, 1.0, 0.5),
                       (12, 21., 1.1, 0.91, 'N', 1.0, 1.0, 1.0, 1.0, 1.5),
                       (13, 31., 2.1, 1.01, 'N', 1.0, 1.0, 1.0, 1.0, 1.0)],
                      dtype=data.dtype.descr + [('FRAC_TLOBS_TILES', 'f8')])
    fitsio.write(input_fn, data, extname='LSS', clobber=True)
    fitsio.write(random_fn, random, extname='LSS', clobber=True)

    args = base_args(
        input_fn, output_fn,
        modes=['bao'], w0=-0.95, wa=0.10,
        random_catalog=str(random_fn), output_random_catalog=str(output_random_fn),
        diagnostic_plot_dir=str(plot_dir), diagnostic_plot_prefix='LRG_NGC',
    )
    summary = driver.run_from_args(args)
    diagnostics = summary['diagnostics']
    assert diagnostics is not None
    assert (plot_dir / 'LRG_NGC_data_redshift_steps.png').exists()
    assert (plot_dir / 'LRG_NGC_random_matching.png').exists()
    assert (plot_dir / 'LRG_NGC_weight_diagnostics.png').exists()
    assert (plot_dir / 'LRG_NGC_diagnostics.json').exists()
    assert diagnostics['matching']['bao_ap_blinded_random_match'] is not None
    # BAO-only diagnostics do not duplicate the final-data panel/metric because
    # final data is the BAO/AP-blinded data unless an additional effect such as
    # RSD is requested.
    assert diagnostics['matching']['final'] is None
    assert diagnostics['plotted_steps']['final_step'] is False
    assert diagnostics['plotted_steps']['weight_histograms_normalized'] is True


def test_reconstruction_catalog_stacks_random_list():
    cat0 = make_catalog([0.5, 0.6])
    cat1 = make_catalog([0.7])
    rec = rsd.to_reconstruction_catalog([cat0, cat1], weight_col='WEIGHT')
    assert len(rec['Z']) == 3
    assert rec['POSITION'].shape == (3, 3)
    np.testing.assert_allclose(rec['INDWEIGHT'], np.ones(3))


def test_driver_uses_lss_like_random_templates_for_multiple_randoms(monkeypatch, tmp_path):
    input_fn = tmp_path / 'input.fits'
    output_fn = tmp_path / 'output.fits'
    random_template = str(tmp_path / 'random_{index}.fits')
    output_random_template = str(tmp_path / 'output_random_{index}.fits')
    recon_random_template = str(tmp_path / 'recon_random_{index}.fits')
    data = np.array([(1, 10., 0., 0.7, 'N', 1.0, 1.0, 1.0, 1.0),
                     (2, 20., 1., 0.9, 'N', 2.0, 1.0, 1.0, 1.0)],
                    dtype=[('TARGETID', 'i8'), ('RA', 'f8'), ('DEC', 'f8'), ('Z', 'f8'), ('PHOTSYS', 'U1'),
                           ('WEIGHT', 'f8'), ('WEIGHT_SYS', 'f8'), ('WEIGHT_COMP', 'f8'), ('WEIGHT_ZFAIL', 'f8')])
    fitsio.write(input_fn, data, extname='LSS', clobber=True)
    for iran in range(2):
        random = np.array([(10 + iran, 11. + iran, 0.1, 0.71, 'N', 1.0, 1.0, 1.0, 1.0, 0.5),
                           (20 + iran, 21. + iran, 1.1, 0.91, 'N', 1.0, 1.0, 1.0, 1.0, 1.5)],
                          dtype=data.dtype.descr + [('FRAC_TLOBS_TILES', 'f8')])
        fitsio.write(random_template.format(index=iran), random, extname='LSS', clobber=True)

    calls = []

    def fake_pyrecon_realspace(data_catalog, random_catalog, **kwargs):
        calls.append({'random_catalog': random_catalog, 'kwargs': kwargs})
        assert isinstance(random_catalog, list)
        assert len(random_catalog) == 2
        realspace = data_catalog.copy()
        realspace['Z'] = np.asarray(data_catalog['Z']) - 0.01
        return realspace

    monkeypatch.setattr(rsd, 'compute_pyrecon_realspace_catalog', fake_pyrecon_realspace)
    args = base_args(
        input_fn, output_fn,
        random_catalog=None, random_catalog_template=random_template, nran=2,
        output_random_catalog=None, output_random_catalog_template=output_random_template,
        save_reconstruction_random_catalog_template=recon_random_template,
        modes=['rsd'], run_pyrecon=True, bias=2.0,
    )
    summary = driver.run_from_args(args)

    assert calls
    assert summary['nran'] == 2
    assert isinstance(summary['random_catalog'], list)
    assert isinstance(summary['output_random_catalog'], list)
    assert len(summary['output_random_catalog']) == 2
    assert summary['random_rows'] == [2, 2]
    assert summary['random_rows_total'] == 4
    assert summary['lss_like_catalogs']['reconstruction_random_resampled'] is True
    for iran in range(2):
        assert (tmp_path / f'output_random_{iran}.fits').exists()
        assert (tmp_path / f'recon_random_{iran}.fits').exists()


def test_driver_fnl_applies_before_final_random_resampling(monkeypatch, tmp_path):
    input_fn = tmp_path / 'input.fits'
    random_fn = tmp_path / 'random.fits'
    output_fn = tmp_path / 'output.fits'
    output_random_fn = tmp_path / 'output_random.fits'
    summary_fn = tmp_path / 'summary.json'
    data = np.array([(1, 10., 0., 0.7, 'N', 1.0, 1.0, 1.0, 1.0),
                     (2, 20., 1., 0.9, 'N', 2.0, 1.0, 1.0, 1.0),
                     (3, 30., 2., 1.0, 'N', 3.0, 1.0, 1.0, 1.0)],
                    dtype=[('TARGETID', 'i8'), ('RA', 'f8'), ('DEC', 'f8'), ('Z', 'f8'), ('PHOTSYS', 'U1'),
                           ('WEIGHT', 'f8'), ('WEIGHT_SYS', 'f8'), ('WEIGHT_COMP', 'f8'), ('WEIGHT_ZFAIL', 'f8')])
    random = np.array([(11, 11., 0.1, 0.71, 'N', 1.0, 1.0, 1.0, 1.0),
                       (12, 21., 1.1, 0.91, 'N', 1.0, 1.0, 1.0, 1.0),
                       (13, 31., 2.1, 1.01, 'N', 1.0, 1.0, 1.0, 1.0)], dtype=data.dtype)
    fitsio.write(input_fn, data, extname='LSS', clobber=True)
    fitsio.write(random_fn, random, extname='LSS', clobber=True)

    factor = np.array([1.2, 0.8, 1.1])
    calls = []

    def fake_fnl_apply(tracer_name, catalog, randoms, *, parameters, **kwargs):
        calls.append({
            'tracer_name': tracer_name,
            'parameters': parameters,
            'random_len': len(randoms),
        })
        out = catalog.copy()
        out['WEIGHT'] = np.asarray(out['WEIGHT']) * factor
        out['WEIGHT_COMP'] = np.asarray(out['WEIGHT_COMP']) * factor
        return out, {'fnl': parameters['fnl'], 'zeff': parameters['zeff'], 'bias': parameters['bias']}, factor

    monkeypatch.setattr(fnl, 'apply_blinding', fake_fnl_apply)
    args = base_args(
        input_fn, output_fn, random_catalog=str(random_fn), output_random_catalog=str(output_random_fn),
        modes=['fnl'], fnl=5.0, zeff=0.8, bias=2.0, compmd='dat', summary_file=str(summary_fn),
        random_seed=0, random_resample_columns=['Z', 'WEIGHT', 'WEIGHT_COMP'],
    )
    summary = driver.run_from_args(args)
    out = fitsio.read(output_fn, ext='LSS')
    outr = fitsio.read(output_random_fn, ext='LSS')

    # The driver first resets post-addnbar WEIGHT to the LSS pre-addnbar
    # component weight, so the fNL factor is applied to WEIGHT_COMP *
    # WEIGHT_SYS * WEIGHT_ZFAIL rather than the input final WEIGHT column.
    np.testing.assert_allclose(out['WEIGHT'], factor)
    np.testing.assert_allclose(out['WEIGHT_COMP'], data['WEIGHT_COMP'] * factor)
    assert not any('BLIND' in name.upper() for name in out.dtype.names)
    assert set(outr['WEIGHT']).issubset(set(out['WEIGHT']))
    assert set(outr['WEIGHT_COMP']).issubset(set(out['WEIGHT_COMP']))
    assert calls == [{
        'tracer_name': 'LRG3',
        'parameters': {'fnl': 5.0, 'zeff': 0.8, 'bias': 2.0},
        'random_len': len(random),
    }]
    assert summary['modes'] == ['fnl']
    assert summary['applied'] == [{'mode': 'fnl', 'parameters': {'fnl': 5.0, 'zeff': 0.8, 'bias': 2.0}}]
    assert summary['lss_like_catalogs']['final_random_resampled'] is True
    assert summary_fn.exists()
