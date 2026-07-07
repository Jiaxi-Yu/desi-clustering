import numpy as np
import pytest
from mockfactory import Catalog

from clustering_statistics.catalog_blinding import bao as catalog_bao_blinding

try:
    from desiblind.catalog_bao import CatalogBAOBlinder
except ImportError:  # pragma: no cover - depends on optional checkout
    CatalogBAOBlinder = None


def test_resolve_options_is_bao_only_and_explicit():
    options = catalog_bao_blinding.resolve_options({'parameters': {'w0': -0.95, 'wa': 0.10}})
    assert options['mode'] == 'bao'
    assert options['parameters'] == {'w0': -0.95, 'wa': 0.10}
    assert options['input_zcol'] == 'Z'
    assert options['output_zcol'] == 'Z'

    with pytest.raises(ValueError, match='only supports desiblind BAO/AP'):
        catalog_bao_blinding.resolve_options({'modes': ['rsd'], 'parameters': {'w0': -0.95, 'wa': 0.10}})


@pytest.mark.skipif(CatalogBAOBlinder is None, reason='desiblind is not importable')
def test_apply_to_catalog_delegates_to_desiblind():
    catalog = Catalog({
        'RA': np.array([10., 20., 30.]),
        'DEC': np.array([0., 5., -5.]),
        'Z': np.array([0.5, 0.7, 0.9]),
    })
    options = catalog_bao_blinding.resolve_options({'parameters': {'w0': -0.95, 'wa': 0.10}})
    blinded = catalog_bao_blinding.apply_to_catalog(catalog, options)
    expected = CatalogBAOBlinder.apply_to_catalog(catalog, {'w0': -0.95, 'wa': 0.10})

    assert blinded is not catalog
    np.testing.assert_allclose(blinded['Z'], expected['Z'], rtol=0, atol=1e-12)
    assert blinded.attrs['catalog_bao_blinding'] == 'desiblind.CatalogBAOBlinder'



def test_compute_stats_suppresses_data_vector_blinding_when_catalog_bao_is_active(monkeypatch, tmp_path):
    import jax
    import clustering_statistics.compute_stats as compute_stats_module
    from clustering_statistics import tools
    from clustering_statistics.compute_stats import compute_stats_from_options

    calls = []
    written = []

    class ToyStat:
        def __init__(self):
            self.attrs = {}

    def fake_read_catalog(kind, **kwargs):
        calls.append(('read', kind, kwargs.get('region')))
        catalog = Catalog({
            'RA': np.array([10., 20.]),
            'DEC': np.array([0., 5.]),
            'Z': np.array([0.5, 0.6]),
            'PHOTSYS': np.array(['N', 'N']),
            'WEIGHT': np.ones(2),
            'WEIGHT_SYS': np.ones(2),
            'WEIGHT_COMP': np.ones(2),
            'WEIGHT_ZFAIL': np.ones(2),
            'FRAC_TLOBS_TILES': np.ones(2),
        })
        if kind == 'randoms' and not kwargs.get('concatenate', True):
            return [catalog]
        return catalog

    def fake_prepare_catalog(catalog, kind, **kwargs):
        calls.append(('prepare', kind, np.asarray(catalog["Z"]).copy() if not isinstance(catalog, list) else None))
        return catalog

    def fake_apply_to_catalogs(catalogs, options):
        calls.append(('catalog_bao', isinstance(catalogs, list)))
        assert not isinstance(catalogs, list), 'randoms should be LSS-resampled from shifted data, not directly BAO-shifted'
        out = catalogs.copy()
        out['Z'] = np.asarray(out['Z']) + 0.01
        out.attrs['catalog_bao_blinding'] = 'test'
        return out

    def fake_get_stats_fn(kind, catalog=None, **kwargs):
        return tmp_path / f'{kind}.npy'

    def fake_write_stats(filename, statistic, mpicomm=None):
        written.append((filename, dict(statistic.attrs)))

    monkeypatch.setattr(compute_stats_module.catalog_bao_blinding, 'apply_to_catalogs', fake_apply_to_catalogs)
    monkeypatch.setattr(tools, 'check_if_stats_requires_blinding', lambda **kwargs: True)
    monkeypatch.setattr(tools, 'apply_blinding', lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('data-vector blinding should not run')))
    monkeypatch.setattr(tools, 'write_stats', fake_write_stats)
    monkeypatch.setattr(compute_stats_module, 'compute_mesh2_spectrum', lambda *args, **kwargs: {'raw': ToyStat()})
    monkeypatch.setattr(jax.experimental.multihost_utils, 'sync_global_devices', lambda name: None)

    compute_stats_from_options(
        ['mesh2_spectrum'],
        read_catalog=fake_read_catalog,
        prepare_catalog=fake_prepare_catalog,
        mask_catalog=lambda catalog, kind, **kwargs: catalog,
        get_stats_fn=fake_get_stats_fn,
        catalog=dict(
            version='data-dr3-test',
            tracer='LRG',
            zrange=(0.4, 0.6),
            region='NGC',
            nran=1,
            catalog_bao_blinding={'parameters': {'w0': -0.95, 'wa': 0.10}},
        ),
        mesh2_spectrum={},
    )

    assert written
    assert any(call[0] == 'catalog_bao' for call in calls)
    prepared_data = [call for call in calls if call[0] == 'prepare' and call[1] == 'data'][0]
    np.testing.assert_allclose(prepared_data[2], [0.51, 0.61])
    assert written[0][1]['catalog_bao_blinding'] == 'desiblind.CatalogBAOBlinder'


def test_compute_stats_rejects_partial_catalog_bao_for_cross_tracer():
    from clustering_statistics.compute_stats import compute_stats_from_options

    with pytest.raises(ValueError, match='must be configured for every tracer'):
        compute_stats_from_options(
            ['mesh2_spectrum'],
            catalog=dict(
                version='data-dr3-test',
                tracer=('LRG', 'ELG'),
                zrange=((0.4, 0.6), (0.8, 1.1)),
                region='NGC',
                nran=1,
                catalog_bao_blinding=({'parameters': {'w0': -0.95, 'wa': 0.10}}, None),
            ),
            mesh2_spectrum={},
        )


def test_resolve_options_rejects_unsafe_alpha_shift():
    with pytest.raises(ValueError, match='outside the allowed DESI 3% alpha-shift region'):
        catalog_bao_blinding.resolve_options({'parameters': {'w0': -0.90, 'wa': 0.26}}, tracer='LRG', zrange=(0.4, 0.6))

    options = catalog_bao_blinding.resolve_options(
        {'parameters': {'w0': -0.90, 'wa': 0.26}, 'validate_alpha_shift': False},
        tracer='LRG',
        zrange=(0.4, 0.6),
    )
    assert options['parameters'] == {'w0': -0.90, 'wa': 0.26}
    assert options['parameter_metadata']['alpha_validation'] is False


@pytest.mark.skipif(CatalogBAOBlinder is None, reason='desiblind is not importable')
def test_resolve_options_loads_desiblind_hashed_parameter_bank(tmp_path):
    params = {'w0': -0.95, 'wa': 0.10}
    parameters_fn = tmp_path / 'catalog_blinding_parameters.npy'
    CatalogBAOBlinder.write_blinded_parameters('LRG1', params, parameters_fn=parameters_fn)

    options = catalog_bao_blinding.resolve_options(
        {'parameter_source': 'desiblind', 'parameters_fn': parameters_fn, 'metadata': 'closed'},
        tracer='LRG',
        zrange=(0.4, 0.6),
    )
    assert options['parameters'] == params
    assert options['parameter_metadata']['parameter_source'] == 'desiblind'
    assert options['parameter_metadata']['name'] == 'LRG1'
    attrs = catalog_bao_blinding.attrs(options)
    assert attrs['catalog_bao_blinding_parameter_source'] == 'desiblind'
    assert 'catalog_bao_blinding_w0' not in attrs
    assert 'catalog_bao_blinding_wa' not in attrs


@pytest.mark.skipif(CatalogBAOBlinder is None, reason='desiblind is not importable')
def test_resolve_options_loads_lss_parameter_bank(tmp_path):
    bank_fn = tmp_path / 'w0wa.txt'
    np.savetxt(bank_fn, np.array([[-0.95, 0.10], [-0.90, 0.03]]))
    filerow = tmp_path / 'filerow.txt'
    filerow.write_text('1\n')

    options = catalog_bao_blinding.resolve_options(
        {'parameter_source': 'lss', 'lss_parameters_fn': bank_fn, 'lss_filerow': filerow},
        tracer='LRG',
        zrange=(0.4, 0.6),
    )
    assert options['parameters'] == {'w0': -0.90, 'wa': 0.03}
    assert options['parameter_metadata']['parameter_source'] == 'lss'
    assert options['parameter_metadata']['index'] == 1
