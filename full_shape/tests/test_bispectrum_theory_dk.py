from pathlib import Path

import pytest

from full_shape import tools
from full_shape.job_scripts import validation_abacus_mocks


def _run_options(bispectrum_theory_dk=None):
    return validation_abacus_mocks._build_run_options(
        stats=['mesh2_spectrum', 'mesh3_spectrum'],
        tracers=['LRG1'],
        version='abacus-2ndgen-dr2-complete',
        covariance='holi-v3-altmtl',
        stats_dir=Path('/tmp'),
        project='full_shape/base',
        theory_model='folpsD',
        bispectrum_theory_dk=bispectrum_theory_dk,
    )


def test_spectrum3_theory_rebin_dynamic_tracks_observable_spacing():
    assert tools._get_spectrum3_theory_rebin(0.005, 0.005) == 1
    assert tools._get_spectrum3_theory_rebin(0.010, 0.005) == 2


def test_spectrum3_theory_rebin_fixed_is_independent_of_observable_spacing():
    assert tools._get_spectrum3_theory_rebin(0.005, 0.005, theory_dk=0.005) == 1
    assert tools._get_spectrum3_theory_rebin(0.010, 0.005, theory_dk=0.005) == 1


def test_spectrum3_grid_info_reports_both_compaction_stages():
    dynamic = tools.get_spectrum3_window_grid_info(0.010, 0.0025)
    fixed = tools.get_spectrum3_window_grid_info(0.010, 0.0025, theory_dk=0.005)
    native = tools.get_spectrum3_window_grid_info(
        0.010, 0.0025, theory_dk=tools.SPECTRUM3_NATIVE_WINDOW_GRID)

    assert dynamic['first_stage_stride'] == 4
    assert dynamic['second_stage_stride'] == 2
    assert dynamic['final_theory_dk'] == pytest.approx(0.020)
    assert fixed['first_stage_stride'] == 2
    assert fixed['final_theory_dk'] == pytest.approx(0.010)
    assert native['compacted'] is False
    assert native['final_theory_dk'] == pytest.approx(0.0025)


@pytest.mark.parametrize(
    ('ostep', 'tstep', 'theory_dk', 'message'),
    [
        (0.010, 0.005, -0.005, 'positive and finite'),
        (0.010, 0.005, 0.0025, 'cannot be finer'),
        (0.010, 0.005, 0.007, 'integer multiple'),
    ],
)
def test_spectrum3_theory_rebin_rejects_invalid_fixed_spacing(ostep, tstep, theory_dk, message):
    with pytest.raises(ValueError, match=message):
        tools._get_spectrum3_theory_rebin(ostep, tstep, theory_dk=theory_dk)


def test_validation_parser_accepts_fixed_bispectrum_theory_spacing():
    args = validation_abacus_mocks._get_parser().parse_args(['--bispectrum-theory-dk', '0.005'])
    assert args.bispectrum_theory_dk == 0.005
    assert validation_abacus_mocks._get_parser().parse_args([]).bispectrum_theory_dk is None


def test_fixed_bispectrum_theory_spacing_reaches_only_bispectrum():
    options = _run_options(bispectrum_theory_dk=0.005)
    observables = {
        observable['stat']['kind']: observable
        for observable in options['likelihoods'][0]['observables']
    }
    assert 'theory_dk' not in observables['mesh2_spectrum']['window']
    assert observables['mesh3_spectrum']['window']['theory_dk'] == 0.005


def test_fixed_bispectrum_theory_spacing_separates_outputs_but_reuses_prepared_stats(tmp_path):
    dynamic = _run_options()
    fixed = _run_options(bispectrum_theory_dk=0.005)

    dynamic_fit = tools.get_fits_fn(fits_dir=tmp_path, kind='profiles', **dynamic)
    fixed_fit = tools.get_fits_fn(fits_dir=tmp_path, kind='profiles', **fixed)
    assert dynamic_fit != fixed_fit

    dynamic_observable = dynamic['likelihoods'][0]['observables'][1]
    fixed_observable = fixed['likelihoods'][0]['observables'][1]
    dynamic_emulator_hash = tools._hash_options(tools._get_emulator_cache_options(dynamic_observable))
    fixed_emulator_hash = tools._hash_options(tools._get_emulator_cache_options(fixed_observable))
    assert dynamic_emulator_hash != fixed_emulator_hash

    dynamic_prepared = tools._get_prepared_cache_options(
        dynamic['likelihoods'][0]['observables'],
        covariance_options=dynamic['likelihoods'][0]['covariance'],
        kind='data',
    )
    fixed_prepared = tools._get_prepared_cache_options(
        fixed['likelihoods'][0]['observables'],
        covariance_options=fixed['likelihoods'][0]['covariance'],
        kind='data',
    )
    assert dynamic_prepared == fixed_prepared
