"""Likelihood metadata and Cobaya mappings for DESI cosmology configs.

This file centralizes the previous split likelihood-binding modules: dataset
metadata, named likelihood combinations, parameterization selection, and the
translation to Cobaya likelihood dictionaries.
"""

import os
from collections.abc import Iterable
from pathlib import Path



# -----------------------------------------------------------------------------
# Former cosmo bindings section: bao
# -----------------------------------------------------------------------------

DEFAULT_BAO_DATA_PATH = Path('/global/cfs/cdirs/desicollab/science/cpe/y3_bao_cosmo/bao_v1p2/bao/cobaya_data')
DEFAULT_BAO_LYA_FS_DATA_PATH = Path('/global/cfs/cdirs/desicollab/science/cpe/dr2_fs/lya_fs/likelihood/cobaya/measurements')

_BAO_DR2_DATASET_SPECS = (
    ('desi-dr2-bao-all', 'desi_dr2_bao_all', 'desi_gaussian_bao_ALL_GCcomb', {}),
    ('desi-dr2-bao-gqc', 'desi_dr2_bao_gqc', 'desi_gaussian_bao_noLya_GCcomb', {}),
    ('desi-dr2-bao-lya-fs', 'desi_dr2_bao_lya_fs', 'desi_gaussian_bao_Lya_GCcomb_with_new_lyman_alpha_fs', {
        'path': DEFAULT_BAO_LYA_FS_DATA_PATH,
        'rs_fid': 1,
        'aliases': ['BAO'],
        'speed': 2000,
    }),
    ('desi-dr2-bao-gqc-lya-fs', 'desi_dr2_bao_gqc_lya_fs', 'desi_gaussian_bao_ALL_GCcomb_with_new_lyman_alpha_fs', {
        'path': DEFAULT_BAO_LYA_FS_DATA_PATH,
        'rs_fid': 1,
        'aliases': ['BAO'],
        'speed': 2000,
    }),
    ('desi-dr2-bao-bgs', 'desi_dr2_bao_bgs_z1', 'desi_gaussian_bao_BGS_BRIGHT-21.35_GCcomb_z0.1-0.4', {}),
    ('desi-dr2-bao-lrg-z1', 'desi_dr2_bao_lrg_z1', 'desi_gaussian_bao_LRG_GCcomb_z0.4-0.6', {}),
    ('desi-dr2-bao-lrg-z2', 'desi_dr2_bao_lrg_z2', 'desi_gaussian_bao_LRG_GCcomb_z0.6-0.8', {}),
    ('desi-dr2-bao-lrgpluselg', 'desi_dr2_bao_lrgpluselg_z1', 'desi_gaussian_bao_LRG+ELG_LOPnotqso_GCcomb_z0.8-1.1', {}),
    ('desi-dr2-bao-elg', 'desi_dr2_bao_elg_z2', 'desi_gaussian_bao_ELG_LOPnotqso_GCcomb_z1.1-1.6', {}),
    ('desi-dr2-bao-qso', 'desi_dr2_bao_qso_z1', 'desi_gaussian_bao_QSO_GCcomb_z0.8-2.1', {}),
    ('desi-dr2-bao-lya', 'desi_dr2_bao_lya', 'desi_gaussian_bao_Lya_GCcomb', {}),
)

BAO_DR2_DATASETS = {}
for _name, _likelihood, _file_root, _extra in _BAO_DR2_DATASET_SPECS:
    _metadata = {'likelihood': _likelihood}
    if 'path' in _extra:
        _metadata['path'] = _extra['path']
    _metadata['measurements_file'] = f'{_file_root}_mean.txt'
    _metadata['cov_file'] = f'{_file_root}_cov.txt'
    _metadata.update({key: value for key, value in _extra.items() if key != 'path'})
    BAO_DR2_DATASETS[_name] = _metadata

def make_list(value):
    """Return ``value`` as a list, treating strings as scalars."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    try:
        return list(value)
    except TypeError:
        return [value]


def normalize_dataset(dataset, datasets=BAO_DR2_DATASETS):
    """Normalize one or more dataset labels to a list of strings."""
    names = make_list(dataset)
    unknown = [name for name in names if name not in datasets]
    if unknown:
        raise ValueError('Unknown BAO dataset(s): {}. Known datasets are {}.'.format(
            ', '.join(unknown), ', '.join(sorted(datasets))))
    return names


def get_bao_data_path(path=None):
    """Return the directory containing BAO mean/covariance files."""
    if path is None:
        path = os.getenv('DESI_CLUSTERING_COSMO_BAO_DATA_PATH', None)
    if path is None:
        path = DEFAULT_BAO_DATA_PATH
    return Path(path)


def get_dataset_metadata(dataset, datasets=BAO_DR2_DATASETS):
    """Return metadata dictionaries for one or more BAO datasets."""
    return [datasets[name] for name in normalize_dataset(dataset, datasets=datasets)]


# -----------------------------------------------------------------------------
# Former cosmo bindings section: bbn
# -----------------------------------------------------------------------------

BBN_LIKELIHOODS = {
    'schoneberg2024-bbn': {'family': 'bbn', 'parameterization': 'background', 'cobaya': {'cosmo.cobaya.external_likelihoods.bbn.schoneberg2024': None}},
    'schoneberg2024-bbn-fixed-nnu': {'family': 'bbn', 'parameterization': 'background', 'cobaya': {'cosmo.cobaya.external_likelihoods.bbn.schoneberg2024_fixed_nnu': None}},
}


# -----------------------------------------------------------------------------
# Former cosmo bindings section: sn
# -----------------------------------------------------------------------------

SN_LIKELIHOODS = {
    'desdovekie': {'family': 'sn', 'parameterization': 'background', 'cobaya': 'sn.desdovekie'},
    'pantheon': {'family': 'sn', 'parameterization': 'background', 'cobaya': 'sn.pantheon'},
    'pantheonplus': {'family': 'sn', 'parameterization': 'background', 'cobaya': 'sn.pantheonplus'},
    'union3': {'family': 'sn', 'parameterization': 'background', 'cobaya': 'sn.union3'},
    'desy5sn': {'family': 'sn', 'parameterization': 'background', 'cobaya': 'sn.desy5'},
    'desy5sn-zmin0.0': {'family': 'sn', 'parameterization': 'background', 'cobaya': {'cosmo.cobaya.external_likelihoods.sn.zmin.DESY5Zmin': {'dataset_file': 'DESY5/config.dataset', 'aliases': ['DESY5'], 'use_abs_mag': False, 'speed': 100, 'zmin': 0.0}}},
    'desy5sn-zmin0.05': {'family': 'sn', 'parameterization': 'background', 'cobaya': {'cosmo.cobaya.external_likelihoods.sn.zmin.DESY5Zmin': {'dataset_file': 'DESY5/config.dataset', 'aliases': ['DESY5'], 'use_abs_mag': False, 'speed': 100, 'zmin': 0.05}}},
    'desy5sn-zmin0.1': {'family': 'sn', 'parameterization': 'background', 'cobaya': {'cosmo.cobaya.external_likelihoods.sn.zmin.DESY5Zmin': {'dataset_file': 'DESY5/config.dataset', 'aliases': ['DESY5'], 'use_abs_mag': False, 'speed': 100, 'zmin': 0.1}}},
    'desy5sn-zmin0.2': {'family': 'sn', 'parameterization': 'background', 'cobaya': {'cosmo.cobaya.external_likelihoods.sn.zmin.DESY5Zmin': {'dataset_file': 'DESY5/config.dataset', 'aliases': ['DESY5'], 'use_abs_mag': False, 'speed': 100, 'zmin': 0.2}}},
    'pantheonplus-zmin0.1': {'family': 'sn', 'parameterization': 'background', 'cobaya': {'cosmo.cobaya.external_likelihoods.sn.zmin.PantheonPlusZmin': {'dataset_file': 'PantheonPlus/config.dataset', 'aliases': ['PantheonPlus'], 'use_abs_mag': False, 'speed': 100, 'zmin': 0.1}}},
    'union3-zmin0.1': {'family': 'sn', 'parameterization': 'background', 'cobaya': {'cosmo.cobaya.external_likelihoods.sn.zmin.Union3Zmin': {'dataset_file': 'Union3/full_long.dataset', 'aliases': ['Union3'], 'use_abs_mag': False, 'speed': 100, 'zmin': 0.1}}},
}


# -----------------------------------------------------------------------------
# Former cosmo bindings section: cmb
# -----------------------------------------------------------------------------

def _spt_candl_dataset_file():
    """Return the installed SPT-3G candl dataset index path."""
    from pathlib import Path
    import spt_candl_data
    return str(Path(spt_candl_data.__file__).parent / 'SPT3G_D1_TnE_v0' / 'SPT3G_D1_TnE_index.yaml')


def _candl_likelihood_class():
    """Return candl's Cobaya likelihood class."""
    from candl.interface import CandlCobayaLikelihood
    return CandlCobayaLikelihood


def _cmb_spa_cobaya():
    """Return the CMB-SPA composite likelihood block."""
    return {
        'planck_2018_lowl.TT': None,
        'planck_2018_lowl.EE_sroll2': None,
        'act_dr6_cmbonly.PlanckActCut': {
            'dataset_params': {
                'use_cl': 'tt te ee',
                'lmin_cuts': '0 0 0',
                'lmax_cuts': '1000 600 600',
            },
            'params': {
                'A_planck': {
                    'value': 'lambda A_act: A_act',
                    'latex': r'A_{\rm planck}',
                    'proposal': 0.003,
                },
            },
        },
        'act_dr6_cmbonly.ACTDR6CMBonly': {
            'stop_at_error': True,
            'ell_cuts': {'EE': [600, 8500], 'TE': [600, 8500], 'TT': [600, 8500]},
            'input_file': 'dr6_data_cmbonly.fits',
            'lmax_theory': 9500,
        },
        'candl_like': {
            'external': _candl_likelihood_class(),
            'data_set_file': _spt_candl_dataset_file(),
            'clear_internal_priors': True,
            'variant': 'lite',
            'feedback': True,
            'wrapper': None,
            'additional_args': {},
        },
        'act_dr6_spt_lenslike.ACTDR6LensLike': {
            'lens_only': False,
            'variant': 'actplanckspt3g_baseline',
            'lmax': 4000,
            'version': 'v1.2',
        },
    }


def _cmb_sp4a_cobaya():
    """Return the CMB-SP4A composite likelihood block from the June 2026 CPE baseline."""
    return {
        'planck_2018_lowl.TT': None,
        'planck_2018_lowl.EE_sroll2': None,
        'camspec_npipe_lite': {
            'ell_cuts': {'TT': [30, 1500], 'TE': [30, 1000], 'EE': [30, 600]},
            'params': {
                'A_planck': {
                    'prior': {'dist': 'norm', 'loc': 1.0, 'scale': 0.0025},
                    'ref': {'dist': 'norm', 'loc': 1.0, 'scale': 0.1},
                    'latex': r'A_{\rm planck}',
                    'proposal': 0.003,
                },
                'calTE': {
                    'prior': {'dist': 'norm', 'loc': 1.0, 'scale': 0.01},
                    'ref': {'dist': 'norm', 'loc': 1.0, 'scale': 0.01},
                    'proposal': 0.01,
                    'latex': r'c_{TE}',
                },
                'calEE': {
                    'prior': {'dist': 'norm', 'loc': 1.0, 'scale': 0.01},
                    'ref': {'dist': 'norm', 'loc': 1.0, 'scale': 0.01},
                    'proposal': 0.01,
                    'latex': r'c_{EE}',
                },
            },
        },
        'act_dr6_cmbonly.ACTDR6CMBonly': {
            'ell_cuts': {'TT': [1500, 6500], 'TE': [1000, 6500], 'EE': [600, 6500]},
            'lmax_theory': 6500,
            'params': {
                'A_act': {
                    'value': 'lambda A_planck: A_planck',
                    'latex': r'A_{\rm ACT}',
                },
                'P_act': {
                    'prior': {'min': 0.9, 'max': 1.1},
                    'ref': {'dist': 'norm', 'loc': 1.0, 'scale': 0.01},
                    'proposal': 0.01,
                    'latex': r'p_{\rm ACT}',
                },
            },
        },
        'candl_like': {
            'external': _candl_likelihood_class(),
            'data_set_file': _spt_candl_dataset_file(),
            'clear_internal_priors': True,
            'variant': 'lite',
            'feedback': True,
            'wrapper': None,
            'additional_args': {},
            'params': {
                'Tcal': {
                    'latex': r'T_{\rm cal}',
                    'prior': {'dist': 'norm', 'loc': 1.0, 'scale': 0.0036},
                    'ref': {'dist': 'norm', 'loc': 1.0, 'scale': 5.0e-05},
                    'proposal': 5.0e-05,
                },
                'Ecal': {
                    'latex': r'E_{\rm cal}',
                    'prior': {'min': 0.8, 'max': 1.2},
                    'ref': 1.0,
                },
            },
        },
        'act_dr6_spt_lenslike.ACTDR6LensLike': {
            'lens_only': False,
            'stop_at_error': True,
            'variant': 'actplanckspt3g_baseline',
            'lmax': 4000,
            'version': 'v1.2',
        },
    }


CMB_SPA_PRIORS = {
    'cal_dip_prior': 'lambda A_act: stats.norm.logpdf(A_act, loc = 1.0, scale = 0.003)',
    'gaussian_Tcal': 'lambda Tcal: stats.norm.logpdf(Tcal, loc=1.0, scale=0.0036)',
}


CMB_LIKELIHOODS = {
    'CMB-SPA': {'family': 'cmb', 'parameterization': 'cmb', 'cobaya': _cmb_spa_cobaya(), 'prior': CMB_SPA_PRIORS},
    'CMB-SPA-tauprior': {'family': 'cmb', 'parameterization': 'cmb', 'cobaya': _cmb_spa_cobaya(), 'prior': CMB_SPA_PRIORS, 'tauprior': True},
    'CMB-SP4A': {'family': 'cmb', 'parameterization': 'cmb', 'cobaya': _cmb_sp4a_cobaya()},
    # Planck 2018 theta*/rdrag priors.
    'planck2018-thetastar-fixed-nnu': {'family': 'cmb_compressed', 'parameterization': 'background', 'cobaya': {'cosmo.cobaya.external_likelihoods.cmb.thetastar_planck2018_fixed_nnu': None}},
    'planck2018-thetastar-varied-nnu': {'family': 'cmb_compressed', 'parameterization': 'background', 'cobaya': {'cosmo.cobaya.external_likelihoods.cmb.thetastar_planck2018_varied_nnu': None}},
    'planck2018-thetastar-fixed-marg-nnu': {'family': 'cmb_compressed', 'parameterization': 'background', 'cobaya': {'cosmo.cobaya.external_likelihoods.cmb.thetastar_planck2018_fixed_marg_nnu': None}},
    'planck2018-rdrag-fixed-nnu': {'family': 'cmb_compressed', 'parameterization': 'background', 'cobaya': {'cosmo.cobaya.external_likelihoods.cmb.rdrag_planck2018_fixed_nnu': None}},

    # Standard external Cobaya CMB likelihood package names.
    'planck2018-lowl-TT-clik': {'family': 'cmb', 'parameterization': 'cmb', 'cobaya': 'planck_2018_lowl.TT_clik'},
    'planck2018-lowl-EE-clik': {'family': 'cmb', 'parameterization': 'cmb', 'cobaya': 'planck_2018_lowl.EE_clik'},
    'planck2018-lowl-TT': {'family': 'cmb', 'parameterization': 'cmb', 'cobaya': 'planck_2018_lowl.TT'},
    'planck2018-lowl-EE': {'family': 'cmb', 'parameterization': 'cmb', 'cobaya': 'planck_2018_lowl.EE'},
    'planck2018-lowl-EE-sroll2': {'family': 'cmb', 'parameterization': 'cmb', 'cobaya': 'planck_2018_lowl.EE_sroll2'},
    'planck2018-highl-plik-TT': {'family': 'cmb', 'parameterization': 'cmb', 'cobaya': 'planck_2018_highl_plik.TT'},
    'planck2018-highl-plik-TTTEEE': {'family': 'cmb', 'parameterization': 'cmb', 'cobaya': 'planck_2018_highl_plik.TTTEEE'},
    'planck2018-highl-plik-TTTEEE-lite': {'family': 'cmb', 'parameterization': 'cmb', 'cobaya': 'planck_2018_highl_plik.TTTEEE_lite'},
    'planck2018-highl-CamSpec-TT-clik': {'family': 'cmb', 'parameterization': 'cmb', 'cobaya': 'planck_2018_highl_CamSpec.TT'},
    'planck2018-highl-CamSpec-TTTEEE-clik': {'family': 'cmb', 'parameterization': 'cmb', 'cobaya': 'planck_2018_highl_CamSpec.TTTEEE'},
    'planck2018-highl-CamSpec-TT': {'family': 'cmb', 'parameterization': 'cmb', 'cobaya': 'planck_2018_highl_CamSpec.TT_native'},
    'planck2018-highl-CamSpec-TTTEEE': {'family': 'cmb', 'parameterization': 'cmb', 'cobaya': 'planck_2018_highl_CamSpec.TTTEEE_native'},
    'planck2018-highl-CamSpec2021-TT': {'family': 'cmb', 'parameterization': 'cmb', 'cobaya': 'planck_2018_highl_CamSpec2021.TT'},
    'planck2018-highl-CamSpec2021-TTTEEE': {'family': 'cmb', 'parameterization': 'cmb', 'cobaya': 'planck_2018_highl_CamSpec2021.TTTEEE'},
    'planck-NPIPE-highl-CamSpec-TT': {'family': 'cmb', 'parameterization': 'cmb', 'cobaya': 'planck_NPIPE_highl_CamSpec.TT'},
    'planck-NPIPE-highl-CamSpec-TE': {'family': 'cmb', 'parameterization': 'cmb', 'cobaya': 'planck_NPIPE_highl_CamSpec.TE'},
    'planck-NPIPE-highl-CamSpec-EE': {'family': 'cmb', 'parameterization': 'cmb', 'cobaya': 'planck_NPIPE_highl_CamSpec.EE'},
    'planck-NPIPE-highl-CamSpec-TTTEEE': {'family': 'cmb', 'parameterization': 'cmb', 'cobaya': {'cosmo.cobaya.external_likelihoods.cmb.planck_npipe.TTTEEENoCache': None}},
    'planck-NPIPE-highl-CamSpec-TTTEEE-ell-max-600': {'family': 'cmb', 'parameterization': 'cmb', 'cobaya': {'cosmo.cobaya.external_likelihoods.cmb.planck_npipe.TTTEEENoCache': {'dataset_file': '/global/cfs/cdirs/desicollab/science/cpe/cmbdata/CamSpec_NPIPE/CamSpec_NPIPE_12_6_cl.dataset', 'dataset_params': {'use_cl': '143x143 217x217 143x217 TE EE', 'use_range': '30-600'}, 'aliases': ['CamSpec_NPIPE_TTTEEE']}}},
    'planck-NPIPE-highl-CamSpec-TTTEEE-cuts-for-act': {'family': 'cmb', 'parameterization': 'cmb', 'cobaya': {'cosmo.cobaya.external_likelihoods.cmb.planck_npipe.TTTEEENoCache': {'dataset_file': '/global/cfs/cdirs/desicollab/science/cpe/cmbdata/CamSpec_NPIPE/CamSpec_NPIPE_12_6_cl.dataset', 'dataset_params': {'use_cl': '143x143 217x217 143x217 TE EE', 'use_range': {'143x143': '30-2000', '217x217': '500-2000', '143x217': '500-2000', 'TE': '30-1000', 'EE': '30-1000'}}, 'aliases': ['CamSpec_NPIPE_TTTEEE']}}},
    'planck2018-lensing-clik': {'family': 'cmb_lensing', 'parameterization': 'cmb', 'cobaya': 'planck_2018_lensing.clik'},
    'planck2018-lensing': {'family': 'cmb_lensing', 'parameterization': 'cmb', 'cobaya': 'planck_2018_lensing.native'},
    'planckpr4lensing': {'family': 'cmb_lensing', 'parameterization': 'cmb', 'cobaya': 'planckpr4lensing.PlanckPR4Lensing'},
    'planckpr4lensingmarged': {'family': 'cmb_lensing', 'parameterization': 'cmb', 'cobaya': 'planckpr4lensing.PlanckPR4LensingMarged'},
    'planck2020-lollipop-lowlE': {'family': 'cmb', 'parameterization': 'cmb', 'cobaya': 'planck_2020_lollipop.lowlE'},
    'planck2020-lollipop-lowlB': {'family': 'cmb', 'parameterization': 'cmb', 'cobaya': 'planck_2020_lollipop.lowlB'},
    'planck2020-lollipop-lowlEB': {'family': 'cmb', 'parameterization': 'cmb', 'cobaya': 'planck_2020_lollipop.lowlEB'},
    'planck2020-hillipop-TT': {'family': 'cmb', 'parameterization': 'cmb', 'cobaya': 'planck_2020_hillipop.TT'},
    'planck2020-hillipop-TE': {'family': 'cmb', 'parameterization': 'cmb', 'cobaya': 'planck_2020_hillipop.TE'},
    'planck2020-hillipop-EE': {'family': 'cmb', 'parameterization': 'cmb', 'cobaya': 'planck_2020_hillipop.EE'},
    'planck2020-hillipop-TTTEEE': {'family': 'cmb', 'parameterization': 'cmb', 'cobaya': 'planck_2020_hillipop.TTTEEE'},
    'wmap-TTTEEE': {'family': 'cmb', 'parameterization': 'cmb', 'cobaya': {'wmaplike.WMAPLike': {'temin': 24, 'params': {'A_sz': {'prior': {'min': 0.0, 'max': 2.0}}}}}},
    'act-dr6-lensing': {'family': 'cmb_lensing', 'parameterization': 'cmb', 'cobaya': {'act_dr6_lenslike_v1_2.ACTDR6LensLike': {'lens_only': False, 'variant': 'act_baseline', 'lmax': 4000, 'version': 'v1.2'}}},
    'planck-act-dr6-lensing': {'family': 'cmb_lensing', 'parameterization': 'cmb', 'cobaya': {'act_dr6_lenslike_v1_2.ACTDR6LensLike': {'lens_only': False, 'variant': 'actplanck_baseline', 'lmax': 4000, 'version': 'v1.2'}}},
    'planck2018-lowl-TTTEEE-sroll2-momento': {'family': 'cmb', 'parameterization': 'cmb', 'cobaya': {'cosmo.cobaya.external_likelihoods.cmb.momento.TTTEEE_SROLL20': None}},
    'planck2018-lowl-TT-11-29-clik': {'family': 'cmb', 'parameterization': 'cmb', 'cobaya': 'planck_2018_lowl.TT_clik'},

    # Compressed CMB likelihoods.
    'CMB-compressed-theta-ombh2-ombch2-marg-nnu': {'family': 'cmb_compressed', 'parameterization': 'background', 'cobaya': {'cosmo.cobaya.external_likelihoods.cmb.CMB_standard_compression_PR4': {'observables': ['thetastar', 'ombh2', 'ombch2'], 'inflate_cov': True}}},
    'CMB-compressed-ombh2-ombch2-marg-nnu': {'family': 'cmb_compressed', 'parameterization': 'background', 'cobaya': {'cosmo.cobaya.external_likelihoods.cmb.CMB_standard_compression_PR4': {'observables': ['ombh2', 'ombch2'], 'inflate_cov': True}}},
    'CMB-compressed-theta-ombh2-marg-nnu': {'family': 'cmb_compressed', 'parameterization': 'background', 'cobaya': {'cosmo.cobaya.external_likelihoods.cmb.CMB_standard_compression_PR4': {'observables': ['thetastar', 'ombh2'], 'inflate_cov': True}}},
    'CMB-compressed-theta-ombh2-ombch2': {'family': 'cmb_compressed', 'parameterization': 'background', 'cobaya': {'cosmo.cobaya.external_likelihoods.cmb.CMB_standard_compression_PR4': {'observables': ['thetastar', 'ombh2', 'ombch2'], 'inflate_cov': False}}},
    'CMB-compressed-ombh2-ombch2': {'family': 'cmb_compressed', 'parameterization': 'background', 'cobaya': {'cosmo.cobaya.external_likelihoods.cmb.CMB_standard_compression_PR4': {'observables': ['ombh2', 'ombch2'], 'inflate_cov': False}}},
    'CMB-compressed-theta-ombh2': {'family': 'cmb_compressed', 'parameterization': 'background', 'cobaya': {'cosmo.cobaya.external_likelihoods.cmb.CMB_standard_compression_PR4': {'observables': ['thetastar', 'ombh2'], 'inflate_cov': False}}},
    'CMB-compressed-theta': {'family': 'cmb_compressed', 'parameterization': 'background', 'cobaya': {'cosmo.cobaya.external_likelihoods.cmb.CMB_standard_compression_PR4': {'observables': ['thetastar'], 'inflate_cov': False}}},
    'CMB-compressed-ombh2': {'family': 'cmb_compressed', 'parameterization': 'background', 'cobaya': {'cosmo.cobaya.external_likelihoods.cmb.CMB_standard_compression_PR4': {'observables': ['ombh2'], 'inflate_cov': False}}},
    'CMB-compressed-ombch2': {'family': 'cmb_compressed', 'parameterization': 'background', 'cobaya': {'cosmo.cobaya.external_likelihoods.cmb.CMB_standard_compression_PR4': {'observables': ['ombch2'], 'inflate_cov': False}}},
    'CMB-compressed-fake-theta-ombh2-ombch2': {'family': 'cmb_compressed', 'parameterization': 'background', 'cobaya': {'cosmo.cobaya.external_likelihoods.cmb.CMB_standard_compression_PR4_DESI_omch2': {'observables': ['thetastar', 'ombh2', 'ombch2'], 'inflate_cov': False}}},
    'CMB-compressed-R-lA-marg-nnu': {'family': 'cmb_compressed', 'parameterization': 'background', 'cobaya': {'cosmo.cobaya.external_likelihoods.cmb.CMB_shift_parameters_compression_PR3': {'observables': ['R', 'lA'], 'inflate_cov': True}}},
    'CMB-compressed-R-lA-ombh2-marg-nnu': {'family': 'cmb_compressed', 'parameterization': 'background', 'cobaya': {'cosmo.cobaya.external_likelihoods.cmb.CMB_shift_parameters_compression_PR3': {'observables': ['R', 'lA', 'ombh2'], 'inflate_cov': True}}},
    'CMB-compressed-R-lA-ombh2-ombch2-marg-nnu': {'family': 'cmb_compressed', 'parameterization': 'background', 'cobaya': {'cosmo.cobaya.external_likelihoods.cmb.CMB_shift_parameters_compression_PR3': {'observables': ['R', 'lA', 'ombh2', 'omch2'], 'inflate_cov': True}}},
    'CMB-compressed-R-lA': {'family': 'cmb_compressed', 'parameterization': 'background', 'cobaya': {'cosmo.cobaya.external_likelihoods.cmb.CMB_shift_parameters_compression_PR3': {'observables': ['R', 'lA'], 'inflate_cov': False}}},
    'CMB-compressed-R-lA-ombh2': {'family': 'cmb_compressed', 'parameterization': 'background', 'cobaya': {'cosmo.cobaya.external_likelihoods.cmb.CMB_shift_parameters_compression_PR3': {'observables': ['R', 'lA', 'ombh2'], 'inflate_cov': False}}},
    'CMB-compressed-R-lA-ombh2-ombch2': {'family': 'cmb_compressed', 'parameterization': 'background', 'cobaya': {'cosmo.cobaya.external_likelihoods.cmb.CMB_shift_parameters_compression_PR3': {'observables': ['R', 'lA', 'ombh2', 'omch2'], 'inflate_cov': False}}},
}


# -----------------------------------------------------------------------------
# Former cosmo bindings section: registry
# -----------------------------------------------------------------------------

BAO_LIKELIHOODS = {
    name: {
        'name': name,
        'family': 'bao',
        'parameterization': 'background',
        **metadata,
    }
    for name, metadata in BAO_DR2_DATASETS.items()
}

from cosmo.desilike.mapping_likelihoods import _FS_TRACERS as _desilike_FS_TRACERS
FS_LIKELIHOODS = {
    fs_name: {'family': 'fs', 'parameterization': 'lss'}
    for fs_name in _desilike_FS_TRACERS
}

LIKELIHOOD_REGISTRY = {
    **BAO_LIKELIHOODS,
    **SN_LIKELIHOODS,
    **BBN_LIKELIHOODS,
    **CMB_LIKELIHOODS,
    **FS_LIKELIHOODS,
}


def normalize_likelihoods(likelihoods=None, dataset=None, default='desi-dr2-bao-all', registry=LIKELIHOOD_REGISTRY):
    """Normalize one or more likelihood names to a list of strings."""
    if likelihoods is None:
        likelihoods = dataset
    elif dataset is not None:
        raise ValueError('Pass either likelihoods or dataset, not both.')
    if likelihoods is None:
        likelihoods = default
    names = make_list(likelihoods)
    unknown = [name for name in names if name not in registry]
    if unknown:
        raise ValueError('Unknown likelihood(s): {}. Known likelihoods are {}.'.format(
            ', '.join(unknown), ', '.join(sorted(registry))))
    return names


def get_likelihood_metadata(likelihoods=None, dataset=None, registry=LIKELIHOOD_REGISTRY):
    """Return metadata dictionaries for registered likelihoods."""
    names = normalize_likelihoods(likelihoods=likelihoods, dataset=dataset, registry=registry)
    metadata = []
    for name in names:
        item = registry[name]
        alias = item.get('alias')
        if alias is not None:
            item = registry[alias]
        metadata.append({**item, 'name': name})
    return metadata

def get_likelihood_families(likelihoods=None, dataset=None, registry=LIKELIHOOD_REGISTRY):
    """Return the likelihood families needed by a likelihood combination."""
    return sorted({metadata['family'] for metadata in get_likelihood_metadata(likelihoods=likelihoods,
                                                                             dataset=dataset,
                                                                             registry=registry)})


_PARAMETERIZATION_PRIORITY = {'background': 0, 'lss': 1, 'cmb': 2}


def get_parameterization(likelihoods=None, dataset=None, registry=LIKELIHOOD_REGISTRY):
    """Return the cosmological parameterization required by likelihoods.

    Priority order: ``'cmb'`` > ``'lss'`` > ``'background'``.  The highest-priority
    parameterization across all listed likelihoods is returned.
    """
    parameterizations = {metadata.get('parameterization', 'background')
                         for metadata in get_likelihood_metadata(likelihoods=likelihoods,
                                                                 dataset=dataset,
                                                                 registry=registry)}
    return max(parameterizations, key=lambda p: _PARAMETERIZATION_PRIORITY.get(p, -1))


# -----------------------------------------------------------------------------
# Former cosmo bindings section: combinations
# -----------------------------------------------------------------------------

LIKELIHOOD_COMBINATIONS = {}


def is_likelihood_combination(name):
    """Return whether *name* is a named likelihood-combination preset."""
    return isinstance(name, str) and name in LIKELIHOOD_COMBINATIONS


def get_likelihood_combination(name):
    """Return the likelihood list for a named preset."""
    try:
        return list(LIKELIHOOD_COMBINATIONS[name])
    except KeyError as exc:
        raise KeyError(f'Unknown likelihood combination {name!r}.') from exc


def _as_sequence(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(',') if item.strip()]
    if isinstance(value, Iterable):
        return list(value)
    return [value]


def normalize_likelihood_combination(value):
    """Expand one preset/list/comma-separated value to likelihood names.

    Non-preset likelihood names are preserved. Presets can also appear inside a
    comma-separated value or explicit list, e.g. ``'bao,pantheonplus'``.
    """
    if is_likelihood_combination(value):
        return get_likelihood_combination(value)
    output = []
    for item in _as_sequence(value):
        if is_likelihood_combination(item):
            output.extend(get_likelihood_combination(item))
        else:
            output.append(item)
    # Preserve order but avoid duplicate entries introduced by combinations like
    # 'bao,pantheonplus'.
    deduped = []
    for item in output:
        if item not in deduped:
            deduped.append(item)
    if len(deduped) == 1:
        return deduped[0]
    return deduped


def normalize_likelihood_combinations(values):
    """Normalize multiple CLI/config likelihood-combination values."""
    values = values or ['desi-dr2-bao-all']
    return [normalize_likelihood_combination(value) for value in values]


# -----------------------------------------------------------------------------
# Former cosmo bindings section: cobaya
# -----------------------------------------------------------------------------

DEFAULT_BAO_LIKELIHOOD_PACKAGE = 'cosmo.cobaya.external_likelihoods.bao.desi_dr2'


def _resolve_config(config, python_path=None):
    """Return a shallow copy of a Cobaya likelihood config."""
    if config is None:
        return None
    return dict(config)

def _get_bao_cobaya_likelihood(metadata, likelihood_package=None, likelihood_path=None, python_path=None):
    """Return one Cobaya likelihood entry for a BAO data product."""
    if likelihood_package is None:
        likelihood_package = DEFAULT_BAO_LIKELIHOOD_PACKAGE
    if likelihood_path is None:
        likelihood_path = metadata.get('path', get_bao_data_path(None))
    else:
        likelihood_path = get_bao_data_path(likelihood_path)
    config = {
        'path': str(likelihood_path),
        'measurements_file': metadata['measurements_file'],
        'cov_file': metadata['cov_file'],
    }
    for key in ['rs_fid', 'aliases', 'speed']:
        if key in metadata:
            config[key] = metadata[key]
    if python_path is not None:
        config['python_path'] = str(python_path)
    return f'{likelihood_package}.{metadata["likelihood"]}', config


def _get_external_cobaya_likelihoods(metadata, python_path=None):
    cobaya = metadata['cobaya']
    if isinstance(cobaya, str):
        return {cobaya: None}
    if isinstance(cobaya, dict):
        return {name: _resolve_config(config, python_path=python_path) for name, config in cobaya.items()}
    raise TypeError(f'Unsupported Cobaya metadata for {metadata.get("name")!r}: {type(cobaya).__name__}')


def get_cobaya_likelihoods(likelihoods=None, dataset=None, likelihood_package=None,
                           likelihood_path=None, python_path=None, registry=LIKELIHOOD_REGISTRY):
    """Return Cobaya likelihood entries for one or more likelihoods.

    Parameters
    ----------
    likelihoods : str, list, optional
        Likelihood names, e.g. ``'desi-dr2-bao-all'`` or a list such as
        ``['desi-dr2-bao-all', 'pantheonplus']``.
    dataset : str, list, optional
        Convenience alias for BAO-only calls.
    likelihood_package : str, optional
        Override the package that contains native Cobaya BAO wrapper classes.
    likelihood_path : str, Path, optional
        Directory containing BAO mean/covariance files.
    python_path : str, Path, optional
        Optional path added by Cobaya before importing likelihood classes.
    """
    output = {}
    for metadata in get_likelihood_metadata(likelihoods=likelihoods, dataset=dataset, registry=registry):
        family = metadata['family']
        if family == 'bao':
            name, config = _get_bao_cobaya_likelihood(metadata, likelihood_package=likelihood_package,
                                                      likelihood_path=likelihood_path,
                                                      python_path=python_path)
            output[name] = config
        elif 'cobaya' in metadata:
            output.update(_get_external_cobaya_likelihoods(metadata, python_path=python_path))
        else:
            raise ValueError(f'No Cobaya binding is registered for likelihood family {family!r}.')
    return output


def get_cobaya_priors(likelihoods=None, dataset=None, registry=LIKELIHOOD_REGISTRY):
    """Return info-level Cobaya priors registered for likelihood combinations."""
    output = {}
    for metadata in get_likelihood_metadata(likelihoods=likelihoods, dataset=dataset, registry=registry):
        output.update(metadata.get('prior', {}) or {})
    return output
