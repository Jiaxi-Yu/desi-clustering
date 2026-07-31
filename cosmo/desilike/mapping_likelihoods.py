"""desilike likelihood mapping.

Maps named likelihoods from the cobaya registry (see
``cosmo/cobaya/mapping_likelihoods.py``) to desilike likelihood instances.

Skipped families / individual names:
  - planck2020-* : no desilike equivalent
  - *-momento : no desilike equivalent
  - planckpr4* : no desilike equivalent
  - wmap-* : no desilike equivalent
  - planck2018-lensing* : no desilike lensing equivalent for Planck 2018
  - planck2018-highl-CamSpec-* (PR3) : no PR3 CamSpec class in desilike
  - planck-NPIPE-highl-CamSpec-TE/EE : no per-spectrum class in desilike
  - CMB-compressed-fake-* : uses a non-existent cobaya class, skipped
"""

import functools
import os
from collections.abc import Iterable
from pathlib import Path


# ---------------------------------------------------------------------------
# Named likelihood combinations
# ---------------------------------------------------------------------------

# Subset of cosmo/cobaya/mapping_likelihoods.py's LIKELIHOOD_COMBINATIONS,
# restricted to presets whose members are all supported by desilike (see the
# skipped families listed above): drops 'bao-bbn-fixed-nnu' (needs
# 'schoneberg2024-bbn-fixed-nnu'), 'bao-planck-npipe-lensing' (needs
# 'planckpr4lensing'), and 'bao-planck-npipe-sroll2-momento' (needs
# 'planck2018-lowl-TTTEEE-sroll2-momento').
LIKELIHOOD_COMBINATIONS = {
    'bao': ['desi-dr2-bao-all'],
    'bao-sn-pantheonplus': ['desi-dr2-bao-all', 'pantheonplus'],
    'bao-sn-union3': ['desi-dr2-bao-all', 'union3'],
    'bao-sn-desy5': ['desi-dr2-bao-all', 'desy5sn'],
    'bao-sn-desdovekie': ['desi-dr2-bao-all', 'desdovekie'],
    'bao-sn-pantheonplus-zmin0.1': ['desi-dr2-bao-all', 'pantheonplus-zmin0.1'],
    'bao-sn-union3-zmin0.1': ['desi-dr2-bao-all', 'union3-zmin0.1'],
    'bao-sn-desy5-zmin0.1': ['desi-dr2-bao-all', 'desy5sn-zmin0.1'],
    'bao-bbn': ['desi-dr2-bao-all', 'schoneberg2024-bbn'],
    'bao-thetastar-fixed-nnu': ['desi-dr2-bao-all', 'planck2018-thetastar-fixed-nnu'],
    'bao-thetastar-varied-nnu': ['desi-dr2-bao-all', 'planck2018-thetastar-varied-nnu'],
    'bao-rdrag-fixed-nnu': ['desi-dr2-bao-all', 'planck2018-rdrag-fixed-nnu'],
    'bao-cmb-compressed-theta': ['desi-dr2-bao-all', 'CMB-compressed-theta'],
    'bao-cmb-compressed-r-la': ['desi-dr2-bao-all', 'CMB-compressed-R-lA'],
    'bao-cmb-compressed-theta-ombh2': ['desi-dr2-bao-all', 'CMB-compressed-theta-ombh2'],
    'bao-cmb-compressed-theta-ombh2-ombch2': ['desi-dr2-bao-all', 'CMB-compressed-theta-ombh2-ombch2'],
    'bao-sn-cmb-compressed-theta': ['desi-dr2-bao-all', 'pantheonplus', 'CMB-compressed-theta'],
    'bao-sn-cmb-compressed-r-la': ['desi-dr2-bao-all', 'pantheonplus', 'CMB-compressed-R-lA'],
    'cmb-spa': ['CMB-SPA'],
    'cmb-spa-tauprior': ['CMB-SPA-tauprior'],
    'bao-cmb-spa': ['desi-dr2-bao-all', 'CMB-SPA'],
    'bao-sn-desdovekie-cmb-spa': ['desi-dr2-bao-all', 'desdovekie', 'CMB-SPA'],
    'bao-planck-npipe': ['desi-dr2-bao-all', 'planck-NPIPE-highl-CamSpec-TTTEEE'],
    'bao-planck-npipe-ell-max-600': ['desi-dr2-bao-all', 'planck-NPIPE-highl-CamSpec-TTTEEE-ell-max-600'],
    'bao-planck-npipe-cuts-for-act': ['desi-dr2-bao-all', 'planck-NPIPE-highl-CamSpec-TTTEEE-cuts-for-act'],
}


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


def get_likelihood_label(likelihoods=None):
    """Return a filesystem-friendly label for a likelihood or list of likelihoods."""
    if likelihoods is None:
        return 'none'
    if isinstance(likelihoods, str):
        return likelihoods
    return '_'.join(likelihoods)


# ---------------------------------------------------------------------------
# BAO
# ---------------------------------------------------------------------------

# Cobaya BAO name → zbins for DESIDR2BAOLikelihood (None = all tracers)
_BAO_ZBINS = {
    'desi-dr2-bao-all':        None,
    'desi-dr2-bao-bgs':        ['BGS'],
    'desi-dr2-bao-lrg-z1':     ['LRG1'],
    'desi-dr2-bao-lrg-z2':     ['LRG2'],
    'desi-dr2-bao-lrgpluselg': ['LRG3+ELG1'],
    'desi-dr2-bao-elg':        ['ELG2'],
    'desi-dr2-bao-qso':        ['QSO'],
    'desi-dr2-bao-lya':        ['Lya'],
}


# Likelihoods backed by external measurement files (mean + covariance txt).
# Built on-the-fly via bao_likelihood_from_files; not downloaded by install_likelihoods().
_BAO_MEASUREMENT_FILES_DIR = Path('/dvs_ro/cfs/cdirs/desicollab/science/cpe/dr2_fs/lya_fs/likelihood/cobaya/measurements')
_BAO_MEASUREMENT_FILES = {
    'desi-dr2-bao-gqc-lya-fs': (
        _BAO_MEASUREMENT_FILES_DIR / 'desi_gaussian_bao_ALL_GCcomb_with_new_lyman_alpha_fs_mean.txt',
        _BAO_MEASUREMENT_FILES_DIR / 'desi_gaussian_bao_ALL_GCcomb_with_new_lyman_alpha_fs_cov.txt',
    ),
    'desi-dr2-bao-lya-fs': (
        _BAO_MEASUREMENT_FILES_DIR / 'desi_gaussian_bao_Lya_GCcomb_with_new_lyman_alpha_fs_mean.txt',
        _BAO_MEASUREMENT_FILES_DIR / 'desi_gaussian_bao_Lya_GCcomb_with_new_lyman_alpha_fs_cov.txt',
    )
}


# ---------------------------------------------------------------------------
# SN
# ---------------------------------------------------------------------------

# Cobaya SN name → (desilike class name, zrange)
_SN_MAP = {
    'desdovekie':           ('DESY5DovekieSNLikelihood', (None, None)),
    'pantheon':             ('PantheonSNLikelihood',     (None, None)),
    'pantheonplus':         ('PantheonPlusSNLikelihood', (None, None)),
    'union3':               ('Union3SNLikelihood',       (None, None)),
    'desy5sn':              ('DESY5v1SNLikelihood',      (None, None)),
    'desy5sn-zmin0.0':      ('DESY5v1SNLikelihood',      (0.0,  None)),
    'desy5sn-zmin0.05':     ('DESY5v1SNLikelihood',      (0.05, None)),
    'desy5sn-zmin0.1':      ('DESY5v1SNLikelihood',      (0.1,  None)),
    'desy5sn-zmin0.2':      ('DESY5v1SNLikelihood',      (0.2,  None)),
    'pantheonplus-zmin0.1': ('PantheonPlusSNLikelihood', (0.1,  None)),
    'union3-zmin0.1':       ('Union3SNLikelihood',       (0.1,  None)),
}


# ---------------------------------------------------------------------------
# Full-shape (FS)
# ---------------------------------------------------------------------------

# Stat abbreviation → generate_likelihood_options_helper stats list
_FS_STAT_SHORTS = {
    's2':         ['mesh2_spectrum'],
    's2-s3':      ['mesh2_spectrum', 'mesh3_spectrum'],
    's2-baor':    ['mesh2_spectrum', 'recon_bao'],
    's2-c2r':     ['mesh2_spectrum', 'recon_particle2_correlation'],
    's2-s3-c2r':  ['mesh2_spectrum', 'mesh3_spectrum', 'recon_particle2_correlation'],
    's2-s3-baor': ['mesh2_spectrum', 'mesh3_spectrum', 'recon_bao'],
}

# Individual tracer strings (used by the "all" variants)
_FS_ALL_TRACERS = ['BGS1', 'LRG1', 'LRG2', 'LRG3', 'ELG2', 'QSO1']
_FS_SHORT_TRACERS = {'BGS1': 'bgs', 'LRG1': 'lrg1', 'LRG2': 'lrg2', 'LRG3': 'lrg3',
                     'ELG2': 'elg2', 'QSO1': 'qso1'}

# DR2 full-shape name → kwargs for generate_likelihood_options_helper.
# Entries without a 'tracer' key represent the "all individual tracers" combination.
# Beyond generate_likelihood_options_helper kwargs, entries may carry post-helper
# tweaks applied by _fs_likelihood_options: 'select' (stat kind → k-range select list),
# 'theory' (merged into each observable's theory options), and 'covariance_options'
# (merged into the covariance options).
_FS_TRACERS = {}
for _stat_short, _stats in _FS_STAT_SHORTS.items():
    _FS_TRACERS[f'desi-dr2-fs-{_stat_short}-all'] = {'stats': _stats}
    for _tracer in _FS_ALL_TRACERS:
        _FS_TRACERS[f'desi-dr2-fs-{_stat_short}-{_FS_SHORT_TRACERS[_tracer]}'] = {'tracer': _tracer, 'stats': _stats}

# Abacus-HF DR2 v2 mock full-shape likelihoods, replicating the setup of
# full_shape/job_scripts/validation_systematic_templates.py with dataset
# 'abacus-hf-dr2-v2-altmtl' (mean of mocks, holi-v3-altmtl mock covariance,
# no systematic templates). BGS is not available in the Abacus-HF mocks;
# _fs_likelihood_options falls back to the second-generation mocks for it.
_FS_SELECT = {
    'mesh2_spectrum': [{'ells': 0, 'k': [0.02, 0.20, 0.01]},
                       {'ells': 2, 'k': [0.02, 0.20, 0.01]}],
    'mesh3_spectrum': [{'ells': (0, 0, 0), 'k': [0.02, 0.20, 0.01]},
                       {'ells': (2, 0, 2), 'k': [0.02, 0.03, 0.01]}],
}
_FS_STATS_DIR = Path('/dvs_ro/cfs/cdirs/desicollab/science/cai/desi-clustering/dr2/summary_statistics')

# Default cache for full-shape likelihoods (prepared stats + emulators), shared
# with full_shape/job_scripts (e.g. validation_systematic_templates.py) so
# products prepared there are reused. None (no $SCRATCH) disables caching, and
# with it the emulator (see _fs_likelihood_options).
DEFAULT_FS_CACHE_DIR = (Path(os.environ['SCRATCH']) / 'desi-clustering/full_shape/job_scripts/_cache'
                        if 'SCRATCH' in os.environ else None)

# Default ACECosmology engine for FolpsD-style full-shape fits (see
# desilike.theories.primordial_cosmology.ACECosmology): jaxace w0waCDM background,
# jaxmapse linear pk, and the local Capse w0waCDM Cl set (the packaged 'camb_lcdm'
# harmonic emulator is LCDM-only).
DEFAULT_ACE_ENGINE = {'background': 'ACE_mnuw0wacdm_ln10As_basis',
                      'fourier': 'mnuw0wacdm_class',
                      'harmonic': 'capse_mnuw0wacdm_250001'}
for _stat_short in ['s2', 's2-s3']:
    for _theory in ['folpsD', 'comet']:
        _FS_TRACERS[f'abacus-dr2-fs-{_stat_short}-all-{_theory}'] = {
            'stats': _FS_STAT_SHORTS[_stat_short],
            'tracer': _FS_ALL_TRACERS,
            'version': 'abacus-hf-dr2-v2-altmtl',
            'covariance': 'holi-v3-altmtl',
            'stats_dir': _FS_STATS_DIR,
            'project': 'full_shape/fiber_assignment_systematics',
            'emulator': _theory != 'comet',
            'select': {stat: _FS_SELECT[stat] for stat in _FS_STAT_SHORTS[_stat_short]},
            'theory': {'model': _theory, 'prior_basis': 'physical_aap'},
            'covariance_options': {'source': 'mock', 'project': 'full_shape/base'},
        }
        for _tracer in _FS_ALL_TRACERS:
            _FS_TRACERS[f'abacus-dr2-fs-{_stat_short}-{_FS_SHORT_TRACERS[_tracer]}-{_theory}'] = _FS_TRACERS[f'abacus-dr2-fs-{_stat_short}-all-{_theory}'] | dict(tracer=[_tracer])


# ---------------------------------------------------------------------------
# CMB
# ---------------------------------------------------------------------------

# Names that have no desilike equivalent and should raise NotImplementedError
_CMB_NOT_IMPLEMENTED = {
    # Planck 2018 PR3 CamSpec (no PR3 CamSpec class in desilike)
    'planck2018-highl-CamSpec-TT',
    'planck2018-highl-CamSpec-TTTEEE',
    'planck2018-highl-CamSpec-TT-clik',
    'planck2018-highl-CamSpec-TTTEEE-clik',
    'planck2018-highl-CamSpec2021-TT',
    'planck2018-highl-CamSpec2021-TTTEEE',
    # NPIPE CamSpec per-spectrum (no TE-only / EE-only class)
    'planck-NPIPE-highl-CamSpec-TE',
    'planck-NPIPE-highl-CamSpec-EE',
    # Planck 2018 lensing (user excluded)
    'planck2018-lensing',
    'planck2018-lensing-clik',
}

# CMB — PR4 standard compression (thetastar, ombh2, ombch2)
_PR4_STANDARD_MAP = {
    'CMB-compressed-theta':                       (['thetastar'],                     False),
    'CMB-compressed-ombh2':                       (['ombh2'],                         False),
    'CMB-compressed-ombch2':                      (['ombch2'],                        False),
    'CMB-compressed-theta-ombh2':                 (['thetastar', 'ombh2'],            False),
    'CMB-compressed-ombh2-ombch2':                (['ombh2', 'ombch2'],               False),
    'CMB-compressed-theta-ombh2-ombch2':          (['thetastar', 'ombh2', 'ombch2'], False),
    'CMB-compressed-theta-ombh2-marg-nnu':        (['thetastar', 'ombh2'],            True),
    'CMB-compressed-ombh2-ombch2-marg-nnu':       (['ombh2', 'ombch2'],               True),
    'CMB-compressed-theta-ombh2-ombch2-marg-nnu': (['thetastar', 'ombh2', 'ombch2'], True),
}

# CMB — PR3 shift-parameter compression (R, lA, ombh2, omch2)
_PR3_SHIFT_MAP = {
    'CMB-compressed-R-lA':                       (['R', 'lA'],               False),
    'CMB-compressed-R-lA-ombh2':                 (['R', 'lA', 'ombh2'],      False),
    'CMB-compressed-R-lA-ombh2-ombch2':          (['R', 'lA', 'ombh2', 'omch2'], False),
    'CMB-compressed-R-lA-marg-nnu':              (['R', 'lA'],               True),
    'CMB-compressed-R-lA-ombh2-marg-nnu':        (['R', 'lA', 'ombh2'],      True),
    'CMB-compressed-R-lA-ombh2-ombch2-marg-nnu': (['R', 'lA', 'ombh2', 'omch2'], True),
}


@functools.lru_cache(maxsize=2)
def _likelihood_map(return_parameterization=False):
    """Return the class-backed likelihood mapping.

    This is the single place where class-backed likelihood names are declared:
    for each name it records both how to build the desilike likelihood
    instance(s) (``cosmo`` is added by the caller at instantiation time) and the
    cosmological parameterization it requires — see :func:`get_parameterization`.
    Full-shape, file-backed-BAO, and not-yet-implemented CMB names are declared
    separately (``_FS_TRACERS``, ``_BAO_MEASUREMENT_FILES``, ``_CMB_NOT_IMPLEMENTED``)
    since they aren't built through this ``(cls, kwargs)`` path.

    If *return_parameterization* is ``False`` (default), return the name →
    ``[(cls, init_kwargs), ...]`` mapping used to build desilike likelihood
    instances. If ``True``, return the name → parameterization mapping instead
    (see :func:`get_parameterization`).
    """
    from desilike.likelihoods.bao import DESIDR2BAOLikelihood
    from desilike.likelihoods.bbn import Schoneberg2024BBNLikelihood, BaseBBNLikelihood
    import desilike.likelihoods.supernovae as _sn_mod
    from desilike.likelihoods.cmb.candl import (
        PlanckPR3LowlTTLikelihood, PlanckPR3LowlEELikelihood, PlanckPR3LowlEESroll2Likelihood,
        PlanckPR3TTLikelihood, PlanckPR3TTTEEELikelihood, PlanckPR3TTTEEELiteLikelihood,
        ACTDR6TTTEEELikelihood, SPT3GD1TnELikelihood,
    )
    from desilike.likelihoods.cmb import (
        TTHighlPlanckNPIPECamspecLikelihood, TTTEEEHighlPlanckNPIPECamspecLikelihood,
        TTTEEEHighlPlanckNPIPECamspecEllMax600Likelihood, TTTEEEHighlPlanckNPIPECamspecCutsForACTLikelihood,
        ACTDR6SPTLensingLikelihood,
    )
    from desilike.likelihoods.cmb.camspec import CamspecNPIPELiteLikelihood
    from cosmo.desilike.external_likelihoods.cmb import (
        PlanckPR3ThetaStarFixedNnuLikelihood, PlanckPR3ThetaStarVariedNnuLikelihood,
        PlanckPR3ThetaStarMargNnuLikelihood, PlanckPR3RdragLikelihood,
        PlanckPR4StandardCompressionLikelihood, PlanckPR3ShiftParameterCompressionLikelihood,
    )

    m = {}

    for bao_name, zbins in _BAO_ZBINS.items():
        m[bao_name] = ('background', [(DESIDR2BAOLikelihood, {'zbins': zbins})])

    m['schoneberg2024-bbn'] = ('background', [(Schoneberg2024BBNLikelihood, {})])

    for sn_name, (cls_name, zrange) in _SN_MAP.items():
        m[sn_name] = ('background', [(getattr(_sn_mod, cls_name), {'zrange': zrange})])

    for alias in ('planck2018-lowl-TT', 'planck2018-lowl-TT-clik', 'planck2018-lowl-TT-11-29-clik'):
        m[alias] = ('cmb', [(PlanckPR3LowlTTLikelihood, {})])
    for alias in ('planck2018-lowl-EE', 'planck2018-lowl-EE-clik'):
        m[alias] = ('cmb', [(PlanckPR3LowlEELikelihood, {})])
    m['planck2018-lowl-EE-sroll2'] = ('cmb', [(PlanckPR3LowlEESroll2Likelihood, {})])

    m['planck2018-highl-plik-TT'] = ('cmb', [(PlanckPR3TTLikelihood, {})])
    m['planck2018-highl-plik-TTTEEE'] = ('cmb', [(PlanckPR3TTTEEELikelihood, {})])
    m['planck2018-highl-plik-TTTEEE-lite'] = ('cmb', [(PlanckPR3TTTEEELiteLikelihood, {})])

    m['planck-NPIPE-highl-CamSpec-TT'] = ('cmb', [(TTHighlPlanckNPIPECamspecLikelihood, {})])
    m['planck-NPIPE-highl-CamSpec-TTTEEE'] = ('cmb', [(TTTEEEHighlPlanckNPIPECamspecLikelihood, {})])
    m['planck-NPIPE-highl-CamSpec-TTTEEE-ell-max-600'] = ('cmb', [(TTTEEEHighlPlanckNPIPECamspecEllMax600Likelihood, {})])
    m['planck-NPIPE-highl-CamSpec-TTTEEE-cuts-for-act'] = ('cmb', [(TTTEEEHighlPlanckNPIPECamspecCutsForACTLikelihood, {})])

    # Planck plik-lite/ACT/SPT-3G ell ranges chosen to avoid overlap between datasets
    _cmb_spa_entries = [
        (PlanckPR3LowlTTLikelihood, {}),
        (PlanckPR3LowlEESroll2Likelihood, {}),
        (PlanckPR3TTTEEELiteLikelihood, {'data_selection': ['TT ell<1001 only', 'TE ell<601 only', 'EE ell<601 only']}),
        (ACTDR6TTTEEELikelihood, {'data_selection': ['ell>600 only']}),
        (SPT3GD1TnELikelihood, {'variant': 'lite'}),
        (ACTDR6SPTLensingLikelihood, {'variant': 'actplanckspt3g_baseline'}),
    ]
    m['CMB-SPA'] = ('cmb', _cmb_spa_entries)
    m['CMB-SPA-tauprior'] = ('cmb', _cmb_spa_entries + [
        (BaseBBNLikelihood, {'mean': [0.051], 'covariance': [[0.006 ** 2]], 'quantities': ['tau_reio']}),
    ])

    m['planck2018-thetastar-fixed-nnu'] = ('background', [(PlanckPR3ThetaStarFixedNnuLikelihood, {})])
    m['planck2018-thetastar-varied-nnu'] = ('background', [(PlanckPR3ThetaStarVariedNnuLikelihood, {})])
    m['planck2018-thetastar-fixed-marg-nnu'] = ('background', [(PlanckPR3ThetaStarMargNnuLikelihood, {})])
    m['planck2018-rdrag-fixed-nnu'] = ('background', [(PlanckPR3RdragLikelihood, {})])

    for cname, (observables, inflate_cov) in _PR4_STANDARD_MAP.items():
        m[cname] = ('background', [(PlanckPR4StandardCompressionLikelihood, {'observables': observables, 'inflate_cov': inflate_cov})])
    for cname, (observables, inflate_cov) in _PR3_SHIFT_MAP.items():
        m[cname] = ('background', [(PlanckPR3ShiftParameterCompressionLikelihood, {'observables': observables, 'inflate_cov': inflate_cov})])

    m['CMB-SP4A'] = ('cmb', [
        (CamspecNPIPELiteLikelihood, {'ell_cuts': {'TT': [30, 1500], 'TE': [30, 1000], 'EE': [30, 600]}}),
        (ACTDR6TTTEEELikelihood, {'ell_cuts': {'TT': [1500, 6500], 'TE': [1000, 6500], 'EE': [600, 6500]}}),
        (SPT3GD1TnELikelihood, {'variant': 'lite'}),
    ])

    m['act-dr6-lensing'] = ('cmb', [(ACTDR6SPTLensingLikelihood, {'variant': 'act_baseline'})])
    m['planck-act-dr6-lensing'] = ('cmb', [(ACTDR6SPTLensingLikelihood, {'variant': 'actplanck_baseline'})])

    if return_parameterization:
        return {name: parameterization for name, (parameterization, _) in m.items()}
    return {name: entries for name, (_, entries) in m.items()}


_PARAMETERIZATION_PRIORITY = {'background': 0, 'lss': 1, 'cmb': 2}


def get_parameterization(likelihoods=None, dataset=None):
    """Return the cosmological parameterization required by likelihoods.

    Priority order: ``'cmb'`` > ``'lss'`` > ``'background'``. The highest-priority
    parameterization across all listed likelihoods is returned.

    Self-contained: does not depend on ``cosmo.cobaya.mapping_likelihoods``.
    Class-backed likelihoods (BAO/SN/BBN/CMB) get their parameterization straight
    from :func:`_likelihood_map`, the single place they are declared. Likelihoods
    built through a different path — full-shape, file-backed BAO, not-yet-implemented
    CMB spectra — are tagged from their own dedicated name tables instead.
    """
    if likelihoods is not None and dataset is not None:
        raise ValueError('Pass either likelihoods or dataset, not both.')
    names = likelihoods if likelihoods is not None else dataset
    if names is None:
        names = 'desi-dr2-bao-all'
    names = [names] if isinstance(names, str) else list(names)
    registry = dict(_likelihood_map(return_parameterization=True))
    for name in _BAO_MEASUREMENT_FILES:
        registry[name] = 'background'
    for name in _FS_TRACERS:
        registry[name] = 'lss'
    for name in _CMB_NOT_IMPLEMENTED:
        registry[name] = 'cmb'
    unknown = [name for name in names if name not in registry]
    if unknown:
        raise ValueError('Unknown likelihood(s): {}. Known likelihoods are {}.'.format(
            ', '.join(unknown), ', '.join(sorted(registry))))
    parameterizations = {registry[name] for name in names}
    return max(parameterizations, key=lambda p: _PARAMETERIZATION_PRIORITY.get(p, -1))


def get_default_engine(likelihoods=None):
    """Return the default cosmology engine for the named likelihoods (used for ``engine=None``).

    Full-shape likelihoods drive the choice, through their theory model
    (``'folpsD'`` when the ``_FS_TRACERS`` entry does not set one, matching
    ``full_shape.tools.fill_fiducial_likelihood_options``):

    * all FS theories are COMET: ``'eisenstein_hu'`` — COMET emulates the power
      spectrum internally from the cosmological parameters, so only cheap
      background quantities are needed;
    * any FolpsD-style FS theory (needs the real linear pk): the
      :data:`DEFAULT_ACE_ENGINE` dict for
      :class:`~desilike.theories.primordial_cosmology.ACECosmology`
      (jaxace w0waCDM background + jaxmapse pk + Capse w0waCDM Cl);
    * no FS likelihood: ``'class'``.
    """
    if isinstance(likelihoods, str):
        likelihoods = [likelihoods]
    models = set()
    for name in likelihoods or []:
        entry = _FS_TRACERS.get(name)
        if entry is not None:
            models.add(entry.get('theory', {}).get('model', 'folpsD'))
    if models and models <= {'comet'}:
        return 'eisenstein_hu'
    if models:
        return dict(DEFAULT_ACE_ENGINE)
    return 'class'


def get_engine_label(engine=None, likelihoods=None):
    """Return a string label for *engine* (e.g. for output paths), resolving
    ``engine=None`` through :func:`get_default_engine`; per-section engine
    dicts (:data:`DEFAULT_ACE_ENGINE`-style) are labelled ``'ace'``."""
    if engine is None:
        engine = get_default_engine(likelihoods)
    return engine if isinstance(engine, str) else 'ace'


def install_likelihoods(names, **installer_kwargs):
    """Download and install data files required by the named likelihood(s).

    Parameters
    ----------
    names : str or list of str
        Likelihood names as defined in the registry.
    **installer_kwargs
        Passed to :class:`desilike.install.Installer`.
    """
    from desilike.install import Installer
    installer = Installer(**installer_kwargs)
    installed = set()
    if isinstance(names, str):
        names = [names]
    for name in names:
        for cls, _ in _likelihood_map().get(name, []):
            if cls not in installed and hasattr(cls, 'install'):
                cls.install(installer)
                installed.add(cls)
        # FS likelihoods: data lives on NERSC disk — no install needed


def bao_likelihood_from_files(mean_fn, cov_fn, cosmo=None, name=None, rs_drag=None):
    """Build a BAO Gaussian likelihood directly from cobaya-style text files.

    Uses the same file format and machinery as :class:`~desilike.likelihoods.bao.DESIDR2BAOLikelihood`
    but accepts arbitrary file paths, making it easy to test alternative measurements
    or redshift binnings without adding entries to the internal registry.

    Parameters
    ----------
    mean_fn : str
        Path to the mean-values file.  Each non-comment line has the format
        ``z  value  quantity``, where *quantity* is one of ``DM_over_rs``,
        ``DH_over_rs``, or ``DV_over_rs``.  Multiple redshifts are supported;
        each distinct redshift becomes a separate observable.
    cov_fn : str
        Path to the covariance file (whitespace-delimited flat list of N² floats,
        reshaped to N×N where N equals the total number of measurement rows in
        *mean_fn*).
    cosmo : BasePrimordialCosmology, optional
        Shared cosmology calculator.  Defaults to
        ``CosmoprimoCosmology(fiducial='DESI')``.
    name : str, optional
        Base name for the observables.  For multi-z files the per-z name is
        ``'{name}/{z_eff}'``.  Defaults to the stem of *mean_fn*.
    rs_drag : Parameter, optional
        Shared ``r_d`` :class:`~desilike.parameter.Parameter` forwarded to each
        :class:`~desilike.theories.galaxy_clustering.template.BAOTheory` (see
        :func:`~cosmo.desilike.parameters.get_cosmology`'s ``cosmo.rs_drag_param``).
        ``None`` (default) computes ``r_d`` from *cosmo* as usual.

    Returns
    -------
    :class:`~desilike.likelihoods.base.ObservablesGaussianLikelihood`
    """
    import os
    from desilike.likelihoods.bao import _read_mean_file, _read_cov_file
    from desilike.observables.galaxy_clustering.compressed import BAOCompressionObservable
    from desilike.likelihoods.base import ObservablesGaussianLikelihood

    if cosmo is None:
        from desilike.theories.primordial_cosmology import CosmoprimoCosmology
        cosmo = CosmoprimoCosmology(fiducial='DESI')
    if name is None:
        name = os.path.splitext(os.path.basename(mean_fn))[0]

    z_groups = _read_mean_file(mean_fn)
    total_params = sum(len(param_names) for _, _, param_names in z_groups)
    covariance = _read_cov_file(cov_fn, total_params)

    observables = []
    for z_eff, meas_values, param_names in z_groups:
        obs_name = f'{name}/{z_eff}' if len(z_groups) > 1 else name
        # A quantity may be measured several times at the same redshift (e.g. F_AP from
        # both BAO and full-shape): BAOCompressionObservable compares each measurement
        # to the same theory prediction (repeated names must be consecutive).
        observables.append(BAOCompressionObservable(
            data=meas_values, parameters=param_names, name=obs_name,
            z=z_eff, cosmo=cosmo, rs_drag=(False if rs_drag is None else rs_drag),
        ))

    return ObservablesGaussianLikelihood(observables, covariance=covariance)


def _fs_likelihood_options(name, cache_dir=None, **kwargs):
    """Build the filled full-shape likelihood options for a registered FS name.

    Combines the ``_FS_TRACERS`` entry with *kwargs* overrides, builds one
    likelihood-options dictionary per tracer through
    ``generate_likelihood_options_helper``, then applies the entry's post-helper
    tweaks: 'select' (stat kind → k-range select list), 'theory' (merged into each
    observable's theory options) and 'covariance_options' (merged into the
    covariance options). For Abacus versions, BGS tracers fall back to the
    second-generation mocks ('abacus-2ndgen-dr2-altmtl' with 'holi-bgs-altmtl'
    covariance), as in full_shape/job_scripts/validation_systematic_templates.py.
    """
    from full_shape.tools import generate_likelihood_options_helper, fill_fiducial_options
    helper_kwargs = {**_FS_TRACERS[name], **kwargs}
    tracer = helper_kwargs.pop('tracer', None)
    select = helper_kwargs.pop('select', {})
    theory = helper_kwargs.pop('theory', {})
    covariance_options = helper_kwargs.pop('covariance_options', {})
    if tracer is None:
        tracers = _FS_ALL_TRACERS
    elif isinstance(tracer, str):
        tracers = [tracer]
    else:
        tracers = list(tracer)
    likelihoods_options = []
    for tracer in tracers:
        tracer_kwargs = dict(helper_kwargs)
        if 'BGS' in tracer and 'abacus' in tracer_kwargs.get('version', ''):
            tracer_kwargs['version'] = 'abacus-2ndgen-dr2-altmtl'
            if 'holi' in str(tracer_kwargs.get('covariance', '')):
                tracer_kwargs['covariance'] = 'holi-bgs-altmtl'
        likelihood_options = generate_likelihood_options_helper(tracer=tracer, **tracer_kwargs)
        for observable_options in likelihood_options['observables']:
            kind = observable_options['stat']['kind']
            if kind in select:
                observable_options['stat']['select'] = [dict(item) for item in select[kind]]
            observable_options['theory'] = {**observable_options.get('theory', {}), **theory}
        likelihood_options['covariance'].update(covariance_options)
        likelihoods_options.append(likelihood_options)
    options = fill_fiducial_options({'likelihoods': likelihoods_options})
    if cache_dir is None:  # full_shape.tools requires a cache_dir to build emulators
        for likelihood_options in options['likelihoods']:
            for observable_options in likelihood_options['observables']:
                observable_options['emulator'] = {'name': ''}
    return options


def get_likelihood(name, cosmo=None, **kwargs):
    """Return the desilike likelihood instance(s) for a named likelihood.

    Parameters
    ----------
    name : str
        Likelihood name as defined in ``cosmo/cobaya/mapping_likelihoods.py``,
        e.g. ``'desi-dr2-bao-all'``, ``'pantheonplus'``, ``'planck2018-lowl-EE-sroll2'``.
    cosmo : BasePrimordialCosmology, optional
        Shared cosmology calculator.  When ``None`` each likelihood constructs
        its own default (``CosmoprimoCosmology(fiducial='DESI')``).
    **kwargs
        Full-shape names only: overrides for the ``_FS_TRACERS`` entry, plus
        ``cache_dir`` — directory caching prepared stats and emulators
        (default :data:`DEFAULT_FS_CACHE_DIR`; pass ``None`` to disable
        caching, which also disables the emulator).

    Returns
    -------
    likelihood or list of likelihoods
        A single desilike likelihood instance, or a list of instances for names
        that decompose into multiple independent likelihoods.

    Raises
    ------
    ValueError
        If *name* is not in the known registry.
    NotImplementedError
        If *name* is known but has no desilike equivalent yet.
    """
    # BAO-alone/SN-alone background fits sample r_d directly rather than through cosmo
    # (see get_cosmology's constrain_rd branch); forward that shared Parameter to every
    # BAO-flavored likelihood construction below so they all read the same r_d.
    rs_drag_param = getattr(cosmo, 'rs_drag_param', None)

    entries = _likelihood_map().get(name)
    if entries is not None:
        from desilike.likelihoods.bao import DESIDR2BAOLikelihood
        likelihoods = []
        for cls, init_kwargs in entries:
            call_kwargs = dict(init_kwargs)
            if rs_drag_param is not None and cls is DESIDR2BAOLikelihood:
                call_kwargs['rs_drag'] = rs_drag_param
            likelihoods.append(cls(cosmo=cosmo, **call_kwargs))
        return likelihoods[0] if len(likelihoods) == 1 else likelihoods

    if name in _CMB_NOT_IMPLEMENTED:
        raise NotImplementedError(f'No desilike equivalent for likelihood {name!r}.')

    # ------------------------------------------------------------------
    # File-backed BAO likelihoods (external measurement files, no install)
    # ------------------------------------------------------------------
    if name in _BAO_MEASUREMENT_FILES:
        mean_fn, cov_fn = _BAO_MEASUREMENT_FILES[name]
        return bao_likelihood_from_files(mean_fn, cov_fn, cosmo=cosmo, name=name, rs_drag=rs_drag_param)

    # ------------------------------------------------------------------
    # Full-shape (FS) — uses full_shape.tools API, not a simple cls(**kwargs)
    # ------------------------------------------------------------------
    if name in _FS_TRACERS:
        from full_shape.tools import get_likelihood as _get_fs_likelihood
        # Prepared stats and emulators are cached under cache_dir (pass
        # cache_dir=None explicitly to disable caching and the emulator).
        cache_dir = kwargs.pop('cache_dir', DEFAULT_FS_CACHE_DIR)
        from desilike.theories import ACECosmology
        if isinstance(cosmo, ACECosmology):
            # ACECosmology is already emulator-based (pure JAX): no theory emulator.
            kwargs.setdefault('emulator', False)
        options = _fs_likelihood_options(name, cache_dir=cache_dir, **kwargs)
        cosmology_options = cosmo if cosmo is not None else options.get('cosmology')
        return _get_fs_likelihood(options['likelihoods'], cosmology_options=cosmology_options, cache_dir=cache_dir)

    try:
        from cosmo.cobaya.mapping_likelihoods import LIKELIHOOD_REGISTRY
        if name in LIKELIHOOD_REGISTRY:
            raise NotImplementedError(f'No desilike equivalent for likelihood {name!r} (yet) --- just ask!.')
        known = sorted(LIKELIHOOD_REGISTRY)
    except ImportError:
        known = list(_BAO_ZBINS) + list(_SN_MAP) + list(_CMB_NOT_IMPLEMENTED)
    raise ValueError(f'Unknown likelihood {name!r}. Known names: {known}.')
