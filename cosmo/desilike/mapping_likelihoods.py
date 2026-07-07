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
# Built on-the-fly via bao_likelihood_from_files; not downloaded by install().
_BAO_MEASUREMENT_FILES_DIR = '/global/cfs/cdirs/desicollab/science/cpe/dr2_fs/lya_fs/likelihood/cobaya/measurements'
_BAO_MEASUREMENT_FILES = {
    'desi-dr2-bao-gqc-lya-fs': (
        'desi_gaussian_bao_ALL_GCcomb_with_new_lyman_alpha_fs_mean.txt',
        'desi_gaussian_bao_ALL_GCcomb_with_new_lyman_alpha_fs_cov.txt',
    ),
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
_FS_STAT_ABBREVS = {
    's2':         ['mesh2_spectrum'],
    's2-s3':      ['mesh2_spectrum', 'mesh3_spectrum'],
    's2-baor':    ['mesh2_spectrum', 'recon_bao'],
    's2-c2r':     ['mesh2_spectrum', 'recon_particle2_correlation'],
    's2-s3-c2r':  ['mesh2_spectrum', 'mesh3_spectrum', 'recon_particle2_correlation'],
    's2-s3-baor': ['mesh2_spectrum', 'mesh3_spectrum', 'recon_bao'],
}

# Individual tracer strings (used by the "all" variants)
_FS_ALL_TRACERS = ['BGS1', 'LRG1', 'LRG2', 'LRG3', 'LRG3xELG1', 'ELG2', 'QSO1']

# DR2 full-shape name → kwargs for generate_likelihood_options_helper.
# Entries without a 'tracer' key represent the "all individual tracers" combination.
_FS_TRACERS = {}
for _stat_abbrev, _stats in _FS_STAT_ABBREVS.items():
    _FS_TRACERS[f'desi-dr2-fs-{_stat_abbrev}-all'] = {'stats': _stats}
    for _short, _tracer in [('bgs', 'BGS1'), ('lrg-z1', 'LRG1'), ('lrg-z2', 'LRG2'),
                             ('lrg-z3', 'LRG3'), ('lrgxelg', 'LRG3xELG1'),
                             ('elg', 'ELG2'), ('qso', 'QSO1')]:
        _FS_TRACERS[f'desi-dr2-fs-{_stat_abbrev}-{_short}'] = {'tracer': _tracer, 'stats': _stats}


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


@functools.lru_cache(maxsize=1)
def _likelihood_map():
    """Return the name → [(cls, init_kwargs)] mapping for all non-FS likelihoods.

    Values are lists of (class, kwargs) pairs; ``cosmo`` is always added by the
    caller at instantiation time.  The result is cached after the first call so
    the imports run only once.
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
        m[bao_name] = [(DESIDR2BAOLikelihood, {'zbins': zbins})]

    m['schoneberg2024-bbn'] = [(Schoneberg2024BBNLikelihood, {})]

    for sn_name, (cls_name, zrange) in _SN_MAP.items():
        m[sn_name] = [(getattr(_sn_mod, cls_name), {'zrange': zrange})]

    for alias in ('planck2018-lowl-TT', 'planck2018-lowl-TT-clik', 'planck2018-lowl-TT-11-29-clik'):
        m[alias] = [(PlanckPR3LowlTTLikelihood, {})]
    for alias in ('planck2018-lowl-EE', 'planck2018-lowl-EE-clik'):
        m[alias] = [(PlanckPR3LowlEELikelihood, {})]
    m['planck2018-lowl-EE-sroll2'] = [(PlanckPR3LowlEESroll2Likelihood, {})]

    m['planck2018-highl-plik-TT'] = [(PlanckPR3TTLikelihood, {})]
    m['planck2018-highl-plik-TTTEEE'] = [(PlanckPR3TTTEEELikelihood, {})]
    m['planck2018-highl-plik-TTTEEE-lite'] = [(PlanckPR3TTTEEELiteLikelihood, {})]

    m['planck-NPIPE-highl-CamSpec-TT'] = [(TTHighlPlanckNPIPECamspecLikelihood, {})]
    m['planck-NPIPE-highl-CamSpec-TTTEEE'] = [(TTTEEEHighlPlanckNPIPECamspecLikelihood, {})]
    m['planck-NPIPE-highl-CamSpec-TTTEEE-ell-max-600'] = [(TTTEEEHighlPlanckNPIPECamspecEllMax600Likelihood, {})]
    m['planck-NPIPE-highl-CamSpec-TTTEEE-cuts-for-act'] = [(TTTEEEHighlPlanckNPIPECamspecCutsForACTLikelihood, {})]

    # Planck plik-lite/ACT/SPT-3G ell ranges chosen to avoid overlap between datasets
    _cmb_spa_entries = [
        (PlanckPR3LowlTTLikelihood, {}),
        (PlanckPR3LowlEESroll2Likelihood, {}),
        (PlanckPR3TTTEEELiteLikelihood, {'data_selection': ['TT ell<1001 only', 'TE ell<601 only', 'EE ell<601 only']}),
        (ACTDR6TTTEEELikelihood, {'data_selection': ['ell>600 only']}),
        (SPT3GD1TnELikelihood, {'variant': 'lite'}),
        (ACTDR6SPTLensingLikelihood, {'variant': 'actplanckspt3g_baseline'}),
    ]
    m['CMB-SPA'] = _cmb_spa_entries
    m['CMB-SPA-tauprior'] = _cmb_spa_entries + [
        (BaseBBNLikelihood, {'mean': [0.051], 'covariance': [[0.006 ** 2]], 'quantities': ['tau_reio']}),
    ]

    m['planck2018-thetastar-fixed-nnu'] = [(PlanckPR3ThetaStarFixedNnuLikelihood, {})]
    m['planck2018-thetastar-varied-nnu'] = [(PlanckPR3ThetaStarVariedNnuLikelihood, {})]
    m['planck2018-thetastar-fixed-marg-nnu'] = [(PlanckPR3ThetaStarMargNnuLikelihood, {})]
    m['planck2018-rdrag-fixed-nnu'] = [(PlanckPR3RdragLikelihood, {})]

    for cname, (observables, inflate_cov) in _PR4_STANDARD_MAP.items():
        m[cname] = [(PlanckPR4StandardCompressionLikelihood, {'observables': observables, 'inflate_cov': inflate_cov})]
    for cname, (observables, inflate_cov) in _PR3_SHIFT_MAP.items():
        m[cname] = [(PlanckPR3ShiftParameterCompressionLikelihood, {'observables': observables, 'inflate_cov': inflate_cov})]

    m['CMB-SP4A'] = [
        (CamspecNPIPELiteLikelihood, {'ell_cuts': {'TT': [30, 1500], 'TE': [30, 1000], 'EE': [30, 600]}}),
        (ACTDR6TTTEEELikelihood, {'ell_cuts': {'TT': [1500, 6500], 'TE': [1000, 6500], 'EE': [600, 6500]}}),
        (SPT3GD1TnELikelihood, {'variant': 'lite'}),
    ]

    m['act-dr6-lensing'] = [(ACTDR6SPTLensingLikelihood, {'variant': 'act_baseline'})]
    m['planck-act-dr6-lensing'] = [(ACTDR6SPTLensingLikelihood, {'variant': 'actplanck_baseline'})]

    return m


def install(names, **installer_kwargs):
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


def bao_likelihood_from_files(mean_fn, cov_fn, cosmo=None, name=None):
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
        observables.append(BAOCompressionObservable(
            data=meas_values, parameters=param_names, name=obs_name,
            z=z_eff, cosmo=cosmo,
        ))

    return ObservablesGaussianLikelihood(observables, covariance=covariance)


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
    entries = _likelihood_map().get(name)
    if entries is not None:
        likelihoods = [cls(cosmo=cosmo, **init_kwargs) for cls, init_kwargs in entries]
        return likelihoods[0] if len(likelihoods) == 1 else likelihoods

    if name in _CMB_NOT_IMPLEMENTED:
        raise NotImplementedError(f'No desilike equivalent for likelihood {name!r}.')

    # ------------------------------------------------------------------
    # File-backed BAO likelihoods (external measurement files, no install)
    # ------------------------------------------------------------------
    if name in _BAO_MEASUREMENT_FILES:
        import os
        mean_fn, cov_fn = _BAO_MEASUREMENT_FILES[name]
        data_dir = kwargs.get('data_dir', _BAO_MEASUREMENT_FILES_DIR)
        return bao_likelihood_from_files(
            os.path.join(data_dir, mean_fn),
            os.path.join(data_dir, cov_fn),
            cosmo=cosmo, name=name,
        )

    # ------------------------------------------------------------------
    # Full-shape (FS) — uses full_shape.tools API, not a simple cls(**kwargs)
    # ------------------------------------------------------------------
    if name in _FS_TRACERS:
        from full_shape.tools import generate_likelihood_options_helper, get_likelihood as _get_fs_likelihood, fill_fiducial_options
        cache_dir = kwargs.get('cache_dir', None)
        helper_kwargs = {**_FS_TRACERS[name], **{k: v for k, v in kwargs.items() if k != 'cache_dir'}}
        tracer = helper_kwargs.pop('tracer', None)
        tracers = _FS_ALL_TRACERS if tracer is None else [tracer]
        options = {'likelihoods': [generate_likelihood_options_helper(tracer=t, **helper_kwargs) for t in tracers]}
        options = fill_fiducial_options(options)
        if cache_dir is None:
            for likelihood_options in options['likelihoods']:
                likelihood_options['emulator'] = {'name': ''}
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
