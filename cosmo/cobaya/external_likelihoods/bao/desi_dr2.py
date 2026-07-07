"""DESI BAO DR2 Cobaya likelihood wrappers.

These classes intentionally contain no data paths. Data and covariance files are
provided by the Cobaya info dictionary built in :mod:`cosmo.cobaya`.
"""

from cobaya.likelihoods.base_classes import BAO


class desi_dr2_bao_all(BAO):
    """DESI BAO DR2 likelihood for all tracers."""


class desi_dr2_bao_bgs_z1(BAO):
    """DESI BAO DR2 likelihood for BGS, 0.1 < z < 0.4."""


class desi_dr2_bao_lrg_z1(BAO):
    """DESI BAO DR2 likelihood for LRG, 0.4 < z < 0.6."""


class desi_dr2_bao_lrg_z2(BAO):
    """DESI BAO DR2 likelihood for LRG, 0.6 < z < 0.8."""


class desi_dr2_bao_lrgpluselg_z1(BAO):
    """DESI BAO DR2 likelihood for LRG+ELG, 0.8 < z < 1.1."""


class desi_dr2_bao_elg_z2(BAO):
    """DESI BAO DR2 likelihood for ELG, 1.1 < z < 1.6."""


class desi_dr2_bao_qso_z1(BAO):
    """DESI BAO DR2 likelihood for QSO, 0.8 < z < 2.1."""


class desi_dr2_bao_lya(BAO):
    """DESI BAO DR2 likelihood for Lyman-alpha."""


class desi_dr2_bao_gqc(BAO):
    """DESI DR2 BAO likelihood for galaxy+QSO combined tracers, excluding Lyman-alpha."""


class desi_dr2_bao_lya_fs(BAO):
    """DESI DR2 BAO likelihood for the updated Lyman-alpha full-shape product."""


class desi_dr2_bao_gqc_lya_fs(BAO):
    """DESI DR2 BAO likelihood for galaxy+QSO combined with updated Lyman-alpha full-shape."""
