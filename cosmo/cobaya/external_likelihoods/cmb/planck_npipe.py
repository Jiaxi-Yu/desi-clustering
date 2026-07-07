"""Planck NPIPE CamSpec likelihood wrappers."""

from cobaya.likelihoods.planck_NPIPE_highl_CamSpec.TTTEEE import TTTEEE as _TTTEEE
from cobaya.likelihoods.base_classes import planck_2018_CamSpec_python as _camspec_base


class TTTEEENoCache(_TTTEEE):
    """Planck NPIPE CamSpec TTTEEE variant used by the DESI cosmology helpers."""

    def init_params(self, ini, silent=False):
        old_use_cache = _camspec_base.use_cache
        _camspec_base.use_cache = False
        try:
            return super().init_params(ini, silent=silent)
        finally:
            _camspec_base.use_cache = old_use_cache
