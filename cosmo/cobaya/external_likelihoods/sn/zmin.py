"""SN zmin variants used in DESI cosmology comparisons."""

from cobaya.likelihoods.sn.desy5 import DESy5
from cobaya.likelihoods.sn.pantheonplus import PantheonPlus
from cobaya.likelihoods.sn.union3 import Union3


class PantheonPlusZmin(PantheonPlus):
    """Pantheon+ SN likelihood with configurable zmin."""

    zmin: float = 0.0

    def configure(self):
        self._apply_mask(zmask=self.zcmb > self.zmin)
        self.pre_vars = 0.0


class Union3Zmin(Union3):
    """Union3 SN likelihood with configurable zmin."""

    zmin: float = 0.0

    def configure(self):
        self._apply_mask(zmask=self.zcmb > self.zmin)
        self.pre_vars = 0.0


class DESY5Zmin(DESy5):
    """DESY5 SN likelihood with configurable zmin."""

    zmin: float = 0.0

    def configure(self):
        self._apply_mask(zmask=self.zcmb > self.zmin)
        self.pre_vars = self.mag_err**2
