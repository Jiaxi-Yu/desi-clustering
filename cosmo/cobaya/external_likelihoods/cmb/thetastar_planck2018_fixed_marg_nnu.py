"""Planck 2018 theta-star prior with marginal Neff uncertainty."""

from .thetastar import thetastar


class thetastar_planck2018_fixed_marg_nnu(thetastar):
    r"""Planck 2018 :math:`	heta_\star` measurement with base-Neff error."""

    quantities = ['thetastar100']
    mean = [1.04110]
    cov = [[2.809e-7]]
    aliases = []
    speed = 4500
