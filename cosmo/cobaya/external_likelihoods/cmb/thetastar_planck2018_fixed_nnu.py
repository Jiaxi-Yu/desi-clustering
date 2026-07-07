"""Planck 2018 theta-star prior with fixed Neff."""

from .thetastar import thetastar


class thetastar_planck2018_fixed_nnu(thetastar):
    r"""Planck 2018 :math:`	heta_\star` measurement."""

    quantities = ['thetastar100']
    mean = [1.04110]
    cov = [[9.61e-8]]
    aliases = []
    speed = 4500
