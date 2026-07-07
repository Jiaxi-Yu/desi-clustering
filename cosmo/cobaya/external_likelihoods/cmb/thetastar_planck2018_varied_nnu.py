"""Planck 2018 theta-star prior with varied Neff."""

from .thetastar import thetastar


class thetastar_planck2018_varied_nnu(thetastar):
    r"""Planck 2018 :math:`	heta_\star` and :math:`N_\mathrm{eff}` constraint."""

    quantities = ['thetastar100', 'nnu']
    mean = [1.041452223230532, 2.8943778320386477]
    cov = [[2.88190908e-07, -8.30199834e-05],
           [-8.30199834e-05, 3.52915264e-02]]
    aliases = []
    speed = 4500
