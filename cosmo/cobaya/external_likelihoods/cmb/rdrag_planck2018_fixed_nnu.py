"""Planck 2018 rdrag prior with fixed Neff."""

from .rdrag import rdrag


class rdrag_planck2018_fixed_nnu(rdrag):
    r"""Planck 2018 :math:`r_\mathrm{drag}` measurement."""

    rdrag_mean = 147.09
    rdrag_std = 0.26
    aliases = []
    speed = 4500
