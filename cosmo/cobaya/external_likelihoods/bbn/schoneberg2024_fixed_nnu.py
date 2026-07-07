"""Schoneberg 2024 BBN likelihood with fixed Neff."""

from .bbn import BBN


class schoneberg2024_fixed_nnu(BBN):
    r"""BBN :math:`\omega_b h^2` constraint with fixed :math:`N_\mathrm{eff}`."""

    quantities = ['omegabh2']
    mean = [0.02218]
    cov = [[3.025e-7]]
    aliases = []
    speed = 4500
