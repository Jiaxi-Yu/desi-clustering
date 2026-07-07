"""Schoneberg 2024 BBN likelihood."""

from .bbn import BBN


class schoneberg2024(BBN):
    r"""BBN joint :math:`\omega_b h^2` and :math:`N_\mathrm{eff}` constraint."""

    quantities = ['omegabh2', 'nnu']
    mean = [0.02196, 2.944]
    cov = [[4.03112260e-07, 7.30390042e-05],
           [7.30390042e-05, 4.52831584e-02]]
    aliases = []
    speed = 4500
