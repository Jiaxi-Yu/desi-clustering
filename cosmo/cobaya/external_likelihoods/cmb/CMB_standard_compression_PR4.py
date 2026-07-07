"""Planck PR4 standard compressed-CMB likelihood."""

from .CMB_compressed import CMB_compressed


class CMB_standard_compression_PR4(CMB_compressed):
    r"""Early-universe LCDM priors from Planck PR4."""

    compression_type = 'standard'
    means = [[0.01041027, 0.02223208, 0.14207901]]
    covs = [[6.62099420e-12, 1.24442058e-10, -1.19287532e-09],
            [1.24442058e-10, 2.13441666e-08, -9.40008323e-08],
            [-1.19287532e-09, -9.40008323e-08, 1.48841714e-06]]
    observables = ['thetastar', 'ombh2', 'ombch2']
    inflate_cov = False
    aliases = ['CMB']
    speed = 4500
