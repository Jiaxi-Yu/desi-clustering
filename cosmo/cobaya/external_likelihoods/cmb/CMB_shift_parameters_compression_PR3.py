"""Planck PR3 shift-parameter compressed-CMB likelihood."""

from .CMB_compressed import CMB_compressed


class CMB_shift_parameters_compression_PR3(CMB_compressed):
    r"""Planck 2018 compressed measurements on R, lA, omegabh2, and omegach2."""

    compression_type = 'shift_parameters'
    means = [[1.75044145e+00, 3.01758927e+02, 2.23714354e-02, 1.20112045e-01]]
    covs = [[1.56398206e-05, 1.46122171e-04, -3.65381050e-07, 4.56358860e-06],
            [1.46122171e-04, 7.64994349e-03, -3.72384417e-06, 3.01192868e-05],
            [-3.65381050e-07, -3.72384417e-06, 2.09989601e-08, -9.29992287e-08],
            [4.56358860e-06, 3.01192868e-05, -9.29992287e-08, 1.37017300e-06]]
    observables = ['R', 'lA', 'ombh2', 'omch2']
    inflate_cov = False
    aliases = ['CMB']
    speed = 4500
