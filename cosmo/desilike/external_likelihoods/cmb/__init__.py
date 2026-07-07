"""Compressed CMB likelihoods for desilike.

Mirrors the cobaya compressed CMB likelihoods in
``cosmo/cobaya/external_likelihoods/cmb/``.

All distances from cosmoprimo are in Mpc/h; the formulas below account for
this convention so that the derived shift parameters match the Planck 2018
values.
"""

import jax.numpy as jnp
import numpy as np
from desilike.base import GaussianLikelihood

_C_KM_S = 299792.458  # speed of light in km/s


# ---------------------------------------------------------------------------
# Planck PR4 standard compression data (thetastar, ombh2, ombch2)
# ---------------------------------------------------------------------------

_PR4_STANDARD_OBSERVABLES = ['thetastar', 'ombh2', 'ombch2']

_PR4_STANDARD_MEANS = np.array([0.01041027, 0.02223208, 0.14207901])

_PR4_STANDARD_COV = np.array([
    [6.62099420e-12, 1.24442058e-10, -1.19287532e-09],
    [1.24442058e-10, 2.13441666e-08, -9.40008323e-08],
    [-1.19287532e-09, -9.40008323e-08, 1.48841714e-06],
])


# ---------------------------------------------------------------------------
# Planck PR3 shift-parameter compression data (R, lA, ombh2, omch2)
# ---------------------------------------------------------------------------

_PR3_SHIFT_OBSERVABLES = ['R', 'lA', 'ombh2', 'omch2']

_PR3_SHIFT_MEANS = np.array([1.75044145e+00, 3.01758927e+02, 2.23714354e-02, 1.20112045e-01])

_PR3_SHIFT_COV = np.array([
    [1.56398206e-05, 1.46122171e-04, -3.65381050e-07, 4.56358860e-06],
    [1.46122171e-04, 7.64994349e-03, -3.72384417e-06, 3.01192868e-05],
    [-3.65381050e-07, -3.72384417e-06, 2.09989601e-08, -9.29992287e-08],
    [4.56358860e-06, 3.01192868e-05, -9.29992287e-08, 1.37017300e-06],
])

_INFLATE_COV_FACTOR = 1.7096774193548387


def _default_cosmo():
    from desilike.theories.primordial_cosmology import CosmoprimoCosmology
    return CosmoprimoCosmology(fiducial='DESI')


class _BaseCMBCompressedLikelihood(GaussianLikelihood):
    """Gaussian compressed CMB likelihood.

    Parameters
    ----------
    observables : list of str
        Ordered subset of observable names to include.
    means_data : array_like
        Full mean vector corresponding to *all_observables*.
    covariance_data : array_like
        Full covariance matrix corresponding to *all_observables*.
    all_observables : list of str
        Ordered list of all possible observables for this compression.
    inflate_cov : bool
        If True, inflate the covariance by ``_INFLATE_COV_FACTOR ** 2``.
    cosmo : BasePrimordialCosmology, optional
    """

    def __init__(self, observables, means_data, covariance_data, all_observables,
                 inflate_cov=False, cosmo=None):
        if cosmo is None:
            cosmo = _default_cosmo()
        self.cosmo = cosmo
        self._observables = list(observables)

        indices = [list(all_observables).index(obs) for obs in observables]
        means_full = np.asarray(means_data, dtype=float)
        cov_full = np.asarray(covariance_data, dtype=float)
        selected_means = means_full[indices]
        selected_cov = cov_full[np.ix_(indices, indices)]

        if inflate_cov:
            selected_cov = selected_cov * _INFLATE_COV_FACTOR ** 2

        self.flatdata = jnp.asarray(selected_means)
        self.precision = jnp.asarray(np.linalg.inv(selected_cov))

    def __post_init__(self, *args, **kwargs):
        reqs = {}
        if any(obs in ('thetastar', 'lA', 'R') for obs in self._observables):
            # theta_star_noreion (z_star without reionization) is consistent across
            # CLASS and CAMB; plain theta_star differs by ~10 sigma between engines
            # because CLASS and CAMB define the recombination redshift differently.
            reqs['thermodynamics.rs_drag'] = None
        if 'R' in self._observables:
            reqs['params.Omega_m'] = None
        if any(obs in ('ombh2', 'ombch2') for obs in self._observables):
            reqs['params.omega_b'] = None
        if any(obs in ('omch2', 'ombch2') for obs in self._observables):
            reqs['params.omega_cdm'] = None
        self.cosmo.add_requirements(reqs)

    def _get_observable(self, obs):
        if obs == 'ombh2':
            return self.cosmo['omega_b']
        if obs == 'omch2':
            return self.cosmo['omega_cdm']
        if obs == 'ombch2':
            return self.cosmo['omega_b'] + self.cosmo['omega_cdm']
        if obs in ('thetastar', 'lA', 'R'):
            thermo = self.cosmo._cosmo.get_thermodynamics()
            if obs == 'thetastar':
                return thermo.theta_star_noreion
            bg = self.cosmo._cosmo.get_background()
            chi_star = bg.comoving_angular_distance(thermo.z_star_noreion)  # Mpc/h
            if obs == 'lA':
                return jnp.pi / thermo.theta_star_noreion
            if obs == 'R':
                return jnp.sqrt(self.cosmo['Omega_m']) * 100.0 * chi_star / _C_KM_S
        raise ValueError(f'Unknown compressed-CMB observable {obs!r}.')

    def __call__(self):
        self.flattheory = jnp.array([self._get_observable(obs) for obs in self._observables])
        return super().__call__()


# ---------------------------------------------------------------------------
# PR4 standard compression
# ---------------------------------------------------------------------------

class PlanckPR4StandardCompressionLikelihood(_BaseCMBCompressedLikelihood):
    r"""Planck PR4 standard compressed CMB likelihood.

    Gaussian constraint on a subset of (theta_star, omega_b, omega_b+omega_cdm)
    from Planck PR4 (NPIPE).

    Parameters
    ----------
    observables : list of str, optional
        Subset of ``['thetastar', 'ombh2', 'ombch2']`` to include.
        Defaults to all three.
    inflate_cov : bool
        Inflate the covariance by ``_INFLATE_COV_FACTOR**2`` to account for
        Neff marginalisation uncertainty.
    cosmo : BasePrimordialCosmology, optional
    """

    def __init__(self, observables=None, inflate_cov=False, cosmo=None):
        if observables is None:
            observables = _PR4_STANDARD_OBSERVABLES
        super().__init__(
            observables=observables,
            means_data=_PR4_STANDARD_MEANS,
            covariance_data=_PR4_STANDARD_COV,
            all_observables=_PR4_STANDARD_OBSERVABLES,
            inflate_cov=inflate_cov,
            cosmo=cosmo,
        )


# ---------------------------------------------------------------------------
# PR3 shift-parameter compression
# ---------------------------------------------------------------------------

class PlanckPR3ShiftParameterCompressionLikelihood(_BaseCMBCompressedLikelihood):
    r"""Planck PR3 shift-parameter compressed CMB likelihood.

    Gaussian constraint on a subset of (R, l_A, omega_b, omega_cdm) from
    Planck 2018 (PR3).

    Parameters
    ----------
    observables : list of str, optional
        Subset of ``['R', 'lA', 'ombh2', 'omch2']`` to include.
        Defaults to all four.
    inflate_cov : bool
        Inflate the covariance by ``_INFLATE_COV_FACTOR**2``.
    cosmo : BasePrimordialCosmology, optional
    """

    def __init__(self, observables=None, inflate_cov=False, cosmo=None):
        if observables is None:
            observables = _PR3_SHIFT_OBSERVABLES
        super().__init__(
            observables=observables,
            means_data=_PR3_SHIFT_MEANS,
            covariance_data=_PR3_SHIFT_COV,
            all_observables=_PR3_SHIFT_OBSERVABLES,
            inflate_cov=inflate_cov,
            cosmo=cosmo,
        )


# ---------------------------------------------------------------------------
# Standalone theta_star likelihoods
# ---------------------------------------------------------------------------

class _BaseThetaStarLikelihood(GaussianLikelihood):
    """Gaussian prior on 100*theta_star (and optionally N_eff)."""

    def __init__(self, mean, covariance, quantities, cosmo=None):
        if cosmo is None:
            cosmo = _default_cosmo()
        self.cosmo = cosmo
        self._quantities = list(quantities)
        self.flatdata = jnp.asarray(mean)
        self.precision = jnp.asarray(np.linalg.inv(np.atleast_2d(covariance)))

    def __post_init__(self, *args, **kwargs):
        reqs = {'thermodynamics.rs_drag': None}
        if 'nnu' in self._quantities:
            reqs['params.N_eff'] = None
        self.cosmo.add_requirements(reqs)

    def __call__(self):
        thermo = self.cosmo._cosmo.get_thermodynamics()
        values = []
        for quantity in self._quantities:
            if quantity == 'thetastar100':
                values.append(100.0 * thermo.theta_star_noreion)
            elif quantity == 'nnu':
                values.append(self.cosmo['N_eff'])
            else:
                raise ValueError(f'Unknown thetastar quantity {quantity!r}.')
        self.flattheory = jnp.array(values)
        return super().__call__()


class PlanckPR3ThetaStarFixedNnuLikelihood(_BaseThetaStarLikelihood):
    r"""Planck 2018 :math:`\theta_\star` prior, fixed :math:`N_\mathrm{eff}`.

    mean = 1.04110, std = sqrt(9.61e-8).
    """

    def __init__(self, cosmo=None):
        super().__init__(
            mean=[1.04110],
            covariance=[[9.61e-8]],
            quantities=['thetastar100'],
            cosmo=cosmo,
        )


class PlanckPR3ThetaStarVariedNnuLikelihood(_BaseThetaStarLikelihood):
    r"""Planck 2018 joint :math:`\theta_\star` and :math:`N_\mathrm{eff}` prior."""

    def __init__(self, cosmo=None):
        super().__init__(
            mean=[1.041452223230532, 2.8943778320386477],
            covariance=[[2.88190908e-07, -8.30199834e-05],
                        [-8.30199834e-05, 3.52915264e-02]],
            quantities=['thetastar100', 'nnu'],
            cosmo=cosmo,
        )


class PlanckPR3ThetaStarMargNnuLikelihood(_BaseThetaStarLikelihood):
    r"""Planck 2018 :math:`\theta_\star` prior, marginalized over :math:`N_\mathrm{eff}`.

    mean = 1.04110, std = sqrt(2.809e-7).
    """

    def __init__(self, cosmo=None):
        super().__init__(
            mean=[1.04110],
            covariance=[[2.809e-7]],
            quantities=['thetastar100'],
            cosmo=cosmo,
        )


# ---------------------------------------------------------------------------
# Standalone r_drag likelihood
# ---------------------------------------------------------------------------

class PlanckPR3RdragLikelihood(GaussianLikelihood):
    r"""Planck 2018 :math:`r_\mathrm{drag}` prior, fixed :math:`N_\mathrm{eff}`.

    mean = 147.09 Mpc, std = 0.26 Mpc.
    """

    def __init__(self, cosmo=None):
        if cosmo is None:
            cosmo = _default_cosmo()
        self.cosmo = cosmo
        self.flatdata = jnp.array([147.09])
        self.precision = jnp.array([[1.0 / 0.26 ** 2]])

    def __post_init__(self, *args, **kwargs):
        self.cosmo.add_requirements({'thermodynamics.rs_drag': None})

    def __call__(self):
        thermo = self.cosmo._cosmo.get_thermodynamics()
        # cosmoprimo returns rs_drag in Mpc/h; convert to Mpc
        rs_drag_mpc = thermo.rs_drag / (self.cosmo['H0'] / 100.0)
        self.flattheory = jnp.array([rs_drag_mpc])
        return super().__call__()
