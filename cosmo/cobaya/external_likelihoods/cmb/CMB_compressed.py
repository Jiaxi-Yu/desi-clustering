"""Compressed CMB likelihoods."""

from typing import List

import numpy as np
from cobaya.conventions import Const
from cobaya.likelihood import Likelihood


class CMB_compressed(Likelihood):
    """Gaussian compressed-CMB likelihood."""

    type = 'CMB_compressed'
    means: np.ndarray
    covs: np.ndarray
    compression_type: str
    observables: List[str]

    def initialize(self):
        self.means = np.array([float(x) for x in self.means[0]])
        self.covs = np.array(self.covs, dtype=float)
        if self.compression_type == 'standard':
            all_parameters = ['thetastar', 'ombh2', 'ombch2']
        elif self.compression_type == 'shift_parameters':
            all_parameters = ['R', 'lA', 'ombh2', 'omch2']
        else:
            raise ValueError(f'Unknown CMB compression type {self.compression_type!r}.')
        self.means, self.covs = self.select_means_and_covs(self.means, self.covs,
                                                           all_parameters, self.observables)
        if self.inflate_cov is True:
            factor = 1.7096774193548387
            self.covs = self.covs * factor**2
        self.invcov = np.linalg.inv(np.atleast_2d(self.covs))

    def get_requirements(self):
        if self.compression_type == 'standard':
            return {'thetastar': None, 'ombh2': None, 'omch2': None}
        if self.compression_type == 'shift_parameters':
            return {'DAstar': None, 'rstar': None, 'omegam': None, 'H0': None, 'ombh2': None}
        raise ValueError(f'Unknown CMB compression type {self.compression_type!r}.')

    def select_means_and_covs(self, means, covs, all_parameters, observables):
        indices = [all_parameters.index(param) for param in observables]
        selected_means = np.array(means)[indices]
        selected_covs = np.array(covs)[np.ix_(indices, indices)]
        return selected_means, selected_covs

    def get_theory(self, observable):
        if observable == 'ombh2':
            return self.provider.get_param('ombh2')
        if observable == 'omch2':
            return self.provider.get_param('omch2')
        if observable == 'ombch2':
            return self.provider.get_param('ombh2') + self.provider.get_param('omch2')
        if observable == 'thetastar':
            return self.provider.get_param('thetastar') / 100.0
        if observable == 'R':
            omegam = self.provider.get_param('omegam')
            H0 = self.provider.get_param('H0')
            DAstar = self.provider.get_param('DAstar')
            return np.sqrt(omegam * H0**2) * DAstar * 1000.0 / Const.c_km_s
        if observable == 'lA':
            rstar = self.provider.get_param('rstar')
            DAstar = self.provider.get_param('DAstar')
            return 1000.0 * np.pi * DAstar / rstar
        raise ValueError(f'Unknown compressed-CMB observable {observable!r}.')

    def logp(self, **params_values):
        theory = np.array([self.get_theory(obs) for obs in self.observables])
        diff = theory - self.means
        return -0.5 * diff.T.dot(self.invcov).dot(diff)
