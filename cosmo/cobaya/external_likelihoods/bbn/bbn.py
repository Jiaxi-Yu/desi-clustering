"""Gaussian BBN likelihood base classes."""

import numpy as np
from cobaya.likelihood import Likelihood


class BBN(Likelihood):
    """Gaussian likelihood for BBN-derived parameter constraints."""

    type = 'bbn'

    def initialize(self):
        self.mean = np.ravel(self.mean)
        self.invcov = np.linalg.inv(np.atleast_2d(self.cov))

    def get_requirements(self):
        return {name: None for name in self.quantities}

    def logp(self, **params_values):
        diff = self.mean - np.array([self.provider.get_param(name) for name in self.quantities])
        return -0.5 * diff.dot(self.invcov).dot(diff)
