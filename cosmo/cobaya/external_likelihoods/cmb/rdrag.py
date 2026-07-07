"""Gaussian rdrag prior likelihoods."""

from cobaya.likelihood import Likelihood


class rdrag(Likelihood):
    r"""Gaussian likelihood for :math:`r_\mathrm{drag}`."""

    type = 'rdrag'

    def initialize(self):
        self.minus_half_invvar = -0.5 / self.rdrag_std**2

    def get_requirements(self):
        return {'rdrag': None}

    def logp(self, **params_values):
        rdrag_theory = self.provider.get_param('rdrag')
        return self.minus_half_invvar * (rdrag_theory - self.rdrag_mean)**2
