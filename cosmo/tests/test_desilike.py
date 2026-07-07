"""Integration tests for the desilike cosmo pipeline."""

from pathlib import Path

import pytest

from cosmo.desilike.run import get_posterior, sample_desilike
from cosmo.desilike.mapping_likelihoods import _BAO_ZBINS, _SN_MAP


def _install_all():
    from desilike.install import Installer
    import desilike.likelihoods.supernovae as sn_mod
    from desilike.likelihoods.bao import DESIDR2BAOLikelihood
    installer = Installer()
    DESIDR2BAOLikelihood.install(installer)
    for cls_name in {v[0] for v in _SN_MAP.values()}:
        getattr(sn_mod, cls_name).install(installer)


@pytest.fixture(scope='session', autouse=True)
def install_data():
    _install_all()


LIKELIHOODS = ['desi-dr2-bao-all', 'desdovekie']
ENGINE = 'eisenstein_hu'

# All likelihoods compatible with eisenstein_hu (no CMB Cl needed).
_LIKELIHOODS_EH = (
    list(_BAO_ZBINS) +
    ['schoneberg2024-bbn'] +
    list(_SN_MAP) +
    [
        # CMB compressed — ombh2/ombch2 only (EH provides these directly)
        'CMB-compressed-ombh2', 'CMB-compressed-ombch2',
        'CMB-compressed-ombh2-ombch2', 'CMB-compressed-ombh2-ombch2-marg-nnu',
        # rdrag prior (EH computes the sound horizon)
        'planck2018-rdrag-fixed-nnu',
        # theta_star / R / lA need full Boltzmann thermodynamics — skip with eisenstein_hu
    ]
)


def test_posterior():
    posterior = get_posterior(LIKELIHOODS, engine=ENGINE)
    assert posterior is not None


def test_sample(tmp_path):
    posterior = get_posterior(LIKELIHOODS, engine=ENGINE)
    init = {'rng': 42, 'nparallel': 1}
    run = {'min_steps': 5, 'gelman_rubin': None, 'ess': None}
    samples = sample_desilike(posterior, kernel='emcee', init=init, run=run,
                              output_dir=tmp_path)
    assert samples is not None


@pytest.mark.parametrize('name', _LIKELIHOODS_EH)
def test_gradient(name):
    import jax
    import jax.numpy as jnp
    from desilike.base import get_params
    posterior = get_posterior([name], engine=ENGINE)
    params = {p.name: jnp.asarray(p._value) for p in get_params(posterior)}
    val, grad = jax.value_and_grad(posterior)(params)
    assert jnp.isfinite(val), f'logpdf not finite: {float(val)}'
    for pname, g in grad.items():
        assert jnp.isfinite(g), f'grad[{pname}] not finite: {float(g)}'


if __name__ == '__main__':
    import tempfile
    _install_all()
    test_posterior()
    with tempfile.TemporaryDirectory() as tmp:
        test_sample(Path(tmp))
    for name in _LIKELIHOODS_EH:
        print(f'Testing gradient for {name!r}...')
        test_gradient(name)
    print('All gradient tests passed.')
