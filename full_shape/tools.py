"""
In this file we propose a dictionary-like interface to building :mod:`desilike` likelihoods.

Important functions are:
- :func:`generate_likelihood_options_helper`, helper to generate dictionary of options
- :func:`get_stats`: read clustering statistics from disk
- :func:`get_theory`: build desilike theory
- :func:`get_single_likelihood` return single desilike likelihood (that can be summed up with others)
- :func:`get_fits_fn`: proposed path to output fits.

Dictionary of options are organized as follows:
- list of dictionaries, one for each independent (summed) likelihood;
- each of this dictionary is {'observables': [observable1, observable2, ...], 'covariance': covariance options}
- each of observable1, observable2, ... is a dictionary that specifies how to build the desilike
observable (data, theory, window): ``{'stat': {'kind': ..., 'basis': ..., 'select': [...]},
'catalog': {'version':, ...}, 'theory': {'model': ...}, 'window': {}}``.

Optional per-observable key ``'redshift_smearing'`` (default: absent, no smearing), either a
mode string or ``{'mode': ..., **kw}`` where the remaining keys locate the PDF through
``get_stats_fn`` (e.g. ``{'mode': 'pdf', 'version': 'data-dr2-v2'}``). ``'pdf'`` folds the
measured redshift-error PDF into the window matrix; ``'pdf+gauss'`` / ``'pdf+lor'``
additionally give the theory a parametric damping with a free ``vsmear`` [Mpc/h], which
folpsD / folpsEFT alone support. Normalised once in :func:`fill_fiducial_observable_options`.
"""

import os
import re
import json
import hashlib
import logging
import numbers
import itertools
import warnings
from pathlib import Path

import numpy as np
import scipy as sp
import lsstypes as types
from lsstypes.utils import get_hartlap2007_factor, get_percival2014_factor, mkdir

from clustering_statistics.tools import (write_stats, float2str, get_full_tracer, get_simple_tracer, _make_tuple,
                                         get_simple_stats, _unzip_catalog_options, default_mpicomm,
                                         rebinning_matrix, setup_logging)
from clustering_statistics import tools as clustering_tools
from full_shape import box_tools


logger = logging.getLogger('tools')
base_fits_dir = Path('/global/cfs/cdirs/desi/science/cai/desi-clustering/dr2/fits/')
base_stats_dir = Path('/dvs_ro') / clustering_tools.base_stats_dir.relative_to('/global')


def _json_attr(value):
    return json.dumps(value, sort_keys=True)


def _get_covariance_display_version(covariance: dict):
    """Return a readable covariance version label for cache filenames."""
    version = covariance.get('version', 'none')
    if (covariance.get('source') == 'mock') and version is None:
        stats_dir = covariance.get('stats_dir', None)
        if stats_dir is not None:
            try:
                parts = Path(stats_dir).parts
                if len(parts) >= 2 and parts[-2:] == ('mock_challenge', 'ezmock'):
                    return 'ezmock-mc'
            except TypeError:
                pass
    return 'none' if version is None else str(version)


_fiducial = None

def get_fiducial():
    global _fiducial
    if _fiducial is None:
        from cosmoprimo.fiducial import DESI
        _fiducial = DESI()
    return _fiducial


def get_cosmology(cosmology_options: dict=None):
    """
    Construct and return a :mod:`desilike` :class:`CosmoprimoCosmology` calculator.

    Returns
    -------
    cosmo : :class:`desilike.theories.galaxy_clustering.CosmoprimoCosmology`
        Instance with configured priors.
    """
    from desilike import VariableCollection, Parameter
    from desilike.theories import PrimordialCosmology, CosmoprimoCosmology
    if isinstance(cosmology_options, PrimordialCosmology):  # already-constructed calculator (CosmoprimoCosmology, ACECosmology, ...)
        return cosmology_options
    cosmology_options = cosmology_options or {}
    model = cosmology_options.get('model', 'base_ns-fixed')
    is_fixed_model = model == 'fixed'
    is_ns_fixed = is_fixed_model or 'ns-fixed' in model
    has_w0wa = any(w in model for w in ['w_wa', 'w0wa'])
    fiducial = get_fiducial()

    params = VariableCollection()
    params.set(Parameter('h', value=fiducial['h'],
                            #prior=dict(limits=[0.2, 1.0]),
                            prior=dict(limits=[0.5, 1.0]),
                            ref=dict(dist='norm', loc=fiducial['h'], scale=0.01),
                            fd_eps=0.03, latex='h'))
    params.set(Parameter('omega_b', value=fiducial['omega_b'],
                            prior=dict(dist='norm', loc=0.02237, scale=0.00055),
                            ref=dict(dist='norm', loc=fiducial['omega_b'], scale=0.0003),
                            fd_eps=0.0015, latex=r'\omega_b'))
    params.set(Parameter('omega_cdm', value=fiducial['omega_cdm'],
                            prior=dict(limits=[0.01, 0.99]),
                            ref=dict(dist='norm', loc=fiducial['omega_cdm'], scale=0.005),
                            fd_eps=0.01, latex=r'\omega_\mathrm{cdm}'))
    params.set(Parameter('logA', value=fiducial['logA'],
                            prior=dict(limits=[1.61, 3.91]),
                            ref=dict(dist='norm', loc=fiducial['logA'], scale=0.1),
                            fd_eps=0.05, latex=r'\ln(10^{10}A_s)'))
    params.set(Parameter('n_s', value=fiducial['n_s'],
                            prior=dict(dist='norm', loc=0.9649, scale=0.042),
                            ref=dict(dist='norm', loc=fiducial['n_s'], scale=0.1),
                            fd_eps=0.005, latex='n_s'))
    params.set(Parameter('m_ncdm', value=fiducial['m_ncdm_tot'], fixed=True,
                            prior=dict(limits=[0., 5.]),
                            ref=dict(dist='norm', loc=fiducial['m_ncdm_tot'], scale=0.12, limits=[0., 10.]),
                            fd_eps=(0.31, 0.15, 0.15), latex=r'm_\mathrm{ncdm}'))
    params.set(Parameter('N_ur', value=fiducial['N_ur'], fixed=True,
                            prior=dict(limits=[0.05, 10.]),
                            ref=dict(dist='norm', loc=fiducial['N_ur'], scale=0.2, limits=[0., 10.]),
                            fd_eps=(0.31, 0.15, 0.15), latex=r'm_\mathrm{ncdm}'))
    params.set(Parameter('w0_fld', value=fiducial['w0_fld'], fixed=True,
                            prior=dict(limits=[-3., 1.]),
                            ref=dict(dist='norm', loc=-1., scale=0.08),
                            fd_eps=0.1, latex=r'w_0'))
    params.set(Parameter('wa_fld', value=fiducial['wa_fld'], fixed=True,
                            prior=dict(limits=[-3., 2.]),
                            ref=dict(dist='norm', loc=0., scale=0.3),
                            fd_eps=0.3, latex=r'w_a'))
    if is_fixed_model:
        for param in params:
            param.update(fixed=True)
    params['n_s'].update(fixed=is_ns_fixed)
    if has_w0wa:
        params['w0_fld'].update(fixed=is_fixed_model)
        params['wa_fld'].update(fixed=is_fixed_model)
    # H0 and Omega_m follow from the sampled parameters alone; sigma8_m, sigma8_cb and rs_drag
    # each require a Boltzmann section (fourier, thermodynamics) to be run. Restrict the set with
    # cosmology_options['derived'] --- comet needs none of the sections, so asking for sigma8 /
    # rs_drag there would only add a computation the theory itself never uses.
    derived = {'H0': 'H_0', 'Omega_m': r'\Omega_\mathrm{m}', 'sigma8_m': r'\sigma_{8,\mathrm{m}}',
               'sigma8_cb': r'\sigma_{8,\mathrm{cb}}', 'rs_drag': 'r_s'}
    requested = cosmology_options.get('derived', None)
    if requested is not None:
        unknown = [name for name in requested if name not in derived]
        if unknown:
            raise ValueError(f'unknown derived cosmology parameters {unknown}; choose from {list(derived)}')
        derived = {name: derived[name] for name in requested}
    for name, latex in derived.items():
        params.set(Parameter(name, derived=True, latex=latex))
    cosmo = CosmoprimoCosmology(engine=cosmology_options.get('engine', 'class'), fiducial=fiducial, params=params)
    return cosmo


def _get_default_ref_from_prior(prior, value=None):
    """Build a compact reference distribution from a prior for sampler initialization."""
    if not prior:
        return None

    dist = prior.get('dist', None)
    limits = prior.get('limits', [-np.inf, np.inf])
    if limits is None:
        limits = [-np.inf, np.inf]
    limits = list(limits)

    if dist == 'norm':
        scale = prior.get('scale', None)
        if scale is None or scale <= 0:
            return None
        return {
            'dist': 'norm',
            'loc': prior.get('loc', value if value is not None else 0.),
            'scale': scale / 5.,
            'limits': limits,
        }

    if dist == 'uniform':
        if len(limits) != 2 or not np.all(np.isfinite(limits)):
            return None
        lo, hi = limits
        return {
            'dist': 'norm',
            'loc': value if value is not None else 0.5 * (lo + hi),
            'scale': (hi - lo) / 20.,
            'limits': [lo, hi],
        }

    return None


def update_theory_nuisance_priors(params, model, stat, prior_basis, coevolution='', tracer=None, marg=False, ells=None, user_params=None):
    """
    Apply default nuisance-parameter priors to a VariableCollection in-place.

    Parameters
    ----------
    params : VariableCollection
        Parameter collection for a theory calculator, e.g. ``desilike.get_params(theory, level=1)``.
    model : str
        Perturbation theory model tag. When 'EFT', FoG parameters are fixed.
    stat : str
        Observable; one of ['mesh2_spectrum', 'mesh3_spectrum', 'recon_particle2_correlation'].
    prior_basis : str
        'physical' or 'physical_aap' uses physical bias parameters (b1p, b2p,...).
        Any other value uses the standard Eulerian basis (b1, b2, ...).
    coevolution : str
        If b3 in string, fix b3 to its co-evolution value.
    marg : bool, optional
        If True, set counter-term and shot-noise parameters to ``derived='marg'``.
    user_params : dict, optional
        Per-parameter config dicts that override the defaults.

    Returns
    -------
    params : VariableCollection
        The same collection, with parameters updated in-place.
    """
    configs = {}
    if model == 'bao':
        tracer = get_simple_tracer(tracer)
        recon = bool(prior_basis)
        if tracer == 'BGS':
            sigmapar, sigmaper = 10., 6.5
            if recon: sigmapar, sigmaper = 8., 3.
        elif tracer in ['LRG', 'LGE']:
            sigmapar, sigmaper = 9., 4.5
            if recon: sigmapar, sigmaper = 6., 3.
        elif tracer == 'LRG+ELG':
            sigmapar, sigmaper = 9., 4.5
            if recon: sigmapar, sigmaper = 6., 3.
        elif tracer == 'ELG':
            sigmapar, sigmaper = 8.5, 4.5
            if recon: sigmapar, sigmaper = 6., 3.
        elif tracer == 'QSO':
            sigmapar, sigmaper = 9., 3.5
            if recon: sigmapar, sigmaper = 6., 3.
        sigmas = {'sigmas': (2., 2.), 'sigmapar': (sigmapar, 2.), 'sigmaper': (sigmaper, 1.)}
        for name, value in sigmas.items():
            configs[name] = {'value': value[0], 'prior': {'dist': 'norm', 'loc': value[0], 'scale': value[1], 'limits': [0., 20.]}}
        if marg:
            for param in params.select(basename=f'*l*_*'):
                param.update(derived='marg')
        if user_params:
            configs = configs | user_params
        for basename, config in configs.items():
            for param in params.select(basename=basename):
                param.update(**config, fixed=('prior' not in config))
        for ell in [0, 2, 4]:
            if ell not in ells:
                for param in params.select(basename=f'*l{ell:d}_*'):
                    param.update(fixed=True, derived=False)
        if len(ells) <= 1:
            for param in params.select(basename='dbeta'):
                param.update(fixed=True)
    elif 'folps' in model:
        # These are the default values in desilike, we just repeat them for clarity
        if 'physical' in prior_basis:
            # ── Bias parameters ───────────────────────────────────────────────
            if 'mesh2_spectrum' in stat:
                configs.update({
                    'b1': {'value': 1.5, 'fixed': False, 'prior': {'dist': 'uniform', 'limits': [0.1, 8.]}},
                    'b2': {'value': 0., 'fixed': False, 'prior': {'dist': 'norm', 'loc': 0., 'scale': 20.}},
                    'bs': {'value': 0., 'fixed': False, 'prior': {'dist': 'norm', 'loc': 0., 'scale': 20.}},
                    'b3': {'value': 0., 'fixed': False, 'prior': {'dist': 'norm', 'loc': 0., 'scale': 1.}},
                    'alpha0': {'value': 0., 'fixed': False, 'prior': {'dist': 'norm', 'loc': 0., 'scale': 50.}},
                    'alpha2': {'value': 0., 'fixed': False, 'prior': {'dist': 'norm', 'loc': 0., 'scale': 50.}},
                    'alpha4': {'value': 0., 'fixed': False, 'prior': {'dist': 'norm', 'loc': 0., 'scale': 50.}},
                    'ct': {'value': 0., 'fixed': True, 'prior': {}},
                    'X_FoG': {'value': 0., 'fixed': True, 'prior': {'dist': 'uniform', 'limits': [0, 20]}},
                    'sn0': {'value': 0., 'fixed': False, 'prior': {'dist': 'norm', 'loc': 0., 'scale': 2.}},
                    'sn2': {'value': 0., 'fixed': False, 'prior': {'dist': 'norm', 'loc': 0., 'scale': 5.}},
                })
                if 'b3' in coevolution:
                    configs['b3p'] = {'fixed': True, 'value': 0.}
            elif 'mesh3_spectrum' in stat:
                configs.update({
                    'b1': {'value': 1.5, 'fixed': False, 'prior': {'dist': 'uniform', 'limits': [0.1, 8.]}},
                    'b2': {'value': 0., 'fixed': False, 'prior': {'dist': 'norm', 'loc': 0., 'scale': 20.}},
                    'bs': {'value': 0., 'fixed': False, 'prior': {'dist': 'norm', 'loc': 0., 'scale': 20.}},
                    'c1': {'value': 0., 'fixed': False, 'prior': {'dist': 'norm', 'loc': 0., 'scale': 20.}},
                    'c2': {'value': 0., 'fixed': True, 'prior': {'dist': 'norm', 'loc': 0., 'scale': 20.}},
                    'sn0': {'value': 0., 'fixed': False, 'prior': {'dist': 'norm', 'loc': 0., 'scale': 2.}},
                    'snb0': {'value': 0., 'fixed': False, 'prior': {'dist': 'norm', 'loc': 0., 'scale': 1.}},
                    'X_FoG': {'value': 0., 'fixed': True, 'prior': {'dist': 'uniform', 'limits': [0, 20]}},
                })

        # ── FoG damping ───────────────────────────────────────────────────
        if 'EFT' in model.upper():
            configs['X_FoG*'] = {'fixed': True}
        else:
            configs['X_FoG*'] = {'fixed': False, 'prior': {'dist': 'uniform', 'limits': [0, 20]}}
        if marg:
            for param in params.select(basename=['alpha*', 'sn2', 'sn4']):
                param.update(derived='marg')
        if user_params:
            configs = configs | user_params
        for basename, config in configs.items():
            for param in params.select(basename=basename):
                param.update(**config)
    elif 'comet' in model:
        if marg:
            # 'a[0:5]' / 'NP*' are comet's own (standard-basis) names; under a physical basis it
            # exposes folps' names instead --- alpha0/2/4, sn2, plus the comet-only sn22 --- so
            # select both sets to marginalize the same sector as folps in either basis.
            for param in params.select(basename=['a[0:5]', 'NP*', 'alpha*', 'sn2', 'sn4', 'sn22']):
                param.update(derived='marg')
        if user_params:
            configs = configs | user_params
        for basename, config in configs.items():
            for param in params.select(basename=basename):
                param.update(**config)
    return params


@default_mpicomm
def get_theory(stat: str, theory_options: dict, cosmology: object=None, data_attrs: dict=None, data=None,
               redshift_smearing: str=None, mpicomm=None):
    """
    Return a configured theory desilike calculator for the requested statistic.

    Parameters
    ----------
    stat : str
        Statistic name, e.g. 'mesh2_spectrum' or 'mesh3_spectrum'.
    theory_options : dict
        Theory options dict containing at least 'model' and possibly other keys. If 'z' is provided, data attribute 'z' will be ignored.
    cosmology : Cosmoprimo
        Cosmology calculator.
    data_attrs : dict
        Data attributes ('z', 'recon_mode', 'recon_smoothing_radius', 'tracers', ...).
    redshift_smearing : str, dict, optional
        Redshift-smearing option, see :func:`_format_redshift_smearing`. The measured PDF
        ('pdf') is folded into the window in :func:`get_stats`, not here; only the parametric
        part of 'pdf+gauss' / 'pdf+lor' is handled, as a kernel with a free ``vsmear``
        [Mpc/h] passed to the theory. Supported by folpsD / folpsEFT only.

    Returns
    -------
    theory : BaseCalculator
        Initialized theory object from desilike for the requested statistic.
    """
    from desilike.theories.galaxy_clustering import (DirectSpectrum2Template, ShapeFitSpectrum2Template, BAOSpectrum2Template,
        REPTVelocileptorsTracerSpectrum2Poles, FOLPSTracerSpectrum2Poles, FOLPSPTSpectrum2Poles,
        FOLPSTracerSpectrum3Poles, COMETPTSpectrum2Poles, COMETTracerSpectrum2Poles, COMETPTSpectrum3Poles, COMETTracerSpectrum3Poles,
        DampedBAOWigglesTracerCorrelation2Poles, DampedBAOWigglesPTSpectrum2Poles, DampedBAOWigglesTracerSpectrum2Poles)
    from desilike.theories.galaxy_clustering.full_shape import get_physical_stochastic_settings
    from desilike.base import params as get_params
    theory_options = dict(theory_options)
    # 'pdf' alone is entirely in the window; only the parametric suffix reaches the theory.
    smearing_form = _get_redshift_smearing_form(redshift_smearing)
    if smearing_form is not None and theory_options['model'] not in ['folpsD', 'folpsEFT']:
        raise NotImplementedError(f"redshift_smearing={redshift_smearing!r} adds a parametric damping, which "
                                  f"model {theory_options['model']!r} does not support; use 'pdf' instead.")
    smearing_kw = {'redshift_smearing': _get_redshift_smearing_kernel(smearing_form)} if smearing_form else {}
    fiducial = get_fiducial()
    template = None
    theory_options.setdefault('cosmology', {'template': 'direct'})
    cosmology_options = theory_options['cosmology']
    z = data_attrs['z']
    tracers = data_attrs['tracers']
    nbar = data_attrs['nbar']
    if all(tracer == tracers[0] for tracer in tracers):
        tracers = tracers[0]
    if 'z' in theory_options:
        z = theory_options['z']
    if mpicomm.rank == 0:
        logger.info(f'theory is evaluated at effective redshift {z:.3f}')
    if cosmology_options['template'] == 'direct':
        template = DirectSpectrum2Template(fiducial=fiducial, cosmo=cosmology, z=z)
    elif cosmology_options['template'] == 'shapefit':
        template = ShapeFitSpectrum2Template(fiducial=fiducial, z=z)
    elif cosmology_options['template'] == 'bao':
        kw = {name: cosmology_options[name] for name in ['apmode'] if name in cosmology_options}
        if 'now' in cosmology_options:
            kw['with_now'] = cosmology_options['now']
        template = BAOSpectrum2Template(fiducial=fiducial, z=z, **kw)
    if template is None:
        raise ValueError(f'template not found for {stat} and {repr(cosmology_options["template"])}')
    theory = None
    if 'mesh2_spectrum' in stat:
        if theory_options['model'] == 'reptvelocileptors':
            theory = REPTVelocileptorsTracerSpectrum2Poles(template=template, **theory_options.get('options', {}))
        elif theory_options['model'] in ['folpsD', 'folpsEFT']:
            A_full = theory_options.get('A_full', True)
            pt = FOLPSPTSpectrum2Poles(A_full=A_full)
            if theory_options['model'] == 'folpsD':
                theory_options.setdefault('damping_method', 'tree+loop')
            # use_GTNS is forwarded so the EFT limit can be compared like-for-like with comet: at
            # X_FoG = 0 there is no tree-level damping to resum GTNS, so folps' default (dropped
            # for every 'tree+...' method) is not the model comet's 'EFT' evaluates, which keeps it.
            kw = {name: theory_options[name] for name in ['damping', 'damping_method', 'use_GTNS', 'prior_basis'] if name in theory_options}
            theory = FOLPSTracerSpectrum2Poles(template=template, pt=pt, tracers=tracers, **kw, **smearing_kw, **theory_options.get('options', {}))
            kw_stoch = get_physical_stochastic_settings(tracer=get_simple_tracer(tracers))
            theory.update(**kw_stoch, nbar=nbar, params=update_theory_nuisance_priors(get_params(theory, level=1), theory_options['model'], stat, kw['prior_basis'], marg=theory_options.get('marg', False), user_params=theory_options.get('params') or None))
        elif theory_options['model'] in ['comet', 'cometEFT']:
            kw = {name: theory_options[name] for name in ['prior_basis'] if name in theory_options}
            # 'comet' is comet's fitted velocity-dispersion kernel VDG_infty; 'cometEFT' the pure
            # EFT model, whose FoG suppression is perturbative only (no avir) --- the counterpart
            # of 'folpsEFT' (X_FoG fixed to 0).
            kw['model'] = {'comet': 'VDG_infty', 'cometEFT': 'EFT'}[theory_options['model']]
            #pt = COMETPTSpectrum2Poles(cosmo=cosmology, z=z) # backend='numpy')
            #theory = COMETTracerSpectrum2Poles(pt=pt, tracers=tracers, **kw, **theory_options.get('options', {}))
            theory = COMETTracerSpectrum2Poles(cosmo=cosmology, tracers=tracers, z=z, **kw, **theory_options.get('options', {}))
            kw_stoch = get_physical_stochastic_settings(tracer=get_simple_tracer(tracers))
            theory.update(**kw_stoch, nbar=nbar, params=update_theory_nuisance_priors(get_params(theory, level=1), theory_options['model'], stat, kw['prior_basis'], marg=theory_options.get('marg', False), user_params=theory_options.get('params') or None))
    elif 'mesh3_spectrum' in stat:
        if theory_options['model'] in ['folpsD', 'folpsEFT']:
            kw = {name: theory_options[name] for name in ['damping', 'A_full', 'prior_basis'] if name in theory_options}
            theory = FOLPSTracerSpectrum3Poles(template=template, tracers=tracers, z=z, **kw, **smearing_kw, **theory_options.get('options', {}))
            kw_stoch = get_physical_stochastic_settings(tracer=get_simple_tracer(tracers))
            theory.update(**kw_stoch, nbar=nbar, params=update_theory_nuisance_priors(get_params(theory, level=1), theory_options['model'], stat, kw['prior_basis'], marg=theory_options.get('marg', False), user_params=theory_options.get('params') or None))
        elif theory_options['model'] in ['comet', 'cometEFT']:
            kw = {name: theory_options[name] for name in ['prior_basis'] if name in theory_options}
            # 'comet' is comet's fitted velocity-dispersion kernel VDG_infty; 'cometEFT' the pure
            # EFT model, whose FoG suppression is perturbative only (no avir) --- the counterpart
            # of 'folpsEFT' (X_FoG fixed to 0).
            kw['model'] = {'comet': 'VDG_infty', 'cometEFT': 'EFT'}[theory_options['model']]
            theory = COMETTracerSpectrum3Poles(cosmo=cosmology, tracers=tracers, **kw, **theory_options.get('options', {}))
            kw_stoch = get_physical_stochastic_settings(tracer=get_simple_tracer(tracers))
            theory.update(**kw_stoch, nbar=nbar, params=update_theory_nuisance_priors(get_params(theory, level=1), theory_options['model'], stat, kw['prior_basis'], marg=theory_options.get('marg', False), user_params=theory_options.get('params') or None))
    elif 'recon_' in stat:
        kw = {name: np.asarray(data_attrs.get(f'recon_{name}', None)).flat[0] for name in ['mode', 'smoothing_radius']}
        kw = kw | {name: theory_options[name] for name in kw if name in theory_options}
        if kw['mode'] is None: kw['mode'] = ''  # no reconstruction
        kw['ells'] = [0, 2, 4]
        if stat == 'recon_particle2_correlation':
            kw['broadband'] = theory_options.get('broadband', 'pcs2')
            theory = DampedBAOWigglesTracerCorrelation2Poles(template=template, **kw)
        elif stat == 'recon_mesh2_spectrum':
            kw['broadband'] = theory_options.get('broadband', 'pcs')
            theory = DampedBAOWigglesTracerSpectrum2Poles(template=template, **kw)
        else:
            raise NotImplementedError(f'cannot fit {stat}')
        # FIXME level=2
        params = update_theory_nuisance_priors(get_params(theory, level=2), theory_options['model'], stat, prior_basis=kw['mode'], tracer=tracers, marg=theory_options.get('marg', False), ells=getattr(data, 'ells', [0, 2, 4]), user_params=theory_options.get('params') or None)
        theory.update(params=params)
    if theory is None:
        raise ValueError(f'theory not found for {stat} and {repr(theory_options)}')
    return theory


def pack_stats(stats, **labels):
    """
    Pack a list of stat-like objects into a single :class:`types.ObservableTree` or :class:`types.WindowMatrix`.

    Parameters
    ----------
    stats : list[ObservableLike, WindowMatrix]
        List of statistics objects to pack.
    labels : mapping
        Labels to attach to the resulting container.
        E.g. ``observables=['spectrum2', 'spectrum3'], tracers=[('LRG', 'LRG'), ('LRG', 'LRG', 'LRG')]``
        for the combined power spectrum and bispectrum.

    Returns
    -------
    Packed types.ObservableTree or types.WindowMatrix
    """
    if isinstance(stats[0], types.ObservableLike):
        return types.ObservableTree(stats, **labels)
    elif isinstance(stats[0], types.WindowMatrix):
        windows = stats
        values = [window.value() for window in windows]
        observables = [window.observable for window in windows]
        theories = [window.theory for window in windows]
        return types.WindowMatrix(
            value=sp.linalg.block_diag(*values),
            observable=pack_stats(observables, **labels),
            theory=pack_stats(theories, **labels),
        )
    else:
        raise ValueError(f'unrecognized type {stats[0]}')


def unpack_stats(stats):
    """
    Unpack packed stats structures into individual windows/observables.

    Parameters
    ----------
    stats : types.ObservableLike | types.WindowMatrix | types.GaussianLikelihood
        If ObservableLike, returns a list of observables
        If WindowMatrix, returns a list of window matrices
        If GaussianLikelihood, returns a tuple[list of observables, list of window matrices, covariance]

    Returns
    -------
    Unpacked statistics.
    """
    if isinstance(stats, types.ObservableLike):
        return stats.flatten(level=1)  # iter over labels
    elif isinstance(stats, types.WindowMatrix):
        window = stats
        windows = []
        for label in window.observable.labels(level=1):
            windows.append(window.at.observable.get(**label).at.theory.get(**label))
        return windows
    elif isinstance(stats, types.GaussianLikelihood):
        likelihood = stats
        observables = unpack_stats(likelihood.observable)
        windows = [None] * len(observables) if likelihood.window is None else unpack_stats(likelihood.window)
        return (observables, windows, likelihood.covariance)


def combine_covariances(covariances, observable):
    """Combine input covariances into a large one, for observable."""
    olabels = observable.labels(level=1)
    nblocks = len(olabels)
    value = [[None for i in range(nblocks)] for i in range(nblocks)]
    observables = [None for i in range(nblocks)]
    for ilabel1, ilabel2 in itertools.product(range(nblocks), repeat=2):
        label1, label2 = (olabels[ilabel] for ilabel in [ilabel1, ilabel2])
        block = None
        for covariance in covariances:
            clabels = covariance.observable.labels(level=1)
            csizes = list(covariance.observable.sizes(level=1))
            cumsizes = np.cumsum([0] + csizes)
            if label1 in clabels and label2 in clabels:
                i1, i2 = clabels.index(label1), clabels.index(label2)
                block = covariance.value()[cumsizes[i1]:cumsizes[i1 + 1], cumsizes[i2]:cumsizes[i2 + 1]]
                observables[i1] = covariance.observable.get([label1])
        if block is None:
            warnings.warn(f'block {label1}, {label2} not found, assuming it is 0')
            shape = tuple(observable.get(**label).size for label in [label1, label2])
            block = np.zeros(shape)
        value[ilabel1][ilabel2] = block
    value = np.block(value)
    if any(observable is None for observable in observables):
        raise ValueError(f'could not find observables for labels {olabels}')
    observable = types.join(observables)
    return types.CovarianceMatrix(observable=observable, value=value)


def _infer_effective_nparams(observables: list[dict]) -> int:
    """Infer effective free-parameter count for covariance corrections.

    Uses a fixed effective count by fit content:
      - 7 for single-stat fits (mesh2-only or mesh3-only)
      - 9 for joint mesh2+mesh3 fits
    """
    stats = {obs['stat']['kind'] for obs in observables}
    has_mesh2 = any('mesh2_spectrum' in stat for stat in stats)
    has_mesh3 = any('mesh3_spectrum' in stat for stat in stats)
    return 9 if (has_mesh2 and has_mesh3) else 7


def _get_covariance_correction_factor(covariance: types.CovarianceMatrix,
                                      observables: list[dict],
                                      covariance_options: dict,
                                      default_corrections=('hartlap', 'percival')):
    """Return multiplicative covariance correction factor and per-term metadata."""
    corrections = covariance_options.get('corrections', default_corrections)
    if isinstance(corrections, str):
        corrections = [corrections]
    corrections = [str(corr).lower() for corr in (corrections or [])]

    factor = 1.
    nbins = int(covariance.value().shape[0])
    nobs = covariance.attrs['nobs']
    metadata = {'nbins': nbins, 'corrections': tuple(corrections)}
    if nobs <= 0:  # analytic covariance matrix
        return factor, metadata | dict(corrections=tuple())

    nobs = int(nobs)
    metadata.update(nobs=nobs)

    if 'hartlap' in corrections:
        hartlap = get_hartlap2007_factor(nobs, nbins)
        factor /= hartlap
        metadata['hartlap_factor'] = float(hartlap)

    if 'percival' in corrections:
        nparams = covariance_options.get('nparams', None)
        if nparams is None:
            nparams = _infer_effective_nparams(observables)
        percival = get_percival2014_factor(nobs, nbins, nparams)
        factor *= percival
        metadata['percival_factor'] = float(percival)
        metadata['nparams'] = int(nparams)

    return factor, metadata


def _get_prepared_cache_options(observables_options: list[dict], covariance_options: dict=None, kind: str=None):
    def window_options(observable_options):
        options = dict(observable_options.get('window', {}))
        options.pop('theory_dk', None)
        return options

    def redshift_smearing_options(observable_options):
        # The cached window carries the measured PDF: key on the file-lookup keys, and on the
        # part of the mode before '+', so 'pdf', 'pdf+gauss' and 'pdf+lor' -- which give the
        # same window, differing only in the theory -- share one cached product.
        redshift_smearing = _format_redshift_smearing(observable_options.get('redshift_smearing', None))
        if redshift_smearing is None:
            return {}
        return {'redshift_smearing': redshift_smearing | {'mode': redshift_smearing['mode'].partition('+')[0]}}

    options = {'observables': [
        {name: dict(observable_options[name]) for name in ['stat', 'catalog']}
        | ({'window': window_options(observable_options)} if 'window' in observable_options else {})
        | redshift_smearing_options(observable_options)
        for observable_options in observables_options
    ]}
    if kind == 'covariance':
        options['covariance'] = covariance_options or {}
    return options


def _stat_is_compressed(stat):
    # Meaning, no window
    return 'bao' in stat


@default_mpicomm
def get_stats(observables_options: list[dict], covariance_options: dict=None, unpack: bool=False,
              get_stats_fn=clustering_tools.get_stats_fn, cache_dir: str | Path=None, cache_mode: str='rw', mpicomm=None):
    """
    Load and assemble measurement products (data, windows, covariance).

    This function:
      - reads per-statistic measurement files determined by `observables`;
      - optionally caches the assembled likelihood;
      - constructs mock-based covariance from available mock files;
      - returns a :class:`types.GaussianLikelihood` (or unpacked components if requested).

    Parameters
    ----------
    observables_options : list[dict]
        List of observable option dicts, one per statistic to load. Each dict must have at
        least the following keys:

        - ``'stat'``: dict with at least ``'kind'`` (e.g. ``'mesh2_spectrum'``,
          ``'mesh3_spectrum'``, ``'recon_particle2_correlation'``).  Optional keys:

          - ``'select'``: list of per-multipole selection dicts, each with an ``'ells'``
            key and coordinate-range entries of the form ``[min, max]`` or
            ``[min, max, step]``.  May also be a callable for custom rebinning.
          - ``'basis'``: string passed through to the stats-file path builder
            (e.g. ``'sugiyama-diagonal'`` for bispectrum).

        - ``'catalog'``: dict of catalog metadata keys used to locate measurement files,
          typically including ``'tracer'``, ``'version'``, ``'zrange'``, ``'region'``,
          ``'weight'``, ``'stats_dir'``, and optionally ``'imock'`` and ``'project'``.

    covariance_options : dict or None
        Options controlling how the covariance matrix is built. Relevant keys:

        - ``'source'``: ``'mock'`` (default), ``'jaxpower'``, or ``'rascalc'``.
        - ``'version'``: version string for mock or analytic covariance files.
        - ``'stats_dir'``: base directory for mock realization files.
        - ``'corrections'``: list of correction names to apply, e.g.
          ``['hartlap', 'percival']``.
        - ``'nparams'``: effective parameter count for the Percival correction
          (inferred automatically when omitted).
        - ``'scale'``: positive multiplicative factor for the final matched
          covariance matrix.

        If ``None`` or ``{}``, the covariance-source dispatch falls through to ``None``
        and no covariance is built.
    unpack : bool, optional
        If ``True`` return unpacked (data, windows, covariance) rather than a :class:`types.GaussianLikelihood`.
    get_stats_fn : callable
        Function used to locate stats files.
    cache_dir : str or Path, optional
        Directory to use for caching assembled likelihoods.
    cache_mode : str, optional
        'rw' for read/write; 'r' for read-only.

    Returns
    -------
    types.GaussianLikelihood or tuple
    """
    covariance_options = covariance_options or {}

    if cache_dir is not None:
        cache_dir = Path(cache_dir) / 'prepared_stats'
    read_cache = cache_dir is not None and 'r' in cache_mode
    write_cache = cache_dir is not None and 'w' in cache_mode

    def get_cache_fn(kind, kwargs):
        if cache_dir is None:
            return None
        _full_options = _get_prepared_cache_options(observables_options, covariance_options, kind=kind)
        _level = {'stat': 1, 'catalog': 2, 'window': 1, 'covariance': 0}
        if kind == 'covariance':
            _level['covariance'] = 1
        _str_from_options = str_from_likelihood_options(_full_options, level=_level)
        _hash = _hash_options(_full_options | kwargs)
        return cache_dir / f'{kind}_{_str_from_options}-{_hash}.h5'

    def get_covariance_manifest_key():
        options = _get_prepared_cache_options(observables_options, covariance_options, kind='covariance')
        return _hash_options(options, length=32)

    def get_covariance_manifest_fn():
        if cache_dir is None:
            return None
        return cache_dir / 'covariance_manifest.json'

    def read_covariance_manifest():
        manifest_fn = get_covariance_manifest_fn()
        if manifest_fn is None or not read_cache:
            return {}
        manifest = {}
        if mpicomm.rank == 0 and manifest_fn.exists():
            with open(manifest_fn, 'r') as file:
                manifest = json.load(file)
        return mpicomm.bcast(manifest, root=0)

    def write_covariance_manifest_entry(cache_fn, imocks):
        manifest_fn = get_covariance_manifest_fn()
        if manifest_fn is None or not write_cache or mpicomm.rank != 0:
            return
        mkdir(manifest_fn.parent)
        if manifest_fn.exists():
            with open(manifest_fn, 'r') as file:
                manifest = json.load(file)
        else:
            manifest = {}
        key = get_covariance_manifest_key()
        try:
            filename = str(cache_fn.relative_to(cache_dir))
        except ValueError:
            filename = str(cache_fn)
        manifest[key] = {'filename': filename, 'imocks': list(imocks)}
        with open(manifest_fn, 'w') as file:
            json.dump(manifest, file, indent=2, sort_keys=True)

    def get_covariance_cache_fn_from_manifest():
        manifest = read_covariance_manifest()
        entry = manifest.get(get_covariance_manifest_key(), None)
        if not entry:
            return None, None
        cache_fn = Path(entry['filename'])
        if not cache_fn.is_absolute():
            cache_fn = cache_dir / cache_fn
        return cache_fn, entry.get('imocks', None)

    def get_from_cache(cache_fn):
        if cache_fn is None or not read_cache:
            return None
        stats = None
        if all(mpicomm.allgather(cache_fn.exists())):
            logger.info(f'Reading cached stats {cache_fn}.')
            stats = types.read(cache_fn)
        return mpicomm.bcast(stats, root=0)

    # Helper: iterate over (stat, tracer) combinations
    def iter_stat_tracer_combinations(observables_options, with_stat_kw=False, catalog_options=None):
        """
        Yield (stat, labels, file_kwargs, observable_options) for each requested observable.

        Compact helper for iterating the user-provided observables and producing file kwargs
        and labeling information used when reading files.
        """
        _catalog_options = dict(catalog_options or {})
        for observable_options in observables_options:
            stat = observable_options['stat']['kind']
            catalog_options = observable_options['catalog'] | _catalog_options
            tracers = _make_tuple(catalog_options['tracer'])
            version = catalog_options.get('version', None)
            full_tracer = get_full_tracer(tracers, version=version)
            nfields = 3 if 'mesh3' in stat else 2
            simple_tracers = get_simple_tracer(tracers)
            simple_tracers += (simple_tracers[-1],) * (nfields - len(simple_tracers))
            labels = {
                'observables': get_simple_stats(stat),
                'tracers': simple_tracers,
            }
            kw = {}
            for name in observable_options['stat']:
                if name in ['basis']:  # shared by stats and covariance
                    kw[name] = observable_options['stat'][name]
                elif with_stat_kw and name not in ['kind', 'select']:
                    kw[name] = observable_options['stat'][name]
            file_kw = kw | catalog_options | {'tracer': full_tracer}
            yield stat, labels, file_kw, dict(observable_options)

    def _with_project(observable: types.ObservableTree):
        return hasattr(observable, 'project')

    def _apply_project(observable: types.ObservableTree, select: list=None):
        # Project correlation function
        data, windows = [], []
        for _select in select:
            _select = dict(_select)
            ells = [_select.pop('ells')]
            correlation = observable
            RR = correlation.get('RR')
            for coord_name, limits in _select.items():
                if len(limits) == 3:  # apply binning only
                    step = limits[2]
                    edge = correlation.edges(coord_name)[0]
                    rebin = int(np.rint(np.mean(step / (edge[..., 1] - edge[..., 0]))) + 0.5)
                    correlation = correlation.select(**{coord_name: slice(0, None, rebin)})
                #correlation = correlation.select(**{coord_name: tuple(limits[:2])})
            pole, window = correlation.project(ells=ells, kw_window=dict(RR=RR))
            data.append(pole)
            windows.append(window)
        data = types.join(data)
        window = types.WindowMatrix(value=np.concatenate([window.value() for window in windows], axis=0),
                                    observable=types.join([window.observable for window in windows]),
                                    theory=windows[0].theory,
                                    attrs=windows[0].attrs)
        return data, window

    def _apply_select(observable: types.ObservableTree, select: list=None):
        """
        Apply a selection (k-range, ell selection) to an observable.

        The selection dict keys are multipoles (ell) and values are slice-like
        specifications or (min, max, [step]) tuples.
        """
        if select is None:
            return observable
        if callable(select):  # custom rebinning
            return select(observable)
        labels = []
        for _select in select:
            _select = dict(_select)
            keys = observable.labels(return_type='keys')
            label = {}
            for key in keys:
                if key in _select:
                    label[key] = _select.pop(key)
            labels.append(label)
            pole = observable.get(**label)
            for coord_name, limits in _select.items():
                if len(limits) == 3:
                    step = limits[2]
                    edge = pole.edges(coord_name)[0]
                    rebin = int(np.rint(np.mean(step / (edge[..., 1] - edge[..., 0]))) + 0.5)
                    pole = pole.select(**{coord_name: slice(0, None, rebin)})
                pole = pole.select(**{coord_name: tuple(limits[:2])})
            observable = observable.at(**label).replace(pole)
        observable = observable.get(labels)
        return observable

    def _get_mock_stats_fn(stat, file_kw):
        file_kw = dict(file_kw)
        stats_dir = Path(file_kw.pop('stats_dir'))
        version = file_kw.pop('version', None)
        project = file_kw.pop('project', '')
        if isinstance(version, str) and version.startswith('ezmock'):
            tracer = get_simple_tracer(_make_tuple(file_kw['tracer']))
            tracer = tracer[0] if isinstance(tracer, tuple) else tracer
            zsnap = float2str(file_kw['zsnap'], 3, 3)
            imock = file_kw['imock']
            kind = {'mesh2_spectrum': 'mesh2_spectrum_poles'}.get(stat, stat)
            if 'mesh3' in kind:
                basis = file_kw.get('basis', None)
                basis = f'_{basis}' if basis else ''
                kind = f'mesh3_spectrum{basis}_poles'
            return stats_dir / version / f'{kind}_{tracer}_z{zsnap}_{imock}.h5'
        def _has_existing(fn):
            if isinstance(fn, list):
                return len(fn) > 0
            return fn.exists()
        base_fn = get_stats_fn(kind=stat, stats_dir=stats_dir, project=project, version=version, **file_kw)
        if _has_existing(base_fn):
            return base_fn
        project_fn = None
        if not project and version is not None and file_kw.get('imock', None) is not None:
            project_fn = get_stats_fn(kind=stat, stats_dir=stats_dir, project=version, **file_kw)
            if _has_existing(project_fn):
                return project_fn
        if version is None:
            return base_fn
        alt_fn = get_stats_fn(kind=stat, stats_dir=stats_dir, **file_kw)
        return alt_fn

    def _window_mode(observable_options):
        return str(observable_options.get('window', {}).get('mode', 'file')).lower()

    covariance_stat_options = covariance_options.get('stat_options', {})
    covariance_file_options = {key: value for key, value in covariance_options.items() if key != 'stat_options'}

    def _format_log_fns(fns):
        if not isinstance(fns, list):
            return str(fns)
        if not fns:
            return '<no files>'
        if len(fns) <= 1:
            return str(fns[0])
        fns = [str(fn) for fn in fns]
        prefix = os.path.commonprefix(fns)
        suffix = os.path.commonprefix([fn[::-1] for fn in fns])[::-1]
        if prefix or suffix:
            return f'{prefix}*{suffix}'
        return '*'

    def _format_missing_data_context(stat, file_kw, fns):
        fields = ['stats_dir', 'version', 'tracer', 'zrange', 'region', 'weight', 'imock']
        context = ', '.join(f'{name}={file_kw[name]!r}' for name in fields if name in file_kw)
        return f'No measurement files found for {stat} ({context}); resolved lookup: {_format_log_fns(fns)}'

    def _format_missing_covariance_context(covariance_options, covariance_log_patterns, observables_options):
        fields = ['stats_dir', 'version', 'tracer', 'zrange', 'region', 'weight']
        contexts = []
        for observable_options in observables_options:
            catalog = observable_options['catalog']
            context = ', '.join(f'{name}={catalog[name]!r}' for name in fields if name in catalog)
            contexts.append(context)
        patterns = '; '.join(f'{stat}: {pattern}' for stat, pattern in covariance_log_patterns)
        contexts = '; '.join(contexts)
        return (
            f"No covariance mock realizations found for source={covariance_options.get('source')!r}, "
            f"version={covariance_options.get('version')!r} (matched 0 realizations). "
            f"Observable context: {contexts}. Lookup patterns: {patterns}"
        )

    # Loading data, window
    all_data_fns, all_imocks, joint_labels, selects = [], [], {'observables': [], 'tracers': []}, []
    window_modes = []
    for stat, labels, file_kw, kw in iter_stat_tracer_combinations(observables_options, with_stat_kw=True):
        file_kw = {'imock': None} | file_kw
        all_imocks.append(file_kw['imock'])
        fn = get_stats_fn(kind=stat, **file_kw)
        if not isinstance(fn, list): fn = [fn]
        all_data_fns.append(fn)
        for name in joint_labels:
            joint_labels[name].append(labels[name])
        selects.append(kw['stat'].get('select', None))
        window_modes.append(_window_mode(kw))
    no_window = all(mode == 'none' for mode in window_modes)
    if any(mode == 'none' for mode in window_modes) and not no_window:
        raise ValueError("window mode 'none' must be used for all observables in a likelihood")
    # The measured PDF is folded into the window, so it needs a mesh spectrum window to fold into.
    for iobs, (stat, labels, file_kw, kw) in enumerate(iter_stat_tracer_combinations(observables_options, with_stat_kw=True)):
        if _format_redshift_smearing(observables_options[iobs].get('redshift_smearing', None)) is None:
            continue
        if window_modes[iobs] == 'none' or _stat_is_compressed(stat) or 'mesh' not in stat:
            raise ValueError(f"redshift_smearing is not supported for {stat} with window mode "
                             f"{window_modes[iobs]!r}: there is no mesh spectrum window to fold the PDF into. "
                             "Pass a kernel to the theory instead, via the 'redshift_smearing' argument of "
                             "FOLPSTracerSpectrum2Poles / FOLPSTracerSpectrum3Poles.")
    cache_data_fn = get_cache_fn('data', dict(imocks=all_imocks))
    cache_window_fn = None if no_window else get_cache_fn('window', dict(imocks=all_imocks))
    data = get_from_cache(cache_data_fn)
    window = None if no_window else get_from_cache(cache_window_fn)
    data_cache_hit = data is not None
    window_cache_hit = no_window or window is not None
    if data is None or (not no_window and window is None):
        for iobs, (stat, labels, file_kw, kw) in enumerate(iter_stat_tracer_combinations(observables_options, with_stat_kw=True)):
            if not all_data_fns[iobs]:
                raise FileNotFoundError(_format_missing_data_context(stat, file_kw, all_data_fns[iobs]))
        if mpicomm.rank == 0:
            data, windows = [], []
            for iobs, (stat, labels, file_kw, kw) in enumerate(iter_stat_tracer_combinations(observables_options, with_stat_kw=True)):
                is_compressed = _stat_is_compressed(stat)
                _data, _windows = [], []
                logger.info(f"Reading data vector for {stat} from {_format_log_fns(all_data_fns[iobs])}")
                for fn in all_data_fns[iobs]:
                    observable = types.read(fn)
                    if _with_project(observable):  # correlation function
                        dw = _apply_project(observable, selects[iobs])
                        _data.append(dw[0])
                        _windows.append(dw[1])
                    else:  # power spectrum
                        _data.append(observable)
                data.append(types.mean(_data))
                if is_compressed:
                    # Placeholder
                    windows.append(types.WindowMatrix(value=np.eye(data[-1].size),
                                                      observable=data.clone(value=np.zeros_like(data.value())),
                                                      theory=data.clone(value=np.zeros_like(data.value()))))
                elif no_window:
                    continue
                elif _windows:
                    windows.append(_windows[0])
                else:
                    file_kw = dict(file_kw)
                    imock = file_kw.get('imock', None)
                    if imock is not None:  # FIXME
                        file_kw['imock'] = 0
                    file_kw.pop('auw', None)  # auw stat has the same window as non-auw stat
                    window_options = {
                        key: value for key, value in observables_options[iobs].get('window', {}).items()
                        if key not in ['mode', 'theory_dk']
                    }
                    file_kw = file_kw | window_options
                    #fn = _get_mock_stats_fn(f'window_{stat}', file_kw) if 'stats_dir' in file_kw else get_stats_fn(kind=f'window_{stat}', **file_kw)
                    fn = get_stats_fn(kind=f'window_{stat}', **file_kw)
                    logger.info(f"Reading window for {stat} from {fn}")
                    window_matrix = types.read(fn)
                    # Fold the measured redshift-error PDF in here, on the native theory grid:
                    # the mixing matrix is block-diagonal in k, so it must be applied before
                    # rebin_spectrum3_window / select_window_theory compact and trim it.
                    redshift_smearing = _format_redshift_smearing(observables_options[iobs].get('redshift_smearing', None))
                    if redshift_smearing is not None:
                        smearing_kw = file_kw | {'stats_dir': base_stats_dir} | {name: value for name, value in redshift_smearing.items() if name != 'mode'}
                        smearing_kw.pop('imock', None)
                        smearing_fn = get_stats_fn(kind='redshift_smearing', **smearing_kw)
                        if not os.path.exists(smearing_fn):
                            context = ', '.join(f'{name}={file_kw[name]!r}' for name in ['tracer', 'zrange'] if name in file_kw)
                            raise FileNotFoundError(f'No redshift-smearing PDF for {stat} ({context}); '
                                                    f'resolved lookup: {smearing_fn}')
                        logger.info(f"Reading redshift-smearing PDF for {stat} from {smearing_fn}")
                        window_matrix = window_with_redshift_smearing(window_matrix, types.read(smearing_fn))
                    windows.append(window_matrix)
            # Join mesh2_spectrum, mesh3_spectrum, etc.
            data = pack_stats(data, **joint_labels)
            window = None if no_window else pack_stats(windows, **joint_labels)
        data, window = mpicomm.bcast((data, window), root=0)
        data_cache_hit = False
        window_cache_hit = no_window
    if write_cache and not data_cache_hit:
        write_stats(cache_data_fn, data)
    if write_cache and not window_cache_hit:
        write_stats(cache_window_fn, window)
    for stat, labels, file_kw, kw in iter_stat_tracer_combinations(observables_options):
        leaf = _apply_select(data.get(**labels), select=kw['stat'].get('select', None))
        data = data.at(**labels).replace(leaf)
        if window is not None:
            window = window.at.observable.at(**labels).match(data.get(**labels))
    # Analytic covariances
    if covariance_options.get('source') in ['jaxpower', 'rascalc']:
        # WARNING: not tested yet!
        full_tracers = []
        for stat, labels, file_kw, kw in iter_stat_tracer_combinations(observables_options):
            imock = file_kw.get('imock', None)
            if imock is not None:  # FIXME
                file_kw['imock'] = 0
            file_kw = file_kw | covariance_options
            full_tracers.append(file_kw['tracer'] + (file_kw['tracer'][-1],) * (len(labels['tracers']) - len(file_kw['tracer'])))
        tracers = sorted({t for tpl in full_tracers for t in tpl})
        all_combinations = []
        for tpl in full_tracers:
            n = len(tpl)
            all_combinations.extend(itertools.product(tracers, repeat=n))
        all_combinations = list(dict.fromkeys(all_combinations))  # remove duplicates
        covariances = []
        # Query all possible cross-covariances
        # FIXME if there are 3pt-covariances
        source = covariance_options['source']
        cov_fns = []
        for tracers in all_combinations:
            if all(tracer == tracers[0] for tracer in tracers):
                tracers = tracers[0]
            fn = get_stats_fn(kind=f'covariance_{stat}', **(file_kw | dict(tracer=tracers)))
            cov_fns.append(str(fn))
            if fn.exists():
                logger.info(f"Reading covariance for {stat} from {_format_log_fns(fn)}")
                covariances.append(types.read(fn))
        if not covariances:
            raise ValueError(f'no covariances found in {cov_fns}')
        covariance = combine_covariances(covariances, data)
        #covariance = covariance.clone(value=covariance.value() / 25.)
        covariance.attrs['nobs'] = -1
    elif covariance_options.get('source') == 'mock':
        # Mock-based covariance
        cache_fn, imocks_exists = get_covariance_cache_fn_from_manifest()
        covariance_cache_hit = False
        covariance = get_from_cache(cache_fn)
        covariance_cache_hit = covariance is not None
        if covariance is not None and mpicomm.rank == 0:
            logger.info(f"Reading covariance cache {cache_fn} from manifest {get_covariance_manifest_fn()}.")
        if covariance is not None and imocks_exists is not None:
            canonical_cache_fn = get_cache_fn('covariance', dict(imocks=imocks_exists))
            if canonical_cache_fn != cache_fn:
                cache_fn = canonical_cache_fn
                covariance_cache_hit = all(mpicomm.allgather(cache_fn.exists()))
        if covariance is None:
            all_fns = []
            covariance_log_patterns = []
            all_imocks = None
            covariance_labels = []
            for stat, labels, file_kw, kw in iter_stat_tracer_combinations(observables_options, catalog_options=covariance_file_options):
                file_kw = file_kw | {'imock': '*'} | covariance_file_options | covariance_stat_options.get(stat, {})
                imocks = file_kw.pop('imock')
                if imocks == '*':
                    imocks = list(range(2001))
                if all_imocks is None:
                    all_imocks = list(imocks)
                else:
                    all_imocks = [imock for imock in all_imocks if imock in imocks]
                stat_fns = [get_stats_fn(kind=stat, **(file_kw | {'imock': imock})) for imock in imocks]
                all_fns.append(stat_fns)
                covariance_log_patterns.append((stat, _format_log_fns(stat_fns)))
            all_fns = list(zip(*all_fns, strict=True))  # get a list of list of file names
            ifns_exists = []
            if mpicomm.rank == 0:
                for stat, pattern in covariance_log_patterns:
                    logger.info(f"Looking for covariance mocks for {stat} at {pattern}")
                for ifn, fns in enumerate(all_fns):
                    if all(fn.exists() for fn in fns):
                        ifns_exists.append(ifn)
            ifns_exists = mpicomm.bcast(ifns_exists, root=0)
            if not ifns_exists:
                raise FileNotFoundError(
                    _format_missing_covariance_context(covariance_options, covariance_log_patterns, observables_options)
                )
            imocks_exists = [all_imocks[ifn] for ifn in ifns_exists]
            cache_fn = get_cache_fn('covariance', dict(imocks=imocks_exists))
            covariance = get_from_cache(cache_fn)
            covariance_cache_hit = covariance is not None
        if covariance is None:
            mocks = []
            if mpicomm.rank == 0:
                covariance_read_fns = [all_fns[ifn] for ifn in ifns_exists]
                logger.info(f"Reading covariance for {len(covariance_read_fns)} mock realizations from "
                            f"{_format_log_fns(covariance_read_fns)}")
                for ifn in ifns_exists:
                    # Join mesh2_spectrum, mesh3_spectrum, etc.
                    observables = [types.read(fn) for fn in all_fns[ifn]]
                    observables = [_apply_project(observable, select)[0] if _with_project(observable) else _apply_select(observable, select)
                                   for observable, select in zip(observables, selects)]
                    mock = types.ObservableTree(observables, **joint_labels)
                    mocks.append(mock)
                covariance = types.cov(mocks)
                covariance.attrs['nobs'] = len(mocks)
            covariance = mpicomm.bcast(covariance, root=0)
            covariance_cache_hit = False
        if write_cache and cache_fn is not None and not covariance_cache_hit:
            write_stats(cache_fn, covariance)
        if cache_fn is not None:
            write_covariance_manifest_entry(cache_fn, imocks_exists)

    covariance = covariance.at.observable.match(data)

    covariance_scale = float(covariance_options.get('scale', 1.))
    if not np.isfinite(covariance_scale) or covariance_scale <= 0.:
        raise ValueError(f'covariance scale must be positive and finite, got {covariance_scale!r}')
    if covariance_scale != 1.:
        covariance = covariance.clone(value=covariance.value() * covariance_scale)
        if mpicomm.rank == 0:
            logger.info(f'Applied covariance scale factor {covariance_scale:.6f}.')
    covariance.attrs['covariance_scale'] = covariance_scale

    factor, metadata = _get_covariance_correction_factor(covariance, observables_options, covariance_options)
    if factor != 1.:
        covariance = covariance.clone(value=covariance.value() * factor)
    covariance.attrs['covariance_correction_factor'] = float(factor)
    for name, value in metadata.items():
        covariance.attrs[name] = value
    if metadata['corrections']:
        info = f"Applied covariance corrections {metadata['corrections']} with factor {factor:.6f}"
        if 'hartlap_factor' in metadata:
            info += f", hartlap={metadata['hartlap_factor']:.6f}"
        if 'percival_factor' in metadata:
            info += f", percival={metadata['percival_factor']:.6f}, nparams={metadata['nparams']}"
        if mpicomm.rank == 0:
            logger.info(info)

    volume_scale = box_tools.get_covariance_volume_scale_factor(covariance_options.get('volume_rescaling', None))
    if volume_scale != 1.:
        covariance = covariance.clone(value=covariance.value() * volume_scale)
    covariance.attrs['volume_rescaling'] = _json_attr(dict(covariance_options.get('volume_rescaling', {}) or {}))
    covariance.attrs['volume_scale_factor'] = float(volume_scale)

    likelihood = types.GaussianLikelihood(
        observable=data,
        window=window,
        covariance=covariance,
    )

    if unpack:
        return unpack_stats(likelihood)
    return likelihood


def _get_spectrum3_theory_rebin(ostep, tstep, theory_dk=None):
    """Return the first-stage bispectrum theory-grid stride."""
    ostep, tstep = float(ostep), float(tstep)
    if not np.isfinite(ostep) or ostep <= 0.:
        raise ValueError(f'Observable bispectrum spacing must be positive and finite; found {ostep}.')
    if not np.isfinite(tstep) or tstep <= 0.:
        raise ValueError(f'Native bispectrum theory spacing must be positive and finite; found {tstep}.')
    if theory_dk is None:
        rebin = int(ostep / tstep)
    else:
        theory_dk = float(theory_dk)
        if not np.isfinite(theory_dk) or theory_dk <= 0.:
            raise ValueError(f'Fixed bispectrum theory spacing must be positive and finite; found {theory_dk}.')
        ratio = theory_dk / tstep
        rebin = int(np.rint(ratio))
        if ratio < 1. or rebin < 1:
            raise ValueError(
                f'Fixed bispectrum theory spacing {theory_dk} cannot be finer than the native spacing {tstep}.'
            )
        if not np.isclose(ratio, rebin, rtol=1e-6, atol=1e-8):
            raise ValueError(
                f'Fixed bispectrum theory spacing {theory_dk} must be an integer multiple '
                f'of the native spacing {tstep}.'
            )
    if rebin < 1:
        raise ValueError(
            f'Observable bispectrum spacing {ostep} cannot be finer than the native theory spacing {tstep}.'
        )
    return rebin


def rebin_spectrum3_window(window, data=None, theory_dk=None):
    """Compact a bispectrum window, optionally using a fixed first-stage theory spacing."""
    if data is None:
        data = window.observable
    # Simplify window matrix
    ostep = min(np.diff(pole.edges('k'), axis=-1).min() for pole in data)
    with_templates = hasattr(window.theory, 'types')
    window_theory = window.at.theory.get(types='theory') if with_templates else window

    tstep = min(np.diff(pole.edges('k'), axis=-1).min() for pole in window_theory.theory)
    rebin = _get_spectrum3_theory_rebin(ostep, tstep, theory_dk=theory_dk)
    requested = 'dynamic' if theory_dk is None else f'{float(theory_dk):g}'
    logger.info(
        'Compactifying bispectrum window theory: observable dk=%g, native theory dk=%g, '
        'requested theory dk=%s, first-stage stride=%d, effective theory dk=%g',
        ostep, tstep, requested, rebin, rebin * tstep,
    )
    window_theory = window_theory.at.theory.select(k=slice(0, None, rebin))
    # Compact non-diagonal term
    rebin = rebinning_matrix(window_theory.theory, new_coords=window_theory.theory.select(k=slice(0, None, 2)),
                             interp_order=3, diag='separate')
    window_theory = window_theory.clone(value=window_theory.value() @ rebin.value(), theory=rebin.theory)

    if with_templates:
        # Re-assemble the window matrix
        window_templates = window.at.theory.get(types=[t for t in window.theory.types if t != 'theory'])
        window = window.clone(value=np.concatenate([window_theory.value(), window_templates.value()], axis=-1),
                              theory=window.theory.at(types='theory').replace(window_theory.theory))
    else:
        window = window_theory
    return window


def _format_redshift_smearing(redshift_smearing):
    """Normalise the ``redshift_smearing`` option to ``{'mode': ..., **kw}``, or None.

    Applied once, in :func:`fill_fiducial_observable_options`; idempotent, so the few places
    that may receive a raw option (``get_stats``, ``get_theory`` called directly) can call it
    defensively. A bare string is shorthand for ``{'mode': <string>}``; the remaining keys are
    passed to :func:`~clustering_statistics.tools.get_stats_fn` to locate the PDF, and default
    to the DR2 products.

    'pdf' folds the measured P(dv) into the window; the '+gauss' / '+lor' suffixes additionally
    give the theory a parametric kernel with a free vsmear, which only folpsD / folpsEFT accept.
    """
    if not redshift_smearing:
        return None
    if isinstance(redshift_smearing, str):
        redshift_smearing = {'mode': redshift_smearing}
    redshift_smearing = dict(redshift_smearing)
    allowed_modes = ('pdf', 'pdf+gauss', 'pdf+lor')
    mode = redshift_smearing.get('mode', None)
    if mode not in allowed_modes:
        raise ValueError(f'redshift_smearing mode must be one of {list(allowed_modes)}, got {mode!r}')
    # the PDF depends on tracer and zrange only, so region is pinned to the one that was produced
    return {'project': 'auxiliary_data/redshift_smearing', 'version': 'data-dr2-v2', 'region': 'GCcomb'} | redshift_smearing


def _get_redshift_smearing_form(redshift_smearing):
    """Parametric damping form ('gauss', 'lor') of a formatted option, or None."""
    redshift_smearing = _format_redshift_smearing(redshift_smearing)
    if redshift_smearing is None:
        return None
    return redshift_smearing['mode'].partition('+')[2] or None


def _get_redshift_smearing_kernel(form):
    """Parametric single-field damping D(k mu), carrying its own free ``vsmear`` [Mpc/h].

    Follows desilike's ``redshift_smearing`` protocol: a jax-traceable ``fun(kmu, *values)``
    whose ``params`` are attached to the theory, which then composes the kernel per statistic
    (``D^2`` for the power spectrum, ``D(k1 mu1) D(k2 mu2) D(k3 mu3)`` for the bispectrum).
    """
    import jax.numpy as jnp
    from desilike.parameter import Parameter

    if form == 'gauss':
        def fun(kmu, vsmear):
            return jnp.exp(-(kmu * vsmear)**2 / 2.)
    elif form == 'lor':
        def fun(kmu, vsmear):
            return jnp.exp(-jnp.abs(kmu * vsmear))
    else:
        raise ValueError(f'unknown parametric redshift-smearing form {form!r}')
    # 4 Mpc/h is ~380 km/s at z = 1.5, comfortably above the measured sigma_68 (161 km/s at
    # most, for QSO around z = 1.65) that the PDF already accounts for.
    # latex must end in '_X' or '_{...}' so Parameter.latex() merges the tracer namespace into
    # the existing subscript; otherwise it appends a second one and mathtext rejects it.
    fun.params = Parameter('vsmear', value=0., prior=dict(dist='uniform', limits=[0., 4.]),
                           ref=dict(dist='norm', loc=0., scale=0.5), latex=r'v_{\mathrm{smear}}')
    return fun


def _get_redshift_smearing_damping(redshift_smearing, zeff, kmu_max=5., nkmu=100001):
    r"""Tabulate the single-field characteristic function of the line-of-sight displacement.

    A redshift error displaces a galaxy by :math:`\epsilon = \delta v / (aH)`, with :math:`aH`
    from the **fiducial** cosmology --- the one used to turn redshifts into distances. Hence

    .. math:: D(k\mu) = \int d\delta v\, \mathcal{P}(\delta v) e^{i k \mu \delta v / (aH)}

    Parameters
    ----------
    redshift_smearing : lsstypes.ObservableLeaf
        PDF of the redshift error, with ``dv`` [km/s] as coordinate and ``pdf`` as value
        (see ``clustering_statistics/helper_scripts/convert_redshift_smearing.py``).
    zeff : float
        Effective redshift at which to evaluate :math:`aH`.

    Returns
    -------
    damping : callable
        ``damping(kmu)`` with ``kmu`` in :math:`h/\mathrm{Mpc}`; :math:`\mathcal{P}` is close
        to symmetric, so only the (real) cosine transform is kept.
    """
    dv, pdf = redshift_smearing.coords('dv'), redshift_smearing.values('pdf')
    # a H(zeff) in km/s per Mpc/h, fiducial cosmology
    aH = 100. * get_fiducial().efunc(zeff) / (1. + zeff)
    weights = np.zeros_like(dv)  # trapezoidal: the dv grid is not uniform
    dgrid = np.diff(dv)
    weights[:-1] += 0.5 * dgrid
    weights[1:] += 0.5 * dgrid
    weights = weights * pdf
    # the direct sum is O(len(dv)) per point, prohibitive on the bispectrum angular grid
    kmu = np.linspace(0., float(kmu_max), int(nkmu))
    damping = np.empty_like(kmu)
    step = max(1, int(2e7 // dv.size))
    for start in range(0, kmu.size, step):
        damping[start:start + step] = np.cos(kmu[start:start + step, None] / aH * dv) @ weights
    damping /= damping[0]
    return lambda x: np.interp(np.abs(x), kmu, damping)


def _get_spectrum2_smearing_matrix(damping, k, ells, nmu=200):
    r"""Multipole-mixing matrix ``M[ell, ellin, k]`` for the power spectrum.

    :math:`D` depends on :math:`k\mu`, so damping :math:`P(k, \mu)` mixes multipoles:

    .. math:: M_{\ell \ell^\prime}(k) = (2\ell + 1) \int_0^1 d\mu L_\ell(\mu) L_{\ell^\prime}(\mu) D^2(k\mu)

    exact given the multipoles *ells* of the theory. Note :math:`D^2`: the two galaxies of a
    pair are displaced independently.
    """
    x, w = np.polynomial.legendre.leggauss(2 * nmu)
    mu, w = x[nmu:], (w[nmu:] + w[nmu - 1::-1]) / 2.  # fold onto [0, 1], the integrand is even
    d2 = damping(np.asarray(k)[:, None] * mu)**2
    legendre = {ell: sp.special.legendre(ell)(mu) for ell in ells}
    return np.array([[(2 * ell + 1) * np.sum(w * legendre[ell] * legendre[ellin] * d2, axis=-1)
                      for ellin in ells] for ell in ells])


# Sugiyama tri-polar basis S_{l1 l2 L}, as functions of the rotational invariants
# x = k1_hat . k2_hat, mu1 = k1_hat . n_hat and mu2 = k2_hat . n_hat, together with
# H = (l1 l2 L; 0 0 0), the Wigner 3j at all m = 0 (folps' normalisation).
# folps' angular integrand is (2 l1 + 1) (2 l2 + 1) (2 L + 1) / (8 pi) * S.
_sugiyama_basis = {
    (0, 0, 0): (1., lambda x, mu1, mu2: np.ones_like(x * mu1 * mu2)),
    (1, 1, 0): (-1. / np.sqrt(3.), lambda x, mu1, mu2: -x / np.sqrt(3.)),
    (2, 2, 0): (1. / np.sqrt(5.), lambda x, mu1, mu2: (3. * x**2 - 1.) / (2. * np.sqrt(5.))),
    (2, 0, 2): (1. / np.sqrt(5.), lambda x, mu1, mu2: (3. * mu1**2 - 1.) / (2. * np.sqrt(5.))),
    (0, 2, 2): (1. / np.sqrt(5.), lambda x, mu1, mu2: (3. * mu2**2 - 1.) / (2. * np.sqrt(5.))),
    (1, 1, 2): (np.sqrt(2. / 15.), lambda x, mu1, mu2: (3. * mu1 * mu2 - x) / np.sqrt(30.)),
    (2, 2, 2): (-2. / np.sqrt(70.), lambda x, mu1, mu2: (3. * (mu1**2 + mu2**2 + x**2) - 9. * mu1 * mu2 * x - 2.) / np.sqrt(70.)),
}


def _get_spectrum3_smearing_matrix(damping, k, ells, precision=(8, 10, 10)):
    r"""Multipole-mixing matrix ``M[ell, ellin, k]`` for the bispectrum, in the Sugiyama basis.

    .. math:: M_{ab}(k_1, k_2) = \frac{H_a}{H_b} \frac{N_a}{8\pi} \int d\Omega\, S_a\, D(k_1\mu_1) D(k_2\mu_2) D(k_3\mu_3)\, S_b

    with :math:`N_a = (2l_1+1)(2l_2+1)(2L+1)` and :math:`d\Omega = dx\, d\mu_1\, d\phi` over
    :math:`[-1, 1] \times [-1, 1] \times [0, 2\pi]`, matching folps' ``Sugiyama_Bell``. One
    factor of :math:`D` per field: the three galaxies are displaced independently, unlike a
    pair, which gives :math:`D^2`. The angular variables are taken pre-AP, as in folps'
    ``bispectrum()``. Sanity check: :math:`M` is the identity for :math:`D \equiv 1`.

    *precision* is folps' ``(Nphi, Nx, Nmu)``, the Gauss-Legendre orders of the three angular
    integrals.
    """
    unknown = [ell for ell in ells if tuple(ell) not in _sugiyama_basis]
    if unknown:
        raise NotImplementedError(f'Sugiyama basis functions are not implemented for {unknown}')
    nphi, nx, nmu = (int(n) for n in precision)
    # Gauss-Legendre nodes: phi on [0, pi] (the phi integrand is even, hence the factor 2
    # below), x and mu1 on [-1, 1]
    phi, wphi = np.polynomial.legendre.leggauss(nphi)
    phi, wphi = (phi + 1.) * np.pi / 2., wphi * np.pi / 2.
    x12, wx = np.polynomial.legendre.leggauss(nx)
    mu1, wmu = np.polynomial.legendre.leggauss(nmu)

    k = np.asarray(k)
    k1, k2 = k[:, 0, None, None, None], k[:, 1, None, None, None]
    x12, mu1, phi = x12[None, :, None, None], mu1[None, None, :, None], phi[None, None, None, :]
    # same geometry as folps' APtransforms, before the AP distortion
    k3 = np.sqrt(k1**2 + k2**2 + 2. * k1 * k2 * x12)
    mu2 = np.sqrt(1. - mu1**2) * np.sqrt(1. - x12**2) * np.cos(phi) + mu1 * x12
    mu3 = -(k1 * mu1 + k2 * mu2) / k3
    weights = 2. * wphi[None, None, None, :] * wmu[None, None, :, None] * wx[None, :, None, None]
    d = damping(k1 * mu1) * damping(k2 * mu2) * damping(k3 * mu3) * weights

    x12, mu1, mu2 = np.broadcast_arrays(x12, mu1, mu2)
    basis = [_sugiyama_basis[tuple(ell)][1](x12, mu1, mu2) for ell in ells]
    hnorm = np.array([_sugiyama_basis[tuple(ell)][0] for ell in ells])
    nnorm = np.array([np.prod([2 * l + 1 for l in ell]) for ell in ells], dtype='f8')
    toret = np.empty((len(ells), len(ells), len(k)), dtype='f8')
    for ill, base in enumerate(basis):
        for illin, basein in enumerate(basis):
            toret[ill, illin] = (hnorm[ill] / hnorm[illin] * nnorm[ill] / (8. * np.pi)
                                 * np.sum(base * d * basein, axis=(1, 2, 3)))
    return toret


def window_with_redshift_smearing(window, redshift_smearing, **kwargs):
    r"""Return *window* with the redshift-smearing damping folded in.

    The damping depends on :math:`k\mu`, so it cannot be applied multipole by multipole --- it
    mixes them. Written as :math:`P^\mathrm{obs}_\ell = \sum_{\ell^\prime} M_{\ell \ell^\prime}
    P_{\ell^\prime}`, with :math:`M` block-diagonal in :math:`k`, it composes with the window
    as :math:`W^\prime = W M`. Folding it in here costs nothing at evaluation time and leaves
    the theory untouched.

    Both the power spectrum (``ells`` are ints) and the bispectrum in the Sugiyama basis
    (``ells`` are ``(l1, l2, L)`` triplets) are supported; the two differ in the number of
    :math:`D` factors, one per field.

    Parameters
    ----------
    window : lsstypes.WindowMatrix
    redshift_smearing : lsstypes.ObservableLeaf
        PDF of the redshift error, see :func:`_get_redshift_smearing_damping`.
    **kwargs : dict
        Passed on to the matrix builder: ``nmu`` for the power spectrum, ``precision`` for the
        bispectrum.

    Returns
    -------
    window : lsstypes.WindowMatrix
    """
    zeff = window.attrs.get('zeff', None)
    if zeff is None:
        zeff = next(iter(window.observable)).attrs['zeff']
    damping = _get_redshift_smearing_damping(redshift_smearing, zeff)

    with_templates = hasattr(window.theory, 'types')
    window_theory = window.at.theory.get(types='theory') if with_templates else window

    ells = list(window_theory.theory.ells)
    coords = [leaf.coords('k') for leaf in window_theory.theory]
    if not all(np.array_equal(coord, coords[0]) for coord in coords):
        raise NotImplementedError('redshift smearing requires all theory multipoles on the same k grid')
    k = coords[0]
    if np.ndim(ells[0]) == 0:  # power spectrum: ells are 0, 2, 4...
        matrix = _get_spectrum2_smearing_matrix(damping, k, ells, **kwargs)
    else:  # bispectrum, Sugiyama basis: ells are (l1, l2, L)
        matrix = _get_spectrum3_smearing_matrix(damping, k, ells, **kwargs)
    logger.info('Folding redshift smearing at zeff=%.4f into the window: ells=%s, %d theory coordinates.',
                zeff, ells, len(k))

    value = window_theory.value()
    value = value.reshape(value.shape[0], len(ells), len(k))
    value = np.einsum('dak,abk->dbk', value, matrix).reshape(value.shape[0], len(ells) * len(k))
    window_theory = window_theory.clone(value=value)

    if with_templates:  # re-assemble the window matrix, as in rebin_spectrum3_window
        window_templates = window.at.theory.get(types=[t for t in window.theory.types if t != 'theory'])
        window = window.clone(value=np.concatenate([window_theory.value(), window_templates.value()], axis=-1),
                              theory=window.theory.at(types='theory').replace(window_theory.theory))
    else:
        window = window_theory
    return window


def select_window_theory(window, data):
    """Restrict window theory to a range close to observed data."""
    coord_name = list(next(iter(data)).coords())
    assert len(coord_name) == 1
    coord_name = coord_name[0]
    data_limits = min(pole.edges(coord_name).min() for pole in data), max(pole.edges(coord_name).max() for pole in data)
    window = window.at.theory.select(coord_name=(data_limits[0] / 1.5, data_limits[1] * 1.5))
    return window


@default_mpicomm
def get_single_likelihood(likelihood_options, stats: types.GaussianLikelihood=None,
                          cosmology_options: dict=None, get_stats_fn=clustering_tools.get_stats_fn,
                          get_theory=get_theory, cache_dir:str | Path=None, cache_mode: str='rw', mpicomm=None):
    """
    Build a single :mod:`desilike` Gaussian likelihood from provided options.

    Parameters
    ----------
    likelihood_options : dict
        Options containing 'observables' list and 'covariance' dict.
    stats : types.GaussianLikelihood or None
        Preloaded measurements (if ``None`` they will be loaded via :func:`get_stats`).
    cosmology_options : optional
        Cosmology options or object or :class:`desilike.theories.Cosmoprimo`.
    get_stats_fn : callable, optional
        Function to locate measurement files.
    cache_dir : str | Path, optional
        Directory used for caching pre-computed emulators.
    cache_mode : str, optional
        'rw' for read/write; 'r' for read-only.

    Returns
    -------
    ObservablesGaussianLikelihood
    """
    from desilike.observables.galaxy_clustering import Spectrum2PolesObservable, Spectrum3PolesObservable, Correlation2PolesObservable
    from desilike.observables.galaxy_clustering.compressed import BAOCompressionObservable
    from desilike.likelihoods import ObservablesGaussianLikelihood
    from desilike import Parameter
    # likelihood_options: {'observables': [observable_options], 'covariance': {}}
    observables_options = likelihood_options['observables']
    covariance_options = likelihood_options.get('covariance', {})
    cosmology = get_cosmology(cosmology_options)
    if stats is None:
        stats = get_stats(observables_options, covariance_options=covariance_options, unpack=False, get_stats_fn=get_stats_fn, cache_dir=cache_dir, cache_mode=cache_mode)
    data, windows, covariance = unpack_stats(stats)
    labels = covariance.observable.labels(level=1)
    observables = []
    nbar = {}
    for observable_options, data, window, label in zip(observables_options, data, windows, labels, strict=True):
        stat = observable_options['stat']['kind']
        data_attrs = dict(data.attrs) | label
        z_source = window.observable if window is not None else data
        for _, pole in z_source.items(level=None):
            if 'zeff' in pole.attrs:
                data_attrs['z'] = pole.attrs['zeff']
            elif 'zsnap' in pole.attrs:
                data_attrs['z'] = pole.attrs['zsnap']
        data_attrs.setdefault('z', observable_options['catalog'].get('zsnap', None))
        # redshift_smearing: 0 --- this is the tracer namespace (it becomes data_attrs['tracers']
        # and the parameter namespace), so it must not carry the systematic model.
        namespace = _str_from_observable_options(observable_options, level={'catalog': 1, 'stat': 0, 'window': 0, 'theory': 0, 'covariance': 0, 'redshift_smearing': 0})
        data_attrs['tracers'] = namespace.split('x')
        if namespace not in nbar and 'spectrum2' in stats.observable.observables:
            nbar[namespace] = 1. / stats.observable.get(observables='spectrum2', tracers=label['tracers'], ells=0).values('shotnoise').mean()
        data_attrs['nbar'] = nbar.get(namespace, None)
        suffix = 'x'.join(data_attrs['tracers'])
        if suffix:
            suffix = '_' + suffix
        observable_name = stat + suffix
        if 'mesh2_spectrum' in stat:
            cls = Spectrum2PolesObservable
        elif 'mesh3_spectrum' in stat:
            cls = Spectrum3PolesObservable
        elif 'particle2_correlation' in stat:
            cls = Correlation2PolesObservable
        elif 'bao' in stat:
            cls = BAOCompressionObservable
        else:
            raise NotImplementedError(stat)
        is_compressed = _stat_is_compressed(stat)
        if is_compressed:  # e.g. BAO
            observable = cls(data=data.value(), z=data_attrs.get('z', data.attrs.get('zeff', None)), parameters=list(data.params))
        else:
            if window is not None:
                for _, pole in window.observable.items(level=None):
                    if 'zeff' in pole.attrs:
                        data_attrs['z'] = pole.attrs['zeff']
                    elif 'zsnap' in pole.attrs:
                        data_attrs['z'] = pole.attrs['zsnap']
            if mpicomm.rank == 0 and data_attrs.get('z', None) is not None:
                logger.info(f'{label}: data effective redshift = {data_attrs["z"]:.3f}')
            theory = get_theory(stat, theory_options=observable_options['theory'], cosmology=cosmology, data_attrs=data_attrs, data=data,
                                redshift_smearing=observable_options.get('redshift_smearing', None))
            if window is not None and cls == Spectrum3PolesObservable:
                # Compactify window theory
                theory_dk = observable_options.get('window', {}).get('theory_dk', None)
                window = rebin_spectrum3_window(window, data=data, theory_dk=theory_dk)
            if window is not None:
                window = select_window_theory(window, data)
            templates = None
            if window is not None and hasattr(window.theory, 'types'):  # with systematic templates
                templates = []
                for label, wtheory in window.theory.items():
                    if label['types'] != 'theory':
                        mean, sigma = wtheory.values('value'), wtheory.values('scale')
                        # FIXME: non-trivial shapes in gelman-rubin
                        assert mean.size == 1
                        prior = dict(dist='norm', loc=mean.flat[0], scale=sigma.flat[0])
                        param = Parameter(label['types'], namespace=namespace, value=mean.flat[0],
                                        ref=prior, prior=prior) #, derived='best')
                        template = window.at.theory.get(**label).value()
                        templates.append((param, template[..., 0]))
                window = window.at.theory.get('theory')  # window becomes the "standard window"
            observable = cls(data=data, window=window, theory=theory, templates=templates, name=observable_name)
        if observable_options['emulator']['name']:
            assert cache_dir is not None, 'cache_dir must be provided for emulator'
            read_cache = cache_dir is not None and 'r' in cache_mode
            write_cache = cache_dir is not None and 'w' in cache_mode
            cache_dir = Path(cache_dir)
            emulator_cache_options = _get_emulator_cache_options(observable_options, observable=observable)
            _hash = _hash_options(emulator_cache_options)
            _str_cosmology = str_from_cosmology_options(observable_options['theory']['cosmology'], level=100)
            _str_cosmology += '_' + observable_options['emulator']['name']
            _str_theory = _str_from_observable_options(
                observable_options,
                # redshift_smearing: 0 --- the emulator wraps theory.pt, upstream of both the
                # window and vsmear, so it does not depend on the smearing; tagging it here
                # would only force a pointless refit.
                level={'stat': 2, 'theory': 100, 'window': 1, 'catalog': 2, 'redshift_smearing': 0},
            )
            cache_fn = cache_dir / f'emulator_{_str_cosmology}' / f'emulator_{_str_theory}_{_hash}.h5'
            from desilike.base import compile
            from desilike.emulators import TaylorEmulator
            expected_theory_size = _get_emulator_theory_size(observable)
            if read_cache and cache_fn.exists():
                msg = f'Reading cached emulator {cache_fn}'
                if expected_theory_size is not None:
                    msg += f' for {expected_theory_size} theory entries'
                logger.info(msg)
                emulator = TaylorEmulator.read(str(cache_fn))
            else:
                msg = f'Fitting emulator {cache_fn}'
                if expected_theory_size is not None:
                    msg += f' for {expected_theory_size} theory entries'
                logger.info(msg)
                pt_graph = compile(observable.theory if is_compressed else theory.pt)
                emulator = TaylorEmulator(pt_graph, order=observable_options['emulator'].get('order', 3))
                emulator.fit()
                if write_cache and mpicomm.rank == 0:
                    mkdir(cache_fn.parent)
                    emulator.write(str(cache_fn))
            emulated_pt = emulator.to_calculator()
            if is_compressed:
                observable.update(theory=emulated_pt)
            else:
                theory.update(pt=emulated_pt)
        observables.append(observable)
    return ObservablesGaussianLikelihood(observables=observables, covariance=covariance.value())


def get_likelihood(likelihoods_options: dict | list[dict], cosmology_options: dict=None, get_stats_fn=clustering_tools.get_stats_fn,
                   get_theory=get_theory, cache_dir:str | Path=None, cache_mode: str='rw'):
    """
    Build a desilike :class:`SumLikelihood, summed over all tracers.

    Parameters
    ----------
    likelihoods_options : dict, list[dict]
        List of options {'observables': [observable_options, ...], 'covariance': {}}.
    cosmology_options : optional
        Cosmology options or object or :class:`desilike.theories.Cosmoprimo`.
    get_stats_fn : callable, optional
        Function to locate measurement files.
    cache_dir : str | Path, optional
        Directory used for caching pre-computed emulators.
    cache_mode : str, optional
        'rw' for read/write; 'r' for read-only.

    Returns
    -------
    SumLikelihood
    """
    from desilike.base import SumLikelihood
    cosmology = get_cosmology(cosmology_options)
    if isinstance(likelihoods_options, dict):
        likelihoods_options = [likelihoods_options]
    likelihoods = []
    for likelihood_options in likelihoods_options:
        stats = likelihood_options.pop('stats', None)
        likelihood = get_single_likelihood(likelihood_options, cosmology_options=cosmology,
                                           stats=stats, get_stats_fn=get_stats_fn, get_theory=get_theory,
                                           cache_dir=cache_dir, cache_mode=cache_mode)
        likelihoods.append(likelihood)
    return SumLikelihood(likelihoods)


def get_prior(likelihood):
    import jax.numpy as jnp
    from desilike import Prior, get_params

    class CustomPrior(Prior):
      """Hard constraint w0 + wa < 0, on top of individual parameter priors."""

      def __call__(self):
          self.logpdf = super().__call__()
          if 'w0_fld' in self.params:
              w0, wa = self.params['w0_fld'], self.params['wa_fld']
              self.logpdf = jnp.where(w0.value + wa.value < 0., self.logpdf, -jnp.inf)
          return self.logpdf

    return CustomPrior(get_params(likelihood))


def get_sampler_cls(name):
    """Return sampler class."""
    from desilike.samplers import Emcee, Zeus, MH, PocoMC, Nautilus, BlackjaxNUTS, NumpyroNUTS, NumpyroBarkerMH
    translate = {'emcee': Emcee, 'zeus': Zeus, 'mh': MH, 'pocomc': PocoMC, 'nautilus': Nautilus, 'nuts': BlackjaxNUTS, 'numpyro_nuts': NumpyroNUTS, 'numpyro_barker': NumpyroBarkerMH}
    return translate[name.lower()]


def get_profiler_cls(name):
    """Return profiler class."""
    from desilike.profilers import Minuit
    translate = {'minuit': Minuit}
    return translate[name.lower()]


def propose_fiducial_observable_options(stat, tracer=None, zrange=None):
    """Propose fiducial fitting options for given statistics and tracer."""
    propose_fiducial = {'stat': {'kind': stat},
                        'catalog': {'weight': 'default-FKP'},
                        'window': {'templates': []},
                        'theory': {},
                        'emulator': {'name': 'taylor', 'order': 3}}
    propose_stat = {'mesh2_spectrum': {'select': [{'ells': ell, 'k': [0.02, 0.2, 0.01]} for ell in [0, 2]]},
                    'mesh3_spectrum': {'select': [{'ells': (0, 0, 0), 'k': [0.02, 0.12, 0.01]}, {'ells': (2, 0, 2), 'k': [0.02, 0.08, 0.01]}],
                                        'basis': 'sugiyama-diagonal'},
                   'recon_particle2_correlation': {'select': [{'ells': ell, 's': [60., 150., 4.]} for ell in [0, 2]]},
                   'recon_bao': {}}
    base_full_shape_theory = {'model': 'folpsD', 'prior_basis': 'physical_aap', 'damping': 'vdg', 'marg': True}
    base_bao_theory = {'model': 'bao', 'broadband': 'pcs2', 'marg': True}
    propose_theory = {'mesh2_spectrum': base_full_shape_theory | {'damping_method': 'tree+loop', 'coevolution': '', 'A_full': False},
                      'mesh3_spectrum': base_full_shape_theory | {'A_full': False},
                      'recon_particle2_correlation': base_bao_theory,
                      'recon_bao': {}}
    for _stat in propose_stat:
        if _stat in stat:
            propose_fiducial['stat'].update(propose_stat[_stat])
            propose_fiducial['theory'].update(propose_theory[_stat])
    return propose_fiducial


def propose_fiducial_covariance_options():
    """Return dictionary of default covariance options."""
    return {'source': 'mock', 'version': 'holi-v3-altmtl', 'corrections': ['hartlap', 'percival']}


def propose_fiducial_cosmology_options():
    """Return dictionary of default cosmology options."""
    return {'model': 'base_ns-fixed', 'template': 'direct'}


def propose_fiducial_sampler_options(sampler=None):
    """Return dictionary of default sampler configuration."""
    if sampler is None:
        sampler = 'emcee'
    init, run = {}, {}
    init['rng'] = 42
    if sampler in ['emcee', 'zeus', 'mhmcmc', 'nuts', 'numpyro_nuts', 'numpyro_barker']:
        init['nparallel'] = 4
        run['min_steps'] = 50
        run['gelman_rubin'] = 1.05
        run['ess'] = 400
    if sampler in ['emcee']:
        init['batch_size'] = 16
        run['thinning'] = 5
    if sampler in ['nuts']:
        init['rescale'] = 'diag'
        init['step_size'] = 0.1
        #run['adaptation'] = dict(initial_step_size=0.01, target_acceptance_rate=0.8, steps=1000, is_mass_matrix_diagonal=False)
        run['adaptation'] = dict(initial_step_size=0.01, target_acceptance_rate=0.8, steps=1000, is_mass_matrix_diagonal=False)
    if sampler in ['numpyro_nuts', 'numpyro_barker']:
        init['rescale'] = 'diag'
        init['step_size'] = 0.1
        run['adaptation'] = dict(steps=500, dense_mass=True)
    if sampler in ['mhmcmc']:
        run['check_every'] = 1000
    if sampler in ['nautilus']:
        init['rescale'] = 'diag'
        init['n_live'] = 1000
        run['n_eff'] = 200
        run['verbose'] = True
    if sampler in ['pocomc']:
        init['batch_size'] = 32
        # Default settings
        #init['n_effective'] = 512
        #init['n_active'] = 256
        #run['n_total'] = 4096  # ESS
        # n_effective *and* n_active must be high enough to get the tails right
        # rescale helps (in case variations of one parameter are much smaller than the others)
        init['rescale'] = 'diag'
        #init['prior'] = 2.
        init['n_effective'] = 1024
        init['n_active'] = 512
        #init['n_effective'] = 2048
        #init['n_active'] = 1024
        init['flow'] = 'nsf6'  # default
        #init['train_config'] = {'epochs': 10000, 'patience': 50, 'batch_size': 512, 'learning_rate': 1e-3}
        run['n_total'] = 2048  # ESS
    fiducial_options = {'sampler': sampler, 'init': init, 'run': run}
    return fiducial_options


def propose_fiducial_profiler_options(profiler=None):
    """Return dictionary of default profiler configuration."""
    if profiler is None:
        profiler = 'minuit'
    fiducial_options = {'profiler': profiler, 'init': {}, 'maximize': {'niterations': 1}}
    return fiducial_options


def fill_fiducial_observable_options(options):
    """Fill missing observable options with fiducial values."""
    options = dict(options)
    stat = options['stat']['kind']
    tracer = options['catalog'].get('tracer', None)
    zrange = options['catalog'].get('zrange', None)
    fiducial_options = propose_fiducial_observable_options(stat, tracer, zrange)
    # ``mesh2_spectrum`` defaults are defined for folpsD: do not propose the folpsD-only
    # damping settings for another model. Dropped from the *defaults*, before the merge below
    # brings them in -- an explicitly requested damping is the caller's business and stands.
    model = options.get('theory', {}).get('model', fiducial_options['theory'].get('model'))
    if model != 'folpsD':
        fiducial_options['theory'].pop('damping', None)
        fiducial_options['theory'].pop('damping_method', None)
    options = fiducial_options | options
    for key, value in fiducial_options.items():
        options[key] = value | options[key]
    # Normalise 'pdf+lor' -> {'mode': 'pdf+lor', <get_stats_fn keys>} once, here, so the rest of
    # the pipeline only reads it. Idempotent, and validating at options-construction time means
    # a bad mode raises before any file is touched.
    redshift_smearing = _format_redshift_smearing(options.get('redshift_smearing', None))
    if redshift_smearing is not None:
        options['redshift_smearing'] = redshift_smearing
    return options


def fill_fiducial_likelihood_options(options):
    """Fill missing likelihood options with fiducial values."""
    if isinstance(options, dict):
        options = dict(options)
        options['observables'] = [fill_fiducial_observable_options(obs_opts) for obs_opts in options['observables']]
        options['covariance'] = propose_fiducial_covariance_options() | (options.get('covariance', {}) or {})
        return options
    return type(options)(fill_fiducial_likelihood_options(opts) for opts in options)


def fill_fiducial_options(options):
    """Fill missing options with fiducial values."""
    options = dict(options)
    options['cosmology'] = propose_fiducial_cosmology_options() | {'template': 'direct'} | options.get('cosmology', {})
    likelihoods = options.get('likelihoods', None)
    if likelihoods is not None:
        options['likelihoods'] = fill_fiducial_likelihood_options(options['likelihoods'])
        # Add cosmology arguments to each observable
        for likelihood_options in options['likelihoods']:
            for observable_options in likelihood_options['observables']:
                observable_options['theory']['cosmology'] = options['cosmology']
    for name in ['sampler', 'profiler']:
        options.setdefault(name, {})
        options[name] = globals()[f'propose_fiducial_{name}_options'](options[name].get(name)) | options[name]
    return options


def generate_likelihood_options_helper(stats=('mesh2_spectrum', 'mesh3_spectrum'),
                                       tracer='LRG', zrange=None, region='GCcomb',
                                       version='abacus-hf-dr2-v2-altmtl',
                                       imock=None,
                                       covariance='holi-v3-altmtl',
                                       stats_dir=base_stats_dir,
                                       project='full_shape/base',
                                       emulator=True):
    """
    Convenience helper that builds a minimal dictionary of likelihood options.

    Parameters
    ----------
    stats : list
        List of statistics in the joint likelihood, from ['mesh2_spectrum', 'mesh3_spectrum']
    tracer : str or tuple (for cross correlation), or list of str or tuple to account for cross-correlation between tracers
        Tracers to fit.
    zrange : tuple
        Redshift range.
    region : str
        Sky region.
    version : str
        Version of data to use.
    imock : int
        If data is a mock, mock number.
    covariance : str
        Version of covariance mocks to use.
    project : str, optional
        Optional project subdirectory passed through to the measurement path builder.

    Returns
    -------
    likelihood_options : dict
        Dictionary with keys ['observables', 'covariance'].
        'covariance' is a dictionary specifying how to construct the covariance matrix.
        'observables' contains a list of dictionary (one for each observable), with keys:
        {'stat': {'kind': ..., 'basis': ..., 'select': [...]}, 'catalog': {'version':, ...}, 'theory': {'model': ...}, 'window': {}}

    """
    if isinstance(stats, str):
        stats = [stats]
    tracers = [tracer] if isinstance(tracer, (str, tuple)) else tracer
    zranges = [zrange] if not isinstance(tracer[0], (tuple, list)) else zrange
    observables = []
    for tracer, stat in itertools.product(tracers, stats):
        tracer, zrange = get_full_tracer_zrange(tracer, zrange)
        catalog = {'version': version, 'tracer': tracer, 'zrange': zrange, 'region': region, 'stats_dir': stats_dir, 'imock': imock}
        if 'data' not in version and imock is None:
            catalog['imock'] = '*'  # read all available mocks
        observable_options = {'stat': {'kind': stat}, 'catalog': catalog}
        if project:
            observable_options['stat']['project'] = project
        if emulator is False: emulator_options = {'name': ''}
        elif emulator is True: emulator_options = {}
        else: emulator_options = dict(emulator)
        observable_options['emulator'] = emulator_options
        observables.append(observable_options)
    covariance = {'version': covariance, 'stats_dir': stats_dir}
    return fill_fiducial_likelihood_options({'observables': observables, 'covariance': covariance})


def get_full_tracer_zrange(tracerz=None, zrange=None):
    """
    Translate simple tracer labels, (e.g. LRG1),
    to full tracer and zrange tuples ('LRG', (0.4, 0.6)).

    Parameters
    ----------
    tracerz : str, tuple, list, None
        If None returns the mapping table. If tracerz is a string returns
        (tracer, zrange) or for compound tracer strings returns zipped tuples.

    Returns
    -------
    tracer, zrange
    """
    translate_zrange = {'BGS1': (0.1, 0.4),
                        'LRG1': (0.4, 0.6), 'LRG2': (0.6, 0.8), 'LRG3': (0.8, 1.1),
                        'LGE1': (0.4, 0.6), 'LGE2': (0.6, 0.8), 'LGE3': (0.8, 1.1),
                        'ELG1': (0.8, 1.1), 'ELG2': (1.1, 1.6),
                        'QSO1': (0.8, 2.1)}
    if tracerz is None:
        return translate_zrange

    def _get_full_tracer_zrange(tracerz, zrange=zrange):
        if 'x' in tracerz:
            return list(zip(*[_get_full_tracer_zrange(t, zrange=zrange) for t in tracerz.split('x')]))
        if tracerz in translate_zrange:
            # Return tracer and z-range from translate_zrange
            tracer = tracerz[:-1]
            zrange = translate_zrange[tracerz]
        else:
            # Not in translate_zrange
            tracer = tracerz
        if zrange is None:
            raise ValueError(f'zrange not found for {tracerz}; choose one from {list(translate_zrange)}')
        return tracer, zrange

    if isinstance(tracerz, str):
        return _get_full_tracer_zrange(tracerz)
    else:  # tuple/list of tracers
        return type(tracerz)(zip(*map(_get_full_tracer_zrange, tracerz)))


def _get_level(level: int | dict=None):
    """Compact helper to normalise verbosity level for string helpers."""
    _default_level = {'stat': 1, 'catalog': 1, 'window': 1, 'redshift_smearing': 1, 'theory': 0, 'covariance': 0, 'cosmology': 1}
    if level is None: level = {}
    if not isinstance(level, dict):
        level = {name: level for name in _default_level}
    level = _default_level | level
    return level


def _base_type_options(options):
    """
    Recursively cast objects of input dictionary ``d`` to Python base types
    so they can be serialized by standard YAML.
    """
    def convert(v):
        if isinstance(v, dict):
            return {k: convert(vv) for k, vv in v.items()}
        if isinstance(v, (list, tuple, set, frozenset)):
            return [convert(vv) for vv in v]
        if isinstance(v, np.ndarray):
            if v.size == 1:
                return convert(v.item())
            return [convert(vv) for vv in v.tolist()]
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, (np.bool_,)):
            return bool(v)
        if v is None or isinstance(v, (bool, numbers.Number, str)):
            return v
        return str(v)
    return convert(options)


def _get_emulator_theory_grid_signature(observable):
    """Return a cache-stable description of the initialized observable theory grid."""
    window = getattr(observable, 'window', None)
    theory = getattr(window, 'theory', None)
    if theory is None:
        theory = getattr(observable, 'theory', None)
    if theory is None:
        return None
    if hasattr(theory, 'items'):
        leaves = []
        for label, leaf in theory.items(level=None):
            coords = {}
            for coord_name in leaf.coords():
                coords[coord_name] = np.asarray(leaf.coords(coord_name))
            leaves.append({'label': dict(label), 'coords': coords})
        return leaves
    signature = {}
    for name in ['k', 's', 'ells']:
        if hasattr(theory, name):
            signature[name] = getattr(theory, name)
    return signature or None


def _get_emulator_theory_size(observable):
    window = getattr(observable, 'window', None)
    theory = getattr(window, 'theory', None)
    if theory is not None and hasattr(theory, 'size'):
        return int(theory.size)
    flatdata = getattr(observable, 'flatdata', None)
    if flatdata is not None:
        return int(np.size(flatdata))
    return None


def _get_emulator_cache_options(observable_options, observable=None):
    """Return options that define the reusable Taylor-emulator domain."""
    options = {
        name: observable_options[name]
        for name in ['stat', 'catalog', 'emulator']
        if name in observable_options
    }
    if 'theory' in observable_options:
        theory_options = dict(observable_options['theory'])
        theory_options.pop('damping', None)
        theory_options.pop('damping_method', None)
        options['theory'] = theory_options
    if 'window' in observable_options:
        options['window'] = observable_options['window']
    if observable is not None:
        grid = _get_emulator_theory_grid_signature(observable)
        if grid is not None:
            options['theory_grid'] = grid
    return options


def _hash_options(options, length=8):
    """Return a short SHA-256 hash of a canonicalized options dict."""
    def _canonical(obj):
        if isinstance(obj, dict):
            return sorted((_canonical(k), _canonical(v)) for k, v in obj.items())
        if isinstance(obj, list):
            return [_canonical(x) for x in obj]
        return obj
    s = json.dumps(_canonical(_base_type_options(options)), sort_keys=True)
    return hashlib.sha256(s.encode()).hexdigest()[:length]


def _str_from_observable_options(options: dict, level: int=None) -> str:
    """Return string identifier given input observable options, with ``level`` of details."""
    level = _get_level(level)
    out_str = []

    # First, catalog
    catalog = _unzip_catalog_options(options['catalog'])

    def _str_zrange(zrange):
        return f'z{float2str(zrange[0], prec_min=1, prec_max=5)}-{float2str(zrange[1], prec_min=1, prec_max=5)}'

    if level['catalog'] >= 1:
        translate_tracerz = get_full_tracer_zrange(tracerz=None)
        catalog_str = []
        for tracer in catalog:
            stracer = get_simple_tracer(tracer)
            catalog_options = catalog[tracer]
            found = False
            if 'zrange' in catalog_options:
                for tracerz, zrange in translate_tracerz.items():
                    if tracerz.startswith(stracer) and np.allclose(catalog_options['zrange'], zrange):
                        stracer = tracerz  # e.g. LRG1
                        found = True
                        break
            tracer_catalog_str = [stracer]
            if 'zrange' in catalog_options:
                if not found or level['catalog'] >= 2:
                    tracer_catalog_str.append(_str_zrange(catalog_options['zrange']))
            elif 'zsnap' in catalog_options:
                tracer_catalog_str.append(f'z{float2str(catalog_options["zsnap"], prec_min=1, prec_max=5)}')
            if level['catalog'] >= 3:
                tracer_catalog_str.append(catalog_options['region'])
            if level['catalog'] >= 4:
                tracer_catalog_str.append('weight-' + catalog_options['weight'])
            catalog_str.append('-'.join(tracer_catalog_str))
        out_str.append('x'.join(catalog_str))

    # Then, stat and select, e.g. S2-ell0-k-0.02-0.2-ell2-k-0.02-0.2
    translate_stat_name = {'S2': ['mesh2_spectrum'],
                           'S3': ['mesh3_spectrum'],
                           'BAOR': ['bao', 'recon'],
                           'C2R': ['particle2_correlation', 'recon']}
    stat_options = options['stat']
    stat = stat_options['kind']
    if level['stat'] >= 1:
        found = None
        for name in translate_stat_name:
            if all(t in stat for t in translate_stat_name[name]):
                found = name
                break
        if found is None:
            raise ValueError(f'could not find shot name for {stat}')
        out_str.append(found)
    if level['stat'] >= 2:
        select_str = []
        select = stat_options.get('select', [])
        if callable(select):  # custom binning
            select_str.append(getattr(select, 'name', 'custom'))
        else:
            def _str_ell(ell):
                if isinstance(ell, (list, tuple)):
                    ell = ''.join([str(ell) for ell in ell])
                else:
                    ell = str(ell)
                return ell

            for _select in select:
                _select = dict(_select)
                label = []
                for key in list(_select):
                    if key == 'ells':
                        label.append('ell' + _str_ell(_select.pop(key)))
                for coord_name, limits in _select.items():
                    prec = dict(prec_min=2, prec_max=3) if name.startswith('S') else dict(prec_min=0, prec_max=0)
                    label.append(coord_name + '-'.join(float2str(lim, **prec) for lim in limits))
                select_str.append('-'.join(label))
        select_str = '-'.join(select_str)
        out_str.append(select_str)
    if level['window'] > 0:
        templates = list(options.get('window', {}).get('templates', []))
        if templates:
            out_str.append('w')
            out_str.extend(templates)

    if level['redshift_smearing'] > 0:
        redshift_smearing = _format_redshift_smearing(options.get('redshift_smearing', None))
        if redshift_smearing is not None:  # only when set, so the default stays byte-identical
            out_str.append('rs-' + redshift_smearing['mode'].replace('+', '-'))

    if level['theory'] > 0:
        out_str.append('th')
        out_str.append(options['theory']['model'])

    return '-'.join(out_str)


def str_from_likelihood_options(likelihood_options, level: int | dict=None):
    """
    Return a compact string identifier for likelihood options.

    Parameters
    ----------
    likelihood_options : dict
        Dictionary with keys 'observables', 'covariance'.
    level : dict
        "Verbosity level". Default is {'stat': 1, 'catalog': 1, 'theory': 0, 'covariance': 0}.
        Increase for more details.
        Covariance level behavior:
        - > 0: include covariance version
        - >= 3: include covariance corrections and optional nparams
    """
    level = _get_level(level)
    out_str = []
    for options in likelihood_options['observables']:
        out_str.append(_str_from_observable_options(options, level=level))
    if level['covariance'] > 0:
        covariance = likelihood_options.get('covariance', {}) or {}
        covariance_str = []
        for name in ['source', 'version']:
            if name == 'version':
                covariance_str.append(_get_covariance_display_version(covariance))
            else:
                value = covariance.get(name, 'none')
                covariance_str.append('none' if value is None else str(value))
        covariance_str = ['cov-' + '-'.join(covariance_str)]
        if level['covariance'] >= 3:
            corrections = covariance.get('corrections', None)
            if isinstance(corrections, str):
                corrections = [corrections]
            corrections = sorted(str(corr).lower() for corr in (corrections or []))
            if corrections:
                covariance_str.append('corr-' + '-'.join(corrections))
            nparams = covariance.get('nparams', None)
            if nparams is not None:
                covariance_str.append(f'nparams-{int(nparams)}')
        out_str.append('-'.join(covariance_str))
    return '+'.join(out_str)


def str_from_cosmology_options(cosmology_options: dict, level: int | dict=None):
    """
    Return a compact string identifier for cosmology options.

    Parameters
    ----------
    cosmology_options : dict
        Dictionary with keys 'model', 'template'.
    level : dict
        "Verbosity level". Default is {'cosmology': 1}.
        Increase for more details.
    """
    level = _get_level(level)
    out_str = []
    if level['cosmology'] >= 1:
        model, template = cosmology_options['model'], cosmology_options['template']
        if template.lower() == 'direct':
            out_str.append(f'cosmo-{model}')
        else:
            out_str.append(f'template-{template}')
    return '-'.join(out_str)


def str_from_options(options: dict, level: int | dict=None):
    """
    Return a compact string identifier for options.

    Parameters
    ----------
    options : dict
        Dictionary with keys 'likelihoods', 'cosmology'.
    level : dict
        "Verbosity level". Default is {'stat': 1, 'catalog': 1, 'theory': 0, 'covariance': 0, 'cosmology': 1}.
        Increase for more details.
    """
    level = _get_level(level)
    out_str = [str_from_cosmology_options(options['cosmology'], level=level)]
    out_str += [str_from_likelihood_options(likelihood_options, level=level) for likelihood_options in options['likelihoods']]
    return '_'.join(out_str)


def get_fits_fn(fits_dir=Path(os.getenv('SCRATCH', '.')) / 'fits',  project='', kind='chain', likelihoods: list=None,
                sampler: dict=None, profiler: dict=None, cosmology: dict=None, ichain: int=None,
                level=None, extra='', ext='h5', **kwargs):
    """
    Construct a file path for fit outputs based on likelihood and run options.

    Parameters
    ----------
    fits_dir : str, Path
        Base directory for fit outputs.
    project : str
        KP analysis to which the measurement corresponds. For example: 'full_shape/base', 'local_png/base', 'bao/base', etc.
    kind : str
        Fitting product. Options are 'samples', 'profiles', etc.
    likelihoods : list
        Likelihood options used to build the filename.
    ichain : int or None
        Optional chain index appended to filename.
    extra : str, optional
        Extra suffix to include in the path.
    ext : str, optional
        File extension. Default is 'h5'.

    Returns
    -------
    fn : Path
        Fit file name.
    """
    fits_dir = Path(fits_dir)
    fits_dir = fits_dir / project
    imock = kwargs.get('imock', None)
    if imock is not None:
        imock = str(imock)
    if project and imock:
        fits_dir = fits_dir / f'mock{imock}'
    options = {'likelihoods': likelihoods, 'cosmology': cosmology}
    _str_from_options = str_from_options(options, level=level)
    _hash = _hash_options(options)
    extra = f'_{extra}' if extra else ''
    ichain = f'_{ichain:d}' if ichain is not None else ''
    dirname_label = f'{_str_from_options}-{_hash}{extra}'.replace('.', 'p')
    dirname = fits_dir / dirname_label
    basename = f'{kind}{ichain}.{ext}'
    if ichain == '_*':
        return dirname.glob(basename)
    return dirname / basename


try:
    import yaml
except ImportError:
    yaml = None


def write_options(filename, options):
    """Write options to ``filename``."""
    options = _base_type_options(options)

    class FlowList(list):
        pass

    def flow_list_representer(dumper, data):
        return dumper.represent_sequence(
            'tag:yaml.org,2002:seq',
            data,
            flow_style=True,
        )

    yaml.add_representer(FlowList, flow_list_representer)

    def mark_flow_lists(obj):
        if isinstance(obj, dict):
            return {k: mark_flow_lists(v) for k, v in obj.items()}
        if isinstance(obj, list):
            obj = [mark_flow_lists(v) for v in obj]
            # choose the lists you want inline
            if all(not isinstance(v, (dict, list)) for v in obj):
                return FlowList(obj)
            return obj
        return obj

    # To use flow style for simple lists
    options = mark_flow_lists(options)
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    with open(filename, 'w') as file:
        yaml.dump(options, file, sort_keys=False, default_flow_style=False)


def read_options(filename):
    """Read options from ``filename``."""

    class YamlLoader(yaml.SafeLoader):
        """
        *yaml* loader that correctly parses numbers.
        Taken from https://stackoverflow.com/questions/30458977/yaml-loads-5e-6-as-string-and-not-a-number.
        """

    # https://stackoverflow.com/questions/30458977/yaml-loads-5e-6-as-string-and-not-a-number
    YamlLoader.add_implicit_resolver(u'tag:yaml.org,2002:float',
                                     re.compile(u'''^(?:
                                     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
                                     |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
                                     |\\.[0-9_]+(?:[eE][-+][0-9]+)?
                                     |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
                                     |[-+]?\\.(?:inf|Inf|INF)
                                     |\\.(?:nan|NaN|NAN))$''', re.X),
                                     list(u'-+0123456789.'))

    YamlLoader.add_implicit_resolver('!none', re.compile('None$'), first='None')

    def none_constructor(loader, node):
        return None

    YamlLoader.add_constructor('!none', none_constructor)

    with open(filename, 'r') as file:
        return yaml.load(file, Loader=YamlLoader)
