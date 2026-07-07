import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


logger = logging.getLogger('PNG fitting tools')


def read_data(data_dir='.', mocks_dir=None, 
              tracer='LRG', zrange=(0.4, 1.1), weight_type='default-fkp-oqe', region='GCcomb', 
              window_extra='', **kwargs):
    """ 
    Read the data from the clustering statistics output. This is a wrapper of clustering_statistics.tools.get_stats_fn.
    """
    import lsstypes
    from clustering_statistics.tools import get_stats_fn

    # Read the data:
    fn = get_stats_fn(kind='mesh2_spectrum', stats_dir=data_dir, tracer=tracer, zrange=zrange, weight=weight_type, region=region)
    pk = lsstypes.read(fn)
    logger.info(f'Reading the data with {weight_type=} from {fn}')
    window = lsstypes.read(fn)

    # Read the window matrix:
    tracer_window = kwargs.get('tracer_window', tracer)
    fn = get_stats_fn(kind='window_mesh2_spectrum', stats_dir=data_dir, tracer=tracer_window, zrange=zrange, weight=weight_type, region=region, extra=window_extra)
    logger.info(f'Reading the window with {tracer_window=}, {weight_type=},{window_extra=} from {fn}')
    window = lsstypes.read(fn)

    # Domitille FM window computation add an artificat on the region k (input space) > k_Nyquist (observable space) that bias the convolved theory at large scales.. (see validation_window.ipynb).
    # This k > k_Nyquist in the input space is irrelevant. Just remove it ! Here we keep only k < 0.1.
    window = window.at.theory.select(k=(1e-4,0.1))

    # Read the analytical covariance matrix:
    try: 
        tracer_cov = kwargs.get('tracer_cov', tracer)
        fn = get_stats_fn(kind='covariance_mesh2_spectrum', stats_dir=data_dir, tracer=tracer_cov, zrange=zrange, weight=weight_type, region=region)
        cov = lsstypes.read(fn)
    except:
        logger.info('Do not find the analytical covariance matrix. Please provide mocks_dir to estimate the covariance matrix from mocks.')
        cov = None

    # Read the mocks:
    mocks = None
    if mocks_dir is not None: 
        weight_type_mocks = kwargs.get('weight_type_mocks', weight_type)
        nmocks = kwargs.get('nmocks', 1000 if weight_type_mocks == 'default-fkp-oqe' else 100)
        tracer_mocks = kwargs.get('tracer_mocks', tracer)
        logger.info(f"Reading {nmocks=} for {tracer=} with {tracer_mocks=} {weight_type_mocks=}.")

        fns_mock = [get_stats_fn(kind='mesh2_spectrum_poles', stats_dir=mocks_dir, project='holi-v3-altmtl', tracer=tracer_mocks, region=region, zrange=zrange, 
                                 weight=weight_type_mocks, imock=imock) for imock in range(nmocks)]    
        mocks = [lsstypes.read(fn) for fn in fns_mock]

    return pk, window, cov, mocks


def rebin_data(pk, window, cov, mocks, tracer='LRG', kmin=1e-3, kmax=0.08, kpivot=[1e-2, 2e-2], nrebin=[2, 2], use_ell2=True, rebin_ell2=True, **kwargs):
    """ 
    Rebin the data with k > kpivot by a factor nrebin. The quadrupole is rebinned again by a factor nrebin for the full range.
    Then, select data in the k range [kmin, kmax]. If use_ell2 is False, we only keep the monopole.
    Finally, we match the size of the window and covariance to the size of the power spectrum.
    Return the rebinned power spectrum, window and covariance.
    """
    if not isinstance(kpivot, (list, tuple)): kpivot = [kpivot]
    if not isinstance(nrebin, (list, tuple)): nrebin = [nrebin]
    assert len(kpivot) == len(nrebin), "kpivot and nrebin should have the same length."
    
    # Let's rebin the power spectrum : 
    for pivot, rebin in zip(kpivot, nrebin):
        pk = pk.map(lambda pole: pole.at(k=(pivot, 1.)).select(k=slice(0, None, rebin)))
        
    if rebin_ell2:
        # Rebin the quadrupole again but for the full range.
        pk = pk.at(2).at(k=(0, 1e-2)).select(k=slice(0, None, 2))  
        pk = pk.at(2).select(k=slice(0, None, 2))  

    # Let's select the k range and ells:
    kmin_ell2, kmax_ell2 = kwargs.get('kmin_ell2', kmin), kwargs.get('kmax_ell2', kmax)
    pk = pk.at(0).select(k=(kmin, kmax))
    pk = pk.at(2).select(k=(kmin_ell2, kmax_ell2))
    pk = pk.get(ells=[0]) if not use_ell2 else pk.get(ells=[0, 2])
 
    # Match the size of wmatrix and covariance: 
    if window is not None: window = window.at.observable.match(pk)
    
    if cov is not None: 
        tracers = tuple(tracer.split('x')) 
        if len(tracers) == 1: tracers *= 2
        tracers = (tracers[0][:3], tracers[1][:3])  # LRG_zcmb -> LRG, ELGnotqso -> ELG, ... 
        cov = cov.at.observable.at(observables='spectrum2', tracers=tracers).match(pk)

    if mocks is not None:
        mocks = [mock.match(pk) for mock in mocks]
    
    logger.debug(f'After rebinning and k range selection: {pk.get(0).k.shape[0]} and {pk.get(2).k.shape[0] if use_ell2 else "Not used"} data points.')

    return pk, window, cov, mocks


def fix_likelihood_bias_and_damping(likelihood, tracer, zeffs, derived_cross_bias=True, nickname=None, available_tracers=None, bias_params=None, **kwargs):
    """Apply bias and damping parameter relations between the paramters of the likelihood both for ell=0/2 and auto/cross power spectrum.

    Parameters
    ----------
    likelihood : desilike likelihood
        Likelihood whose parameters are updated in place.
    tracer : str
        name of the tracers: 'LRGxLRG', 'LRGxQSO', ...
    nickname : str, optional
        Suffix inserted between the tracer shortname and '_ell{ell}' in the cross-correlation theory
        parameter names (e.g. nickname='LRGxELG' -> 'ELG_LRGxELG_ell0'). Must match the nickname used in
        get_obervable_and_likelihood. Default is None (no suffix, legacy behaviour).
    available_tracers : list or set, optional
        List of tracers for which complete data (including auto-correlations) is available.
        Used to determine which cross-correlation biases can be derived. If None, assumes all
        auto-correlations are available.
    bias_params : dict or tuple, optional
        Override for the fiducial b(z) = alpha * (1 + z)**2 + beta model used to relate biases across
        multipoles and across auto/cross at different effective redshifts. Either a dict mapping the
        3-letter tracer ('LRG', 'QSO', ...) to (alpha, beta), or a single (alpha, beta) tuple applied to
        all tracers. For mocks without redshift evolution, pass alpha=0 (e.g. (0., 1.)) so every derived
        ratio is 1. If None (default), the data-fiducial clustering_statistics.tools.bias is used.
        Note: this only changes the likelihood; the OQE weights and window matrix (which used the data
        fiducial b(z)) are left untouched.

    Returns
    -------
    likelihood
        Same likelihood object, updated in place.
    """
    from desilike import get_params
    all_params = get_params(likelihood)

    def _rescale_bias_params(tracer, zeff):
        """ 
        Fix the bias parameters in the likelihood according to the redshift dependence of the bias.
        Only modifies the parameter if both source and target exist in the likelihood.

        Args:
            likelihood: The likelihood object.
            tracer: The tracer for which to fix the bias parameters. tracer[0] is source, tracer[1] is target.
            zeff: The effective redshifts.
        """
        from clustering_statistics import tools
        source_param_name = f"{tracer[0]}.b1"
        target_param_name = f"{tracer[1]}.b1"

        # Both parameters must exist to create a derived relationship
        if source_param_name not in all_params or target_param_name not in all_params:
            logger.debug(f"Skipping derived relationship: {source_param_name} -> {target_param_name} (missing parameter)")
            return

        # b(z) = alpha * (1 + z)**2 + beta
        tt = tracer[0][:3]
        if bias_params is not None:
            alpha, beta = bias_params[tt] if isinstance(bias_params, dict) else bias_params
        else:
            alpha, beta = tools.bias(1, tracer=tt, return_params=True)
        factor = (alpha * (1 + zeff[1])**2 + beta) / (alpha * (1 + zeff[0])**2 + beta)
        all_params[target_param_name].update(derived=f'b1 * {factor}', depends={'b1': all_params[source_param_name]}, prior=None)
        logger.debug(f"Derived relationship: {target_param_name} = {source_param_name} * {factor}")

    tracers = tracer.split('x')

    # Auto-correlation: link ell2 bias / damping to ell0.
    if tracers[0] == tracers[1]:
        if len(zeffs[tracer]) > 1:
            zeff = [zeffs[tracer][ell] for ell in [0, 2]]
            _rescale_bias_params(tracer=[f"{tracers[0]}_ell0", f"{tracers[0]}_ell2"], zeff=zeff)
            # logger.warning('we neglect the redshift dependence of the damping term, for now') 
            param_name_ell0, param_name_ell2 = f"{tracers[0]}_ell0.sigmas", f"{tracers[0]}_ell2.sigmas"
            all_params[param_name_ell2].update(derived='sigmas', depends={'sigmas': all_params[param_name_ell0]}, prior=None)
            logger.debug(f"Derived damping: {param_name_ell2} = {param_name_ell0}")

    # Cross-correlation: 
    # if derived_cross_bias is False: let free only one cross-correlation bias and fix the other one to break degeneracy.
    # if derived_cross_bias is True: By default, link the bias of the cross-correlation to their auto-correlation counterparts. 
    #                                If the auto-correlation is not available in available_tracers, use one of the cross-correlation bias as default to link #                                all the others one to him.
    if tracers[0] != tracers[1]:
        cross_suffix = f'_{nickname}' if nickname is not None else ''
        for i, tt in enumerate(tracers):
            if derived_cross_bias and (available_tracers is not None):
                # if auto-tracer is available in available_tracers:
                if 'x'.join([tt, tt]) in available_tracers:
                    # derived the bias from the auto-correlation bias, taking into account the different effective redshifts of the auto and cross correlation.
                    zeff = [zeffs['x'.join([tt, tt])][0], zeffs[tracer][0]]
                    _rescale_bias_params(tracer=[f"{tt}_ell0", f'{tt}{cross_suffix}_ell0'], zeff=zeff)
                else:
                    # determine default tracer to link the bias parameters, if auto-correlation data is not available.
                    default_tracer = sorted([tracer for tracer in available_tracers if tt in tracer.split('x')])[0]
                    if tracer == default_tracer:
                        logger.debug(f'This parameter is free ({tt}, {tracer}), and it will be used as default to link the other cross-correlation bias parameters.')
                    else:
                        logger.debug(f'This parameter is free ({tt}, {tracer}), but it will be linked to {default_tracer} bias parameters to break degeneracy, since auto-tracer data for {tt} is not available.')
                        zeff = [zeffs[default_tracer][0], zeffs[tracer][0]]
                        _rescale_bias_params(tracer=[f"{tt}_{default_tracer}_ell0", f'{tt}{cross_suffix}_ell0'], zeff=zeff)

            else:
                # let free the cross-correlation bias, but fix one of the two biases to break degeneracy.
                # the first linear bias parameter can be set with kwargs.          
                if i == 0:      
                    default_b1 = kwargs.get(f"{tt}{cross_suffix}_ell0.b1", 1)
                    all_params[f"{tt}{cross_suffix}_ell0.b1"].update(value=default_b1, fixed=True) 
    
            if len(zeffs[tracer]) > 1:
                zeff = [zeffs[tracer][ell] for ell in [0, 2]]
                _rescale_bias_params(likelihood, tracer=[f"{tt}{cross_suffix}_ell0", f"{tt}{cross_suffix}_ell2"], zeff=zeff)
                try:
                    # logger.warning('we neglect the redshift dependence of the damping term, for now')
                    # Note: the first damping term is fixed to 0 so only need to update the second one.
                    # prior=None to not double-count priors
                    if i == 1: all_params[f"{tt}{cross_suffix}_ell2.sigmas"].update(derived='sigmas', depends={'sigmas': all_params[f"{tt}{cross_suffix}_ell0.sigmas"]}, prior=None)
                except KeyError as e:
                    # It can happen now when removing the quadrupole from the cross-correlation in the join fit.
                    # (i could update zeff but okay just add more flexibility here)
                    logger.debug(f"Skipping derived relationship: {e} (missing parameter)")


_PNG_CROSS_TWO_REDSHIFT_CLS = None


def _get_png_cross_two_redshift_cls():
    """Lazily build (and cache) the two-redshift local-PNG cross calculator.

    Defined lazily so importing this module does not require desilike. The returned class is a
    subclass of desilike's ``PNGTracerPowerSpectrumMultipoles`` that evaluates the two tracers of a
    cross-correlation at two *different* (snapshot) redshifts: the primary template sits at the
    first tracer's redshift ``z_x`` and the second tracer's redshift ``z_y`` is reached by
    linear growth-rescaling of the same template's matter power and PNG kernel -- exact for two
    snapshots that share initial conditions. The matter cross-power is the geometric mean
    ``D(z_x) D(z_y) P(k, 0) = sqrt(P_dd(z_x) P_dd(z_y))``.

    The FoG keeps desilike's product-of-Lorentzians form; the caller fixes the first tracer's
    ``sigmas`` to 0 so it reduces to a single Lorentzian (one Sigma), matching the cross convention
    used elsewhere in ``get_observable_and_likelihood``. In the limit ``z_y == z_x`` this reproduces
    the standard single-template ``PNGTracerPowerSpectrumMultipoles`` cross.
    """
    global _PNG_CROSS_TWO_REDSHIFT_CLS
    if _PNG_CROSS_TWO_REDSHIFT_CLS is not None:
        return _PNG_CROSS_TWO_REDSHIFT_CLS

    import jax.numpy as jnp
    from desilike.parameter import VariableCollection
    from desilike.theories.galaxy_clustering.png import (
        PNGTracerSpectrum2Poles, FixedSpectrum2Template, _alpha_png, _delta_c, _interp_loglog
    )
    from desilike.theories.galaxy_clustering.power_template import ProjectToPoles

    class PNGTracerCrossTwoRedshift(PNGTracerSpectrum2Poles):
        """Local-PNG cross power spectrum of two tracers at two different snapshot redshifts."""
    
        def __init__(self, k=None, ells=(0, 2), method='prim', mu=10, mode='b-p',
                     tracers=None, nbar=1e-4, params=None, templates=None):
            if mode not in ('b-p', 'bphi', 'bfnl'):
                raise ValueError(f"mode must be one of 'b-p', 'bphi', 'bfnl'; got {mode!r}")
        
            if templates is None or len(templates) != 2:
                raise ValueError('PNGTracerCrossTwoRedshift requires exactly two templates.')
        
            vc = type(self).propose_params(tracers=tracers, mode=mode)
            if params is not None:
                vc = vc + VariableCollection(params)
            assign_params(self, vc, tracers)
        
            self.templates = list(templates)
        
            k_arr = np.linspace(0.01, 0.2, 101) if k is None else np.asarray(k, dtype='f8')
            kin_fine = np.geomspace(min(1e-4, k_arr[0] / 2.), max(1., k_arr[-1] * 2.), 1000)
        
            for template in self.templates:
                template.update(k=kin_fine)
    
        def __post_init__(self, k=None, ells=(0, 2), method='prim', mu=10, mode='b-p',
                          tracers=None, nbar=1e-4, params=None, templates=None):
            if k is None:
                k = np.linspace(0.01, 0.2, 101)
        
            self.k = np.asarray(k, dtype='f8')
            self.ells = tuple(ells)
            self._mode = str(mode)
            self._method = str(method)
            self._z = [float(template.z) for template in self.templates]
            self._nbar = float(nbar)
            self._to_poles = ProjectToPoles(mu=mu, ells=self.ells)
        
            for template in self.templates:
                reqs = {'primordial.pk': [{'k': template.k}]}
                if self._method == 'transfer':
                    reqs.update({
                        'background.growth_factor': [{'z': template.z}, {'z': 10.}],
                        'params.Omega_m': None,
                    })
                template.cosmo.add_requirements(reqs)
                template.cosmo()

        def __call__(self):
            if not isinstance(self.b1, tuple):
                raise ValueError('PNGTracerCrossTwoRedshift requires two tracers.')
        
            templates = self.templates
            if len(templates) != 2:
                raise ValueError('PNGTracerCrossTwoRedshift requires exactly two templates.')
        
            k = self.k[:, None]
            mu = self._to_poles.mu
        
            jac_x, kap_x, muap_x = templates[0].ap_k_mu(k, mu)
            jac_y, kap_y, muap_y = templates[1].ap_k_mu(k, mu)
        
            pk_dd_x = jac_x * _interp_loglog(kap_x, templates[0].k, templates[0].pk_dd)
            pk_dd_y = jac_y * _interp_loglog(kap_y, templates[1].k, templates[1].pk_dd)
            pk_dd = jnp.sqrt(pk_dd_x * pk_dd_y)
        
            pk_prim_x = templates[0].cosmo.get('primordial.pk', k=templates[0].k)
            pk_prim_y = templates[1].cosmo.get('primordial.pk', k=templates[1].k)
        
            h_x = templates[0].cosmo['h']
            h_y = templates[1].cosmo['h']
        
            if self._method == 'transfer':
                Omega_m_x = templates[0].cosmo.get('params.Omega_m')
                Omega_m_y = templates[1].cosmo.get('params.Omega_m')
                growth_x = templates[0].cosmo.get('background.growth_factor', z=self._z[0])
                growth_y = templates[1].cosmo.get('background.growth_factor', z=self._z[1])
                growth_norm_x = templates[0].cosmo.get('background.growth_factor', z=10.)
                growth_norm_y = templates[1].cosmo.get('background.growth_factor', z=10.)
            else:
                Omega_m_x = Omega_m_y = None
                growth_x = growth_y = None
                growth_norm_x = growth_norm_y = None
        
            alpha_fine_x = _alpha_png(templates[0].k, templates[0].pk_dd, pk_prim_x, h_x, self._method,
                                      Omega0_m=Omega_m_x, growth_factor_z=growth_x,
                                      growth_factor_znorm=growth_norm_x)
            alpha_fine_y = _alpha_png(templates[1].k, templates[1].pk_dd, pk_prim_y, h_y, self._method,
                                      Omega0_m=Omega_m_y, growth_factor_z=growth_y,
                                      growth_factor_znorm=growth_norm_y)
        
            alpha_x = _interp_loglog(kap_x, templates[0].k, alpha_fine_x)
            alpha_y = _interp_loglog(kap_y, templates[1].k, alpha_fine_y)
        
            b1_x, b1_y = self.b1
            sigmas_x, sigmas_y = self.sigmas
        
            if self._mode == 'b-p':
                p_x, p_y = self.p
                bfnl_loc_x = 2. * _delta_c * (b1_x - p_x) * self.fnl_loc
                bfnl_loc_y = 2. * _delta_c * (b1_y - p_y) * self.fnl_loc
            elif self._mode == 'bphi':
                bphi_x, bphi_y = self.bphi
                bfnl_loc_x = bphi_x * self.fnl_loc
                bfnl_loc_y = bphi_y * self.fnl_loc
            else:
                bfnl_loc_x, bfnl_loc_y = self.bfnl_loc
        
            b_eff_x = b1_x + bfnl_loc_x * alpha_x
            b_eff_y = b1_y + bfnl_loc_y * alpha_y
        
            f_x = templates[0].f
            f_y = templates[1].f
        
            fog_x = 1. / (1. + sigmas_x**2 * kap_x**2 * muap_x**2 / 2.)
            fog_y = 1. / (1. + sigmas_y**2 * kap_y**2 * muap_y**2 / 2.)
        
            pkmu = fog_x * fog_y * (b_eff_x + f_x * muap_x**2) * (b_eff_y + f_y * muap_y**2) * pk_dd
        
            sn = jnp.array([(ell == 0) for ell in self.ells], dtype='f8')[:, None] * self.sn0 / self._nbar
            self.poles = self._to_poles(pkmu) + sn
            return self.poles

    _PNG_CROSS_TWO_REDSHIFT_CLS = PNGTracerCrossTwoRedshift
    return _PNG_CROSS_TWO_REDSHIFT_CLS


def get_observable_and_likelihood(pk, window, cov, tracer='LRG', zeffs={'LRGxLRG': {0: 0.7, 2: 0.7}}, p={'LRG': 1., 'ELG': 1., 'QSO': 1.4},
                                  fix_fnl=False, engine='camb', scale_covariance=1, nickname=None, bias_params=None, pk_zeff=None, **kwargs):
    """
    Get the observable and likelihood for a given tracer. Each multipole is treated as a different observable, but they share the same parameters in the theory.

    Args:
        pk: lsstypes Mesh2SpectrumPoles observable containing the power spectrum multipoles.
        window: lsstypes WindowMatrix matches the power spectrum multipoles.
        cov: lsstypes CovarianceMatrix matches the power spectrum multipoles.
        tracer: name of the tracer used to define the parameters in the theory and avoid duplicates when combining the different observables.
        p (dict, optional): Value of p parameter. Defaults to {'LRG': 1., 'ELG': 1., 'QSO': 1.4}.
        fix_fnl (bool, optional): If true do not fit for $f_{\rm NL}^{\rm loc}$. Defaults to False.
        engine (str, optional): Solver for perturbation theory computation ('class', 'camb' or either that works in cosmoprimo). Defaults to 'class'.
        nickname (str, optional): Suffix inserted between the tracer shortname and '_ell{ell}' in cross-correlation theory
            parameter names for cross-correlations. Use this to avoid parameter name collisions when combining
            multiple cross-correlations that share a tracer (e.g. LRGxELG and ELGxQSO both have ELG).
            With nickname='LRGxELG', ELG parameters become 'ELG_LRGxELG_ell0.b1'. Default is None.
        
        pk_zeff (float or dict, optional): Redshift at which the power-spectrum template is evaluated
            (FixedSpectrum2Template). Use this to evaluate P(k) at a fixed redshift instead of the
            per-multipole OQE effective redshift carried by `zeffs`. Either a scalar (applied to every
            tracer and multipole), or a dict keyed by the tracer pair ('LRGxLRG', 'LRGxQSO', ...) or by
            the 3-letter auto tracer ('LRG', 'QSO'). Keys that are absent fall back to `zeffs`. If None
            (default), `zeffs` is used as before. This only changes the template redshift; the window
            (OQE weights) and the `zeffs` used for the bias relations are left untouched.

    Kwargs:
         kwargs[f"{tt}_{nickname}_ell0.b1"] (or f"{tt}_ell0.b1" if no nickname) value used to fix
         one of the two b1 in the cross-correlation, otherwise fixed to 1.

    Returns:
       observables: list of monopole and quadrupole (if used) desilike observables.
       likelihood: desilike likelihood to run profer or MCMC on.
    """
    from desilike.theories.galaxy_clustering import FixedSpectrum2Template, PNGTracerSpectrum2Poles
    from desilike.observables.galaxy_clustering import Spectrum2PolesObservable
    from desilike.likelihoods import ObservablesGaussianLikelihood
    from desilike import get_params
    from cosmoprimo.fiducial import DESI

    tracers = tuple(tracer.split('x'))
    if len(tracers) == 1: tracers *= 2
    tracers = (tracers[0][:3], tracers[1][:3])  # LRG_zcmb -> LRG, ELGnotqso -> ELG, ...
    cross_correlation = (tracers[0] != tracers[1])
    pair = 'x'.join(tracers)

    def _template_z(ell):
        """Redshift used to evaluate the P(k) template: `pk_zeff` override if given, else the OQE zeff."""
        if pk_zeff is None:
            return zeffs[pair][ell]
        if not isinstance(pk_zeff, dict):
            return pk_zeff
        if pair in pk_zeff:
            return pk_zeff[pair]
        if not cross_correlation and tracers[0] in pk_zeff:
            return pk_zeff[tracers[0]]
        return zeffs[pair][ell]

    def _cross_two_redshift():
        """Return ``(z_x, z_y)`` for a two-snapshot-redshift cross, or ``None`` to use a single
        template. Triggered for a cross when ``pk_zeff`` is a per-tracer dict that gives the two
        tracers distinct redshifts and does NOT carry an explicit pair key (e.g. 'LRGxQSO'), which
        would instead request a single effective redshift (handled by ``_template_z``)."""
        if not (cross_correlation and isinstance(pk_zeff, dict)):
            return None
        if pair in pk_zeff:
            return None
        if tracers[0] in pk_zeff and tracers[1] in pk_zeff:
            zx, zy = pk_zeff[tracers[0]], pk_zeff[tracers[1]]
            if zx != zy:
                return (zx, zy)
        return None

    two_z = _cross_two_redshift()

    observables = []
    for ell in pk.ells:
        cross_suffix = f'_{nickname}' if (nickname is not None and cross_correlation) else ''
        if cross_correlation:
            tracers_theo = [f'{tt}{cross_suffix}_ell{ell}' for tt in tracers]
        else:
            tracers_theo = [f'{tt}_ell{ell}' for tt in tracers]
        if not cross_correlation: tracers_theo = tracers_theo[:1]
        if two_z is not None:
            logger.info(f'{tracers_theo=}, {ell=}, two-redshift cross: z_x={two_z[0]:2.4} ({tracers[0]}), z_y={two_z[1]:2.4} ({tracers[1]})')
        else:
            logger.info(f'{tracers_theo=}, {ell=}, zeff={zeffs[pair][ell]:2.4}, template z={_template_z(ell):2.4}')

        # extract only the mulitpole ell:
        data = pk.get(ells=[ell])
        wmatrix = window.at.observable.match(data)

        # Define Template and Theory:
        if two_z is not None:
            # cross with the two tracers at two different snapshot redshifts (z_x for tracers[0],
            # z_y for tracers[1]); single Lorentzian enforced below by fixing the first sigmas to 0.
            templates = [FixedSpectrum2Template(fiducial=DESI(), engine=engine, z=zz) for zz in two_z]
            theory = _get_png_cross_two_redshift_cls()(templates=templates, mode="b-p", tracers=tracers_theo)
        else:
            template = FixedSpectrum2Template(fiducial=DESI(), engine=engine, z=_template_z(ell))
            theory = PNGTracerSpectrum2Poles(template=template, mode="b-p", tracers=tracers_theo)
        # Fix some parameters:
        params = get_params(theory)
        params['fnl_loc'].update(value=0.0, fixed=fix_fnl)
        if ell == 2: 
            params[f"{'x'.join(tracers_theo)}.sn0"].update(value=0, fixed=True)
        for tracer in tracers_theo:
            params[f'{tracer}.p'].update(value=p[tracer.split('_')[0]], fixed=True)
        # Use only one damping term for the cross-correlation
        if cross_correlation: 
            params[f"{tracers_theo[0]}.sigmas"].update(value=0, fixed=True)
        theory.update(params=params)
        # Don't forget to give different name for the observable in order to stack them together in the likelihood:
        name = f"pk_{'x'.join(tracers)}_ell{ell}"
        observables += [Spectrum2PolesObservable(name=name, data=data, window=wmatrix, theory=theory)] 

    if isinstance(cov, list):
        import lsstypes
        logger.info('Using mocks to estimate the covariance matrix.')
        covariance = lsstypes.cov([mock.match(pk) for mock in cov]).value()
        correction_covariance = {'correction': 'hartlap2007+percival2014', 'nobs': len(cov)}
    else:
        logger.info('Using analytical covariance matrix.')
        covariance = cov.at.observable.get(tracers=tracers).value()
        correction_covariance = None

    likelihood = ObservablesGaussianLikelihood(observables=observables, covariance=covariance, 
                                               correct_covariance=correction_covariance, scale_covariance=scale_covariance)
    fix_likelihood_bias_and_damping(likelihood, tracer='x'.join(tracers), zeffs=zeffs, derived_cross_bias=False, nickname=nickname, bias_params=bias_params, **kwargs)

    return observables, likelihood


def combine_analytical_covariances(pks, covs, order=['LRGxLRG', 'LRGxQSO', 'QSOxQSO'], fiducial=None):
    """ 
    Combine the analytical covariance matrices for the auto power spectra and the cross power spectra into a single covariance matrix for the combined data vector.
    
    #'fiducial should be a dictionary containing the redshift ranges for each tracer, (see propose_fiducial() function)'

    Remarks: 
        * The off-diagonal blocks are estimated from the cross-correlation covariance matrices, and rescaled to account for the difference in effective volume between the auto and cross power spectra. 
        * The rescaling is mandatory to avoid non definite positive covariance matrices.
        * Effective volumes are pre-computed for specific redshift ranges and tracers, and are hard-coded in the function... -> Update if needed.
    """
    from lsstypes import ObservableTree

    veffs = {'LRG_z0.4-1.0': 9.41, 'LRG_z0.4-1.1': 11.201, 'LRG_z0.8-1.0': 4.075, 'LRG_z0.8-1.1': 5.866,
             'ELG_z0.8-1.0': 3.637, 'ELG_z0.8-1.1': 5.663, 'ELG_z0.8-1.6': 16.011,
             'QSO_z0.8-1.0': 1.129, 'QSO_z0.8-1.1': 1.831, 'QSO_z0.8-1.6': 6.494, 'QSO_z0.8-3.5': 14.722, 
             'LRGxQSO_z0.8-1.0': 2.143, 'LRGxQSO_z0.8-1.1': 3.262,
             'ELGxLRG_z0.8-1.0': 3.849, 'ELGxLRG_z0.8-1.1': 5.751,
             'ELGxQSO_z0.8-1.6': 10.151}

    def _extract_offdiag_block(pks, covs, tracer1, tracer2):
        """ 
        Extract the off-diagonal block from the covariance matrices for tracer1 x tracer2 (for instance: LRGxLRG x LRGxQSO).
        We look for the block in all the covariance matrices available, and we match the observables to extract the correct block in the correct order ! 
        If the block is not found, we return a block of zeros.
        """
        block = np.zeros((pks[tracer1].size, pks[tracer2].size))
        for source_key, cov in covs.items(): 
            try:   
                local_tracers = [tuple(tracer1.split('x')), tuple(tracer2.split('x'))]
                local_obs = ObservableTree([pks[tracer1], pks[tracer2]], observables=['spectrum2', 'spectrum2'], tracers=local_tracers)
                block = cov.at.observable.get(observables=['spectrum2', 'spectrum2'], tracers=local_tracers).at.observable.match(local_obs)
                # extract only the off-diagonal block (always the lower-left block, because of the .match() order):
                block = block.value()[:pks[tracer1].size, pks[tracer1].size:]
                # If we get here, the block was found and matched successfully!
                logger.debug(f'Found block for {tracer1} x {tracer2} in covariance "{source_key}".')
            except Exception:
                pass
        return block
    
    def _get_rescale_factor(tracer1, tracer2):
        """ 
        Get the rescale factor for the off-diagonal block between tracer1 and tracer2 (for instance: LRGxLRG x LRGxQSO).
        Redshift range for the full-range and cross-range of tracer are determined from the fiducial dictionary, and the corresponding effective volumes are pre-computed in hardcoded veffs dictionary.
        The rescale factor is given by (V_eff_cross / V_eff1) * (V_eff_cross / V_eff2), where V_eff_cross is the effective volume of the cross-correlation, and V_eff1 and V_eff2 are the effective volumes of the auto-correlations for tracer1 and tracer2.

        Remark: 
            * if tracer1 = 'LRGxQSO' then V_eff_cross = V_eff1 -> no rescaling for this part. 
            * if three or more tracers are in tracer1 and tracer2, we raise a ValueError. We neglect this case because we do not have any estimation of the covariance for this case. 
        """
        tt11, tt12 = tracer1.split('x')
        tt21, tt22 = tracer2.split('x')

        if tt11 == tt12:
            zrange = fiducial[tt11]['zrange']
        else: 
            zrange = fiducial[tracer1]['zrange']
        veff1 = veffs[f'{tt11}_z{zrange[0]}-{zrange[1]}']

        if tt21 == tt22:
            zrange = fiducial[tt21]['zrange']
        else: 
            zrange = fiducial[tracer2]['zrange']
        veff2 = veffs[f'{tt21}_z{zrange[0]}-{zrange[1]}']

        # From which cross-correlation matrix the off-diagonal block is estimated:
        tracer_cross = 'x'.join(np.unique([tt11, tt12, tt21, tt22]))
        if len(tracer_cross.split('x')) > 2:
            raise ValueError(f'Unexpected tracer_cross: {tracer_cross} from tracer1: {tracer1} and tracer2: {tracer2}')
        zrange = fiducial[tracer_cross]['zrange']
        veff1_cross, veff2_cross = veffs[f'{tracer_cross}_z{zrange[0]}-{zrange[1]}'], veffs[f'{tracer_cross}_z{zrange[0]}-{zrange[1]}']
    
        return veff1_cross / veff1 * veff2_cross / veff2
    
    missing_pks = [label for label in order if label not in pks]
    if missing_pks:
        raise ValueError(f'Missing pk blocks for {missing_pks}. Available: {list(pks.keys())}')

    observable_tot = ObservableTree([pks[label] for label in order], observables=['spectrum2'] * len(order), tracers=order)

    matrix = np.zeros((observable_tot.size, observable_tot.size))
    offsets = np.concatenate(([0], np.cumsum([observable_tot.get(tracers=order[i]).size for i in range(len(order))])))
    missing_offdiag = []

    for i in range(len(order)):
        for j in range(len(order)):
            # print(f'  Pair (i, j) = ({i}, {j}): {order[i]} x {order[j]}:', offsets[i], offsets[i + 1], offsets[j], offsets[j + 1])
            if i == j:
                block = covs[order[i]].at.observable.get(observables='spectrum2', tracers=tuple(order[i].split('x')))
                block = block.at.observable.match(observable_tot.get(observables='spectrum2', tracers=order[i])).value()
            else:
                block = _extract_offdiag_block(pks, covs, order[i], order[j])

                # Do not forget to rescale the off-diagonal block to account for the difference in effective volume between the auto and cross power. 
                # This is mandatory because we use a larger volume for the auto while the cross block is estimated from the commom volume.
                try: 
                    rescale_factor = _get_rescale_factor(order[i], order[j])
                except ValueError:
                    rescale_factor = 0
                logger.debug(f'Rescale factor for block {order[i]} x {order[j]}: {rescale_factor:.3f}')
                block = rescale_factor * block 

                if np.sum(block) == 0:
                    missing_offdiag.append(f'No block found or no rescaling available for {order[i]} x {order[j]}')
            
            # Insert the block into the matrix:
            matrix[offsets[i]:offsets[i + 1], offsets[j]:offsets[j + 1]] = block

    logger.debug('Missing off-diagonal blocks:', missing_offdiag)

    covariance_tot = covs[order[0]].clone(observable=observable_tot, value=matrix)
    logger.debug(covariance_tot.observable)
    try: 
        np.linalg.cholesky(covariance_tot.value())
    except Exception as e:
        logger.warning('Covariance matrix is not positive definite:', e)
        
    return covariance_tot


def build_total_likelihood(order, pks, observables, covs, zeffs, fiducial, scale_covariance=1, bias_params=None):
    """ 
    Build the total likelihood for the combined data vector, by stacking the observables and using the combined covariance matrix.
    Were are using order to stack the observable in the same order as the covariance matrix. 
    Note: pks, observables, covs, zeffs, fiducial are dictionaries with keys corresponding to the labels in order.
    Note2: fiducial is used only to extract the redshift ranges for the different tracers, which are needed to rescale the off-diagonal blocks of the covariance matrix in combine_analytical_covariances.
    """
    from desilike.likelihoods import ObservablesGaussianLikelihood
    from tools import combine_analytical_covariances, fix_likelihood_bias_and_damping

    total_observables = [observable for tracer in order for observable in observables[tracer]]
    logger.debug([ff.name for ff in total_observables])

    if isinstance(covs[order[0]], list):
        logger.info('Using mocks to estimate the covariance matrix.')
        covariance = np.cov(np.transpose([np.concatenate([covs[tt][i].match(pks[tt]).value() for tt in order]) for i in range(len(covs[order[0]]))]))
        correction_covariance = {'correction': 'hartlap2007+percival2014', 'nobs': len(covs[order[0]])}
    else:
        logger.info('Using analytical covariance matrix.')
        covariance = combine_analytical_covariances(pks, covs, order=order, fiducial=fiducial).value()
        correction_covariance = None
    
    total_likelihood = ObservablesGaussianLikelihood(observables=total_observables, covariance=covariance, 
                                                     correct_covariance=correction_covariance, scale_covariance=scale_covariance)
    for tracer in order: 
        # We do not link the damping term from the cross-correlation and the auto-correlation
        # Because they are different effective redshifts and we do not know the a priori.
        fix_likelihood_bias_and_damping(total_likelihood, tracer=tracer, zeffs=zeffs, derived_cross_bias=True, nickname=tracer, available_tracers=order, bias_params=bias_params)

    return total_likelihood


def run_profiler(likelihood, fn_output=None):
    """ 
    Run the iminuit profiler on the likelihood, the results are saved in a text file if output_name is provided. fn_output should be a .h5 file. 
    """
    from desilike import Posterior, compile
    from desilike.profilers import Profiler, Minuit

    posterior = compile(Posterior(likelihood=likelihood))
    profiler = Profiler(posterior, kernel=Minuit(), rng=7)
    profiles = profiler.maximize(niterations=10)
    logger.info(f'\n{profiler.profiles.to_stats(tablefmt="pretty")}')
    best = profiler.profiles.choice(index='argmax', squeeze=True).select(input=True).best
    # To set internal arrays
    compile(likelihood)(best)

    if fn_output is not None:
        profiles.write(fn_output)
    return profiler


def run_mcmc(likelihood, dir_output='tmp/', resume=False, nchains=1, max_steps=20000, check_every=1000):
    """Run the MCMC sampler on the likelihood, the results are saved as hdf5. 

    Args:
        likelihood: Desilike Likelihood object to run the MCMC on.
        dir_output (str, optional): Where the chains will be saved. Defaults to 'tmp/samples_*.h5'.
        resume (bool, optional): If True, it will extend the existing chains (saved in dir_output) by running new iterations. Defaults to False.
        nchains (int, optional): Number of chains to run. Defaults to 1.
        max_steps (int, optional): Maximum number of steps to run. Defaults to 1e5.
        check_every (int, optional): How often to check the convergence + save the current state of the chains. Defaults to 1000.

    """
    from desilike.distributed import get_mpicomm
    from desilike import Posterior, compile
    from desilike.samplers import Sampler, Emcee

    posterior = compile(Posterior(likelihood=likelihood))
    mpicomm = get_mpicomm()
    if not resume and mpicomm.rank == 0:  # just remove directory
         for path in Path(dir_output).glob('*'):
            if path.name != 'profiles.h5':
                shutil.rmtree(path) if path.is_dir() else path.unlink()
    sampler = Sampler(posterior, kernel=Emcee(), rng=31, output_dir=dir_output, nparallel=nchains)  
    sampler.run(max_steps=max_steps, check_every=check_every, save_every=check_every)

    return sampler


def plot_observables(observables, figsize=(6, 4), ylims=None, show=True, fn_output=None):
    """ 
    Plot the observables (power spectrum multipoles) with their theory predictions and residuals.

    Parameters
    ----------
    observables : dict
        Mapping tracer -> observables.
    ylims : sequence, optional
        Y-axis limits for the monopole and quadrupole panels.
    profile : object, optional
        Profiler or profile-like object. If provided, an extra column is added to display
        its summary table.
    """
    fig, axs = plt.subplots(2, 2, figsize=figsize, sharex=True, sharey=False, gridspec_kw={'height_ratios': (3, 1)}, squeeze=True)
    fig.subplots_adjust(hspace=0.1)

    translator = {'LRGxLRG': 'L', 'LRG': 'L', 'ELGxELG': 'E', 'ELG': 'E', 'QSOxQSO': 'Q', 'QSO': 'Q', 'LRGxELG': 'LxE', 'LRGxQSO': 'LxQ', 'ELGxQSO': 'ExQ'}

    for tracer in observables.keys():
        for obs in observables[tracer]:
            j = 1 if 'ell2' in obs.name else 0

            wtheory = obs.data.clone(value=obs.flattheory)
           
            data_pole = obs.data.get()
            wtheory_pole = wtheory.get()
            x = data_pole.coords('k')
            std = obs.covariance.at.observable.get().std()

            scale = 1.
            axs[0, j].errorbar(x, scale * data_pole.value(), yerr=scale * std, linestyle='none', marker='o', markersize=4, label=rf'{translator.get(tracer)}')
            axs[0, j].loglog(x, scale * wtheory_pole.value(), ls='-', c='k')

            axs[1, j].plot(x, (data_pole.value() - wtheory_pole.value()) / std)
            axs[1, j].set_ylim(-4, 4)
            for offset in [-2., 2.]: axs[1, j].axhline(offset, color='k', linestyle='--')
            for offset in [-1., 1.]: axs[1, j].axhline(offset, color='lightgray', linestyle=':')

    if ylims is not None:
        axs[0, 0].set_ylim(*ylims[0])
        axs[0, 1].set_ylim(*ylims[1])
    else:
        axs[0, 0].set_ylim(1e4, 8e4)
        axs[0, 1].set_ylim(2e3, 5e4)

    axs[0, 0].legend()
    axs[0, 1].legend()
    axs[0, 0].set_title(r'$\ell = 0$', fontsize=10)
    axs[0, 1].set_title(r'$\ell = 2$', fontsize=10)
    axs[0, 0].set_ylabel(r'$P_{\ell}(k)$ [$(\mathrm{Mpc}/h)^{3}$]')
    axs[1, 0].set_ylabel(r'$\Delta P_{\ell} / \sigma (P_{\ell})$')
    axs[1, 0].set_xlabel(r'$k$ [$h/\mathrm{Mpc}$]')
    axs[1, 1].set_xlabel(r'$k$ [$h/\mathrm{Mpc}$]')

    plt.tight_layout()
    if fn_output is not None: plt.savefig(fn_output)
    if show:
        plt.show()
    else:
        plt.close()


def get_getdist_plotter(fig_width_inch=5, fontsize=14, legend_fontsize=12, axes_labelsize=12, axes_fontsize=14, line_lables=True):
    """Wrapper around getdist to get a plotter with the desired settings."""
    from getdist import plots as gdplt

    plotter = gdplt.get_subplot_plotter()
    plotter.settings.fig_width_inch = fig_width_inch
    plotter.settings.fontsize = fontsize
    plotter.settings.legend_fontsize = legend_fontsize
    plotter.settings.axes_labelsize = axes_labelsize
    plotter.settings.axes_fontsize = axes_fontsize
    plotter.settings.line_labels = line_lables
    plotter.settings.alpha_filled_add = 0.8
    plotter.settings.legend_loc = 'upper right'
    plotter.settings.figure_legend_ncol = 1
    plotter.settings.legend_colored_text = False

    return plotter


def plot_triangle(chains, params, legend_labels=None, xlabels=[r'$f_{\rm NL}^{\rm loc}$', r'$b_1$', r'$s_{n,0}$', r'$\Sigma_s$'], 
                  contour_colors=None, filled=True, contour_ls=None, g=None, fn_output=None, return_fig=False):
    """ Wrapper around getdist's plot_triangle to plot the contours of the different chains."""

    # Compatibility shim: matplotlib >= 3.10 removed QuadContourSet.collections.
    # In 3.10+ QuadContourSet IS a Collection itself, so returning [self] restores
    # the iteration pattern used by getdist (for c in cs.collections: c.set_dashes(...)).
    import matplotlib.contour as _mcontour
    if not hasattr(_mcontour.QuadContourSet, 'collections'):
        _mcontour.QuadContourSet.collections = property(lambda self: [self])

    # Avoid getdist print statements about loading chains (burning ..), since we are building the chains ourselves.
    import getdist.chains
    getdist.chains.print_load_details = False

    from desilike.samples import plotting

    if g is None: g = get_getdist_plotter()
    plotting.plot_triangle(chains, params, legend_labels=legend_labels,
                           contour_colors=contour_colors, filled=filled, contour_ls=contour_ls,
                           g=g, show=False, fn=None)

    if xlabels is not None:
        for i in range(len(xlabels)):
            g.subplots[len(xlabels)-1, i].set_xlabel(xlabels[i])
            if i > 0:
                g.subplots[i, 0].set_ylabel(xlabels[i])

    if fn_output is not None: plt.savefig(fn_output)
    
    if return_fig:
        return g.fig
    else:
        plt.show()


def run_profiling_one_mock(mocks, windows, covs, tracer, region='GCcomb', imock=0, alternative_mocks=None, kmin=1e-3, drop_ell2_cross=True, 
                           analytical_covariance=True, 
                           force_profiling=False, base_dir=None, fiducial=None, extra_fn='', return_profiler=False, save_plot=False):
    """Run the profiler on a single mock realisation and save the result to disk.

    Parameters
    ----------
    mocks : list
        List of mock power spectrum observables.
    window : lsstypes WindowMatrix
        Window matrix for the tracer.
    cov : lsstypes CovarianceMatrix
        Analytical covariance matrix for the tracer.
    tracer : str
        Tracer name (e.g. 'LRGxLRG').
    region : str, optional
        Region name (e.g. 'GCcomb'). Default is 'GCcomb'.
    imock : int, optional
        Index of the mock to fit. Default is 0.
    kmin : float, optional
        Minimum k value for the fit. Default is 1e-3.
    analytical_covariance : bool, optional
        If True, use the analytical covariance. If False, use mock covariance. Default is True.
    force_profiling : bool, optional
        If True, rerun even if output file already exists. Default is False.
    base_dir : str, optional
        Base directory. Profile outputs are written under the corresponding profiles directory.
    """
    import os
    from desilike import compile

    # from clustering_statistics.tools import bias
    # tt1 = short_tracer.split('x')[0]
    # kwargs = {f'{tt1}_{short_tracer}_ell0.b1': bias(zeffs[region][short_tracer][0], tracer=tt1)}
    kwargs = {'LRG_LRGxQSO_ell0.b1': 2.25, 'LRG_LRGxELG_ell0.b1': 2.24, 'ELG_ELGxQSO_ell0.b1': 1.42, 'scale_covariance': 1}

    fn_profile = Path(base_dir) /  f"mock{imock}/bestfit_{tracer}_{region}_{'analytical_cov' if analytical_covariance else 'mock_cov'}_kmin-{kmin}{extra_fn}.h5"
    exists = fn_profile.is_file()
    if force_profiling or not exists:
        fn_profile.parent.mkdir(parents=True, exist_ok=True)

        tracers = tracer.split('-')

        obs, lik, zeffs = {}, {}, {}
        pks = {}
        mocks_cov = {}
        for tt in tracers:
            mocks_cov[tt] = mocks[tt].copy()  # Avoid modifying the original mock observable.

            pks[tt] = mocks_cov[tt].pop(imock).select(k=(kmin, 1))  # remove the used mock from the covariance matrix.
            if (alternative_mocks is not None) and (tt in alternative_mocks) and (alternative_mocks[tt] is not None):
                # use an alternative mock for the fit, but keep the covariance from the original mocks.
                pks[tt] = alternative_mocks[tt].copy().pop(imock).select(k=(kmin, 1))
            if drop_ell2_cross and (tt.split('x')[0] != tt.split('x')[1]):  # if it's a cross-correlation and we want to drop the ell2:
                logger.info(f"{tt}: Dropping ell=2 for the cross-correlation to reduce hartlap factor and speed up the fit.")
                pks[tt] = pks[tt].get(ells=[0])  # keep only the monopole for the fit.
            
            window = windows[tt].at.observable.match(pks[tt])

            zeffs[tt] = {ell: window.observable.get(ell).attrs['zeff'] for ell in pks[tt].ells}  # Keep only the zeff for the used multipoles.

            if analytical_covariance:
                covariance = covs[tt].at.observable.at(observables='spectrum2', tracers=tuple(tt.split("x"))).match(pks[tt])
            else:
                mocks_cov[tt] = [mm.match(pks[tt]) for mm in mocks_cov[tt]]
                covariance = mocks_cov[tt]

            obs[tt], lik[tt] = get_observable_and_likelihood(pks[tt], window, covariance, tt, zeffs, fix_fnl=False, nickname=tt, **kwargs)

        if len(tracers) > 1:
            lik = build_total_likelihood(tracers, pks, obs, covs if analytical_covariance else mocks_cov, zeffs, fiducial)
        else:
            obs, lik = obs[tracers[0]], lik[tracers[0]]

        profiler = run_profiler(lik, fn_output=fn_profile)

        if save_plot:
            ylims = [(2e3, 4e4), (2e3, 4e4)] if tracer in ['ELGxELG', 'ELGxQSO'] else None
            fn_obs = base_dir / f"mock{imock}/bestfit_{tracer}_{'' if analytical_covariance else 'mock'}_kmin-{kmin}{extra_fn}.png"
            if len(tracers) > 1:
                plot_observables({tt: obs[tt] for tt in tracers}, ylims=ylims, fn_output=fn_obs, show=True)
            else: 
                plot_observables({tracer: obs}, ylims=ylims, fn_output=fn_obs, show=True)

        if return_profiler:
            return profiler
