
_fiducial = None


def get_fiducial():
    global _fiducial
    if _fiducial is None:
        from cosmoprimo.fiducial import DESI
        _fiducial = DESI()
    return _fiducial


def get_cosmology(model=None, engine='class', parameterization='background', likelihoods=None):
    """
    Construct and return a :mod:`desilike` :class:`CosmoprimoCosmology` calculator.

    Parameters
    ----------
    model : str, optional
        Cosmological model string.  Default is ``'base'``.  Supported tokens:
        ``'w_wa'`` (free dark energy equation of state), ``'_w'`` (free ``w_0``
        only), ``'fixed'`` (fix all parameters).
    engine : str, optional
        Boltzmann solver: ``'class'`` (default) or ``'camb'``.
    parameterization : str, optional
        ``'background'`` (default) samples ``h``, ``omega_b``, ``omega_cdm``
        only; ``logA``, ``n_s`` and ``tau_reio`` are fixed and ``sigma8`` is not
        tracked.  ``'lss'`` frees ``logA`` and ``n_s`` (tau fixed) and adds
        ``sigma8_m`` / ``sigma8_cb`` as derived outputs.  ``'cmb'`` additionally
        frees ``tau_reio``.
    likelihoods : list of str, optional
        Likelihood names, reserved for future use (e.g. freeing ``N_ur`` when
        ``'varied-nnu'`` likelihoods are present).

    Returns
    -------
    cosmo : :class:`desilike.theories.primordial_cosmology.CosmoprimoCosmology`
    """
    from desilike import VariableCollection, Parameter
    from desilike.theories import CosmoprimoCosmology
    if isinstance(model, CosmoprimoCosmology):
        return model
    is_cmb = parameterization == 'cmb'
    is_lss = parameterization in ('lss', 'cmb')  # lss: logA/n_s free, tau fixed; cmb: all three free
    if model is None:
        model = 'base'
    is_fixed_model = model == 'fixed'
    has_w0wa = 'w_wa' in model  # base_w_wa: both w0 and wa free
    has_w0 = not has_w0wa and '_w' in model  # base_w: only w0 free
    fiducial = get_fiducial()

    params = VariableCollection()
    params.set(Parameter('h', value=fiducial['h'],
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
    if is_cmb:
        params.set(Parameter('tau_reio', value=fiducial['tau_reio'],
                                prior=dict(limits=[0.01, 0.8]),
                                ref=dict(dist='norm', loc=fiducial['tau_reio'], scale=0.006),
                                fd_eps=0.003, latex=r'\tau_\mathrm{reio}'))
    params.set(Parameter('m_ncdm', value=fiducial['m_ncdm_tot'], fixed=True,
                            prior=dict(limits=[0., 5.]),
                            ref=dict(dist='norm', loc=fiducial['m_ncdm_tot'], scale=0.12, limits=[0., 10.]),
                            fd_eps=(0.31, 0.15, 0.15), latex=r'm_\mathrm{ncdm}'))
    params.set(Parameter('N_ur', value=fiducial['N_ur'], fixed=True,
                            prior=dict(limits=[0.05, 10.]),
                            ref=dict(dist='norm', loc=fiducial['N_ur'], scale=0.2, limits=[0., 10.]),
                            fd_eps=(0.31, 0.15, 0.15), latex=r'N_\mathrm{ur}'))
    params.set(Parameter('w0_fld', value=fiducial['w0_fld'], fixed=True,
                            prior=dict(limits=[-3., 1.]),
                            ref=dict(dist='norm', loc=-1., scale=0.08),
                            fd_eps=0.1, latex=r'w_0'))
    params.set(Parameter('wa_fld', value=fiducial['wa_fld'], fixed=True,
                            prior=dict(limits=[-3., 2.]),
                            ref=dict(dist='norm', loc=0., scale=0.3),
                            fd_eps=0.3, latex=r'w_a'))
    if is_fixed_model:
        for name in params:
            params[name].update(fixed=True)
    params['logA'].update(fixed=is_fixed_model or not is_lss)
    params['n_s'].update(fixed=is_fixed_model or not is_lss)
    if has_w0wa:
        params['w0_fld'].update(fixed=is_fixed_model)
        params['wa_fld'].update(fixed=is_fixed_model)
    elif has_w0:
        params['w0_fld'].update(fixed=is_fixed_model)
    params.set(Parameter('H0', derived=True, latex='H_0'))
    params.set(Parameter('Omega_m', derived=True, latex=r'\Omega_\mathrm{m}'))
    params.set(Parameter('Omega_Lambda', derived=True, latex=r'\Omega_\Lambda'))
    params.set(Parameter('Omega_k', derived=True, latex=r'\Omega_k'))
    params.set(Parameter('rs_drag', derived=True, latex=r'r_s'))
    params.set(Parameter('age', derived=True, latex=r't_0'))
    if is_lss:
        params.set(Parameter('sigma8_m', derived=True, latex=r'\sigma_{8,\mathrm{m}}'))
        params.set(Parameter('sigma8_cb', derived=True, latex=r'\sigma_{8,\mathrm{cb}}'))
    cosmo = CosmoprimoCosmology(engine=engine, fiducial=fiducial, params=params)
    return cosmo


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