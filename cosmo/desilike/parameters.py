
_fiducial = None


def get_fiducial():
    global _fiducial
    if _fiducial is None:
        from cosmoprimo.fiducial import DESI
        _fiducial = DESI()
    return _fiducial


def get_cosmology(model=None, engine=None, parameterization='background', likelihoods=None, truncate_priors=False):
    """
    Construct and return a :mod:`desilike` :class:`CosmoprimoCosmology` calculator.

    Parameters
    ----------
    model : str, optional
        Cosmological model string.  Default is ``'base'``.  Supported tokens:
        ``'w_wa'`` (free dark energy equation of state), ``'_w'`` (free ``w_0``
        only), ``'fixed'`` (fix all parameters).
    engine : str or dict, optional
        Boltzmann solver: ``'class'`` or ``'camb'``; ``'ace'`` for the packaged
        jaxace / jaxmapse / jaxcapse neural-network emulators (pure JAX, differentiable;
        LCDM-only Cl, parameters NaN-masked outside the emulator training ranges); or an
        :class:`~desilike.theories.primordial_cosmology.ACECosmology` per-section engine
        dict. ``None`` (default) resolves from *likelihoods* via
        :func:`~cosmo.desilike.mapping_likelihoods.get_default_engine`: ``'eisenstein_hu'``
        for COMET-only full-shape fits,
        :data:`~cosmo.desilike.mapping_likelihoods.DEFAULT_ACE_ENGINE` (jaxace w0waCDM
        background + jaxmapse pk + Capse w0waCDM Cl) when a FolpsD-style full-shape
        likelihood is present, ``'class'`` otherwise.
    parameterization : str, optional
        ``'background'`` (default) samples ``Omega_m`` only, plus the
        per-family absolute-scale anchor: ``r_d`` sampled directly for BAO
        (BAO alone only constrains :math:`H_0 r_d`; see
        :class:`~desilike.theories.galaxy_clustering.template.BAOTheory`'s
        notes on ``rs_drag``), the magnitude (``Mb``/``dM``) proposed by the
        SN likelihoods themselves. ``omega_b`` is freed (and ``r_d`` derived
        from the cosmology instead of sampled) only when a calibrating
        likelihood pins it: BBN (optionally with free ``N_eff``), or a
        CMB-compressed/thetastar/rdrag prior; ``h`` additionally requires BAO
        (without BAO nothing constrains it). ``logA``, ``n_s`` and
        ``tau_reio`` are fixed and ``sigma8`` is not tracked.
        ``'lss'`` samples ``omega_cdm`` directly, frees ``logA`` and ``n_s``
        (tau fixed), and adds ``sigma8_m`` / ``sigma8_cb`` as derived outputs.
        ``'cmb'`` additionally frees ``tau_reio``.
    likelihoods : list of str, optional
        Likelihood names. Drives the background-only branching described above,
        and frees ``N_ur`` when a ``'varied-nnu'``/``'marg-nnu'`` likelihood
        (or ``'schoneberg2024-bbn'``) is present.
    truncate_priors : bool, optional
        ACE engines only (ignored otherwise): intersect each parameter's prior
        with the packaged emulators' training ranges (e.g. ``h`` in (0.5, 0.9),
        ``omega_cdm`` in (0.08, 0.18)). Outside those ranges
        :class:`~desilike.theories.primordial_cosmology.ACECosmology` NaN-masks
        its results, which the :class:`~desilike.base.Posterior` maps to ``-inf``
        — an effective prior truncation either way; enabling this makes it
        explicit, so samplers drawing from the prior (e.g. pocomc) start every
        particle at a finite log-likelihood. Default is ``False``.

    Returns
    -------
    cosmo : :class:`desilike.theories.primordial_cosmology.CosmoprimoCosmology`
        When *parameterization* triggers the background-only degeneracy-breaking
        branch above, the shared, free ``r_d`` :class:`~desilike.parameter.Parameter`
        is also attached as ``cosmo.rs_drag_param`` (``None`` otherwise), for
        :func:`~cosmo.desilike.mapping_likelihoods.get_likelihood` to forward to
        :class:`~desilike.likelihoods.bao.DESIDR2BAOLikelihood`.
    """
    from desilike import VariableCollection, Parameter
    from desilike.theories import PrimordialCosmology, CosmoprimoCosmology, ACECosmology
    if isinstance(model, PrimordialCosmology):
        return model
    if engine is None:
        from .mapping_likelihoods import get_default_engine
        engine = get_default_engine(likelihoods)
    is_cmb = parameterization == 'cmb'
    is_lss = parameterization in ('lss', 'cmb')  # lss: logA/n_s free, tau fixed; cmb: all three free
    is_background = not is_lss
    if model is None:
        model = 'base'
    is_fixed_model = model == 'fixed'
    has_w0wa = 'w_wa' in model  # base_w_wa: both w0 and wa free
    has_w0 = not has_w0wa and '_w' in model  # base_w: only w0 free
    fiducial = get_fiducial()

    likelihood_names = set(likelihoods or [])
    labels = ' '.join(likelihood_names)
    has_bao = 'bao' in labels
    has_varied_nnu = ('schoneberg2024-bbn' in likelihood_names
                       or any(token in name for name in likelihood_names for token in ('varied-nnu', 'marg-nnu')))
    # Likelihoods that pin h and omega_b separately: BBN (omega_b, optionally with free
    # N_eff), and the CMB-compressed/thetastar/rdrag priors (which read theta_star/ombh2/
    # r_d off the cosmology, so they are inert unless h and omega_b are sampled).
    has_calibrator = any(token in labels for token in ('bbn', 'CMB-compressed', 'thetastar', 'rdrag'))
    # Background parameterization samples Omega_m only, plus the per-family absolute-scale
    # anchor: r_d sampled directly for BAO (BAO alone only constrains H_0*r_d; see
    # BAOTheory's rs_drag notes), the magnitude (Mb/dM) proposed by the SN likelihoods
    # themselves. h and omega_b are freed only when a calibrator is present.
    constrain_rd = is_background and has_bao and not has_calibrator

    params = VariableCollection()
    params.set(Parameter('h', value=fiducial['h'],
                            prior=dict(limits=[0.5, 1.0]),
                            ref=dict(dist='norm', loc=fiducial['h'], scale=0.01),
                            fd_eps=0.03, latex='h'))
    params.set(Parameter('omega_b', value=fiducial['omega_b'],
                            prior=dict(limits=[0.005, 0.1]),
                            ref=dict(dist='norm', loc=fiducial['omega_b'], scale=0.0003),
                            fd_eps=0.0015, latex=r'\omega_b'))
    if is_background:
        params.set(Parameter('Omega_m', value=fiducial['Omega_m'],
                                prior=dict(limits=[0.01, 0.99]),
                                ref=dict(dist='norm', loc=fiducial['Omega_m'], scale=0.001),
                                fd_eps=0.002, latex=r'\Omega_\mathrm{m}'))
    else:
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

    rs_drag_param = None
    if is_background and not has_calibrator:
        # No calibrator: h and omega_b are unconstrained (the SN magnitude absorbs H0, BAO
        # only measures H_0*r_d), so fix both to fiducial; for BAO, anchor the absolute
        # scale by sampling r_d directly instead.
        params['h'].update(fixed=True)
        params['omega_b'].update(fixed=True)
        if has_bao:
            from desilike.theories.galaxy_clustering.template import BAOTheory
            rs_drag_param = BAOTheory.propose_params(rs_drag=True, fiducial=fiducial)['rs_drag']
    elif is_background and not has_bao:
        # Calibrator present but no BAO (e.g. SN + BBN): omega_b is pinned (by BBN) but
        # nothing constrains h -- keep it fixed to avoid a flat direction.
        params['h'].update(fixed=True)

    if is_fixed_model:
        for name in params:
            params[name].update(fixed=True)
    params['logA'].update(fixed=is_fixed_model or not is_lss)
    params['n_s'].update(fixed=is_fixed_model or not is_lss)
    if has_varied_nnu:
        params['N_ur'].update(fixed=is_fixed_model)
    if has_w0wa:
        params['w0_fld'].update(fixed=is_fixed_model)
        params['wa_fld'].update(fixed=is_fixed_model)
    elif has_w0:
        params['w0_fld'].update(fixed=is_fixed_model)
    if truncate_priors and (engine == 'ace' or isinstance(engine, dict)):
        # Intersect the priors with the emulators' training ranges: outside them ACECosmology
        # NaN-masks its results (mapped to -inf by the Posterior), so the priors are
        # effectively truncated anyway — making it explicit keeps prior draws (e.g.
        # pocomc's beta = 0 particles) at finite log-likelihood.
        ACECosmology.truncate_priors(params, engine=engine)
    params.set(Parameter('H0', derived=True, latex='H_0'))
    if is_background:
        params.set(Parameter('omega_cdm', derived=True, latex=r'\omega_\mathrm{cdm}'))
    else:
        params.set(Parameter('Omega_m', derived=True, latex=r'\Omega_\mathrm{m}'))
    params.set(Parameter('Omega_Lambda', derived=True, latex=r'\Omega_\Lambda'))
    params.set(Parameter('Omega_k', derived=True, latex=r'\Omega_k'))
    if not constrain_rd:
        # When constrain_rd, 'rs_drag' is instead the free, directly-sampled rs_drag_param
        # (see below); defining both here too would register two distinct Parameter
        # objects under the same name.
        params.set(Parameter('rs_drag', derived=True, latex=r'r_s'))
    params.set(Parameter('age', derived=True, latex=r't_0'))
    if is_lss:
        params.set(Parameter('sigma8_m', derived=True, latex=r'\sigma_{8,\mathrm{m}}'))
        params.set(Parameter('sigma8_cb', derived=True, latex=r'\sigma_{8,\mathrm{cb}}'))
    if engine == 'ace' or isinstance(engine, dict):
        # Packaged jaxace / jaxmapse / jaxcapse neural-network emulators (pure JAX end-to-end,
        # differentiable), 'ace' or a per-section {'background'/'fourier'/'harmonic': name}
        # dict; see desilike.theories.primordial_cosmology.ACECosmology.
        cosmo = ACECosmology(engine=engine, fiducial=fiducial, params=params)
    else:
        cosmo = CosmoprimoCosmology(engine=engine, fiducial=fiducial, params=params)
    cosmo.rs_drag_param = rs_drag_param
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