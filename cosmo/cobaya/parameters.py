"""Cobaya parameter and theory helpers for DESI cosmology fits."""


SUPPORTED_MODELS = {'base', 'base_w', 'base_w_wa'}
SUPPORTED_THEORIES = {'camb'}


def fix_parameter(params, name):
    """Replace a sampled parameter by its reference value."""
    config = dict(params[name])
    ref = config.get('ref', None)
    if isinstance(ref, dict):
        config['value'] = ref.get('loc')
    elif ref is not None:
        config['value'] = ref
    else:
        raise ValueError(f'Cannot fix parameter {name!r}; no reference value is defined.')
    for key in ['prior', 'ref', 'proposal']:
        config.pop(key, None)
    params[name] = config


def _check_model_theory(model, theory):
    if model not in SUPPORTED_MODELS:
        raise ValueError(f'Unsupported model {model!r}. Supported models: {sorted(SUPPORTED_MODELS)}')
    if theory not in SUPPORTED_THEORIES:
        raise ValueError(f'Unsupported theory {theory!r}. Supported theories: {sorted(SUPPORTED_THEORIES)}')


def get_background_cobaya_params(model='base', theory='camb', likelihoods=None):
    """Return Cobaya params and CAMB extra args for background-only fits."""
    _check_model_theory(model, theory)
    likelihood_names = set(likelihoods or [])
    labels = ' '.join(likelihood_names)
    has_bao = 'bao' in labels
    has_varied_nnu = (
        'schoneberg2024-bbn' in likelihood_names
        or any(token in label for label in likelihood_names for token in ['varied-nnu', 'marg-nnu'])
    )
    constrain_rd = has_bao and not any(token in labels for token in ['bbn', 'CMB-compressed', 'thetastar', 'rdrag'])
    params = {
        'H0': {
            'prior': {'min': 20, 'max': 100},
            'ref': {'dist': 'norm', 'loc': 67.36, 'scale': 0.01},
            'latex': r'H_0',
        },
        'ombh2': {
            'prior': {'min': 0.005, 'max': 0.1},
            'ref': {'dist': 'norm', 'loc': 0.02237, 'scale': 0.0001},
            'proposal': 0.0001,
            'latex': r'\Omega_\mathrm{b} h^2',
        },
        'mnu': {'value': 0.06, 'latex': r'\sum m_\nu'},
        'nnu': {'value': 3.044, 'latex': r'N_\mathrm{eff}'},
        'omm': {
            'prior': {'min': 0.01, 'max': 0.99},
            'ref': {'dist': 'norm', 'loc': 0.3152, 'scale': 0.001},
            'proposal': 0.0005,
            'drop': True,
        },
        'omch2': {
            'value': 'lambda omm, mnu, ombh2, H0: omm*(H0/100)**2 - mnu / 93.14 - ombh2',
            'latex': r'\Omega_\mathrm{c} h^2',
        },
        'hrdrag': {
            'prior': {'min': 10., 'max': 1000.},
            'ref': {'dist': 'norm', 'loc': 99.079, 'scale': 1.},
            'proposal': 1.,
            'latex': r'hr_\mathrm{d}',
        },
        'rdrag': {
            'value': 'lambda hrdrag, H0: 100 * hrdrag / H0',
            'latex': r'r_\mathrm{d}',
        },
    }
    # BAO-only/SN-only background fits do not constrain H0 and omega_b separately from r_d.
    # BBN, theta/rdrag and compressed-CMB priors constrain H0 and omega_b separately.
    if constrain_rd:
        fix_parameter(params, 'H0')
        fix_parameter(params, 'ombh2')
    else:
        params.pop('hrdrag')
        if has_bao:
            params['rdrag'] = {'latex': r'r_\mathrm{d}'}
        else:
            params.pop('rdrag')
            fix_parameter(params, 'H0')
            fix_parameter(params, 'ombh2')

    if model in {'base_w', 'base_w_wa'}:
        params['w'] = {
            'prior': {'min': -3., 'max': 1.},
            'ref': {'dist': 'norm', 'loc': -1., 'scale': 0.02},
            'proposal': 0.02,
            'latex': r'w',
        }
    if model == 'base_w_wa':
        params['wa'] = {
            'prior': {'min': -3., 'max': 2.},
            'ref': {'dist': 'norm', 'loc': 0., 'scale': 0.05},
            'proposal': 0.05,
            'latex': r'w_a',
        }

    if has_varied_nnu:
        params['nnu'] = {
            'prior': {'min': 0.05, 'max': 10.},
            'ref': {'dist': 'norm', 'loc': 3.044, 'scale': 0.05},
            'proposal': 0.05,
            'latex': r'N_\mathrm{eff}',
        }

    extra_args = {
        'bbn_predictor': 'PArthENoPE_880.2_standard.dat',
        'dark_energy_model': 'ppf',
        'num_massive_neutrinos': 1,
    }
    return params, extra_args


def get_cmb_cobaya_params(model='base', theory='camb', likelihoods=None):
    """Return Cobaya params and CAMB extra args for full-CMB-capable fits.

    This branch supports the baseline models ``base``, ``base_w`` and
    ``base_w_wa`` with the full-CMB parameterization used by Cobaya/CAMB.
    """
    _check_model_theory(model, theory)
    labels = ' '.join(likelihoods or [])
    params = {
        'logA': {
            'prior': {'min': 1.61, 'max': 3.91},
            'ref': {'dist': 'norm', 'loc': 3.036, 'scale': 0.001},
            'proposal': 0.001,
            'latex': r'\ln(10^{10} A_\mathrm{s})',
            'drop': True,
        },
        'As': {'value': 'lambda logA: 1e-10*np.exp(logA)', 'latex': r'A_\mathrm{s}'},
        'ns': {
            'prior': {'min': 0.8, 'max': 1.2},
            'ref': {'dist': 'norm', 'loc': 0.9649, 'scale': 0.004},
            'proposal': 0.002,
            'latex': r'n_\mathrm{s}',
        },
        'theta_MC_100': {
            'prior': {'min': 0.5, 'max': 10.},
            'ref': {'dist': 'norm', 'loc': 1.04109, 'scale': 0.0004},
            'proposal': 0.0002,
            'latex': r'100\theta_\mathrm{MC}',
            'drop': True,
            'renames': 'theta',
        },
        'cosmomc_theta': {'value': 'lambda theta_MC_100: 1.e-2*theta_MC_100', 'derived': False},
        'H0': {'latex': r'H_0'},
        'ombh2': {
            'prior': {'min': 0.005, 'max': 0.1},
            'ref': {'dist': 'norm', 'loc': 0.02237, 'scale': 0.0001},
            'proposal': 0.0001,
            'latex': r'\Omega_\mathrm{b} h^2',
        },
        'omch2': {
            'prior': {'min': 0.001, 'max': 0.99},
            'ref': {'dist': 'norm', 'loc': 0.12, 'scale': 0.001},
            'proposal': 0.0005,
            'latex': r'\Omega_\mathrm{c} h^2',
        },
        'tau': {
            'prior': {'min': 0.01, 'max': 0.8},
            'ref': {'dist': 'norm', 'loc': 0.0544, 'scale': 0.006},
            'proposal': 0.003,
            'latex': r'\tau_\mathrm{reio}',
        },
        'mnu': {
            'prior': {'min': 0., 'max': 5.},
            'ref': {'dist': 'norm', 'loc': 0.06, 'scale': 0.05},
            'proposal': 0.01,
            'latex': r'\sum m_\nu',
        },
        'nnu': {
            'prior': {'min': 0.05, 'max': 10.},
            'ref': {'dist': 'norm', 'loc': 3.044, 'scale': 0.05},
            'proposal': 0.05,
            'latex': r'N_\mathrm{eff}',
        },
        'w': {
            'prior': {'min': -3., 'max': 1.},
            'ref': {'dist': 'norm', 'loc': -1., 'scale': 0.02},
            'proposal': 0.02,
            'latex': r'w_0',
        },
        'wa': {
            'prior': {'min': -3., 'max': 2.},
            'ref': {'dist': 'norm', 'loc': 0., 'scale': 0.05},
            'proposal': 0.05,
            'latex': r'w_a',
        },
        'omk': {
            'prior': {'min': -0.3, 'max': 0.3},
            'ref': {'dist': 'norm', 'loc': 0., 'scale': 0.01},
            'proposal': 0.01,
            'latex': r'\Omega_\mathrm{k}',
        },
        'omegam': {'latex': r'\Omega_\mathrm{m}'},
        'omegamh2': {'derived': 'lambda omegam, H0: omegam*(H0/100)**2',
                     'latex': r'\Omega_\mathrm{m} h^2'},
        'omegal': {'latex': r'\Omega_\Lambda'},
        'zrei': {'latex': r'z_\mathrm{reio}'},
        'YHe': {'latex': r'Y_\mathrm{P}'},
        'Y_p': {'latex': r'Y_P^\mathrm{BBN}'},
        'DHBBN': {'derived': 'lambda DH: 10**5*DH', 'latex': r'10^5 \mathrm{D}/\mathrm{H}'},
        'sigma8': {'latex': r'\sigma_8'},
        's8h5': {'derived': 'lambda sigma8, H0: sigma8*(H0*1e-2)**(-0.5)',
                 'latex': r'\sigma_8/h^{0.5}'},
        's8omegamp5': {'derived': 'lambda sigma8, omegam: sigma8*omegam**0.5',
                       'latex': r'\sigma_8 \Omega_\mathrm{m}^{0.5}'},
        's8omegamp25': {'derived': 'lambda sigma8, omegam: sigma8*omegam**0.25',
                        'latex': r'\sigma_8 \Omega_\mathrm{m}^{0.25}'},
        'A': {'derived': 'lambda As: 1e9*As', 'latex': r'10^9 A_\mathrm{s}'},
        'clamp': {'derived': 'lambda As, tau: 1e9*As*np.exp(-2*tau)',
                  'latex': r'10^9 A_\mathrm{s} e^{-2\tau}'},
        'age': {'latex': r'\mathrm{Age}/\mathrm{Gyr}'},
        'rdrag': {'latex': r'r_\mathrm{d}'},
        'zdrag': {'latex': r'z_\mathrm{d}'},
        'H0rdrag': {'derived': 'lambda H0, rdrag: H0 * rdrag', 'latex': r'H_0 r_\mathrm{d}'},
    }

    # Baseline models fix neutrino mass, Neff and curvature.
    fix_parameter(params, 'mnu')
    fix_parameter(params, 'nnu')
    fix_parameter(params, 'omk')
    if model == 'base':
        fix_parameter(params, 'w')
        fix_parameter(params, 'wa')
    elif model == 'base_w':
        fix_parameter(params, 'wa')
    elif model == 'base_w_wa':
        pass

    extra_args = {
        'bbn_predictor': 'PArthENoPE_880.2_standard.dat',
        'dark_energy_model': 'ppf',
        'theta_H0_range': [20, 100],
        'num_massive_neutrinos': 1,
    }
    if 'CMB-SPA-tauprior' in labels:
        params['tau'] = {
            'prior': {'dist': 'norm', 'loc': 0.051, 'scale': 0.006},
            'proposal': 0.00295852905,
            'ref': {'dist': 'norm', 'loc': 0.050880188, 'scale': 0.001479264525},
            'latex': r'\tau_\mathrm{reio}',
        }
    if 'CMB-SPA' in labels:
        params.update({
            'A_act': {
                'prior': {'min': 0.5, 'max': 1.5},
                'ref': {'dist': 'norm', 'loc': 1.0, 'scale': 0.01},
                'proposal': 0.003,
                'latex': r'A_{\rm ACT}',
            },
            'P_act': {
                'prior': {'min': 0.9, 'max': 1.1},
                'ref': {'dist': 'norm', 'loc': 1.0, 'scale': 0.01},
                'proposal': 0.01,
                'latex': r'p_{\rm ACT}',
            },
            'Tcal': {
                'prior': {'min': 0.8, 'max': 1.2},
                'ref': {'dist': 'norm', 'loc': 1.0, 'scale': 5.0e-05},
                'proposal': 5.0e-05,
                'latex': r'T_{\rm cal}',
            },
            'Ecal': {
                'prior': {'min': 0.8, 'max': 1.2},
                'ref': 1.0,
                'latex': r'E_{\rm cal}',
            },
        })
        extra_args.update({
            'lmax': 9500,
            'recombination_model': 'CosmoRec',
            'DoLateRadTruncation': False,
            'min_l_logl_sampling': 9500,
        })
    if 'CMB-SP4A' in labels:
        extra_args.update({
            'halofit_version': 'mead2020',
            'AccuracyBoost': 1,
            'lAccuracyBoost': 1.2,
            'lSampleBoost': 1,
            'nonlinear': True,
            'kmax': 10,
            'k_per_logint': 130,
            'lens_margin': 2050,
            'lens_potential_accuracy': 8,
            'lmax': 9500,
            'recombination_model': 'CosmoRec',
            'DoLateRadTruncation': False,
            'min_l_logl_sampling': 6000,
        })
    if 'lensing' in labels:
        extra_args.update({
            'halofit_version': 'mead2016',
            'lmax': 4000,
            'lens_margin': 1250,
            'lens_potential_accuracy': 4,
            'AccuracyBoost': 1,
            'lSampleBoost': 1,
            'lAccuracyBoost': 1,
        })
    if 'act-dr6-TTTEEE' in labels:
        extra_args.update({
            'kmax': 10,
            'k_per_logint': 130,
            'nonlinear': True,
            'lens_potential_accuracy': 8,
            'lens_margin': 2050,
            'lAccuracyBoost': 1.2,
            'min_l_logl_sampling': 6000,
            'DoLateRadTruncation': False,
            'recombination_model': 'CosmoRec',
            'halofit_version': 'mead2020',
        })
    return params, extra_args


def get_cobaya_params(model='base', theory='camb', parameterization='background', likelihoods=None):
    """Return Cobaya parameters and theory ``extra_args``."""
    if parameterization == 'general':
        parameterization = 'cmb'
    if parameterization == 'background':
        return get_background_cobaya_params(model=model, theory=theory, likelihoods=likelihoods)
    if parameterization == 'cmb':
        return get_cmb_cobaya_params(model=model, theory=theory, likelihoods=likelihoods)
    raise ValueError(f'Unsupported parameterization {parameterization!r}.')
