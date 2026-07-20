
import numpy as np
import pandas as pd
from IPython.display import display

PARAM_LABELS = {
    'h': r'$h$',
    'omega_cdm': r'$\omega_{\rm cdm}$',
    'omega_b': r'$\omega_b$',
    'logA': r'$\log(10^{10} A_s)$',
    'n_s': r'$n_s$',
    'H0': r'$H_0$',
    'Omega_m': r'$\Omega_m$',
    'sigma8_m': r'$\sigma_8$',
}

PLANCK_COSMOLOGY = {
    "Omega_m": 0.315191868,
    "H_0": 67.36,
    "H0": 67.36,
    "omega_b": 0.02237,
    "omega_cdm": 0.1200,
    "sigma8_m": 0.80758,
    "h": 0.6736,
    "A_s": 2.083e-9,
    "logA": 3.034,
    "n_s": 0.9649,
    "N_ur": 2.0328,
    "N_ncdm": 1.0,
    "omega_ncdm": 0.0006442,
    "w_0": -1,
    "w_a": 0.0
}

COLOR_TRACERS = dict(BGS1='green', 
                    LRG1='orange', LRG2='orangered', LRG3='firebrick',
                    ELG1='steelblue', ELG2= 'blue',
                    QSO1='purple')

# COLOR_TRACERS = dict(BGS1='yellowgreen', 
#                     LRG1='orange', LRG2='orangered', LRG3='firebrick',
#                     ELG1='skyblue', ELG2= 'steelblue',
#                     QSO1='purple')

COSMO_CHAIN_PARAMS = ['h', 'omega_cdm',  'omega_b', 'logA', 'n_s']

def make_key(tracer, region, stats, theory=None, pk_kmax=None, bk_kmax=None, **kwargs):
    if pk_kmax is None or bk_kmax is None:
        if 'auw' in kwargs and kwargs['auw'] == True:
            return f'{tracer}_{region}_auw'
        return f'{tracer}_{region}'
    if 'mesh3_spectrum' in stats:
        return f'{tracer}_{region}_{theory[:4]}_S2_{pk_kmax:.2f}_S3_{bk_kmax:.2f}'
    return f'{tracer}_{region}_{theory[:4]}_S2_{pk_kmax:.2f}'

def parse_chain_key(key):
    parts = key.split("_")
    tracer = parts[0]
    sampler = parts[1]
    config = "_".join([parts[0], *parts[2:]])  # everything except sampler
    return tracer, sampler, config

def read_fit_result(kind, fn, **kwargs):
    if isinstance(fn, list):
        return [read_fit_result(kind, one_fn, **kwargs) for one_fn in fn]
    if kind == 'profiles':
        try:
            from desilike.samples import Profiles
        except ImportError:
            if str(fn).endswith(('.h5', '.hdf5')):
                obj = read_hdf5_profile(fn)
            else:
                from desilike.samples.profiles import Profiles
                reader = getattr(Profiles, 'read', None) or getattr(Profiles, 'load')
                obj = reader(fn)
        else:
            reader = getattr(Profiles, 'read', None) or getattr(Profiles, 'load')
            try:
                obj = reader(fn)
            except Exception:
                if str(fn).endswith(('.h5', '.hdf5')):
                    obj = read_hdf5_profile(fn)
                else:
                    raise
    elif kind == 'chain':
        try:
            from desilike.samples import Chain
        except ImportError:
            from desilike.samples.chain import Chain
        remove_burnin = kwargs.get('remove_burnin', 0.5)
        ravel = kwargs.get('ravel', False)
        step = kwargs.get('step', 1)
        if str(fn).endswith(('.h5', '.hdf5')):
            obj = read_hdf5_chain(fn)
        else:
            obj = Chain.load(fn)
        obj = obj.remove_burnin(remove_burnin)[::step]
        if ravel: obj = obj.ravel()
    else:
        raise ValueError(f'Unknown kind: {kind}')
    return obj

class HDF5ProfileTable(dict):
    """Minimal desilike-like profile table backed by saved HDF5 arrays."""

    def __init__(self, data, names=None):
        super().__init__({str(name): np.asarray(value) for name, value in data.items()})
        self._names = [str(name) for name in (names or self.keys())]

    def choice(self, input=True, index='argmax', squeeze=True, **kwargs):
        return self

    def params(self, varied=False, input=False, **kwargs):
        names = [str(name) for name in self._names]
        if varied:
            skip = {'logposterior', 'logprior', 'loglikelihood', 'aweight'}
            names = [name for name in names if name not in skip]
        return names

class HDF5Profile:
    """Small profile wrapper with the fields used by the plotting notebook."""

    def __init__(self, bestfit, error=None, fn=None):
        self.bestfit = bestfit
        self.best = bestfit
        self.error = error
        self.fn = fn

    def __contains__(self, name):
        return getattr(self, str(name), None) is not None

def _read_hdf5_names(group):
    if group is None or '__names__' not in group:
        return None
    names = group['__names__']
    try:
        return [str(name) for name in names.asstr()[()]]
    except Exception:
        return [name.decode() if isinstance(name, bytes) else str(name) for name in names[()]]

def _read_hdf5_array_group(group):
    data = {}
    if group is None:
        return data
    for name, value in group.items():
        if name == '__names__':
            continue
        if hasattr(value, 'keys') and 'value' in value:
            data[str(name)] = np.asarray(value['value'])
        elif not hasattr(value, 'keys'):
            data[str(name)] = np.asarray(value)
    return data

def read_hdf5_profile(fn):
    """Read enough of a saved desilike Profiles HDF5 file for notebook plots."""
    import h5py
    with h5py.File(fn, 'r') as stream:
        names = _read_hdf5_names(stream.get('params'))
        best_data = _read_hdf5_array_group(stream.get('best'))
        if not best_data:
            best_data = _read_hdf5_array_group(stream.get('params'))
        error_data = _read_hdf5_array_group(stream.get('error'))
    bestfit = HDF5ProfileTable(best_data, names=names or best_data.keys())
    error = HDF5ProfileTable(error_data, names=error_data.keys()) if error_data else None
    return HDF5Profile(bestfit=bestfit, error=error, fn=fn)

def read_hdf5_chain(fn):
    """Read sampler chain files written as one HDF5 group per parameter."""
    import h5py
    try:
        from desilike.samples import Chain
    except ImportError:
        from desilike.samples.chain import Chain
    with h5py.File(fn, 'r') as stream:
        if '__names__' in stream:
            names = stream['__names__'].asstr()[()]
        else:
            names = [name for name, value in stream.items() if hasattr(value, 'keys') and 'value' in value]
        data = {
            str(name): np.asarray(stream[str(name)]['value'])
            for name in names
            if str(name) in stream and 'value' in stream[str(name)]
        }
    return Chain(data=data)

def _param_name_from_state(param):
    name = param.get('name', None)
    if name:
        return str(name)
    basename = str(param.get('basename', ''))
    namespace = str(param.get('namespace', ''))
    return f'{namespace}.{basename}' if namespace else basename

def _param_label_from_state(param):
    return str(param.get('latex', _param_name_from_state(param))).strip('$')

def read_chain_arrays(fn):
    """Read either HDF5 sampler chains or desilike numpy chain states into arrays."""
    fn = str(fn)
    labels = {}
    if fn.endswith(('.h5', '.hdf5')):
        import h5py
        with h5py.File(fn, 'r') as stream:
            if '__names__' in stream:
                names = [str(name) for name in stream['__names__'].asstr()[()]]
            else:
                names = [name for name, value in stream.items() if hasattr(value, 'keys') and 'value' in value]
            data = {
                name: np.asarray(stream[name]['value'])
                for name in names
                if name in stream and 'value' in stream[name]
            }
        return data, labels

    state = np.load(fn, allow_pickle=True)[()]
    data = {}
    for item in state['data']:
        name = _param_name_from_state(item['param'])
        data[name] = np.asarray(item['value'])
        labels[name] = _param_label_from_state(item['param'])
    return data, labels

def _trim_chain_array(array, remove_burnin=0.5, step=1):
    array = np.asarray(array)
    if remove_burnin:
        nremove = int(array.shape[0] * remove_burnin) if 0 < remove_burnin < 1 else int(remove_burnin)
        array = array[nremove:]
    return array.reshape(-1)[::step]

def read_getdist_chain(fn, params, label=None, remove_burnin=0.5, step=1, settings=None):
    """Read a saved chain file directly into a getdist.MCSamples object."""
    from getdist import MCSamples
    data, labels = read_chain_arrays(fn)
    missing = [param for param in params if param not in data]
    if missing:
        raise ValueError(f'{fn} is missing requested params: {missing}')
    samples = np.column_stack([_trim_chain_array(data[param], remove_burnin, step) for param in params])
    nsamples = samples.shape[0]
    weights = None
    for weight_name in ('weight', 'weights', 'aweight', 'fweight'):
        if weight_name in data:
            weights = _trim_chain_array(data[weight_name], remove_burnin, step)[:nsamples]
            break
    loglikes = None
    if 'logposterior' in data:
        loglikes = -_trim_chain_array(data['logposterior'], remove_burnin, step)[:nsamples]
    gd_chain = MCSamples(
        samples=samples,
        weights=weights,
        loglikes=loglikes,
        names=params,
        labels=[PARAM_LABELS.get(param, labels.get(param, param)).strip('$') for param in params],
        label=label,
    )
    if settings:
        gd_chain.updateSettings(settings)
    return gd_chain

class SavedChain:
    """Small chain wrapper for plotting/summary helpers, backed by saved arrays."""

    def __init__(self, data, labels=None):
        self.data = {str(name): np.asarray(value) for name, value in data.items()}
        self.labels = dict(labels or {})
        first = next(iter(self.data.values()))
        self.shape = first.shape
        self.ndim = first.ndim
        self.size = int(first.size)

    def __getitem__(self, name):
        return self.data[str(name)]

    def __contains__(self, name):
        return str(name) in self.data

    def params(self, varied=False, input=False, **kwargs):
        skip = {'weight', 'weights', 'aweight', 'fweight', 'logposterior', 'logprior', 'loglikelihood'}
        names = [name for name in self.data if name not in skip]
        if not varied:
            names += [name for name in self.data if name in {'logposterior', 'logprior', 'loglikelihood', 'aweight'}]
        return names

    def _values(self, param):
        return np.asarray(self.data[str(param)]).reshape(-1)

    @property
    def weight(self):
        weights = None
        for name in ('weight', 'weights', 'aweight', 'fweight'):
            if name in self.data:
                current = self._values(name).astype(float)
                weights = current if weights is None else weights * current
        if weights is None:
            weights = np.ones(self.size, dtype=float)
        return weights

    @property
    def logposterior(self):
        return self._values('logposterior') if 'logposterior' in self.data else np.zeros(self.size)

    def mean(self, params):
        values = self._values(params)
        weights = self.weight[:values.size]
        good = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
        if not np.any(good):
            return np.nan
        return float(np.average(values[good], weights=weights[good]))

    def median(self, params):
        values = self._values(params)
        values = values[np.isfinite(values)]
        return float(np.median(values)) if values.size else np.nan

    def quantile(self, params, q=(0.1587, 0.8413)):
        values = self._values(params)
        values = values[np.isfinite(values)]
        if not values.size:
            return tuple(np.nan for _ in q)
        return tuple(float(value) for value in np.quantile(values, q))

    def to_getdist(self, params=None, label=None, settings=None, **kwargs):
        from getdist import MCSamples
        params = list(params or self.params(varied=True))
        samples = np.column_stack([self._values(param) for param in params])
        nsamples = samples.shape[0]
        gd_chain = MCSamples(
            samples=samples,
            weights=self.weight[:nsamples],
            loglikes=-self.logposterior[:nsamples],
            names=params,
            labels=[PARAM_LABELS.get(param, self.labels.get(param, param)).strip('$') for param in params],
            label=label,
            **kwargs,
        )
        if settings:
            gd_chain.updateSettings(settings)
        return gd_chain

def read_saved_chain(fn, remove_burnin=0.5, step=1):
    """Read saved .h5/.npy chain products into a lightweight chain object."""
    data, labels = read_chain_arrays(fn)
    data = {name: _trim_chain_array(value, remove_burnin, step) for name, value in data.items()}
    return SavedChain(data, labels=labels)

def _fiducial_params(chain, tracer, cosmo_params=COSMO_CHAIN_PARAMS):
    available = [str(param) for param in chain.params(varied=True)]
    params = [param for param in cosmo_params if param in available]
    params += [param for param in available if param.startswith(f'{tracer}.')]
    return params

def _chain_GR_coeff(chains, tracer, params=None, all_check=False):
    from desilike.samples import diagnostics
    if not isinstance(chains, list):
        chains = [chains]
    gr = {}
    if params is None: params = _fiducial_params(chains[0], tracer=tracer)
    for param in params:
        rhat = diagnostics.gelman_rubin(chains,param,method="diag",)
        gr[str(param)] = rhat
    if all_check:
        for p, rhat in gr.items():
            print(f"{p:30s} Rhat = {rhat:.5f}, Rhat - 1 = {rhat - 1:.5f}")
    max_gr = max(rhat - 1 for rhat in gr.values())
    print(f"max (Rhat-1):", max_gr)
    return gr, max_gr

def chain_fit_results(chain, params=None):
    fit_results = {}
    if params is None:
        params = chain.params(varied=True)
    for param in params:
        if str(param) not in [str(p) for p in chain.params()]:
            continue
        q16, q84 = chain.quantile(params=param, q=(0.1587, 0.8413))
        fit_results[str(param)] = {
            'mean': chain.mean(params=param),
            'median': chain.median(params=param),
            'q16': q16,
            'q84': q84,
        }
    return fit_results

def _chain_sigma(chain, param, q=(0.1587, 0.8413)):
    qlo, qhi = chain.quantile(params=param, q=q)
    return 0.5 * float(qhi - qlo)

def _chain_center(chain, param):
    return float(chain.median(params=param))

DEFAULT_STAT_KMAX = {
    'mesh2_spectrum': 0.35,
    'mesh3_spectrum': 0.20,
}

def _finite_float_or_none(value):
    if value is None or value is False:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None

def _stat_label_from_entry(entry):
    if not isinstance(entry, dict):
        return 'unknown'
    run_options = entry.get('run_options', {})
    stats = run_options.get('stats', [])
    if isinstance(stats, str):
        stats = [stats]
    pk_kmax = _finite_float_or_none(run_options.get('s2_kmax'))
    bk_kmax = _finite_float_or_none(run_options.get('s3_kmax'))
    if pk_kmax is None and 'mesh2_spectrum' in stats:
        pk_kmax = DEFAULT_STAT_KMAX['mesh2_spectrum']
    if bk_kmax is None and 'mesh3_spectrum' in stats:
        bk_kmax = DEFAULT_STAT_KMAX['mesh3_spectrum']
    if 'mesh3_spectrum' in stats and bk_kmax is not None:
        if 'mesh2_spectrum' in stats and pk_kmax is not None:
            return f"S2_{pk_kmax:.2f} + S3_{bk_kmax:.2f}"
        return f"S3_{bk_kmax:.2f}"
    if 'mesh2_spectrum' in stats and pk_kmax is not None:
        return f"S2_{pk_kmax:.2f}"
    return '+'.join(map(str, stats)) if stats else 'unknown'


def build_bestfits_from_chains(chain_dict, params=None, center='mean'):
    rows = []
    for key, entry in chain_dict.items():
        run_options = entry.get('run_options', {})
        tracer = run_options.get('tracers', ['unknown'])[0]
        region = run_options.get('regions', ['unknown'])[0]
        all_fit_results = entry.get('fit_results', {})
        if params is None:
            params_use = list(all_fit_results.keys())
        else:
            params_use = [str(param) for param in params]
        fit_results = {param: all_fit_results[param] for param in params_use if param in all_fit_results}
        row = {
            'key': str(key),
            'tracer': str(tracer),
            'region': str(region),
            'bestfit': fit_results,
            'stat_label': _stat_label_from_entry(entry),
        }
        for param, values in fit_results.items():
            if isinstance(values, dict):
                row[param] = values.get(center, np.nan)
            else:
                row[param] = np.nan
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=['key', 'tracer', 'region', 'bestfit', 'stat_label'])
    return pd.DataFrame(rows)

def _profile_bestfit_choice(bestfit):
    if hasattr(bestfit, 'choice'):
        for kwargs in ({'input': True, 'index': 'argmax'}, {'index': 'argmax'}, {}):
            try:
                return bestfit.choice(**kwargs)
            except TypeError:
                continue
            except Exception:
                continue
    return bestfit

def _profile_param_value(bestfit, param):
    if bestfit is None:
        return np.nan

    candidates = []
    if hasattr(bestfit, 'choice'):
        for kwargs in ({'input': True, 'index': 'argmax'}, {'index': 'argmax'}, {}):
            try:
                candidates.append(bestfit.choice(**kwargs))
            except TypeError:
                continue
            except Exception:
                continue
    candidates.append(bestfit)

    for candidate in candidates:
        value = None
        if isinstance(candidate, dict):
            value = candidate.get(param, None)
        else:
            try:
                value = candidate[param]
            except Exception:
                value = getattr(candidate, param, None)
        if value is None:
            continue
        if hasattr(value, 'value'):
            value = value.value
        values = np.asarray(value)
        if values.size == 0:
            continue
        values = values.reshape(-1)
        finite = np.isfinite(values)
        if finite.any():
            return float(values[finite][0])
    return np.nan

def _profile_bestfit_value(bestfit, param, index='argmax'):
    """Return one scalar from a desilike profile bestfit table."""
    if bestfit is None:
        return np.nan
    for choice_kwargs in ({'index': index}, {}):
        try:
            choice = bestfit.choice(**choice_kwargs)
            value = choice[param] if isinstance(choice, dict) else choice[param]
            if hasattr(value, 'value'):
                value = value.value
            values = np.asarray(value)
            if values.size:
                return float(values.reshape(-1)[0])
        except Exception:
            pass
    try:
        value = bestfit[param]
        if hasattr(value, 'value'):
            value = value.value
        values = np.asarray(value)
        if values.size:
            values = values.reshape(-1)
            finite = np.isfinite(values)
            if finite.any():
                if param in {'loglikelihood', 'logposterior'}:
                    return float(np.nanmax(values[finite]))
                return float(values[finite][0])
    except Exception:
        pass
    return np.nan

def get_profile_bestfit(profiles):
    """Return the bestfit container across desilike profile API versions."""
    if profiles is None:
        return None
    bestfit = getattr(profiles, 'bestfit', None)
    if bestfit is None:
        bestfit = getattr(profiles, 'best', None)
    return bestfit

def _profile_nparams(profile_entry, chain_entry=None):
    if chain_entry is not None:
        try:
            return len(chain_entry['chain'].params(varied=True))
        except Exception:
            pass
    profiles = profile_entry.get('profiles', profile_entry) if isinstance(profile_entry, dict) else profile_entry
    bestfit = get_profile_bestfit(profiles)
    if bestfit is None and isinstance(profile_entry, dict):
        bestfit = profile_entry.get('profile_results', {}).get('bestfit', None)
    try:
        return len(bestfit.params(varied=True))
    except Exception:
        pass
    try:
        names = [str(param) for param in bestfit.params()]
        names = [name for name in names if not name.startswith('log')]
        return len(names)
    except Exception:
        return np.nan

def _profile_lookup_key(profiles_by_key, tracer, region, stats):
    for key in (make_key(tracer, region, stats), f'{tracer}_{region}', f'{tracer}_{region}_auw'):
        if key in profiles_by_key:
            return key
    prefix = f'{tracer}_{region}'
    for key, entry in profiles_by_key.items():
        entry_tracer, entry_region = _profile_tracer_region_from_entry(key, entry)
        if entry_tracer == str(tracer) and entry_region == str(region):
            return key
        if str(key) == prefix or str(key).startswith(f'{prefix}_'):
            return key
    return None

def _profile_tracer_region_from_entry(key, entry=None):
    if isinstance(entry, dict):
        run_options = entry.get('run_options', {})
        tracer = run_options.get('tracers', [None])[0]
        region = run_options.get('regions', [None])[0]
        if tracer is not None and region is not None:
            return str(tracer), str(region)
    if isinstance(key, tuple) and len(key) >= 2:
        return str(key[0]), str(key[1])
    tracer, region = _tracer_region_from_key(key)
    if region.endswith('_auw'):
        region = region[:-4]
    return tracer, region

def _profile_table_row(key, profile_entry, tracer, region, kranges, chains_by_key):
    from scipy.stats import chi2 as chi2_distribution

    run_options = profile_entry.get('run_options', {}) if isinstance(profile_entry, dict) else {}
    chain_entry = chains_by_key.get(key)
    if not run_options and chain_entry is not None:
        run_options = chain_entry.get('run_options', {})
    profiles = profile_entry.get('profiles', profile_entry) if isinstance(profile_entry, dict) else profile_entry
    bestfit = get_profile_bestfit(profiles)
    if bestfit is None and isinstance(profile_entry, dict):
        bestfit = profile_entry.get('profile_results', {}).get('bestfit', None)
    max_loglikelihood = _profile_bestfit_value(bestfit, 'loglikelihood')
    ndata = _ndata_from_run_options(run_options, kranges=kranges)
    nparams = _profile_nparams(profile_entry, chain_entry=chain_entry)
    dof = int(ndata - nparams) if np.isfinite(nparams) and ndata else np.nan
    chi2 = -2.0 * max_loglikelihood if np.isfinite(max_loglikelihood) else np.nan
    return {
        'tracer': tracer,
        'region': region,
        'max_loglikelihood': max_loglikelihood,
        'ndata': ndata,
        'nparams': nparams,
        'chi2': chi2,
        'dof': dof,
        'chi2_dof': chi2 / dof if np.isfinite(chi2) and np.isfinite(dof) and dof > 0 else np.nan,
        'p_value': chi2_distribution.sf(chi2, dof) if np.isfinite(chi2) and np.isfinite(dof) and dof > 0 else np.nan,
    }

def build_profile_chi2_dof_table(profiles_by_key, tracers, regions, stats, kranges,
                                 chains_by_key=None):
    rows = []
    chains_by_key = chains_by_key or {}
    tracers = [str(tracer) for tracer in tracers]
    regions = [str(region) for region in regions]
    for tracer in tracers:
        for region in regions:
            key = _profile_lookup_key(profiles_by_key, tracer, region, stats)
            if key is None:
                continue
            rows.append(_profile_table_row(key, profiles_by_key[key], tracer, region, kranges, chains_by_key))

    seen = {(row['tracer'], row['region']) for row in rows}
    for key, profile_entry in profiles_by_key.items():
        tracer, region = _profile_tracer_region_from_entry(key, profile_entry)
        if tracer not in tracers or region not in regions or (tracer, region) in seen:
            continue
        rows.append(_profile_table_row(key, profile_entry, tracer, region, kranges, chains_by_key))
        seen.add((tracer, region))

    columns = ['tracer', 'region', 'max_loglikelihood', 'ndata', 'nparams', 'chi2', 'dof', 'chi2_dof', 'p_value']
    return pd.DataFrame(rows, columns=columns)

def _tracer_region_from_key(key):
    parts = str(key).split('_')
    tracer = parts[0] if parts else 'unknown'
    region = '_'.join(parts[1:]) if len(parts) > 1 else 'unknown'
    return tracer, region

def build_bestfits_from_profiles(profile_dict, params=None, chain_dict=None):
    rows = []
    chain_dict = chain_dict or {}
    for key, entry in profile_dict.items():
        entry = entry if isinstance(entry, dict) else {'profile_results': {'bestfit': entry}}
        run_options = entry.get('run_options', {})
        if not run_options and key in chain_dict and isinstance(chain_dict[key], dict):
            run_options = chain_dict[key].get('run_options', {})
        tracer = run_options.get('tracers', [None])[0]
        region = run_options.get('regions', [None])[0]
        if tracer is None or region is None:
            tracer, region = _tracer_region_from_key(key)
        profile_results = entry.get('profile_results', entry)
        bestfit = profile_results.get('bestfit', entry.get('bestfit', None))
        if bestfit is None:
            continue
        params_use = [str(param) for param in params] if params is not None else [str(param) for param in getattr(bestfit, 'params', lambda: [])()]
        fit_results = {}
        row = {
            'key': str(key),
            'tracer': str(tracer),
            'region': str(region),
            'bestfit': fit_results,
            'stat_label': _stat_label_from_entry({'run_options': run_options}),
        }
        for param in params_use:
            value = _profile_param_value(bestfit, param)
            if np.isfinite(value):
                fit_results[param] = {'bestfit': value}
                row[param] = value
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=['key', 'tracer', 'region', 'bestfit', 'stat_label'])
    return pd.DataFrame(rows)

def _chain_param_values(chain, param):
    values = chain[param]
    try:
        if hasattr(chain, 'ndim') and np.ndim(values) > chain.ndim:
            values = values[()]
    except Exception:
        pass
    if hasattr(values, 'value'):
        values = values.value
    return np.asarray(values, dtype=float).reshape(-1)

def _chain_weight_field(chain, name, size):
    try:
        values = getattr(chain, name)
        values = values() if callable(values) else values
    except Exception:
        try:
            values = chain[name]
        except Exception:
            return None
    if hasattr(values, 'value'):
        values = values.value
    values = np.asarray(values, dtype=float).reshape(-1)
    return values if values.size == size else None

def chain_sample_weights(chain, size):
    aweight = _chain_weight_field(chain, 'aweight', size)
    fweight = _chain_weight_field(chain, 'fweight', size)
    if aweight is not None and fweight is not None:
        weights = aweight * fweight
    else:
        weights = None
        for name in ('weight', 'weights', 'aweight', 'fweight'):
            weights = _chain_weight_field(chain, name, size)
            if weights is not None:
                break
    if weights is None:
        weights = np.ones(size, dtype=float)
    weights = np.where(np.isfinite(weights), weights, 0.0)
    if np.sum(weights) <= 0:
        weights = np.ones(size, dtype=float)
    return weights / np.sum(weights)

def weighted_chain_covariance(chain, params):
    available = {str(param) for param in chain.params()}
    used = [str(param) for param in params if str(param) in available]
    samples = [_chain_param_values(chain, param) for param in used]
    if not samples:
        return [], np.empty((0, 0))

    nsamples = min(len(sample) for sample in samples)
    samples = np.column_stack([sample[:nsamples] for sample in samples])
    weights = chain_sample_weights(chain, nsamples)
    good = np.isfinite(samples).all(axis=1) & np.isfinite(weights) & (weights > 0)
    samples, weights = samples[good], weights[good]
    if samples.shape[0] < 2 or np.sum(weights) <= 0:
        return used, np.full((len(used), len(used)), np.nan)

    weights = weights / np.sum(weights)
    mean = np.average(samples, axis=0, weights=weights)
    centered = samples - mean
    cov = (centered * weights[:, None]).T @ centered
    correction = 1.0 - np.sum(weights**2)
    if correction > 0:
        cov *= 1.0 / correction
    return used, np.atleast_2d(cov)

def _chain_max_loglikelihood(chain):
    try:
        loglikelihood = _chain_param_values(chain, 'loglikelihood')
    except Exception:
        return np.nan
    return np.nanmax(loglikelihood) if np.isfinite(loglikelihood).any() else np.nan

def chain_bestfit_sample(chain, score='loglikelihood'):
    """Return the chain sample with the largest stored score column."""
    score_values = _chain_param_values(chain, score)
    if not np.isfinite(score_values).any():
        raise ValueError(f'No finite {score!r} values found in chain.')
    index = int(np.nanargmax(score_values))
    sample = {}
    try:
        params = chain.params(varied=True, input=True)
    except Exception:
        params = chain.params(varied=True)
    if not len(params):
        params = chain.params(varied=True)
    for param in params:
        name = str(param)
        try:
            values = _chain_param_values(chain, name)
        except Exception:
            continue
        if index < values.size and np.isfinite(values[index]):
            sample[name] = float(values[index])
    return sample, index, float(score_values[index])

def _likelihood_params(likelihood):
    params_by_name = {}
    for attr in ('varied_params', 'all_params', 'params'):
        params = getattr(likelihood, attr, None)
        if params is None:
            continue
        try:
            params = params() if callable(params) else params
            for param in params:
                params_by_name.setdefault(str(param), param)
        except Exception:
            continue
    for sublikelihood in getattr(likelihood, 'likelihoods', []):
        params_by_name.update(_likelihood_params(sublikelihood))
    return params_by_name

def _evaluate_likelihood_at_sample(likelihood, sample):
    likelihood_params = _likelihood_params(likelihood)
    names = set(likelihood_params)
    params = {name: value for name, value in sample.items() if not names or name in names}
    sample_by_basename = {}
    for name, value in sample.items():
        basename = name.split('.')[-1]
        sample_by_basename.setdefault(basename, []).append((name, value))
    exact_matches = set(params)
    basename_matches = {}
    for name, param in likelihood_params.items():
        if name in params:
            continue
        basename = getattr(param, 'basename', name.split('.')[-1])
        candidates = sample_by_basename.get(str(basename), [])
        if len(candidates) == 1:
            params[name] = candidates[0][1]
            basename_matches[name] = candidates[0][0]
    missing = sorted(name for name in names if name not in params)
    for caller in (
        lambda: likelihood(**params),
        lambda: likelihood(params),
        lambda: likelihood.loglikelihood(**params),
    ):
        try:
            return caller(), params, {
                'nexact_matches': len(exact_matches),
                'nbasename_matches': len(basename_matches),
                'missing_likelihood_params': missing,
                'basename_matches': basename_matches,
            }
        except Exception:
            continue
    raise RuntimeError('Could not evaluate likelihood at the supplied chain sample.')

def _iter_leaf_likelihoods(likelihood):
    likelihoods = getattr(likelihood, 'likelihoods', None)
    if likelihoods:
        for sublikelihood in likelihoods:
            yield from _iter_leaf_likelihoods(sublikelihood)
    else:
        yield likelihood

def _chi2_from_likelihood_precision(likelihood):
    chi2_parts = []
    for sublikelihood in _iter_leaf_likelihoods(likelihood):
        flatdiff = np.asarray(getattr(sublikelihood, 'flatdiff'), dtype=float).reshape(-1)
        precision = np.asarray(getattr(sublikelihood, 'precision'), dtype=float)
        if precision.ndim == 1:
            chi2_parts.append(float((flatdiff * precision) @ flatdiff))
        else:
            chi2_parts.append(float(flatdiff @ precision @ flatdiff))
    return float(np.sum(chi2_parts)), chi2_parts

def _flat_observable_array(obj):
    for attr in ('flatdata', 'flattheory', 'flatarray'):
        if hasattr(obj, attr):
            value = getattr(obj, attr)
            value = value() if callable(value) else value
            return np.asarray(value, dtype=float).reshape(-1)
    if hasattr(obj, 'value'):
        for kwargs in ({'concatenate': True}, {}):
            try:
                return np.asarray(obj.value(**kwargs), dtype=float).reshape(-1)
            except TypeError:
                continue
            except Exception:
                continue
    return np.asarray(obj, dtype=float).reshape(-1)

def _collect_likelihood_vectors(likelihood):
    data, theory, covariance = [], [], []
    likelihoods = getattr(likelihood, 'likelihoods', None)
    if likelihoods:
        for sublikelihood in likelihoods:
            subdata, subtheory, subcov = _collect_likelihood_vectors(sublikelihood)
            data.append(subdata)
            theory.append(subtheory)
            covariance.append(subcov)
        import scipy as sp
        return np.concatenate(data), np.concatenate(theory), sp.linalg.block_diag(*covariance)

    observables = getattr(likelihood, 'observables', None)
    if observables is None:
        observable = getattr(likelihood, 'observable', None)
        observables = [observable] if observable is not None else []
    for observable in observables:
        if hasattr(observable, 'flatdata') and hasattr(observable, 'flattheory'):
            data.append(_flat_observable_array(getattr(observable, 'flatdata')))
            theory.append(_flat_observable_array(getattr(observable, 'flattheory')))
        else:
            data_obj = getattr(observable, 'data', None)
            theory_obj = getattr(observable, 'theory', None)
            data.append(_flat_observable_array(data_obj))
            theory.append(_flat_observable_array(theory_obj))
    cov = getattr(likelihood, 'covariance', None)
    if hasattr(cov, 'value'):
        cov = cov.value()
    return np.concatenate(data), np.concatenate(theory), np.asarray(cov, dtype=float)

def _dof_from_vectors(ndata, nparams):
    return int(ndata - nparams) if np.isfinite(nparams) and ndata else np.nan

def chain_entry_direct_chi2(entry, kranges=None, cache_dir=None, cache_mode='r',
                            score='loglikelihood', rcond=None):
    """
    Rebuild the saved fit likelihood and compute chi2 from data-model residuals.

    This avoids relying on the arbitrary zero point of the stored chain
    ``loglikelihood`` column.
    """
    from full_shape import tools
    import full_shape.job_scripts.test_data_splits as tds

    run_options = dict(entry.get('run_options', {}))
    if not run_options:
        raise ValueError('entry must contain run_options.')
    if kranges is not None:
        tds.KRANGES = kranges
    options = tds._build_run_options(**run_options)
    covariance_options = options['likelihoods'][0].get('covariance', {}) if options.get('likelihoods') else {}
    sample, index, max_score = chain_bestfit_sample(entry['chain'], score=score)
    likelihood = tools.get_likelihood(
        options['likelihoods'],
        cosmology_options=options.get('cosmology', None),
        cache_dir=cache_dir,
        cache_mode=cache_mode,
    )
    _, used_params, match_info = _evaluate_likelihood_at_sample(likelihood, sample)
    chi2, chi2_parts = _chi2_from_likelihood_precision(likelihood)
    data, theory, covariance = _collect_likelihood_vectors(likelihood)
    if data.shape != theory.shape:
        raise ValueError(f'Data/model length mismatch: {data.size} != {theory.size}.')
    if covariance.shape != (data.size, data.size):
        raise ValueError(f'Covariance shape {covariance.shape} does not match data length {data.size}.')
    residual = data - theory
    dof = _dof_from_vectors(data.size, len(used_params))
    return {
        'chi2': chi2,
        'dof': dof,
        'chi2_dof': chi2 / dof if np.isfinite(dof) and dof > 0 else np.nan,
        'ndata': int(data.size),
        'nparams': int(len(used_params)),
        'nchain_params': int(len(sample)),
        'nunused_chain_params': int(len(set(sample) - set(used_params))),
        'nexact_param_matches': int(match_info['nexact_matches']),
        'nbasename_param_matches': int(match_info['nbasename_matches']),
        'nmissing_likelihood_params': int(len(match_info['missing_likelihood_params'])),
        'missing_likelihood_params': match_info['missing_likelihood_params'],
        'covariance_region': covariance_options.get('region', None),
        'covariance_scale': covariance_options.get('scale', np.nan),
        'use_scale_covariance': bool(run_options.get('use_scale_covariance', False)),
        'chain_index': int(index),
        f'max_{score}': max_score,
        'loglikelihood_eval': float(np.asarray(getattr(likelihood, 'loglikelihood', np.nan)).reshape(-1)[0]),
        'chi2_from_loglikelihood_eval': -2.0 * float(np.asarray(getattr(likelihood, 'loglikelihood', np.nan)).reshape(-1)[0]),
        'chi2_parts': chi2_parts,
    }

def build_chain_direct_chi2_dof_table(chains_by_key, tracers, regions, stats, kranges=None,
                                      cache_dir=None, cache_mode='r', score='loglikelihood',
                                      rcond=None, raise_errors=False):
    rows = []
    for tracer in tracers:
        for region in regions:
            key = make_key(tracer, region, stats)
            entry = chains_by_key.get(key)
            if entry is None:
                continue
            row = {'tracer': tracer, 'region': region}
            try:
                row.update(chain_entry_direct_chi2(
                    entry, kranges=kranges, cache_dir=cache_dir, cache_mode=cache_mode,
                    score=score, rcond=rcond,
                ))
                row['error'] = ''
            except Exception as exc:
                if raise_errors:
                    raise
                row.update({'chi2': np.nan, 'dof': np.nan, 'chi2_dof': np.nan,
                            'ndata': np.nan, 'nparams': np.nan, 'nchain_params': np.nan,
                            'nunused_chain_params': np.nan, 'nexact_param_matches': np.nan,
                            'nbasename_param_matches': np.nan, 'nmissing_likelihood_params': np.nan,
                            'missing_likelihood_params': None, 'covariance_region': None,
                            'covariance_scale': np.nan, 'use_scale_covariance': np.nan,
                            'chain_index': np.nan,
                            f'max_{score}': np.nan, 'loglikelihood_eval': np.nan,
                            'chi2_from_loglikelihood_eval': np.nan, 'chi2_parts': None,
                            'error': str(exc)})
            rows.append(row)
    columns = ['tracer', 'region', 'chi2', 'dof', 'chi2_dof', 'ndata', 'nparams',
               'nchain_params', 'nunused_chain_params', 'nexact_param_matches',
               'nbasename_param_matches', 'nmissing_likelihood_params',
               'missing_likelihood_params', 'covariance_region',
               'covariance_scale', 'use_scale_covariance', 'chain_index', f'max_{score}',
               'loglikelihood_eval', 'chi2_from_loglikelihood_eval', 'chi2_parts', 'error']
    return pd.DataFrame(rows, columns=columns)

def debug_chain_entry_parameter_matching(entry, kranges=None, cache_dir=None, cache_mode='r',
                                         score='loglikelihood', max_names=30):
    """Return parameter-matching diagnostics for one chain entry."""
    from full_shape import tools
    import full_shape.job_scripts.test_data_splits as tds

    run_options = dict(entry.get('run_options', {}))
    if kranges is not None:
        tds.KRANGES = kranges
    options = tds._build_run_options(**run_options)
    sample, index, max_score = chain_bestfit_sample(entry['chain'], score=score)
    likelihood = tools.get_likelihood(
        options['likelihoods'],
        cosmology_options=options.get('cosmology', None),
        cache_dir=cache_dir,
        cache_mode=cache_mode,
    )
    _, used_params, match_info = _evaluate_likelihood_at_sample(likelihood, sample)
    chi2, chi2_parts = _chi2_from_likelihood_precision(likelihood)
    likelihood_names = sorted(_likelihood_params(likelihood))
    sample_names = sorted(sample)
    return {
        'chain_index': index,
        f'max_{score}': max_score,
        'chi2': chi2,
        'chi2_parts': chi2_parts,
        'loglikelihood_eval': float(np.asarray(getattr(likelihood, 'loglikelihood', np.nan)).reshape(-1)[0]),
        'n_sample_params': len(sample_names),
        'n_likelihood_params': len(likelihood_names),
        'n_used_params': len(used_params),
        'nexact_param_matches': match_info['nexact_matches'],
        'nbasename_param_matches': match_info['nbasename_matches'],
        'nmissing_likelihood_params': len(match_info['missing_likelihood_params']),
        'sample_params_head': sample_names[:max_names],
        'likelihood_params_head': likelihood_names[:max_names],
        'missing_likelihood_params_head': match_info['missing_likelihood_params'][:max_names],
        'basename_matches_head': dict(list(match_info['basename_matches'].items())[:max_names]),
    }

def _count_k_points(k_range):
    kmin, kmax, dk = map(float, k_range)
    return int(np.rint((kmax - kmin) / dk)) if dk > 0 and kmax > kmin else 0

def _ndata_from_run_options(run_options, kranges):
    stats_use = run_options.get('stats', [])
    if isinstance(stats_use, str):
        stats_use = [stats_use]
    total = 0
    for stat in stats_use:
        kmax_override = {
            'mesh2_spectrum': run_options.get('s2_kmax'),
            'mesh3_spectrum': run_options.get('s3_kmax'),
        }.get(stat)
        for item in kranges.get(stat, []):
            k_range = list(item['k'])
            if kmax_override is not None:
                k_range[1] = kmax_override
            total += _count_k_points(k_range)
    return total

def _ndata_nparams_dof_from_chain_entry(entry, kranges):
    chain = entry['chain']
    ndata = _ndata_from_run_options(entry.get('run_options', {}), kranges=kranges)
    try:
        nvaried = len(chain.params(varied=True))
    except Exception:
        nvaried = np.nan
    dof = int(ndata - nvaried) if np.isfinite(nvaried) and ndata else np.nan
    return ndata, nvaried, dof

def _dof_from_chain_entry(entry, kranges):
    _, _, dof = _ndata_nparams_dof_from_chain_entry(entry, kranges=kranges)
    return dof

def build_chain_chi2_dof_table(chains_by_key, tracers, regions, stats, kranges):
    from scipy.stats import chi2 as chi2_distribution

    rows = []
    for tracer in tracers:
        for region in regions:
            key = make_key(tracer, region, stats)
            entry = chains_by_key.get(key)
            if entry is None:
                continue
            max_loglikelihood = _chain_max_loglikelihood(entry['chain'])
            ndata, nparams, dof = _ndata_nparams_dof_from_chain_entry(entry, kranges=kranges)
            chi2 = -2.0 * max_loglikelihood if np.isfinite(max_loglikelihood) else np.nan
            p_value = chi2_distribution.sf(chi2, dof) if np.isfinite(chi2) and np.isfinite(dof) and dof > 0 else np.nan
            rows.append({
                'tracer': tracer,
                'region': region,
                'max_loglikelihood': max_loglikelihood,
                'ndata': ndata,
                'nparams': nparams,
                'chi2': chi2,
                'dof': dof,
                'chi2_dof': chi2 / dof if np.isfinite(chi2) and np.isfinite(dof) and dof > 0 else np.nan,
                'p_value': p_value,
            })
    columns = ['tracer', 'region', 'max_loglikelihood', 'ndata', 'nparams', 'chi2', 'dof', 'chi2_dof', 'p_value']
    return pd.DataFrame(rows, columns=columns)

def plot_chain_chi2_dof_heatmap(chi2_table, tracers, regions, cmap='viridis', vmin=0,
                                title=r'Chain max-loglikelihood bestfit: $\chi^2 / \mathrm{dof}$',
                                annotate=True, **imshow_kwargs):
    from matplotlib import pyplot as plt
    if chi2_table.empty:
        raise ValueError('chi2_table is empty. Re-run the result-loading cell and check that the table keys match tracers/regions.')
    if 'chi2_dof' not in chi2_table:
        raise ValueError('chi2_table must contain a chi2_dof column.')
    if chi2_table['chi2_dof'].isna().all():
        detail = ''
        if 'error' in chi2_table:
            errors = chi2_table['error'].dropna()
            errors = [str(error) for error in errors if str(error)]
            if errors:
                detail = f' First error: {errors[0]}'
        raise ValueError(f'No finite chi2/dof values found in the table.{detail}')

    grid = chi2_table.pivot(index='region', columns='tracer', values='chi2_dof').reindex(index=regions, columns=tracers)
    fig, ax = plt.subplots(figsize=(0.8 * len(tracers) + 2.0, 0.38 * len(regions) + 2.0),
                           constrained_layout=True)
    image = ax.imshow(grid.to_numpy(dtype=float), cmap=cmap, vmin=vmin, aspect='auto', **imshow_kwargs)
    ax.set_title(title)

    x_centers = np.arange(len(tracers))
    x_edges = np.arange(-0.5, len(tracers) + 0.5, 1)
    ax.set_xticks(x_edges)
    ax.set_xticklabels([''] * len(x_edges))
    ax.tick_params(axis='x', which='major', length=5)
    secax = ax.secondary_xaxis('bottom')
    secax.set_xticks(x_centers)
    secax.set_xticklabels(tracers, rotation=0, ha='center')
    secax.tick_params(axis='x', length=0, pad=10)
    secax.spines['bottom'].set_visible(False)

    ax.set_yticks(np.arange(len(regions)))
    ax.set_yticklabels(regions)
    if annotate:
        for i, region in enumerate(regions):
            for j, tracer in enumerate(tracers):
                value = grid.loc[region, tracer]
                if pd.notna(value):
                    ax.text(j, i, f'{value:.2f}', ha='center', va='center', fontsize=10,
                            color='white' if value > 1.2 else 'black')

    cbar = fig.colorbar(image, ax=ax, shrink=0.9, pad=0.02)
    cbar.set_label(r'$\chi^2 / \mathrm{dof}$')
    return fig, ax, grid

def bestfit_vector(key, params, chains_by_key, profiles_by_key=None, fallback_center='mean'):
    profile_entry = (profiles_by_key or {}).get(key, {})
    profile_results = profile_entry.get('profile_results', profile_entry)
    profile_bestfit = profile_results.get('bestfit', None)

    used, values = [], []
    for param in map(str, params):
        value = _profile_param_value(profile_bestfit, param) if profile_bestfit is not None else np.nan
        if not np.isfinite(value):
            fit_result = chains_by_key.get(key, {}).get('fit_results', {}).get(param, {})
            value = float(fit_result.get(fallback_center, np.nan))
        if np.isfinite(value):
            used.append(param)
            values.append(value)
    return used, np.asarray(values, dtype=float)

def build_fob_from_gccomb_bestfit(chains_by_key, stats, profiles_by_key=None, tracers=None, regions=None,
                                  params=('H0', 'Omega_m', 'sigma8_m'), reference_region='GCcomb',
                                  fallback_center='mean'):
    rows = []
    params = [str(param) for param in params]
    if tracers is None:
        tracers = sorted({entry['run_options']['tracers'][0] for entry in chains_by_key.values()})
    if regions is None:
        regions = sorted({entry['run_options']['regions'][0] for entry in chains_by_key.values()})

    for tracer in tracers:
        ref_key = make_key(tracer, reference_region, stats)
        ref_params, ref_bestfit = bestfit_vector(
            ref_key, params, chains_by_key, profiles_by_key, fallback_center=fallback_center)
        if not ref_params:
            print(f'Missing fiducial bestfit for {tracer}: {ref_key}')
            continue

        for region in regions:
            if region == reference_region:
                continue
            key = make_key(tracer, region, stats)
            if key not in chains_by_key:
                continue

            region_params, region_bestfit = bestfit_vector(
                key, params, chains_by_key, profiles_by_key, fallback_center=fallback_center)
            cov_params, cov = weighted_chain_covariance(chains_by_key[key]['chain'], params)
            common = [param for param in params if param in ref_params and param in region_params and param in cov_params]
            if not common:
                continue

            ref_idx = [ref_params.index(param) for param in common]
            region_idx = [region_params.index(param) for param in common]
            cov_idx = [cov_params.index(param) for param in common]
            delta = region_bestfit[region_idx] - ref_bestfit[ref_idx]
            region_cov = cov[np.ix_(cov_idx, cov_idx)]
            if not (np.isfinite(delta).all() and np.isfinite(region_cov).all()):
                continue

            fob = np.sqrt(max(delta @ np.linalg.pinv(region_cov) @ delta, 0.0))
            rows.append({
                'tracer': tracer,
                'region': region,
                'params': ', '.join(common),
                'ndim': len(common),
                'FoB': float(fob),
            })
    return pd.DataFrame(rows)

def plot_fob_heatmap(fob_table, tracers=None, regions=None, reference_region='GCcomb',
                     cmap='Blues', vmin=0, annotate=True, **imshow_kwargs):
    from matplotlib import pyplot as plt
    if fob_table.empty:
        raise ValueError('fob_table is empty; no FoB values available to plot')
    if tracers is None:
        tracers = list(fob_table['tracer'].dropna().unique())
    if regions is None:
        regions = list(fob_table['region'].dropna().unique())
    regions = [region for region in regions if region != reference_region]
    grid = fob_table.pivot(index='region', columns='tracer', values='FoB').reindex(index=regions, columns=tracers)

    fig, ax = plt.subplots(figsize=(0.8 * len(tracers) + 2.0, 0.38 * len(regions) + 2.0),
                           constrained_layout=True)
    image = ax.imshow(grid.to_numpy(dtype=float), cmap=cmap, vmin=vmin, aspect='auto', **imshow_kwargs)
    ax.set_title(f'FoB vs {reference_region} bestfit')

    x_centers = np.arange(len(tracers))
    x_edges = np.arange(-0.5, len(tracers) + 0.5, 1)
    ax.set_xticks(x_edges)
    ax.set_xticklabels([''] * len(x_edges))
    ax.tick_params(axis='x', which='major', length=5)
    secax = ax.secondary_xaxis('bottom')
    secax.set_xticks(x_centers)
    secax.set_xticklabels(tracers, rotation=0, ha='center')
    secax.tick_params(axis='x', length=0, pad=10)
    secax.spines['bottom'].set_visible(False)

    ax.set_yticks(np.arange(len(regions)))
    ax.set_yticklabels(regions)
    if annotate:
        for i, region in enumerate(regions):
            for j, tracer in enumerate(tracers):
                value = grid.loc[region, tracer]
                if pd.notna(value):
                    ax.text(j, i, f'{value:.1f}', ha='center', va='center', fontsize=12, color='white')
    cbar = fig.colorbar(image, ax=ax, shrink=0.9, pad=0.02)
    cbar.set_label('FoB')
    return fig, ax, grid

def _chain_error_yerr(bestfit_param, center='mean', low='q16', high='q84'):
    if not isinstance(bestfit_param, dict):
        return None
    if center not in bestfit_param:
        return None
    if low not in bestfit_param or high not in bestfit_param:
        return None
    c = float(bestfit_param[center])
    qlow = float(bestfit_param[low])
    qhigh = float(bestfit_param[high])
    return np.array([[c - qlow], [qhigh - c]])

def plot_chain_comparison_by_stat(chains_dict, params=('h', 'omega_cdm', 'logA'), tracers=None, stat_labels=None, center='mean', low='q16', high='q84', **plot_kwargs):
    from matplotlib import pyplot as plt
    markers = plot_kwargs.get('markers', ['o', 's', 'D', '^', 'v'])
    bestfits = build_bestfits_from_chains(chain_dict=chains_dict, params=params)
    if bestfits.empty:
        raise ValueError('bestfits is empty; no chain results available to plot')
    required_cols = {'key', 'tracer', 'stat_label', 'bestfit'}
    missing_cols = required_cols - set(bestfits.columns)
    if missing_cols:
        raise ValueError(f"bestfits is missing required columns: {sorted(missing_cols)}")
    table = bestfits.copy()
    table['key'] = table['key'].astype(str)
    table['tracer'] = table['tracer'].astype(str)
    table['stat_label'] = table['stat_label'].astype(str)
    if tracers is None:
        tracers = list(table['tracer'].dropna().unique())
    else:
        tracers = [str(tracer) for tracer in tracers]
        table = table.loc[table['tracer'].isin(tracers)]
    if stat_labels is None:
        stat_labels = list(table['stat_label'].dropna().unique())
    else:
        stat_labels = [str(stat_label) for stat_label in stat_labels]
        table = table.loc[table['stat_label'].isin(stat_labels)]
    params = [str(param) for param in params if str(param) in table.columns]
    if not params:
        raise ValueError('None of the requested parameters are in bestfits')
    if not tracers:
        raise ValueError('No tracers found in bestfits')
    if not stat_labels:
        raise ValueError('No statistic labels found in bestfits')
    fig, axes = plt.subplots(len(params), 1, figsize=(8.0, 1.7 * len(params) + 0.8), sharex=False,)
    axes = np.atleast_1d(axes)
    offsets = np.linspace(-0.1, 0.1, max(len(stat_labels), 1))
    x_positions = np.arange(len(tracers))
    points_plotted = 0
    for ax, param in zip(axes, params):
        for istat, stat_label in enumerate(stat_labels):
            for itracer, tracer in enumerate(tracers):
                rows = table.loc[(table['tracer'] == str(tracer)) & (table['stat_label'] == str(stat_label))]
                if rows.empty: continue
                row = rows.iloc[0]
                if pd.isna(row[param]): continue
                x = x_positions[itracer] + offsets[istat]
                param_result = row['bestfit'].get(param, {})
                if center in param_result:
                    y = float(param_result[center])
                else:
                    y = float(row[param])
                norm = plt.Normalize(vmin=0, vmax=max(len(stat_labels) - 1, 1))
                cmap = plt.colormaps[COLOR_TRACERS.get(str(tracer), 'viridis')]
                shade = 0.20 + 0.20 * norm(istat)
                color = cmap(shade)
                yerr = _chain_error_yerr(param_result, center=center, low=low, high=high,)
                # ax.scatter(x, y,  edgecolors=color, facecolors='none', s=34, zorder=3,)
                if yerr is not None:
                    ax.errorbar(x, y, yerr=yerr, color=color, marker=markers[istat % len(markers)], ls='', capsize=2, lw=1.0, zorder=2, label=stat_label if itracer == 0 else None,)
                points_plotted += 1
        ax.set_ylabel(PARAM_LABELS.get(param, param))
        ax.set_xticks(x_positions)
        ax.set_xticklabels(tracers)
        ax.set_xlim(x_positions[0]-0.5,x_positions[-1]+0.5)
        ax.grid(axis='y', alpha=0.2)
    if points_plotted == 0:
        raise ValueError('No matching bestfit points found to plot')
    axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), fontsize=10, ncol=min(len(stat_labels), 3), frameon=False,)
    fig.tight_layout()
    return fig

def plot_chain_comparison_by_region(chains_dict, params=('h', 'omega_cdm', 'logA'), tracers=None, regions=None,
                                    profiles=None, profiles_by_key=None, stats='S2_0.20',
                                    center='mean', low='q16', high='q84', reference_region='GCcomb',
                                    **plot_kwargs):
    from matplotlib import pyplot as plt
    from matplotlib import colors as mcolors
    markers = plot_kwargs.get('markers', ['o', 's', 'D', '^', 'v'])
    figsize = plot_kwargs.get('figsize', None)
    title_prefix = plot_kwargs.get('title_prefix', 'blinded')
    bestfits = build_bestfits_from_chains(chain_dict=chains_dict, params=params)
    if bestfits.empty:
        raise ValueError('bestfits is empty; no chain results available to plot')
    required_cols = {'key', 'tracer', 'region', 'bestfit', 'stat_label'}
    missing_cols = required_cols - set(bestfits.columns)
    if missing_cols:
        raise ValueError(f"bestfits is missing required columns: {sorted(missing_cols)}")
    table = bestfits.copy()
    table['key'] = table['key'].astype(str)
    table['tracer'] = table['tracer'].astype(str)
    table['region'] = table['region'].astype(str)
    table['stat_label'] = table['stat_label'].astype(str)
    if profiles is None and profiles_by_key is not None:
        profiles = profiles_by_key
    profile_table = None
    if profiles is not None:
        profile_table = build_bestfits_from_profiles(profiles, params=params, chain_dict=chains_dict)
        if not profile_table.empty:
            profile_table['key'] = profile_table['key'].astype(str)
            profile_table['tracer'] = profile_table['tracer'].astype(str)
            profile_table['region'] = profile_table['region'].astype(str)
            profile_table['stat_label'] = profile_table['stat_label'].astype(str)
    if tracers is None:
        tracers = list(table['tracer'].dropna().unique())
    else:
        tracers = [str(tracer) for tracer in tracers]
        table = table.loc[table['tracer'].isin(tracers)]
        if profile_table is not None and not profile_table.empty:
            profile_table = profile_table.loc[profile_table['tracer'].isin(tracers)]
    if regions is None:
        regions = list(table['region'].dropna().unique())
    else:
        regions = [str(region) for region in regions]
        table = table.loc[table['region'].isin(regions)]
        if profile_table is not None and not profile_table.empty:
            profile_table = profile_table.loc[profile_table['region'].isin(regions)]
    if stats is not None:
        available_stats = list(table['stat_label'].dropna().unique())
        table = table.loc[table['stat_label'].isin([str(stats)])]
        if profile_table is not None and not profile_table.empty:
            profile_table = profile_table.loc[profile_table['stat_label'].isin([str(stats)])]
        if table.empty:
            raise ValueError(f"No rows found for stats={stats}. Available stat labels: {available_stats}")
    params = [str(param) for param in params if str(param) in table.columns]
    if not params:
        raise ValueError('None of the requested parameters are in bestfits')
    def tracer_color(tracer):
        color = COLOR_TRACERS.get(str(tracer), 'C0')
        if color in plt.colormaps():
            return plt.colormaps[color](0.55)
        if mcolors.is_color_like(color):
            return color
        return 'C0'
    def profile_value_for_plot(tracer, region, param):
        if profiles is not None:
            key = _profile_lookup_key(profiles, tracer, region, stats)
            if key is not None:
                entry = profiles[key]
                if isinstance(entry, dict):
                    bestfit = entry.get('profile_results', {}).get('bestfit', None)
                    if bestfit is None:
                        bestfit = get_profile_bestfit(entry.get('profiles', None))
                else:
                    bestfit = get_profile_bestfit(entry)
                value = _profile_param_value(bestfit, param)
                if np.isfinite(value):
                    return value
        if profile_table is not None and not profile_table.empty and param in profile_table.columns:
            profile_rows = profile_table.loc[(profile_table['tracer'] == str(tracer)) & (profile_table['region'] == str(region))]
            if not profile_rows.empty:
                value = profile_rows.iloc[0][param]
                if not pd.isna(value):
                    return float(value)
        return np.nan
    region_labels = [region[:-4] if region.endswith('Scomb') else region for region in regions]
    y_positions = np.arange(1, len(regions) + 1)
    figures = {}
    for param in params:
        fig_width = 2.0 * len(tracers)
        fig_height = max(1.8, 0.35 * (len(regions) + 2))
        fig, axes = plt.subplots(1, len(tracers), figsize=figsize or (fig_width, fig_height),
                                 sharey=True, squeeze=False, gridspec_kw={'wspace': 0.1},)
        axes = axes[0]
        points_plotted = 0
        profile_points_plotted = 0
        for itracer, (ax, tracer) in enumerate(zip(axes, tracers)):
            color = tracer_color(tracer)
            ax.set_title(f'{title_prefix} {tracer}', fontsize=10)
            ax.set_xlabel(PARAM_LABELS.get(param, param), fontsize=12)
            ax.set_yticks(y_positions)
            if itracer == 0:
                ax.set_yticklabels(region_labels, fontsize=10)
            else:
                ax.tick_params(axis='y', labelleft=False)
            for y, region in zip(y_positions, regions):
                # ax.axhline(y, color='0.88', lw=0.3, zorder=0)
                profile_x = profile_value_for_plot(tracer, region, param)
                if np.isfinite(profile_x):
                    ax.scatter(float(profile_x), y, marker='^', color=color, s=46, zorder=4,
                               label='profile bestfit' if profile_points_plotted == 0 and itracer == 0 else None)
                    profile_points_plotted += 1
                rows = table.loc[(table['tracer'] == str(tracer)) & (table['region'] == str(region))]
                if rows.empty:
                    continue
                row = rows.iloc[0]
                if pd.isna(row[param]):
                    continue
                param_result = row['bestfit'].get(param, {})
                if center in param_result:
                    x = float(param_result[center])
                else:
                    x = float(row[param])
                xerr = _chain_error_yerr(param_result, center=center, low=low, high=high)
                if xerr is not None:
                    ax.errorbar(x,y,xerr=xerr,color=color,
                                # marker=markers[itracer % len(markers)],
                                marker = 'o',
                                markerfacecolor='none',
                                markeredgecolor=color,
                                ls='', capsize=2, lw=1.0, zorder=2,)
                    if region == reference_region:
                        ax.axvspan(x - xerr[0, 0], x + xerr[1, 0], color=color, alpha=0.08, zorder=0)
                else:
                    ax.scatter(x, y, edgecolors=color, facecolors='none', s=40, zorder=2)
                points_plotted += 1
            ax.set_ylim(0, len(regions) + 1)
            ax.invert_yaxis()
            # ax.grid(axis='x', alpha=0.2)
        if points_plotted == 0:
            raise ValueError(f'No matching bestfit points found to plot for {param}')
        fig.tight_layout()
        figures[param] = fig
    return figures

def format_result(d, ndigits=4):
    mean = d["mean"]
    q16 = d["q16"]
    q84 = d["q84"]
    return f"{mean:.{ndigits}f} -{mean - q16:.{ndigits}f} +{q84 - mean:.{ndigits}f}"

def get_sigma(d):
    return 0.5 * (d["q84"] - d["q16"])

def sigma_difference(d1, d2):
    delta = d1["mean"] - d2["mean"]
    sigma = np.hypot(get_sigma(d1), get_sigma(d2))
    nsigma = delta / sigma if sigma > 0 else np.nan
    return delta, nsigma, abs(nsigma)


def get_error_ratio_table(sampler_results, params, reference_sampler="emcee"):
    rows = []
    if reference_sampler not in sampler_results:
        return pd.DataFrame()
    for param in params:
        if param not in sampler_results[reference_sampler]:
            continue

        sigma_ref = get_sigma(sampler_results[reference_sampler][param])

        for sampler, results in sampler_results.items():
            if param not in results:
                continue
            sigma = get_sigma(results[param])
            rows.append({
                "param": param,
                "sampler": sampler,
                "sigma": sigma,
                "sigma_ref": sigma_ref,
                "error_ratio": sigma / sigma_ref if sigma_ref > 0 else np.nan,
            })
    return pd.DataFrame(rows)


def _result_sigma(result):
    return 0.5 * (float(result['q84']) - float(result['q16']))

def build_region_sigma_deviations(chains_by_key, table, tracers=None, regions=None, stats=None, 
                                  params=('h', 'omega_cdm', 'logA'),
                                  reference_region='GCcomb', center='mean'):
    rows = []
    if tracers is None:
        tracers = sorted({key.split('_')[0] for key in chains_by_key})
    if regions is None:
        regions = sorted({entry['run_options']['regions'][0] for entry in chains_by_key.values()})

    for tracer in tracers:
        ref_key = make_key(tracer, reference_region, stats)
        if ref_key not in chains_by_key:
            print(f'Missing reference chain for {tracer}: {ref_key}')
            continue
        ref_results = chains_by_key[ref_key]['fit_results']
        for region in regions:
            if region == reference_region:
                continue
            key = make_key(tracer, region, stats)
            if key not in chains_by_key:
                continue
            results = chains_by_key[key]['fit_results']
            for param in params:
                if param not in results or param not in ref_results:
                    continue
                value = float(results[param][center])
                ref_value = float(ref_results[param][center])
                sigma = _result_sigma(results[param])
                ref_sigma = _result_sigma(ref_results[param])
                r = table.loc[(table["name"] == tracer) & (table["region"] == region), "r"].iloc[0]
                combined_sigma = np.sqrt((sigma**2 + ref_sigma**2 - 2 * r * sigma * ref_sigma))
                rows.append({
                    'tracer': tracer,
                    'region': region,
                    'param': param,
                    'delta': value - ref_value,
                    'sigma_region': sigma,
                    'sigma_GCcomb': ref_sigma,
                    'combined_sigma': combined_sigma,
                    'nsigma': (value - ref_value) / combined_sigma if combined_sigma > 0 else np.nan,
                })
    return pd.DataFrame(rows)



'''

def diagnose_chain(
    chain,
    key=None,
    params=('h', 'omega_cdm', 'logA'),
    chains=None,
    rhat=None,
    thresholds=None,
    verbose=True,
):
    """
    Diagnose whether a posterior chain is safe to use for constraints.

    Parameters
    ----------
    chain : desilike.samples.Chain
        Concatenated/raveled chain used for final constraints.
    key : str, optional
        Label for printout.
    params : list[str]
        Parameters to test constraint stability for.
    chains : list[desilike.samples.Chain], optional
        Individual chain files before concatenation. Used for per-chain checks.
    rhat : dict, optional
        Dict mapping param -> Rhat or Rhat-1.
    thresholds : dict, optional
        Override default quality thresholds.
    """

    defaults = {
        # logposterior quality
        'max_minus_p5_max': 50.0,
        'max_minus_p1_max': 100.0,
        'frac_below_max_minus_50_max': 0.01,
        'frac_below_max_minus_100_max': 0.001,

        # parameter stability across separate chains
        'per_chain_center_shift_max': 0.35,  # in combined sigma units
        'per_chain_sigma_ratio_max': 1.25,

        # Rhat
        'rhat_minus_one_max': 0.05,
    }
    if thresholds:
        defaults.update(thresholds)
    th = defaults

    name = key or 'chain'
    rows = []
    failures = []

    lp = np.asarray(chain['logposterior'])
    lp_pct = np.percentile(lp, [0, 1, 5, 16, 50, 84, 95, 99, 100])
    lp_min, lp_p1, lp_p5, lp_p16, lp_med, lp_p84, lp_p95, lp_p99, lp_max = lp_pct

    lp_checks = {
        'max_minus_p5': lp_max - lp_p5,
        'max_minus_p1': lp_max - lp_p1,
        'frac_below_max_minus_50': np.mean(lp < lp_max - 50.0),
        'frac_below_max_minus_100': np.mean(lp < lp_max - 100.0),
    }

    if lp_checks['max_minus_p5'] > th['max_minus_p5_max']:
        failures.append('logpost p5 is too far below max')
    if lp_checks['max_minus_p1'] > th['max_minus_p1_max']:
        failures.append('logpost p1 is too far below max')
    if lp_checks['frac_below_max_minus_50'] > th['frac_below_max_minus_50_max']:
        failures.append('too many samples below max-50')
    if lp_checks['frac_below_max_minus_100'] > th['frac_below_max_minus_100_max']:
        failures.append('too many samples below max-100')

    rows.append({
        'test': 'logposterior',
        'metric': 'percentiles [0,1,5,16,50,84,95,99,100]',
        'value': lp_pct,
        'pass': len(failures) == 0,
    })

    for metric, value in lp_checks.items():
        rows.append({
            'test': 'logposterior',
            'metric': metric,
            'value': value,
            'pass': value <= th.get(metric + '_max', np.inf),
        })

    # Per-chain stability, if individual chain files are provided
    if chains is not None:
        for param in params:
            centers = []
            sigmas = []

            for c in chains:
                cr = c.ravel()
                centers.append(_chain_center(cr, param))
                sigmas.append(_chain_sigma(cr, param))

            centers = np.asarray(centers)
            sigmas = np.asarray(sigmas)

            full_center = _chain_center(chain, param)
            full_sigma = _chain_sigma(chain, param)

            max_center_shift = np.max(np.abs(centers - full_center) / full_sigma)
            sigma_ratio = np.max(sigmas) / np.min(sigmas)

            pass_center = max_center_shift <= th['per_chain_center_shift_max']
            pass_width = sigma_ratio <= th['per_chain_sigma_ratio_max']

            if not pass_center:
                failures.append(f'{param}: per-chain centers disagree')
            if not pass_width:
                failures.append(f'{param}: per-chain widths disagree')

            rows.append({
                'test': 'per-chain center',
                'metric': param,
                'value': max_center_shift,
                'pass': pass_center,
            })
            rows.append({
                'test': 'per-chain width',
                'metric': param,
                'value': sigma_ratio,
                'pass': pass_width,
            })

    # Rhat check, if provided
    if rhat is not None:
        for param, value in rhat.items():
            rminus1 = float(value - 1.0) if value > 0.5 else float(value)
            passed = rminus1 <= th['rhat_minus_one_max']

            if not passed:
                failures.append(f'{param}: Rhat-1 too large')

            rows.append({
                'test': 'Rhat',
                'metric': str(param),
                'value': rminus1,
                'pass': passed,
            })

    table = pd.DataFrame(rows)
    passed = len(failures) == 0

    if verbose:
        print(f'\n{name}')
        print('PASS' if passed else 'FAIL')
        if failures:
            print('Failures:')
            for item in sorted(set(failures)):
                print(' -', item)

        # print('\nlogposterior percentiles:')
        # print(pd.Series(
        #     lp_pct,
        #     index=['min', 'p1', 'p5', 'p16', 'p50', 'p84', 'p95', 'p99', 'max'],
        # ))

    return {
        'key': name,
        'pass': passed,
        'failures': sorted(set(failures)),
        'table': table,
        'logposterior_percentiles': lp_pct,
    }

'''
