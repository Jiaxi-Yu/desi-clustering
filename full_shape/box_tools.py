"""Helpers specific to full-shape fits of cubic box measurements."""
from pathlib import Path

from clustering_statistics.tools import get_simple_tracer, _make_tuple


def get_default_box_mesh3_basis(version='abacus-hf-v2'):
    """Return the mesh3 basis used by available box measurements."""
    return 'sugiyama' if version == 'abacus-2ndgen' else 'sugiyama-diagonal'


def get_lsstypes_covariance_defaults(tracer, stats=('mesh2_spectrum', 'mesh3_spectrum')):
    """Return lsstypes EZmock covariance options matched to the fitted tracer."""
    covariance_zsnaps = {'BGS': 0.200, 'LRG': 0.800, 'ELG': 0.950, 'QSO': 1.400}
    covariance_tracer = get_simple_tracer(tracer)
    if isinstance(covariance_tracer, tuple) and len(covariance_tracer) == 1:
        covariance_tracer = covariance_tracer[0]
    if covariance_tracer not in covariance_zsnaps:
        raise ValueError(f'no lsstypes EZmock covariance defaults for tracer {tracer!r}')
    options = {
        'tracer': covariance_tracer,
        'zsnap': covariance_zsnaps[covariance_tracer],
        'imock': '*',
    }
    if 'mesh3_spectrum' in stats:
        options['stat_options'] = {'mesh3_spectrum': {'basis': 'sugiyama-diagonal'}}
    return options


def get_covariance_volume_scale_factor(config=None):
    """Return covariance volume scaling factor from box-size rescaling options."""
    config = dict(config or {})
    if not config or not config.get('enabled', False):
        return 1.0
    source = float(config['source_boxsize_gpch'])
    target = float(config['target_boxsize_gpch'])
    if source <= 0. or target <= 0.:
        raise ValueError(f'box sizes must be positive, got source={source}, target={target}')
    return (source / target)**3


def generate_box_likelihood_options_helper(
        stats=('mesh2_spectrum',),
        tracer='LRG',
        zsnap=0.800,
        cosmo='000',
        hod='',
        los='z',
        version='abacus-2ndgen',
        imocks='*',
        selects=None,
        stat_options=None,
        window_mode='file',
        covariance_version=None,
        covariance_tracer=None,
        covariance_zsnap=None,
        covariance_cosmo='000',
        covariance_hod='',
        covariance_los='z',
        covariance_imocks='*',
        covariance_stat_options=None,
        covariance_volume_rescaling=None,
        stats_dir=Path('/dvs_ro/cfs/cdirs/desicollab/mocks/cai/LSS/DA2/mocks/desipipe/box'),
        covariance_stats_dir=None,
        emulator=True):
    """
    Convenience helper that builds likelihood options for cubic box mock fits.

    Use ``covariance_version=None`` for flat lsstypes box measurement directories.
    """
    if isinstance(stats, str):
        stats = [stats]
    if imocks == 'all':
        imocks = '*'
    if covariance_imocks == 'all':
        covariance_imocks = '*'
    selects = selects or {}
    stat_options = stat_options or {}
    covariance_stat_options = covariance_stat_options or {}
    tracers = _make_tuple(tracer)
    data_tracer = tracer if any('lorentzian' in t for t in tracers) else get_simple_tracer(tracers)
    if covariance_tracer is None:
        covariance_tracer = get_simple_tracer(tracers)
    if covariance_zsnap is None:
        covariance_zsnap = zsnap
    observables = []
    for stat in stats:
        catalog = {
            'tracer': data_tracer,
            'zsnap': zsnap,
            'cosmo': cosmo,
            'hod': hod,
            'los': los,
            'version': version,
            'imock': imocks,
            'stats_dir': stats_dir,
        }
        _stat_options = {'kind': stat}
        _stat_options.update(stat_options.get(stat, {}))
        if stat in selects:
            _stat_options['select'] = selects[stat]
        observable_options = {'stat': _stat_options, 'catalog': catalog, 'window': {'mode': window_mode}}
        if emulator is False:
            emulator_options = {'name': ''}
        elif emulator is True:
            emulator_options = {}
        else:
            emulator_options = dict(emulator)
        observable_options['emulator'] = emulator_options
        observables.append(observable_options)
    if covariance_stats_dir is None:
        covariance_stats_dir = stats_dir
    covariance = {'source': 'mock', 'version': covariance_version, 'stats_dir': covariance_stats_dir,
                  'tracer': covariance_tracer, 'zsnap': covariance_zsnap, 'cosmo': covariance_cosmo,
                  'hod': covariance_hod, 'los': covariance_los, 'imock': covariance_imocks,
                  'corrections': ['hartlap', 'percival']}
    if covariance_stat_options:
        covariance['stat_options'] = {key: dict(value) for key, value in covariance_stat_options.items()}
    if covariance_volume_rescaling is not None:
        covariance['volume_rescaling'] = dict(covariance_volume_rescaling)

    from .tools import fill_fiducial_likelihood_options
    return fill_fiducial_likelihood_options({'observables': observables, 'covariance': covariance})
