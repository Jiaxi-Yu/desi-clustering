"""Compare generated Cobaya configs against the DESI Y3 reference.

This is a lightweight config-generation comparison. It does not initialize
Cobaya likelihoods, so it can be used even when external Planck data/cache setup
is not writable.
"""

import argparse
import sys
from pathlib import Path


DEFAULT_Y3_REPO = Path('/global/homes/u/uendert/repos/desi/desi-y3-kp')
DEFAULT_CASES = [
    'desi-dr2-bao-all',
    'desi-dr2-bao-all,pantheonplus',
    'desi-dr2-bao-all,schoneberg2024-bbn',
    'desi-dr2-bao-all,CMB-compressed-theta',
    'desi-dr2-bao-all,planck-NPIPE-highl-CamSpec-TTTEEE',
    'desi-dr2-bao-all,pantheonplus,planck-NPIPE-highl-CamSpec-TTTEEE',
    'desi-dr2-bao-all,planck-NPIPE-highl-CamSpec-TTTEEE,planckpr4lensing',
]


def _load_y3_get_cobaya_info(y3_repo):
    sys.path.insert(0, str(y3_repo))
    from scripts.y3_bao_cosmo_tools import get_cobaya_info as get_y3_cobaya_info
    return get_y3_cobaya_info


def _load_new_get_cobaya_info():
    from cosmo.cobaya import get_cobaya_info
    return get_cobaya_info


def _sampled_params(info):
    return sorted(name for name, config in info.get('params', {}).items()
                  if isinstance(config, dict) and 'prior' in config)


def _comparison_sampled_params(info, likelihoods):
    sampled = _sampled_params(info)
    # The reference base BAO+Schoneberg varied-BBN config omits nnu even though
    # the likelihood requires it at runtime. The native desi-clustering config
    # samples nnu so Cobaya can initialize. Do not flag this intentional runtime
    # fix as a reference-comparison mismatch.
    if 'schoneberg2024-bbn' in likelihoods:
        sampled = [name for name in sampled if name != 'nnu']
    return sorted(sampled)


def _fixed_params(info):
    return sorted(name for name, config in info.get('params', {}).items()
                  if isinstance(config, dict) and 'value' in config and 'prior' not in config)


def _extra_args(info):
    theory = info.get('theory', {})
    if not theory:
        return {}
    first = next(iter(theory.values()))
    return first.get('extra_args', {}) or {}


def _canonical_likelihood(name):
    # Canonicalize native desi-clustering classes against reference classes.
    replacements = {
        'desi_y3_cosmo_bindings.cobaya_likelihoods.bao_likelihoods_v1p2.desi_bao_all': 'DESI_BAO_ALL',
        'cosmo.cobaya.external_likelihoods.bao.desi_dr2.desi_dr2_bao_all': 'DESI_BAO_ALL',
        'desi_y3_cosmo_bindings.cobaya_likelihoods.bbn_likelihoods.schoneberg2024': 'BBN_SCHONEBERG2024',
        'cosmo.cobaya.external_likelihoods.bbn.schoneberg2024': 'BBN_SCHONEBERG2024',
        'desi_y3_cosmo_bindings.cobaya_likelihoods.bbn_likelihoods.schoneberg2024_fixed_nnu': 'BBN_SCHONEBERG2024_FIXED_NNU',
        'cosmo.cobaya.external_likelihoods.bbn.schoneberg2024_fixed_nnu': 'BBN_SCHONEBERG2024_FIXED_NNU',
        'desi_y3_cosmo_bindings.cobaya_likelihoods.cmb_likelihoods.CMB_standard_compression_PR4': 'CMB_STANDARD_COMPRESSION_PR4',
        'cosmo.cobaya.external_likelihoods.cmb.CMB_standard_compression_PR4': 'CMB_STANDARD_COMPRESSION_PR4',
        'desi_y3_cosmo_bindings.cobaya_likelihoods.cmb_likelihoods.CMB_shift_parameters_compression_PR3': 'CMB_SHIFT_PARAMETERS_COMPRESSION_PR3',
        'cosmo.cobaya.external_likelihoods.cmb.CMB_shift_parameters_compression_PR3': 'CMB_SHIFT_PARAMETERS_COMPRESSION_PR3',
        'planck_NPIPE_highl_CamSpec.TTTEEE': 'PLANCK_NPIPE_CAMSPEC_TTTEEE',
        'cosmo.cobaya.external_likelihoods.cmb.planck_npipe.TTTEEENoCache': 'PLANCK_NPIPE_CAMSPEC_TTTEEE',
    }
    return replacements.get(name, name)


def _canonical_likelihoods(info):
    return sorted(_canonical_likelihood(name) for name in info.get('likelihood', {}))


def _print_diff(title, y3, new):
    y3_set, new_set = set(y3), set(new)
    if y3_set == new_set:
        print(f'  {title}: OK')
        return
    print(f'  {title}: DIFFER')
    only_y3 = sorted(y3_set - new_set)
    only_new = sorted(new_set - y3_set)
    if only_y3:
        print(f'    only Y3:  {only_y3}')
    if only_new:
        print(f'    only new: {only_new}')


def compare_case(case, y3_get, new_get, model='base', sampler='evaluate'):
    likelihoods = case.split(',') if isinstance(case, str) else list(case)
    print('\nCASE', likelihoods)
    y3_info = y3_get(model=model, dataset=likelihoods, sampler=sampler)
    new_info = new_get(model=model, likelihoods=likelihoods, sampler=sampler, output=False)

    _print_diff('likelihoods', _canonical_likelihoods(y3_info), _canonical_likelihoods(new_info))
    _print_diff('sampled params', _comparison_sampled_params(y3_info, likelihoods),
                _comparison_sampled_params(new_info, likelihoods))
    _print_diff('theory extra_args keys', sorted(_extra_args(y3_info)), sorted(_extra_args(new_info)))

    y3_extra, new_extra = _extra_args(y3_info), _extra_args(new_info)
    extra_value_mismatches = {
        key: (y3_extra[key], new_extra[key])
        for key in sorted(set(y3_extra) & set(new_extra))
        if y3_extra[key] != new_extra[key]
    }
    if extra_value_mismatches:
        print(f'  theory extra_args values: DIFFER {extra_value_mismatches}')
    else:
        print('  theory extra_args values: OK')

    # Report bookkeeping count differences without treating them as failures.
    print(f'  param counts: Y3={len(y3_info.get("params", {}))} new={len(new_info.get("params", {}))}')
    print(f'  fixed counts: Y3={len(_fixed_params(y3_info))} new={len(_fixed_params(new_info))}')


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--y3-repo', default=str(DEFAULT_Y3_REPO))
    parser.add_argument('--model', default='base')
    parser.add_argument('--sampler', default='evaluate')
    parser.add_argument('--case', action='append', dest='cases', help='Comma-separated likelihood list. Can be repeated.')
    ns = parser.parse_args(args=args)

    y3_get = _load_y3_get_cobaya_info(Path(ns.y3_repo))
    new_get = _load_new_get_cobaya_info()
    for case in ns.cases or DEFAULT_CASES:
        compare_case(case, y3_get, new_get, model=ns.model, sampler=ns.sampler)


if __name__ == '__main__':
    main()
