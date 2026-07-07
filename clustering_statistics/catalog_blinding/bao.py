"""BAO/AP catalog-level blinding adapters.

This module adapts ``compute_stats_from_options`` catalog/options conventions to
``desiblind.catalog_bao.CatalogBAOBlinder``. It deliberately does **not**
implement DESI catalog blinding itself; the physics and validation live in the
separate ``desiblind`` package.

Scope
-----

- BAO/AP redshift remapping only.
- In-memory use during ``compute_stats.py`` measurements.
- Saved-catalog workflows should call the data-transform helpers here, then use
  :mod:`.lss_catalogs` to make random catalogs follow the shifted data.
- No RSD or fNL; those live in :mod:`.rsd` and :mod:`.fnl`.

This is separate from the existing statistic/data-vector blinding in
``clustering_statistics.tools.apply_blinding``. The two should not be stacked.
"""


FIDUCIAL_TRACER_BINS = {
    'BGS': [((0.1, 0.4), 'BGS1')],
    'LRG': [((0.4, 0.6), 'LRG1'), ((0.6, 0.8), 'LRG2'), ((0.8, 1.1), 'LRG3')],
    'LGE': [((0.4, 0.6), 'LRG1'), ((0.6, 0.8), 'LRG2'), ((0.8, 1.1), 'LRG3')],
    'ELG': [((0.8, 1.1), 'ELG1'), ((1.1, 1.6), 'ELG2')],
    'QSO': [((0.8, 2.1), 'QSO1')],
}


def _get_desiblind_bao_blinder():
    try:
        from desiblind.catalog_bao import CatalogBAOBlinder
    except ImportError as exc:
        raise ImportError(
            'catalog_bao_blinding requires desiblind. Install desiblind or add its checkout to PYTHONPATH.'
        ) from exc
    return CatalogBAOBlinder


def normalize_parameters(parameters):
    """Return explicit BAO/AP parameters as ``{'w0': ..., 'wa': ...}``."""
    parameters = dict(parameters or {})
    missing = [name for name in ['w0', 'wa'] if parameters.get(name) is None]
    if missing:
        raise ValueError(f'BAO/AP catalog blinding requires parameters: {missing}')
    return {'w0': float(parameters['w0']), 'wa': float(parameters['wa'])}


def apply_blinding(tracer_name, catalog, *, parameters, input_zcol='Z', output_zcol='Z', copy=True):
    """Apply ``desiblind.catalog_bao.CatalogBAOBlinder`` to one catalog."""
    blinder = _get_desiblind_bao_blinder()
    return blinder.apply_blinding(
        tracer_name, catalog, parameters=normalize_parameters(parameters),
        input_zcol=input_zcol, output_zcol=output_zcol, copy=copy,
    )


def _simple_tracer_name(tracer):
    text = str(tracer)
    if 'BGS' in text:
        return 'BGS'
    if 'LRG' in text:
        return 'LRG'
    if 'LGE' in text:
        return 'LGE'
    if 'ELG' in text:
        return 'ELG'
    if 'QSO' in text:
        return 'QSO'
    raise ValueError(f'Cannot infer simple tracer name from {tracer!r}')


def infer_tracerbin_name(tracer, zrange=None):
    """Infer canonical desiblind tracer-bin name from a tracer and zrange.

    This helper covers the standard DR1/Y1 fiducial bins used by DESI LSS and
    desi-clustering. Users can always pass ``name``/``tracerbin`` explicitly for
    nonstandard bins.
    """
    simple = _simple_tracer_name(tracer)
    bins = FIDUCIAL_TRACER_BINS[simple]
    if zrange is None:
        if len(bins) == 1:
            return bins[0][1]
        raise ValueError(f'Cannot infer unique tracer-bin name for {tracer!r} without zrange')
    zrange = tuple(map(float, zrange))
    for candidate_zrange, name in bins:
        if all(abs(a - b) < 1e-6 for a, b in zip(zrange, candidate_zrange)):
            return name
    raise ValueError(f'Cannot infer tracer-bin name for tracer={tracer!r}, zrange={zrange!r}; pass name explicitly')


def _normalize_bool(value):
    if isinstance(value, str):
        return value.strip().lower() not in {'0', 'false', 'no', 'n', 'off'}
    return bool(value)


def _resolve_parameter_source(options):
    source = options.get('parameter_source', options.get('source', None))
    if source is None:
        if 'parameters' in options or 'w0' in options or 'wa' in options or 'specified_w0' in options or 'specified_wa' in options:
            source = 'explicit'
        elif any(key in options for key in ['lss_parameters_fn', 'lss_w0wa_bank', 'lss_parameter_index', 'parameter_index', 'lss_filerow', 'filerow']):
            source = 'lss'
        elif any(key in options for key in ['parameters_fn', 'parameter_bank', 'save_dir', 'bid']):
            source = 'desiblind'
        else:
            source = 'explicit'
    return str(source).lower().replace('-', '_')


def resolve_options(options, tracer=None, zrange=None):
    """Resolve desiblind BAO/AP catalog-blinding options.

    Supported parameter sources are:

    ``explicit``
        Direct ``parameters={'w0': ..., 'wa': ...}`` or top-level ``w0``/``wa``.
        Useful for open demos.
    ``desiblind``
        Preferred closed-production style. Load from a desiblind hashed parameter
        bank with ``parameters_fn`` or ``save_dir``, plus ``name``/``tracerbin``
        and optional ``bid``.
    ``lss``
        Historical compatibility with the LSS plain-text 1000-row bank using
        ``lss_parameters_fn`` plus ``lss_parameter_index`` or ``lss_filerow``.
    """
    if not options:
        return None
    if options is True:
        raise ValueError('catalog_bao_blinding must be a dictionary with desiblind BAO/AP parameters')
    options = dict(options)

    mode = options.get('mode', options.get('modes', 'bao'))
    if isinstance(mode, str):
        modes = [mode]
    else:
        modes = list(mode)
    aliases = {'ap': 'bao', 'bao_ap': 'bao'}
    modes = tuple(aliases.get(str(mode).lower(), str(mode).lower()) for mode in modes)
    if modes != ('bao',):
        raise ValueError(
            f'catalog_bao_blinding only supports desiblind BAO/AP catalog blinding; got modes={modes}. '
            'Do not use this BAO/AP on-the-fly path for RSD/fNL or statistic/data-vector blinding.'
        )

    blinder = _get_desiblind_bao_blinder()
    source = _resolve_parameter_source(options)
    metadata = options.get('metadata', 'open')
    validate_alpha_shift = _normalize_bool(options.get('validate_alpha_shift', True))
    alpha_zrange = options.get('alpha_zrange', options.get('validation_zrange', (0.4, 2.1)))
    max_alpha_shift = float(options.get('max_alpha_shift', 0.03))
    alpha_nz = int(options.get('alpha_nz', 100))

    name = options.get('name', options.get('tracerbin', options.get('observable', None)))
    if name is None and source in {'desiblind', 'hash', 'hashed', 'parameter_bank'}:
        name = infer_tracerbin_name(tracer, zrange=zrange)

    explicit_parameters = dict(options.get('parameters', {}))
    w0 = options.get('w0', options.get('specified_w0', explicit_parameters.get('w0', None)))
    wa = options.get('wa', options.get('specified_wa', explicit_parameters.get('wa', None)))
    if w0 is not None:
        explicit_parameters['w0'] = w0
    if wa is not None:
        explicit_parameters['wa'] = wa

    parameters, parameter_metadata = blinder.load_parameters(
        name=name,
        parameters=explicit_parameters if explicit_parameters else None,
        parameters_fn=options.get('parameters_fn', options.get('parameter_bank', None)),
        save_dir=options.get('save_dir', None),
        bid=options.get('bid', None),
        source=source,
        lss_parameters_fn=options.get('lss_parameters_fn', options.get('lss_w0wa_bank', None)),
        lss_parameter_index=options.get('lss_parameter_index', options.get('parameter_index', None)),
        lss_filerow=options.get('lss_filerow', options.get('filerow', None)),
        validate_alpha_shift=validate_alpha_shift,
        alpha_zrange=alpha_zrange,
        max_alpha_shift=max_alpha_shift,
        alpha_nz=alpha_nz,
    )

    from .lss_catalogs import DEFAULT_REDSHIFT_COLUMNS, DEFAULT_SPLIT_COLUMNS

    return {
        'mode': 'bao',
        'parameters': parameters,
        'parameter_metadata': parameter_metadata,
        'input_zcol': options.get('input_zcol', options.get('bao_input_zcol', 'Z')),
        'output_zcol': options.get('output_zcol', options.get('bao_output_zcol', 'Z')),
        'metadata': metadata,
        'output_version_suffix': options.get('output_version_suffix', 'desiblind-bao-blinded'),
        'random_seed': int(options.get('random_seed', 0)),
        'random_resample_columns': tuple(options.get('random_resample_columns', DEFAULT_REDSHIFT_COLUMNS)),
        'random_split_columns': tuple(options.get('random_split_columns', DEFAULT_SPLIT_COLUMNS)),
        'random_compmd': options.get('random_compmd', options.get('compmd', 'ran')),
        'apply_nz_reweight': _normalize_bool(options.get('apply_nz_reweight', True)),
        'nz_zmin': options.get('nz_zmin', None),
        'nz_zmax': options.get('nz_zmax', None),
        'nz_dz': float(options.get('nz_dz', 0.01)),
        'nbar_dz': float(options.get('nbar_dz', options.get('nz_dz', 0.01))),
        'randens': float(options.get('randens', 2500.)),
        'save_catalog_dir': options.get('save_catalog_dir', None),
        'save_catalog_prefix': options.get('save_catalog_prefix', None),
        'save_randoms': options.get('save_randoms', 'diagnostic'),
        'save_random_index': int(options.get('save_random_index', 0)),
    }


def output_version(version, options):
    """Return an output-only version name for desiblind-blinded measurements."""
    if not options:
        return version
    suffix = options.get('output_version_suffix', 'desiblind-bao-blinded')
    if suffix is False:
        return version
    suffix = str(suffix).strip('-')
    if not suffix:
        return version
    if version is None:
        return suffix
    if suffix in str(version):
        return version
    return f'{version}-{suffix}'


def attrs(options):
    """Return metadata attrs describing the desiblind BAO/AP on-the-fly path."""
    if not options:
        return {}
    parameter_metadata = dict(options.get('parameter_metadata', {}))
    out = {
        'catalog_bao_blinding': 'desiblind.CatalogBAOBlinder',
        'catalog_bao_blinding_mode': 'bao',
        'catalog_bao_blinding_input_zcol': options['input_zcol'],
        'catalog_bao_blinding_output_zcol': options['output_zcol'],
        'catalog_bao_blinding_metadata': options.get('metadata', 'open'),
        'catalog_bao_blinding_parameter_source': parameter_metadata.get('parameter_source', 'unknown'),
    }
    if 'max_alpha_shift' in parameter_metadata:
        out['catalog_bao_blinding_max_alpha_shift'] = parameter_metadata['max_alpha_shift']
        out['catalog_bao_blinding_alpha_shift_validated'] = True
    elif parameter_metadata.get('alpha_validation') is False:
        out['catalog_bao_blinding_alpha_shift_validated'] = False
    if options.get('metadata', 'open') == 'open':
        out['catalog_bao_blinding_w0'] = options['parameters']['w0']
        out['catalog_bao_blinding_wa'] = options['parameters']['wa']
        for key in ['max_abs_alpha_parallel_minus_one', 'max_abs_alpha_perp_minus_one']:
            if key in parameter_metadata:
                out[f'catalog_bao_blinding_{key}'] = parameter_metadata[key]
        for key in ['name', 'bid', 'index', 'filerow', 'parameters_fn', 'save_dir']:
            if parameter_metadata.get(key, None) is not None:
                out[f'catalog_bao_blinding_{key}'] = parameter_metadata[key]
    return out


def apply_to_catalog(catalog, options):
    """Apply ``desiblind.catalog_bao.CatalogBAOBlinder`` to one catalog."""
    if not options:
        return catalog
    blinder = _get_desiblind_bao_blinder()
    blinded = blinder.apply_to_catalog(
        catalog,
        parameters=options['parameters'],
        input_zcol=options['input_zcol'],
        output_zcol=options['output_zcol'],
        copy=True,
    )
    if hasattr(blinded, 'attrs'):
        blinded.attrs.update(attrs(options))
    return blinded


def apply_to_catalogs(catalogs, options):
    """Apply desiblind BAO/AP catalog blinding to a catalog or list of catalogs."""
    if not options:
        return catalogs
    if isinstance(catalogs, (list, tuple)):
        return [apply_to_catalog(catalog, options) for catalog in catalogs]
    return apply_to_catalog(catalogs, options)
