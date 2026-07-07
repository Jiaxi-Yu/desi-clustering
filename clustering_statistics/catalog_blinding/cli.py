"""Command-line interface for preparing saved blinded catalog trees.

This backs the ``clustering-catalog-blinding`` console script. It orchestrates
LSS-like catalog handling while delegating BAO/RSD transforms to ``desiblind``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import bao, rsd
from .diagnostics import write_diagnostic_plots
from .lss_catalogs import (
    DEFAULT_REDSHIFT_COLUMNS,
    add_nbar_fkp,
    apply_bao_nz_reweight,
    fiducial_fkp_p0,
    fiducial_nbar_zrange,
    read_fits_catalog,
    resample_randoms_from_data,
    set_lss_pre_addnbar_weight,
    split_columns_for_region,
    write_fits_catalog,
)

SUPPORTED_MODES = ('bao', 'rsd')
FUTURE_MODES = ('fnl',)


def normalize_modes(modes):
    """Return normalized catalog-level blinding modes in LSS pipeline order."""
    if isinstance(modes, str):
        modes = modes.replace(',', ' ').split()
    aliases = {'ap': 'bao', 'bao_ap': 'bao'}
    normalized = []
    for mode in modes:
        value = aliases.get(str(mode).lower(), str(mode).lower())
        if value in FUTURE_MODES:
            raise NotImplementedError(
                'Catalog-level fNL blinding is not implemented yet. Add the desiblind fNL blinder first, '
                'then wire it through clustering_statistics.catalog_blinding.fnl and this saved-catalog driver.'
            )
        if value not in SUPPORTED_MODES:
            raise ValueError(f'Unsupported catalog blinding mode {mode!r}; supported modes are {SUPPORTED_MODES}')
        if value not in normalized:
            normalized.append(value)
    return tuple(mode for mode in SUPPORTED_MODES if mode in normalized)


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Prepare saved catalog-level blinded catalogs using desiblind transforms and LSS-like random matching.'
    )
    parser.add_argument('--input-catalog', required=True, help='Input observed data catalog FITS file.')
    parser.add_argument('--output-catalog', required=True, help='Output final blinded data catalog FITS file.')
    parser.add_argument('--random-catalog', default=None,
                        help='Input random catalog FITS file. Required for computed RSD reconstruction and for saving matched randoms. For multiple randoms, prefer --random-catalog-template.')
    parser.add_argument('--random-catalog-template', default=None,
                        help='Template for LSS-style input random FITS files, formatted with the random index. Supports {index}, {}, or %%d.')
    parser.add_argument('--nran', type=int, default=None,
                        help='Number of random catalogs used with --random-catalog-template and output templates.')
    parser.add_argument('--output-random-catalog', default=None,
                        help='Optional output final random catalog matched to the final blinded data. For multiple randoms, prefer --output-random-catalog-template.')
    parser.add_argument('--output-random-catalog-template', default=None,
                        help='Template for LSS-style output final random FITS files, formatted with the random index. Supports {index}, {}, or %%d.')
    parser.add_argument('--save-reconstruction-random-catalog', default=None,
                        help='Optional output random catalog matched to the post-BAO/pre-RSD data and used for reconstruction. For multiple randoms, prefer --save-reconstruction-random-catalog-template.')
    parser.add_argument('--save-reconstruction-random-catalog-template', default=None,
                        help='Template for post-BAO/pre-RSD reconstruction random outputs, formatted with the random index. Supports {index}, {}, or %%d.')
    parser.add_argument('--realspace-catalog', default=None,
                        help='Precomputed reconstructed-realspace FITS file for RSD mode.')
    parser.add_argument('--run-pyrecon', action='store_true',
                        help='For RSD mode, compute the reconstructed-realspace catalog with direct pyrecon, without importing LSS.')
    parser.add_argument('--run-jaxrecon', action='store_true',
                        help='For RSD mode, compute the reconstructed-realspace catalog with the JAX-native desi-clustering reconstruction path.')
    parser.add_argument('--save-realspace-catalog', default=None,
                        help='Optional FITS output for the computed reconstructed-realspace catalog.')
    parser.add_argument('--fits-ext', default='LSS')
    parser.add_argument('--modes', nargs='+', default=['bao'],
                        help='Catalog-level blinding modes. Supported modes are bao/ap and rsd. If both are supplied, BAO/AP is applied before RSD.')
    parser.add_argument('--tracer-name', default='LRG3', help='Canonical desiblind tracer/bin name.')
    parser.add_argument('--region', default=None,
                        help='Sky region for LSS-like random resampling. Use NGC to apply the LSS DEC=32.375 split.')
    parser.add_argument('--input-zcol', default='Z')
    parser.add_argument('--output-zcol', default='Z')
    parser.add_argument('--realspace-zcol', default='Z')
    parser.add_argument('--w0', type=float, default=None)
    parser.add_argument('--wa', type=float, default=None)
    parser.add_argument('--zeff', type=float, default=None)
    parser.add_argument('--bias', type=float, default=None)
    parser.add_argument('--fiducial-f', type=float, default=0.8)
    parser.add_argument('--random-seed', type=int, default=0,
                        help='Seed used for LSS-like random redshift-column resampling.')
    parser.add_argument('--random-resample-columns', nargs='*', default=list(DEFAULT_REDSHIFT_COLUMNS),
                        help='Columns sampled from data into randoms, following LSS mkclusran/clusran_resamp.')
    parser.add_argument('--random-split-columns', nargs='*', default=['PHOTSYS'],
                        help='Columns used to split data/random resampling, e.g. PHOTSYS.')
    parser.add_argument('--skip-bao-nz-reweight', action='store_true',
                        help='Do not apply the internal BAO n(z)_in/n(z)_out correction when rebuilding WEIGHT.')
    parser.add_argument('--skip-final-random-resample', action='store_true',
                        help='Do not resample final randoms from the final blinded data.')
    parser.add_argument('--skip-final-nbar', action='store_true',
                        help='Do not recompute simple NZ/NX/WEIGHT/WEIGHT_FKP columns on final data/randoms.')
    parser.add_argument('--nz-zmin', type=float, default=None)
    parser.add_argument('--nz-zmax', type=float, default=None)
    parser.add_argument('--nz-dz', type=float, default=0.01)
    parser.add_argument('--p0', type=float, default=None,
                        help='FKP P0. Defaults to the LSS tracer value when omitted.')
    parser.add_argument('--randens', type=float, default=2500.,
                        help='Parent imaging random density per square degree used for LSS-like nbar area.')
    parser.add_argument('--compmd', choices=['ran', 'dat'], default='ran')
    parser.add_argument('--recon-bias', type=float, default=None,
                        help='Bias used for computed reconstruction. Defaults to --bias.')
    parser.add_argument('--recon-method', choices=['iterative_fft', 'multigrid'], default='iterative_fft',
                        help='Direct-pyrecon reconstruction method.')
    parser.add_argument('--recon-smoothing-radius', type=float, default=15.)
    parser.add_argument('--recon-threshold-randoms', type=float, default=0.01)
    parser.add_argument('--recon-threshold-randoms-method', choices=['mean', 'noise'], default='mean',
                        help='Random-density threshold convention. pyrecon scalar thresholds use mean; jaxrecon scalar default is noise.')
    parser.add_argument('--recon-growth-rate', type=float, default=None,
                        help='Growth rate used by computed reconstruction. Defaults to --fiducial-f for RSD.')
    parser.add_argument('--recon-cellsize', type=float, default=None)
    parser.add_argument('--recon-meshsize', type=int, default=None)
    parser.add_argument('--recon-boxsize', type=float, default=None)
    parser.add_argument('--recon-boxcenter', nargs=3, type=float, default=None,
                        help='Cartesian reconstruction box center (x y z). Useful for exact pyrecon/jaxrecon mesh parity tests.')
    parser.add_argument('--recon-nthreads', type=int, default=64,
                        help='Number of FFTW threads for --run-pyrecon.')
    parser.add_argument('--recon-weight-col', default='WEIGHT')
    parser.add_argument('--fgrowth-blind', type=float, default=None,
                        help='Optional externally prepared RSD fgrowth_blind override.')
    parser.add_argument('--max-df-fraction', type=float, default=0.1)
    parser.add_argument('--diagnostic-plot-dir', default=None,
                        help='Optional directory for real-run diagnostic plots comparing data/random redshift and weight distributions at each blinding step.')
    parser.add_argument('--diagnostic-plot-prefix', default='catalog_blinding',
                        help='Filename prefix for --diagnostic-plot-dir outputs.')
    parser.add_argument('--summary-file', default=None)
    parser.add_argument('--clobber', action='store_true')
    return parser.parse_args(args=args)


def _format_indexed_path(template, index):
    """Format an LSS-style indexed path template with a random index."""
    template = str(template)
    if '{index}' in template:
        return template.format(index=index)
    if '{:d}' in template or '{}' in template:
        return template.format(index)
    if '%d' in template:
        return template % index
    raise ValueError(
        f'Random catalog template {template!r} must contain one of {{index}}, {{}}, {{:d}}, or %d'
    )


def _indexed_paths(template, nran):
    if template is None:
        return None
    if nran is None:
        raise ValueError('--nran is required when using random catalog templates')
    if int(nran) <= 0:
        raise ValueError('--nran must be positive when using random catalog templates')
    return [_format_indexed_path(template, iran) for iran in range(int(nran))]


def _as_random_list(randoms):
    if randoms is None:
        return []
    return list(randoms) if isinstance(randoms, (list, tuple)) else [randoms]


def _first_random(randoms):
    random_list = _as_random_list(randoms)
    return random_list[0] if random_list else None


def _random_row_summary(randoms):
    random_list = _as_random_list(randoms)
    if not random_list:
        return None, None
    rows = [int(len(random)) for random in random_list]
    return (rows[0] if len(rows) == 1 else rows), int(sum(rows))


def _maybe_read_random(args):
    template = getattr(args, 'random_catalog_template', None)
    if template is not None:
        return [read_fits_catalog(filename, ext=args.fits_ext)
                for filename in _indexed_paths(template, getattr(args, 'nran', None))]
    if args.random_catalog is None:
        return None
    return read_fits_catalog(args.random_catalog, ext=args.fits_ext)


def _random_input_paths(args):
    template = getattr(args, 'random_catalog_template', None)
    if template is not None:
        return [str(Path(filename).expanduser().resolve(strict=True))
                for filename in _indexed_paths(template, getattr(args, 'nran', None))]
    return None if args.random_catalog is None else str(Path(args.random_catalog).expanduser().resolve(strict=True))


def _write_random_catalogs(randoms, *, output_catalog=None, output_template=None, fits_ext='LSS', clobber=False):
    random_list = _as_random_list(randoms)
    if output_template is not None:
        paths = _indexed_paths(output_template, len(random_list))
        if len(paths) != len(random_list):  # pragma: no cover - defensive
            raise RuntimeError('internal error formatting output random template')
        return [str(write_fits_catalog(random, path, ext=fits_ext, clobber=clobber))
                for random, path in zip(random_list, paths)]
    if output_catalog is not None:
        if len(random_list) != 1:
            raise ValueError('Use an output random template when writing multiple random catalogs')
        return str(write_fits_catalog(random_list[0], output_catalog, ext=fits_ext, clobber=clobber))
    return None


def _resolved_nbar_zrange(args):
    tracer_zrange = fiducial_nbar_zrange(getattr(args, 'tracer_name', None), zrange=None)
    zmin = args.nz_zmin if args.nz_zmin is not None else (tracer_zrange[0] if tracer_zrange is not None else None)
    zmax = args.nz_zmax if args.nz_zmax is not None else (tracer_zrange[1] if tracer_zrange is not None else None)
    return zmin, zmax


def _resolved_p0(args):
    if getattr(args, 'p0', None) is not None:
        return float(args.p0)
    return fiducial_fkp_p0(getattr(args, 'tracer_name', None))


def _resolved_random_split_columns(args):
    return split_columns_for_region(
        getattr(args, 'region', None), tracer=getattr(args, 'tracer_name', None),
        split_columns=getattr(args, 'random_split_columns', None),
    )


def run_from_args(args):
    modes = normalize_modes(args.modes)
    input_data = read_fits_catalog(args.input_catalog, ext=args.fits_ext)
    random = _maybe_read_random(args)
    bao_parameters = {'w0': args.w0, 'wa': args.wa}
    data = input_data.copy()
    reconstruction_data = None
    reconstruction_random = None
    final_random = None
    applied = []
    realspace = None
    realspace_source = None
    jax_boxcenter = None
    bao_nz_reweight = None
    post_bao_data = None
    nbar_zmin, nbar_zmax = _resolved_nbar_zrange(args)
    p0 = _resolved_p0(args)
    random_split_columns = _resolved_random_split_columns(args)

    if 'bao' in modes:
        data = bao.apply_blinding(
            args.tracer_name, data, parameters=bao_parameters,
            input_zcol=args.input_zcol, output_zcol=args.output_zcol, copy=True,
        )
        params = bao.normalize_parameters(bao_parameters)
        applied.append({'mode': 'bao', 'parameters': params})
        if not args.skip_bao_nz_reweight:
            data, bao_nz_reweight = apply_bao_nz_reweight(
                input_data, data, zcol_before=args.input_zcol, zcol_after=args.output_zcol,
                zmin=nbar_zmin, zmax=nbar_zmax, dz=args.nz_dz, copy=False,
            )
        # Match LSS mkclusdat state before mkclusran/reconstruction: reset the
        # total WEIGHT from component weights and the internal BAO n(z) factor,
        # before final addnbar divides by per-NTILE completeness and recomputes FKP.
        bao_nz_extra_weight = None if bao_nz_reweight is None else bao_nz_reweight.get('correction')
        data = set_lss_pre_addnbar_weight(data, extra_weight=bao_nz_extra_weight, copy=False)
        post_bao_data = data.copy()
        if random is not None:
            reconstruction_random = resample_randoms_from_data(
                random, data, columns=args.random_resample_columns,
                split_columns=random_split_columns, seed=args.random_seed,
                compmd=args.compmd, copy=True,
            )
        reconstruction_data = data.copy()
    else:
        data = set_lss_pre_addnbar_weight(data, copy=False)
        reconstruction_data = data.copy()
        post_bao_data = None
        if random is not None and 'rsd' in modes:
            reconstruction_random = resample_randoms_from_data(
                random, data, columns=args.random_resample_columns,
                split_columns=random_split_columns, seed=args.random_seed,
                compmd=args.compmd, copy=True,
            )
        else:
            reconstruction_random = random.copy() if random is not None and hasattr(random, 'copy') else random

    if 'rsd' in modes:
        computed_backends = [name for name, enabled in [('pyrecon', args.run_pyrecon), ('jaxrecon', args.run_jaxrecon)] if enabled]
        if len(computed_backends) > 1:
            raise ValueError('Use only one reconstruction backend: --run-pyrecon or --run-jaxrecon')
        if computed_backends and args.realspace_catalog is not None:
            raise ValueError('Use either a computed reconstruction backend or --realspace-catalog for RSD, not both')
        if computed_backends:
            backend = computed_backends[0]
            if random is None:
                raise ValueError('--random-catalog is required with --run-pyrecon or --run-jaxrecon')
            if reconstruction_random is None:
                reconstruction_random = random
            recon_bias = args.recon_bias if args.recon_bias is not None else args.bias
            if recon_bias is None:
                raise ValueError('computed RSD reconstruction requires --recon-bias or --bias')
            recon_growth_rate = args.recon_growth_rate if args.recon_growth_rate is not None else args.fiducial_f
            recon_threshold_randoms = args.recon_threshold_randoms if args.recon_threshold_randoms_method == 'mean' else ('noise', args.recon_threshold_randoms)
            jax_threshold_randoms = (args.recon_threshold_randoms_method, args.recon_threshold_randoms)
            if backend == 'jaxrecon':
                mattrs = {}
                jax_cellsize = None if (args.recon_meshsize is not None and args.recon_boxsize is not None) else args.recon_cellsize
                if jax_cellsize is not None:
                    mattrs['cellsize'] = jax_cellsize
                if args.recon_meshsize is not None:
                    mattrs['meshsize'] = args.recon_meshsize
                if args.recon_boxsize is not None:
                    mattrs['boxsize'] = args.recon_boxsize
                jax_boxcenter = args.recon_boxcenter
                if jax_boxcenter is None:
                    jax_boxcenter = rsd.compute_data_boxcenter(reconstruction_data, weight_col=args.recon_weight_col)
                mattrs['boxcenter'] = jax_boxcenter
                realspace = rsd.compute_jaxrecon_realspace_catalog(
                    reconstruction_data, reconstruction_random,
                    bias=recon_bias, smoothing_radius=args.recon_smoothing_radius,
                    growth_rate=recon_growth_rate,
                    threshold_randoms=jax_threshold_randoms,
                    mattrs=mattrs or None, weight_col=args.recon_weight_col,
                    zcol=args.realspace_zcol,
                )
            else:
                realspace = rsd.compute_pyrecon_realspace_catalog(
                    reconstruction_data, reconstruction_random,
                    bias=recon_bias, smoothing_radius=args.recon_smoothing_radius,
                    growth_rate=recon_growth_rate,
                    boxsize=args.recon_boxsize,
                    boxcenter=args.recon_boxcenter,
                    nmesh=args.recon_meshsize,
                    cellsize=args.recon_cellsize if args.recon_cellsize is not None else 7.,
                    threshold_randoms=recon_threshold_randoms,
                    nthreads=args.recon_nthreads,
                    method=args.recon_method,
                    weight_col=args.recon_weight_col,
                    zcol=args.realspace_zcol,
                )
            realspace_source = backend
            if args.save_realspace_catalog is not None:
                write_fits_catalog(realspace, args.save_realspace_catalog, ext=args.fits_ext, clobber=args.clobber)
            _write_random_catalogs(
                reconstruction_random,
                output_catalog=args.save_reconstruction_random_catalog,
                output_template=getattr(args, 'save_reconstruction_random_catalog_template', None),
                fits_ext=args.fits_ext,
                clobber=args.clobber,
            )
        else:
            if args.realspace_catalog is None:
                raise ValueError('--realspace-catalog, --run-pyrecon, or --run-jaxrecon is required when --modes includes rsd')
            realspace = read_fits_catalog(args.realspace_catalog, ext=args.fits_ext)
            realspace_source = 'file'
        if len(realspace) != len(data):
            raise ValueError(f'RSD realspace catalog row count differs: {len(realspace)} != {len(data)}')
        rsd_parameters = {
            'w0': args.w0,
            'wa': args.wa,
            'zeff': args.zeff,
            'bias': args.bias,
            'fiducial_f': args.fiducial_f,
            'fgrowth_blind': args.fgrowth_blind,
            'max_df_fraction': args.max_df_fraction,
        }
        data, normalized_rsd = rsd.apply_blinding(
            args.tracer_name, data, realspace, parameters=rsd_parameters,
            zcol=args.output_zcol, realspace_zcol=args.realspace_zcol,
            output_zcol=args.output_zcol, copy=False,
        )
        applied.append({'mode': 'rsd', 'parameters': normalized_rsd})

    if random is not None:
        if 'rsd' in modes and not args.skip_final_random_resample:
            final_random = resample_randoms_from_data(
                random, data, columns=args.random_resample_columns,
                split_columns=random_split_columns, seed=args.random_seed,
                compmd=args.compmd, copy=True,
            )
        elif reconstruction_random is not None:
            final_random = reconstruction_random.copy() if hasattr(reconstruction_random, 'copy') else reconstruction_random
        else:
            final_random = random.copy() if hasattr(random, 'copy') else random

    if not args.skip_final_nbar:
        data, final_random, final_nbar = add_nbar_fkp(
            data, final_random, zcol=args.output_zcol, zmin=nbar_zmin,
            zmax=nbar_zmax, dz=args.nz_dz, p0=p0,
            compmd=args.compmd, randens=getattr(args, 'randens', 2500.),
            data_extra_weight=(None if bao_nz_reweight is None else bao_nz_reweight.get('correction')),
            copy=False,
        )
    else:
        final_nbar = None

    diagnostic_summary = None
    if getattr(args, 'diagnostic_plot_dir', None) is not None:
        diagnostic_summary = write_diagnostic_plots(
            args.diagnostic_plot_dir,
            input_data=input_data,
            bao_ap_blinded_data=post_bao_data,
            final_data=data,
            input_random=_first_random(random),
            reconstruction_random=_first_random(reconstruction_random) if 'bao' in modes and random is not None else None,
            final_random=_first_random(final_random),
            modes=modes,
            zcol=args.output_zcol,
            zmin=args.nz_zmin,
            zmax=args.nz_zmax,
            dz=args.nz_dz,
            prefix=getattr(args, 'diagnostic_plot_prefix', 'catalog_blinding'),
            bao_nz_reweight=bao_nz_reweight,
        )

    output = write_fits_catalog(data, args.output_catalog, ext=args.fits_ext, clobber=args.clobber)
    output_random = None
    if args.output_random_catalog is not None or getattr(args, 'output_random_catalog_template', None) is not None:
        if final_random is None:
            raise ValueError('Output random catalogs require --random-catalog or --random-catalog-template')
        output_random = _write_random_catalogs(
            final_random,
            output_catalog=args.output_random_catalog,
            output_template=getattr(args, 'output_random_catalog_template', None),
            fits_ext=args.fits_ext,
            clobber=args.clobber,
        )

    random_rows, random_rows_total = _random_row_summary(final_random)

    summary = {
        'input_catalog': str(Path(args.input_catalog).expanduser().resolve(strict=True)),
        'output_catalog': str(output),
        'random_catalog': _random_input_paths(args),
        'output_random_catalog': output_random,
        'realspace_catalog': None if args.realspace_catalog is None else str(Path(args.realspace_catalog).expanduser().resolve(strict=True)),
        'realspace_source': realspace_source,
        'saved_realspace_catalog': None if args.save_realspace_catalog is None else str(Path(args.save_realspace_catalog).expanduser().resolve(strict=False)),
        'saved_reconstruction_random_catalog': (
            None if args.save_reconstruction_random_catalog is None and getattr(args, 'save_reconstruction_random_catalog_template', None) is None
            else ([_format_indexed_path(args.save_reconstruction_random_catalog_template, iran) for iran in range(len(_as_random_list(reconstruction_random)))]
                  if getattr(args, 'save_reconstruction_random_catalog_template', None) is not None
                  else str(Path(args.save_reconstruction_random_catalog).expanduser().resolve(strict=False)))
        ),
        'modes': list(modes),
        'tracer_name': args.tracer_name,
        'input_zcol': args.input_zcol,
        'output_zcol': args.output_zcol,
        'realspace_zcol': args.realspace_zcol,
        'rows': int(len(data)),
        'random_rows': random_rows,
        'random_rows_total': random_rows_total,
        'nran': len(_as_random_list(final_random)) if final_random is not None else len(_as_random_list(random)),
        'applied': applied,
        'lss_like_catalogs': {
            'bao_nz_reweight': bool(bao_nz_reweight is not None),
            'reconstruction_random_resampled': bool(random is not None and ('bao' in modes or 'rsd' in modes)),
            'final_random_resampled': bool(random is not None and 'rsd' in modes and not args.skip_final_random_resample),
            'final_nbar_fkp': bool(final_nbar is not None),
            'random_resample_columns': list(args.random_resample_columns),
            'random_split_columns': list(random_split_columns),
            'region': getattr(args, 'region', None),
            'random_seed': args.random_seed,
            'nbar_zmin': nbar_zmin,
            'nbar_zmax': nbar_zmax,
            'p0': p0,
            'randens': getattr(args, 'randens', 2500.),
        },
        'diagnostics': diagnostic_summary,
        'reconstruction': None if realspace_source not in ['pyrecon', 'jaxrecon'] else {
            'backend': realspace_source,
            'method': args.recon_method if realspace_source == 'pyrecon' else None,
            'bias': args.recon_bias if args.recon_bias is not None else args.bias,
            'smoothing_radius': args.recon_smoothing_radius,
            'growth_rate': args.recon_growth_rate if args.recon_growth_rate is not None else args.fiducial_f,
            'threshold_randoms': args.recon_threshold_randoms,
            'threshold_randoms_method': args.recon_threshold_randoms_method,
            'cellsize': (None if (realspace_source == 'jaxrecon' and args.recon_meshsize is not None and args.recon_boxsize is not None) else args.recon_cellsize) if realspace_source == 'jaxrecon' else (args.recon_cellsize if args.recon_cellsize is not None else 7.),
            'meshsize': args.recon_meshsize,
            'boxsize': args.recon_boxsize,
            'boxcenter': (jax_boxcenter if realspace_source == 'jaxrecon' else args.recon_boxcenter),
            'nthreads': args.recon_nthreads if realspace_source == 'pyrecon' else None,
            'weight_col': args.recon_weight_col,
        },
    }
    if args.summary_file is not None:
        summary_file = Path(args.summary_file).expanduser().resolve(strict=False)
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        summary_file.write_text(json.dumps(summary, indent=2, sort_keys=True))
        summary['summary_file'] = str(summary_file)
    return summary


def main(args=None):
    parsed = parse_args(args=args)
    summary = run_from_args(parsed)
    print('clustering_catalog_blinding=PASS')
    print(f"output_catalog={summary['output_catalog']}")
    if summary.get('output_random_catalog') is not None:
        print(f"output_random_catalog={summary['output_random_catalog']}")
    if summary.get('diagnostics') is not None:
        print(f"diagnostics={summary['diagnostics']['summary_file']}")
    if 'summary_file' in summary:
        print(f"summary={summary['summary_file']}")
    print(f"modes={','.join(summary['modes'])} rows={summary['rows']}")
    return summary


if __name__ == '__main__':
    main()
