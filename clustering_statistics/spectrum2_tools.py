"""
Fourier-space 2-point clustering measurements.

Main functions
--------------
* `prepare_jaxpower_particles`: Convert catalogs into mesh-ready particle inputs.
* `compute_mesh2_spectrum`: Main `P(k)` measurement backend.
* `compute_window_mesh2_spectrum`: Compute the power spectrum window matrix.
* `compute_window_mesh2_spectrum_fm`: Build forward-model window matrix.
* `compute_covariance_mesh2_spectrum`: Estimate Fourier-space covariance.
* `run_preliminary_fit_mesh2_spectrum`: Run preliminary fits used in covariance matrix.
"""
import time
import logging
import functools
import operator
from collections.abc import Callable

import numpy as np
import jax
from jax import numpy as jnp
from jax.sharding import PartitionSpec as P
import lsstypes as types

from .tools import default_mpicomm, _format_bitweights, compute_fkp_effective_redshift, combine_stats


logger = logging.getLogger('spectrum2')


@default_mpicomm
def prepare_jaxpower_particles(*get_data_randoms, mattrs=None, add_data=tuple(), add_randoms=tuple(), check=True, **kwargs):
    """
    Prepare :class:`jaxpower.ParticleField` objects from data and randoms catalogs.

    Parameters
    ----------
    get_data_randoms : callables
        Functions that return dict of 'data' (optionally 'randoms', 'shifted') catalogs.
        Each catalog must contain 'POSITION' and 'INDWEIGHT', and optionally 'BITWEIGHT' for bitwise weights and 'TARGETID'
        for randoms IDs to allow process-invariant random split in bispectrum normalization.
    mattrs : dict, optional
        Mesh attributes ('boxsize', 'meshsize' or 'cellsize', 'boxcenter') to define the :class:`ParticleField` objects. If ``None``, default attributes are used.
    kwargs : dict, optional
        Additional keyword arguments to pass to :class:`ParticleField`.

    Returns
    -------
    all_particles : list of dictionaries
        List of dictionaries of :class:`ParticleField`  'data' (optionally 'randoms', 'shifted') objects for each input catalog.
    """
    # Import mesh attribute computation and particle field creation from jaxpower
    from jaxpower.mesh import get_mesh_attrs, ParticleField
    # Use MPI backend for distributed particle processing across processes
    backend = 'mpi'
    # Extract MPI communicator from kwargs (added by @default_mpicomm decorator)
    mpicomm = kwargs['mpicomm']

    # Load all catalogs by calling the provided functions
    all_catalogs = [_get_data_randoms() for _get_data_randoms in get_data_randoms]

    # Define the mesh attributes; pass in positions only
    # check=True validates that all positions are within mesh bounds
    mattrs = get_mesh_attrs(*[catalog["POSITION"] for catalogs in all_catalogs for catalog in catalogs.values()], check=check, **(mattrs or {}))
    if jax.process_index() == 0:
        logger.info(f'Using mesh {mattrs}.')

    # Use IDS instead
    def collective_arange(local_size):
        # Compute global array indices across all MPI processes
        # This allows each process to know its global position in the distributed array
        sizes = mpicomm.allgather(local_size)
        return sum(sizes[:mpicomm.rank]) + np.arange(local_size)

    all_particles = []
    # Dictionary mapping 'data' and 'randoms' to their respective extra columns to load
    add = {'data': add_data, 'randoms': add_randoms}
    for catalogs in all_catalogs:
        particles = {}
        for name, catalog in catalogs.items():
            extra = {}
            # Start with individual weights from catalog
            indweights = catalog['INDWEIGHT']
            if name == 'data':
                # Extract and process bitwise weights (fiber weights, completeness, etc.)
                bitweights = None
                with_bitweights = 'BITWEIGHT' in catalog and 'BITWEIGHT' in add[name]
                if with_bitweights:
                    # WARNING: indweights is assumed not to contain completeness correction (this is in BITWEIGHT)
                    # Parse bitwise weight array into individual weight components
                    bitweights = _format_bitweights(catalog['BITWEIGHT'])
                    from cucount.jax import BitwiseWeight
                    # Compute individual inverse probability weight (IIP) from bitwise components
                    # p_correction_nbits=False: no impact on IIP computation
                    iip = BitwiseWeight(weights=bitweights, p_correction_nbits=False)(bitweights)
                    # Store original bitweights in extra
                    extra['BITWEIGHT'] = jnp.column_stack(bitweights)
                    extra['INDWEIGHT_NO_COMP'] = indweights
                    # Multiply individual weights by IIP to correct fiber assignment at large scales
                    indweights = indweights * iip
                # Add any additional columns (e.g., Z, WEIGHT_FKP) to extra dictionary
                for column in add[name]:
                    if column != 'BITWEIGHT': extra[column] = catalog[column]
            elif name == 'randoms':
                # Extract target IDs from random catalog for reproducible random splitting
                if 'TARGETID' in catalog and 'IDS' in add[name]:
                    extra['IDS'] = catalog['TARGETID']
                # Add other requested columns to extra dictionary
                for column in add[name]:
                    if column != 'IDS': extra[column] = catalog[column]
            # Create ParticleField object: positions + weights + mesh attributes
            # exchange=True: distribute particles across MPI processes by spatial location
            # This ensures load balancing across processes
            particle = ParticleField(catalog['POSITION'], indweights, attrs=mattrs, exchange=True, backend=backend, extra=extra, **kwargs)
            particles[name] = particle
        all_particles.append(particles)
    if jax.process_index() == 0:
        logger.info(f'All particles on the device')

    return all_particles


def _get_jaxpower_attrs(*all_particles):
    """Return summary attributes from :class:`jaxpower.ParticleField` objects: total weight and size."""
    # Get mesh attributes from first particle set (same for all)
    mattrs = next(iter(all_particles[0].values())).attrs
    # Creating FKP fields
    attrs = {}
    for particles in all_particles:
        for name in particles:
            if particles[name] is not None:
                # Store total weight sum for each particle type
                if f'wsum_{name}' not in attrs:
                    #attrs[f'size_{name}'] = [[]]  # size is process-dependent
                    attrs[f'wsum_{name}'] = [[]]
                #attrs[f'size_{name}'][0].append(particles[name].size)
                # Sum weights across all processes using MPI
                attrs[f'wsum_{name}'][0].append(particles[name].sum())
    # Extract and preserve mesh geometric information for output
    for name in ['boxsize', 'boxcenter', 'meshsize']:
        attrs[name] = mattrs[name]
    return attrs


def compute_mesh2_spectrum_close_pair_correction(*get_data_randoms, spectrum, auw=None, cut=None, los='firstpoint', optimal_weights=None, **kwargs):
    """
    Compute and apply close-pair corrections to 2-point spectrum.

    Parameters
    ----------
    get_data_randoms : callables
        Functions returning dicts with 'data', 'randoms' (optionally 'shifted').
        Catalogs must contain 'POSITION', 'INDWEIGHT', optionally 'BITWEIGHT'.
    spectrum : ObservableTree
        Input spectrum to add close pair correction to.
    auw : Mesh2SpectrumPoles, optional
        Angular upweights to apply.
    cut : bool, optional
        If provided, apply a theta-cut of (0, 0.05) degrees.
    los : {'local', 'firstpoint', 'x', 'y', 'z', array-like}, optional
        Line-of-sight definition. 'local' uses local LOS, 'firstpoint' uses the position of the first point in the pair,
        'x', 'y', 'z' use fixed axes, or provide a 3-vector.
    optimal_weights : callable or None, optional
        Function taking (ell, catalog) as input and returning total weights to apply to data and randoms.
        It can have an optional attribute 'columns' that specifies which additional columns are needed to compute the optimal weights.
        As a default, ``optimal_weights.columns = ['Z']`` to indicate that redshift information is needed.
        A dictionary ``catalog`` of columns is provided, containing 'INDWEIGHT' and the requested columns.
        If ``None``, no optimal weights are applied.

    Returns
    -------
    spectrum : Mesh2SpectrumPoles
    """
    from cucount.jax import create_sharding_mesh, WeightAttrs
    from jaxpower import MeshAttrs, BinMesh2SpectrumPoles
    from .correlation2_tools import prepare_cucount_particles

    columns_optimal_weights = get_optimal_weight_columns(optimal_weights)

    with create_sharding_mesh() as sharding_mesh:
        if callable(get_data_randoms[0]):
            all_particles = prepare_cucount_particles(*get_data_randoms, concatenate=True)
            if jax.process_index() == 0: logger.info('All particles on the device')

    ells = spectrum.ells
    edges = next(iter(spectrum)).edges('k')
    mattrs = MeshAttrs(**{name: spectrum.attrs[name] for name in ['boxsize', 'boxcenter', 'meshsize']})
    bin = BinMesh2SpectrumPoles(mattrs, edges=edges, ells=ells)
    corrections = {'auw': auw, 'cut': cut}

    def _compute_corrections(all_particles, ells, fields=None):
        results = {'raw': spectrum}

        for correction_name, correction_input in corrections.items():
            if correction_input is None:
                continue
            if correction_name == 'cut':
                wattrs = WeightAttrs()
                all_particles_cut = []
                for particles in all_particles:
                    data = particles['data']
                    data_weights = wattrs(data)
                    data = data.clone(weights=data_weights)
                    for name in ['shifted', 'randoms']:
                        if name in particles:
                            randoms = particles[name]
                            data_weights, randoms_weights = wattrs(data), wattrs(randoms)
                            data = data.concatenate([data, randoms.clone(weights=-data_weights.sum() / randoms_weights.sum() * randoms_weights)], local=True)
                            break
                    all_particles_cut.append(data)
                particles = all_particles_cut
            else:
                all_particles_auw = []
                for particles in all_particles:
                    particles = [particles['data'] for particles in all_particles]

            correction = _compute_mesh2_spectrum_close_pair_correction(particles, bin=bin, los=los, fields=fields, **{correction_name: correction_input})
            results[correction_name] = _apply_mesh2_spectrum_close_pair_correction(spectrum, correction)

        return results

    if optimal_weights is None:
        results = _compute_corrections(all_particles, ells=ells)

    else:
        fields = tuple(range(len(all_particles)))
        fields = fields + (fields[-1],) * max(0, 2 - len(fields))

        all_particles = tuple(all_particles)
        all_particles = all_particles + (all_particles[-1],) * max(0, len(fields) - len(all_particles))

        def compute(ell):
            result_ell = {}

            # Apply optimal weights to the data particles used by the close-pair correction.
            for weighted_particles in iter_optimal_weighted_particles(ell, all_particles, optimal_weights, columns_optimal_weights):
                _result = _compute_corrections(weighted_particles, ells=[ell], fields=fields)

                for key, value in _result.items():
                    result_ell.setdefault(key, [])
                    result_ell[key].append(value)

            # Sum symmetric optimal-weight realizations for this ell.
            return {key: combine_stats(value) for key, value in result_ell.items()}

        def join(results):
            # Join per-ell outputs into one spectrum-like object per key.
            for key in results:
                results[key] = types.join(results[key])
            return results

        results = loop_over_optimal_weights(ells, compute, join)

    if len(results) == 1:
        return next(iter(results.values()))
    return results


def _compute_mesh2_spectrum_close_pair_correction(all_particles, bin=None, auw=None, cut=None, los=None, fields=None):
    """Compute and apply close-pair corrections."""
    from jaxpower import BinParticle2CorrelationPoles, BinParticle2SpectrumPoles, compute_particle2, compute_particle2_shotnoise

    all_particles = list(all_particles)
    mattrs, edges, k, ells = bin.mattrs, bin.edges, bin.xavg, bin.ells
    results = {}
    # First compute the theta-cut (close-pair) contribution for contamination correction
    if cut is not None:
        # Define angular selection: only pairs separated by < 0.05 degrees
        sattrs = {'theta': (0., 0.05)}
        #pbin = BinParticle2SpectrumPoles(mattrs, edges=bin.edges, xavg=bin.xavg, sattrs=sattrs, ells=ells)
        # Use correlation binning for close pairs (finer radial bins for accuracy)
        pbin = BinParticle2CorrelationPoles(mattrs, edges={'step': 0.1}, sattrs=sattrs, ells=ells)
        # Count close pairs directly (no mesh needed, exact calculation)
        close = compute_particle2(*all_particles, bin=pbin, los=los)
        # Attach normalization and shot noise, then convert to power spectrum
        close = close.clone(num_shotnoise=compute_particle2_shotnoise(*all_particles, bin=pbin, fields=fields))
        # Convert correlation poles to power spectrum (multiply by bin centers)
        close = close.to_spectrum(k)
        # Store negative contribution (contamination to subtract)
        return close.clone(value=-close.value())

    # Then compute the AUW-weighted (angular upweight) pairs and bitwise-weighted pairs
    with_bitweights = bool(all_particles[0].get('bitwise_weight'))
    if auw is not None or with_bitweights:
        from cucount.jax import WeightAttrs, BitwiseWeight
        # Define angular selection for close pairs (< 0.1 degrees for bitwise weights)
        sattrs = {'theta': (0., 0.1)}
        bitwise = angular = None
        if with_bitweights:
            # Weights for fiber collision corrections
            # 1) systematic weights --- without completeness
            # 2) bitweights
            # 3) weights to subtract off (already in the mesh-based P(k) estimation)
            # Extract bitwise weight structure (sets nrealizations based on BITWEIGHT size, fine to use the first)
            bitwise = dict(weights=all_particles[0].get('bitwise_weight'))
            if jax.process_index() == 0:
                logger.info(f'Applying PIP weights {bitwise}.')
            # No bitwise weights, remove individual weights from AUW * individual_weight
        # Apply angular upweights if provided (fiber collision corrections)
        if auw is not None:
            # Extract angular separation and weight values from pre-computed AUW
            angular = dict(sep=auw.get('DD').coords('theta'), weight=auw.get('DD').value())
            if jax.process_index() == 0:
                logger.info(f'Applying AUW {angular}.')
        wattrs = WeightAttrs(bitwise=bitwise)
        # Set default negative_weight = individual_weight
        for i, particles in enumerate(all_particles):
            if not particles.get('negative_weight'):
                negative_weight = wattrs(particles)
                all_particles[i] = particles.clone(weights=particles.get('weights') + [negative_weight], index_value=particles.index_value.clone(negative_weight=1))
        # Set up weight attributes for pair counting
        wattrs = WeightAttrs(bitwise=bitwise, angular=angular)
        # Create binning for close-pair data-data counts with weights
        pbin = BinParticle2SpectrumPoles(mattrs, edges=edges, xavg=k, sattrs=sattrs, wattrs=wattrs, ells=ells)
        # Count weighted pairs directly
        DD = compute_particle2(*all_particles, bin=pbin, los=los)
        # Attach normalization and shot noise
        DD = DD.clone(num_shotnoise=compute_particle2_shotnoise(*all_particles, bin=pbin, fields=fields))
        return DD


def _apply_mesh2_spectrum_close_pair_correction(spectrum, correction):
    """Apply additive corrections to a :class:`Mesh2SpectrumPoles`."""
    # norm
    def sum_counts(leaves):
        return leaves[0].clone(value=leaves[0].value() + leaves[1].value() / leaves[0].values('norm'))

    return types.tree_map(sum_counts, [spectrum, correction])



def get_optimal_weight_columns(optimal_weights):
    # No optimal weights requested, so no extra particle columns are needed.
    if optimal_weights is None:
        return []

    # By convention, optimal weights need redshift unless explicitly specified.
    return list(getattr(optimal_weights, 'columns', ['Z']))


def iter_optimal_weighted_particles(ell, all_particles, optimal_weights, columns):
    # Handle optional catalogues such as shifted=None.
    # This keeps zip(...) loops alive while preserving the None entry.
    if all_particles[0] is None:
        while True:
            yield tuple(None for _ in all_particles)

    # Build the lightweight catalog dictionaries passed to optimal_weights.
    catalogs = [{'INDWEIGHT': p.weights} | {column: p.extra[column] for column in columns} for p in all_particles]

    # optimal_weights may yield one or several weight combinations.
    for all_weights in optimal_weights(ell, catalogs):
        yield tuple(p.clone(weights=w) for p, w in zip(all_particles, all_weights))


def loop_over_optimal_weights(ells, compute, join):
    results = {}

    # Optimal weights depend on ell, so each multipole is computed separately.
    for ell in ells:
        if jax.process_index() == 0:
            logger.info(f'Applying optimal weights for ell = {ell:d}')

        # Caller owns the actual optimal-weight loop, since the particle structure can differ by function.
        result_ell = compute(ell)

        # Accumulate one combined result per ell.
        for key, value in result_ell.items():
            results.setdefault(key, [])
            results[key].append(value)

    # Caller decides how to join multipoles for its output type.
    return join(results)


def compute_mesh2_spectrum(*get_data_randoms, mattrs=None, cut=None, auw=None,
                           ells=(0, 2, 4), edges=None, los='firstpoint', optimal_weights=None,
                           norm: dict=None, cache=None):
    r"""
    Compute the 2-point spectrum multipoles using mesh-based FKP fields with :mod:`jaxpower`.

    Parameters
    ----------
    get_data_randoms : callables
        Functions that return dict of 'data', 'randoms' (optionally 'shifted') catalogs.
        See :func:`prepare_jaxpower_particles` for details.
    mattrs : dict, optional
        Mesh attributes to define the :class:`jaxpower.ParticleField` objects,
        'boxsize', 'meshsize' or 'cellsize', 'boxcenter'. If ``None``, default attributes are used.
    cut : bool, optional
        If True, apply a theta-cut of (0, 0.05) in degrees.
    auw : ObservableTree, optional
        Angular upweights to apply. If ``None``, no angular upweights are applied.
    ells : list of int, optional
        List of multipole moments to compute. Default is (0, 2, 4).
    edges : dict, optional
        Edges for the binning; array or dictionary with keys 'start' (minimum :math:`k`), 'stop' (maximum :math:`k`), 'step' (:math:`\Delta k`).
        If ``None``, default step of :math:`0.001 h/\mathrm{Mpc}` is used.
        See :class:`jaxpower.BinMesh2SpectrumPoles` for details.
    los : {'local', 'firstpoint', 'x', 'y', 'z', array-like}, optional
        Line-of-sight definition. 'local' uses local LOS, 'firstpoint' uses the position of the first point in the pair,
        'x', 'y', 'z' use fixed axes, or provide a 3-vector.
    optimal_weights : callable or None, optional
        Function taking (ell, catalog) as input and returning total weights to apply to data and randoms.
        It can have an optional attribute 'columns' that specifies which additional columns are needed to compute the optimal weights.
        As a default, ``optimal_weights.columns = ['Z']`` to indicate that redshift information is needed.
        A dictionary ``catalog`` of columns is provided, containing 'INDWEIGHT' and the requested columns.
        If ``None``, no optimal weights are applied.
    norm : dict, optional
        Optional arguments for computing normalization.
        Default is ``{'cellsize': 10.}`` (density computed with ``cellsize = 10.``)
    cache : dict, optional
        Cache to store binning class (can be reused if ``meshsize`` and ``boxsize`` are the same).
        If ``None``, a new cache is created.

    Returns
    -------
    spectrum : Mesh2SpectrumPoles or dict of Mesh2SpectrumPoles
        The computed 2-point spectrum multipoles. If `cut` or `auw` are provided, returns a dict with keys 'raw', 'cut', and/or 'auw'.
    """

    # Import FKP field, power spectrum computation, and binning tools from jaxpower
    from jaxpower import (create_sharding_mesh, FKPField, compute_fkp2_normalization, compute_fkp2_shotnoise, BinMesh2SpectrumPoles, compute_mesh2_spectrum,
                          BinParticle2SpectrumPoles, BinParticle2CorrelationPoles, compute_particle2, compute_particle2_shotnoise)

    # Collect column names needed for optimal weight computation
    columns_optimal_weights = get_optimal_weight_columns(optimal_weights)
    mattrs = mattrs or {}
    # Set up distributed mesh computation across JAX devices (multi-GPU/CPU)
    with create_sharding_mesh(meshsize=mattrs.get('meshsize', None)):
        # Load particles and prepare for FKP field creation
        # add_data=['BITWEIGHT'] for fiber collision corrections
        all_particles = prepare_jaxpower_particles(*get_data_randoms, mattrs=mattrs, add_data=['BITWEIGHT'] + columns_optimal_weights, add_randoms=columns_optimal_weights)

        # Initialize or retrieve cached binning object from previous runs
        if cache is None: cache = {}
        # Set default k-space binning step (0.001 h/Mpc)
        if edges is None: edges = {'step': 0.001}
        # Set default normalization computation parameters (density from 10 Mpc/h cells)
        if norm is None: norm = {'cellsize': 10.}
        kw_norm = dict(norm)

        def _compute_spectrum_ell(all_particles, ells, fields=None):
            # Compute power spectrum for input given multipoles
            # Gather particle attributes (weights, mesh info) for output metadata
            attrs = _get_jaxpower_attrs(*all_particles)
            # Store line-of-sight direction in attributes
            attrs.update(los=los)
            # Get mesh attributes from first data particle
            mattrs = all_particles[0]['data'].attrs

            # Define the binner for k-space binning
            key = 'bin_mesh2_spectrum_{}'.format('_'.join(map(str, ells)))
            bin = cache.get(key, None)
            # Create new binning if not cached or if mesh parameters changed
            if bin is None or not np.all(bin.mattrs.meshsize == mattrs.meshsize) or not np.allclose(bin.mattrs.boxsize, mattrs.boxsize):
                bin = BinMesh2SpectrumPoles(mattrs, edges=edges, ells=ells)
            # Store binning in cache for future use
            cache.setdefault(key, bin)

            all_fkp = [FKPField(particles['data'], particles['randoms']) for particles in all_particles]
            # Computing normalization: integral of density^2, splitting randoms ('split') to avoid common noise
            norm = compute_fkp2_normalization(*all_fkp, bin=bin, **kw_norm)

            # Computing shot noise from shifted catalogs (reconstruction) or use randoms if no shifted available
            all_fkp = [FKPField(particles['data'], particles['shifted'] if particles.get('shifted', None) is not None else particles['randoms']) for particles in all_particles]
            del all_particles
            # Shot noise computed from (shifted) randoms
            num_shotnoise = compute_fkp2_shotnoise(*all_fkp, bin=bin, fields=fields)

            # Wait for normalization and shot noise to complete on all devices
            jax.block_until_ready((norm, num_shotnoise))
            if jax.process_index() == 0:
                logger.info('Normalization and shotnoise computation finished')

            with_bitweights = 'BITWEIGHT' in all_fkp[0].data.extra

            # Galaxy pairs at small angular separation
            results = {}
            corrections = {'auw': auw, 'cut': cut}
            if any(corrections.values()):
                from jaxpower.particle2 import convert_particles
                for name in corrections:
                    if corrections[name] is not None:
                        if name == 'cut':
                            all_particles = [convert_particles(fkp.particles) for fkp in all_fkp]
                        else:
                            if with_bitweights:
                                all_particles = [convert_particles(fkp.data, weights=[fkp.data.extra['INDWEIGHT_NO_COMP']] + list(jnp.unstack(fkp.data.extra['BITWEIGHT'], axis=-1)) + [fkp.data.weights], exchange_weights=False) for fkp in all_fkp]
                            else:
                                all_particles = [convert_particles(fkp.data) for fkp in all_fkp]
                        results[name] = _compute_mesh2_spectrum_close_pair_correction(all_particles, bin=bin, **{name: corrections[name]}, los=los, fields=fields)

            # Wait for particle-based calculations to complete
            jax.block_until_ready(results)
            if jax.process_index() == 0:
                logger.info(f'Particle-based calculation finished')

            # Paint particles onto mesh grids for Fourier-space power spectrum computation
            kw = dict(resampler='tsc', interlacing=3, compensate=True)
            # out='real' to save memory (store as real arrays instead of complex)
            all_mesh = [fkp.paint(**kw, out='real') for fkp in all_fkp]
            # Free memory from FKP fields
            del all_fkp

            # JIT the mesh-based spectrum computation; helps with memory footprint
            jitted_compute_mesh2_spectrum = jax.jit(compute_mesh2_spectrum, static_argnames=['los'])
            #jitted_compute_mesh2_spectrum = compute_mesh2_spectrum
            # Compute power spectrum from painted mesh grids via FFT
            spectrum = jitted_compute_mesh2_spectrum(*all_mesh, bin=bin, los=los)
            # Attach normalization and shot noise to spectrum
            spectrum = spectrum.clone(norm=norm, num_shotnoise=num_shotnoise)
            # Propagate particle attributes to each multipole
            spectrum = spectrum.map(lambda pole: pole.clone(attrs=attrs))
            # Also attach attributes at spectrum level
            spectrum = spectrum.clone(attrs=attrs)
            # Wait for spectrum computation to complete on all devices
            jax.block_until_ready(spectrum)
            if jax.process_index() == 0:
                logger.info('Mesh-based computation finished')

            for name in results:
                results[name] = _apply_mesh2_spectrum_close_pair_correction(spectrum, results[name])
            results['raw'] = spectrum

            return results

        if optimal_weights is None:
            # Standard case: no optimal weights → compute all multipoles in one shot
            results = _compute_spectrum_ell(all_particles, ells=ells)

        else:
            # Names of particle types, e.g. ['data', 'randoms', 'shifted']
            # Each catalog dict has these keys
            names = list(all_particles[0].keys())
            # Field labels used by jaxpower multi-tracer logic.
            fields = tuple(range(len(all_particles)))
            # Pad fields to at least nfields, e.g. 1 catalogue -> (0, 0).
            fields = fields + (fields[-1],) * max(0, 2 - len(fields))
            # Pad particle inputs similarly for auto/cross compatibility.
            all_particles = tuple(all_particles)
            all_particles = all_particles + (all_particles[-1],) * max(0, len(fields) - len(all_particles))

            def compute(ell):
                # Collect results for this ell across all weight realizations
                result_ell = {}
                # Build one iterator per particle type (data/randoms/shifted)
                # Each iterator yields weighted versions of that particle type
                iterators = [iter_optimal_weighted_particles(ell,
                        [particles[name] for particles in all_particles],  # same type across catalogs
                        optimal_weights,
                        columns_optimal_weights,
                    ) for name in names]

                # Zip the iterators → one "weight realization"
                # weighted_by_name looks like:
                #   [(data1, data2), (randoms1, randoms2), (shifted1, shifted2)]
                for weighted_by_name in zip(*iterators):

                    # Reorganize from grouping by particle type → grouping by catalog
                    # Before: [(data1, data2), (randoms1, randoms2), (shifted1, shifted2)]
                    # After:  [(data1, randoms1, shifted1), (data2, randoms2, shifted2)]
                    weighted_by_catalog = list(zip(*weighted_by_name))

                    # Convert tuples → dicts expected by _compute_spectrum_ell
                    weighted_by_catalog = [dict(zip(names, particles)) for particles in weighted_by_catalog]

                    # Compute spectrum for this weight realization (single ell)
                    _result = _compute_spectrum_ell(weighted_by_catalog, ells=[ell], fields=fields)

                    # Accumulate results (raw / cut / auw)
                    for key, value in _result.items():
                        result_ell.setdefault(key, [])
                        result_ell[key].append(value)

                # Combine all weight realizations for this ell (e.g. symmetric tracer permutations)
                return {key: combine_stats(value) for key, value in result_ell.items()}

            def join(results):
                # Join results across multipoles: results[key] = [ell0, ell2, ell4, ...] → concatenate
                for key in results:
                    results[key] = types.join(results[key])
                return results

            # Main driver: loops over ell and calls compute(...)
            results = loop_over_optimal_weights(ells, compute, join)

    # Return single result or dictionary of variants
    if len(results) == 1:
        return next(iter(results.values()))
    return results


def compute_window_mesh2_spectrum(*get_data_randoms, spectrum: types.Mesh2SpectrumPoles, optimal_weights: Callable=None,
                                  cut: bool=None, zeff: dict=None, method: str='smooth_mesh', split_randoms: int=None):
    r"""
    Compute the 2-point spectrum window with :mod:`jaxpower`.

    Parameters
    ----------
    get_data_randoms : callables
        Functions that return dict of 'randoms' catalogs.
        See :func:`prepare_jaxpower_particles` for details.
    spectrum : Mesh2SpectrumPoles
        Measured 2-point spectrum multipoles.
    optimal_weights : callable or None, optional
        Function taking (ell, catalog) as input and returning total weights to apply to data and randoms.
    cut : bool, optional
        Whether to compute the theta-cut contribution.
    zeff : dict, optional
        Optional arguments for computing effective redshift.
        Default is ``{'cellsize': 10.}``.
    method : string, optional
        ``'smooth_mesh'`` to use the "smooth" method with 1D window correlation computed with FFTs on the mesh,
        ``'smooth_particle'`` for particle counts, or ``'exact'`` for the exact mesh window.

    Returns
    -------
    results : dict
        Dictionary containing the computed window matrices and optional raw/cut correlations.
    """
    from jaxpower import (create_sharding_mesh, BinMesh2SpectrumPoles, BinParticle2CorrelationPoles, compute_particle2,
                          compute_particle2_shotnoise, compute_smooth2_spectrum_window,
                          split_particles, compute_mesh2_spectrum_window, interpolate_window_function, get_smooth2_window_bin_attrs)

    assert method in ['smooth_mesh', 'smooth_particle', 'exact'], method

    ells = spectrum.ells
    mattrs = {name: spectrum.attrs[name] for name in ['boxsize', 'boxcenter', 'meshsize']}
    los = spectrum.attrs['los']
    ellsin = [0, 2, 4]

    if zeff is None:
        zeff = {'cellsize': 10.}
    kw_zeff = dict(zeff)

    columns_optimal_weights = get_optimal_weight_columns(optimal_weights)
    mattrs = mattrs or {}

    with create_sharding_mesh(meshsize=mattrs.get('meshsize', None)):
        all_particles = prepare_jaxpower_particles(*get_data_randoms, mattrs=mattrs, add_randoms=['IDS'] + columns_optimal_weights)
        all_randoms = [particles['randoms'] for particles in all_particles]
        del all_particles
        fields = list(range(len(all_randoms)))
        fields = fields + [fields[-1]] * (2 - len(fields))

        stop, step = -np.inf, np.inf
        for pole in spectrum:
            edges = pole.edges('k')
            stop = max(edges.max(), stop)
            step = min(np.nanmin(np.diff(edges, axis=-1)), step)

        edgesin = np.arange(0., 1.2 * stop, step)
        edgesin = jnp.column_stack([edgesin[:-1], edgesin[1:]])

        def _compute_window_ell(all_randoms, ells, isum=0, fields=None):
            all_randoms = list(all_randoms)
            seed = [(42, randoms.extra['IDS']) for randoms in all_randoms]
            mattrs = all_randoms[0].attrs
            pole = spectrum.get(ells[0])
            bin = BinMesh2SpectrumPoles(mattrs, edges=pole.edges('k'), ells=ells)
            norm = jnp.concatenate([spectrum.get(ell).values('norm') for ell in ells], axis=0)
            wsum_data = pole.attrs['wsum_data'][isum]

            if fields is None:
                fields = list(range(len(all_randoms)))
                fields = fields + [fields[-1]] * (2 - len(fields))

            seed = [(42, randoms.extra['IDS']) for randoms in all_randoms]
            zeff, norm_zeff = compute_fkp_effective_redshift(*all_randoms, order=2, split=seed, fields=fields,
                                                                return_fraction=True, **kw_zeff)
            kw_window = get_smooth2_window_bin_attrs(ells, ellsin)
            results = {}
            if 'smooth' in method:
                ellsw = kw_window['ells']
                correlation = compute_smooth2_spectrum_window_correlation(*all_randoms, spectrum=spectrum, ells=ellsw, zeff=None, wsum_data=wsum_data, method=method, split_randoms=split_randoms)

                results[f'window_{method}2_correlation_raw'] = types.ObservableTree([correlation], oells=[ells[0] if len(ells) == 1 else tuple(ells)])

                window = compute_smooth2_spectrum_window(correlation, edgesin=edgesin, ellsin=ellsin, bin=bin, flags=('fftlog',))
                window = window.clone(value=window.value() / (norm[..., None] / np.mean(norm)))

            elif method == 'exact':
                kw_paint = dict(resampler='tsc', interlacing=3, compensate=True)
                all_mesh = []

                if jax.process_index() == 0:
                    logger.info(f'Using {mattrs}')

                for iran, randoms in enumerate(split_particles(all_randoms + [None] * (2 - len(all_randoms)), seed=seed, fields=fields)):
                    randoms = randoms.clone(attrs=mattrs).exchange(backend='mpi')
                    alpha = wsum_data[min(iran, len(wsum_data) - 1)] / randoms.weights.sum()
                    all_mesh.append(alpha * randoms.paint(**kw_paint, out='real'))

                window = compute_mesh2_spectrum_window(*all_mesh, edgesin=edgesin, ellsin=(ellsin, 'local'),
                                                       los=los, bin=bin, pbar=False, flags=('infinite',), norm=1.)
                window = window.clone(value=window.value().real / norm[..., None])

            observable = window.observable.map(
                lambda pole, label: pole.clone(norm=spectrum.get(**label).values('norm'), attrs=pole.attrs),
                input_label=True,
            )
            results['raw'] = window.clone(observable=observable)
            results['raw'].attrs.update(zeff=zeff / norm_zeff, norm_zeff=norm_zeff)

            if cut:
                from jaxpower.particle2 import convert_particles

                sattrs = {'theta': (0., 0.05)}
                sedges = np.arange(0., sum(mattrs.boxsize**2)**0.5, 0.1)
                pbin = BinParticle2CorrelationPoles(mattrs, edges=sedges, sattrs=sattrs, **kw_window)

                all_particles = []
                #for iran, randoms in enumerate(all_randoms):
                for iran, randoms in enumerate(split_particles(all_randoms + [None] * (2 - len(all_randoms)), seed=seed, fields=fields)):
                    alpha = wsum_data[min(iran, len(wsum_data) - 1)] / randoms.weights.sum()
                    all_particles.append(convert_particles(randoms.clone(weights=alpha * randoms.weights)))

                correlation = compute_particle2(*all_particles, bin=pbin, los=los)

                from cucount.jax import BinAttrs, count2_analytic
                battrs = BinAttrs(s=sedges)
                RR0 = count2_analytic(mattrs=1., battrs=battrs)

                # Divide by volume factor and normalization
                def renormalize(pole):
                    return pole.clone(norm=np.mean(norm) * RR0)

                correlation = correlation.map(renormalize)
                #correlation = correlation.clone(
                #    num_shotnoise=compute_particle2_shotnoise(*all_particles, bin=pbin, fields=fields),
                #    norm=[np.mean(norm)] * len(pbin.ells),
                #)
                results['window_smooth_particle2_correlation_cut'] = types.ObservableTree(
                    [correlation],
                    oells=[ells[0] if len(ells) == 1 else tuple(ells)],
                )
                edgesin_extended = jnp.arange(edgesin.max(), 10., 0.02)
                edgesin_extended = jnp.column_stack([edgesin_extended[:-1], edgesin_extended[1:]])
                edgesin_extended = jnp.concatenate([edgesin, edgesin_extended], axis=0)
                window_cut = compute_smooth2_spectrum_window(correlation, edgesin=edgesin_extended, ellsin=ellsin, bin=bin, flags=('rect',), batch_size=20)
                #results['cut'] = window_cut
                results['cut'] = results['raw'].at.theory.match(window_cut.theory).clone(theory=window_cut.theory)  # extend theory of raw to that of cut
                results['cut'] = results['cut'].clone(value=results['cut'].value() - window_cut.value() / (norm[..., None] / np.mean(norm)))

            return results

        if optimal_weights is None:
            seed = [(42, randoms.extra['IDS']) for randoms in all_randoms]
            results = _compute_window_ell(all_randoms, ells=ells, fields=fields)

            for key in results:
                if 'correlation' not in key:
                    observable = results[key].observable
                    observable = observable.map(lambda pole: pole.clone(attrs=results[key].attrs | pole.attrs))
                    results[key] = results[key].clone(observable=observable)

        else:
            all_randoms = tuple(all_randoms)
            seed = [(42, randoms.extra['IDS']) for randoms in all_randoms]
            all_randoms = split_particles(list(all_randoms) + [None] * (len(fields) - len(all_randoms)), seed=seed, fields=fields)

            def compute(ell):
                result_ell = {}

                for isum, _all_randoms in enumerate(iter_optimal_weighted_particles(ell, all_randoms, optimal_weights,
                                                                                    columns_optimal_weights)):

                    # Compute window for this ell and this weighted realization
                    # fields = None as cross (2 randoms given)
                    _result = _compute_window_ell(_all_randoms, ells=[ell], isum=isum, fields=None)

                    # Attach zeff to matrix outputs; leave correlation debug outputs untouched.
                    for key in _result:
                        if 'correlation' not in key:
                            observable = _result[key].observable
                            observable = observable.map(lambda pole: pole.clone(attrs=_result[key].attrs | pole.attrs))
                            _result[key] = _result[key].clone(observable=observable)

                        result_ell.setdefault(key, [])
                        result_ell[key].append(_result[key])

                # Combine symmetric optimal-weight realizations, then preserve the summed window value.
                combined = {}
                for key, windows in result_ell.items():
                    window = combine_stats(windows)
                    window = window.clone(value=sum(window.value() for window in windows))
                    combined[key] = window
                return combined

            def join(results):
                for key in results:
                    if 'correlation' in key:
                        # Join debug correlation outputs along multipoles
                        results[key] = types.join(results[key])
                    else:
                        # Join window matrices along the output multipole axis
                        observables = [window.observable for window in results[key]]
                        observable = types.join(observables)
                        value = np.concatenate([window.value() for window in results[key]], axis=0)
                        results[key] = results[key][0].clone(value=value, observable=observable)
                return results

            results = loop_over_optimal_weights(ells, compute, join)

    return results


def compute_smooth2_spectrum_window_correlation(*get_data_randoms, spectrum: types.Mesh2SpectrumPoles, ells=None, zeff: dict=None, wsum_data=None, method: str='smooth_mesh', split_randoms: int=None, fields: list=None):
    r"""
    Compute the 2-point window correlation with :mod:`jaxpower` or :mod:`cucount`.

    Parameters
    ----------
    get_data_randoms : callables
        Functions that return dict of 'randoms' catalogs.
        See :func:`prepare_jaxpower_particles` for details.
    spectrum : Mesh2SpectrumPoles
        Measured 2-point spectrum multipoles.
    optimal_weights : callable or None, optional
        Function taking (ell, catalog) as input and returning total weights to apply to data and randoms.
        It can have an optional attribute 'columns' that specifies which additional columns are needed to compute the optimal weights.
        As a default, ``optimal_weights.columns = ['Z']`` to indicate that redshift information is needed.
        A dictionary ``catalog`` of columns is provided, containing 'INDWEIGHT' and the requested columns.
        If ``None``, no optimal weights are applied.
    zeff : dict, optional
        Optional arguments for computing effective redshift.
        Default is ``{'cellsize': 10.}`` (density computed with ``cellsize = 10.``)
    method : string, optional
        ``'smooth_mesh'`` to use the "smooth" method with 1D window correlation computed with FFTs on the mesh,
        ``'smooth_particle'`` for particle counts.
    split_randoms : float, tuple
        If provided, number of subsets to split the random catalogs into.
        If a tuple, (number of splits, used number of splits).
        (e.g. (10, 5) will just use the first 5 splits out of 10).

    Returns
    -------
    window : ObservableTree
        The computed 2-point window correlation.
    """
    # Import window and correlation computation tools from jaxpower
    from jaxpower import (create_sharding_mesh, BinMesh2CorrelationPoles, compute_mesh2_correlation,
                         get_smooth2_window_bin_attrs, interpolate_window_function, split_particles)

    # Extract multipole moments from input spectrum
    # Extract mesh parameters from spectrum attributes
    mattrs = {name: spectrum.attrs[name] for name in ['boxsize', 'boxcenter', 'meshsize']}
    # Extract line-of-sight direction
    los = spectrum.attrs['los']
    # Mesh painting parameters: TSC kernel with 3-fold interlacing for aliasing correction
    kw_paint = dict(resampler='tsc', interlacing=3, compensate=True)
    # Set default effective redshift computation parameters
    if zeff is None: kw_zeff = None
    else: kw_zeff = dict(zeff)

    mattrs = mattrs or {}
    # Set up distributed computation mesh
    with create_sharding_mesh(meshsize=mattrs.get('meshsize', None)) as sharding_mesh:
        # Load random catalogs and prepare particles
        if callable(get_data_randoms[0]):
            all_particles = prepare_jaxpower_particles(*get_data_randoms, mattrs=mattrs, add_randoms=['IDS'])
            all_randoms = [particles['randoms'] for particles in all_particles]
            del all_particles
        else:
            all_randoms = list(get_data_randoms)
        # Map particle indices to catalog indices (cycle if fewer catalogs than 2 fields)
        if fields is None:
            fields = list(range(len(all_randoms)))
        fields = list(fields)
        fields += [fields[-1]] * (2 - len(fields))

        # Compute effective redshift for this weighted realization
        seed = [(42, randoms.extra['IDS']) for randoms in all_randoms]
        if kw_zeff is not None:
            zeff, norm_zeff = compute_fkp_effective_redshift(*all_randoms, order=2, split=seed, fields=fields, return_fraction=True, **kw_zeff)

        mattrs = all_randoms[0].attrs

        pole = next(iter(spectrum))
        # Get normalization from input power spectrum
        norm = jnp.concatenate([pole.values('norm') for pole in spectrum], axis=0)

        # Get window basis attributes (which multipoles to compute in correlation space)
        kw_window = get_smooth2_window_bin_attrs(spectrum.ells, ellsin=[0, 2, 4])
        if ells is not None:
            kw_window['ells'] = ells
        if wsum_data is None:
            assert len(pole.attrs['wsum_data']) == 1
            wsum_data = pole.attrs['wsum_data'][0]
        # Use logarithmic s-grid for window interpolation (and FFTlog)
        coords = jnp.logspace(-3, 5, 4 * 1024)

        if method == 'smooth_particle':
            from jaxpower.particle2 import convert_particles
            from cucount.jax import BinAttrs, SelectionAttrs, WeightAttrs
            from cucount.types import count2, count2_analytic, compute_norm2
            from jaxpower.mesh import create_sharded_random, _process_seed
            from .correlation3_tools import _digitize_cartesian, _remove_phantom_particles

            all_particles = []
            # Paint random catalogs on coarse mesh
            for iran, randoms in enumerate(split_particles(all_randoms + [None] * (2 - len(all_randoms)),
                                                            seed=seed, fields=fields)):
                # Normalize by data/random weight ratio
                alpha = wsum_data[min(iran, len(wsum_data) - 1)] / randoms.weights.sum()
                # Paint random on mesh scaled by alpha
                randoms = randoms.clone(weights=alpha * randoms.weights)
                all_particles.append(convert_particles(randoms))
            del all_randoms

            edges = np.arange(0., jnp.sqrt(jnp.sum(mattrs.boxsize**2)), mattrs.cellsize.min())
            battrs = BinAttrs(s=edges, pole=(tuple(kw_window['ells']), 'firstpoint'))

            sepmax = edges.max()
            limits = [0., 200., 500.]
            limits = [lim for lim in limits if lim < sepmax] + [sepmax]
            resols = [None, 40., 100.]
            nsplits, max_nsplits = 1, 1
            if split_randoms is not None:
                if isinstance(split_randoms, tuple):
                    nsplits, max_nsplits = split_randoms
                else:
                    nsplits = max_nsplits = split_randoms

            wattrs = WeightAttrs()
            prod = functools.partial(functools.reduce, operator.mul)

            def count2split(*particles, sattrs=None, norm_ref=1.):
                kw = dict(wattrs=wattrs, battrs=battrs, sattrs=sattrs, norm=1.)
                nsplits = [getattr(p, 'nsplits', 0) for p in particles]
                if any(nsplits):
                    nsplit = next(n for n in nsplits if n)
                    particle_iters = []
                    for particle in particles:
                        if getattr(particle, 'nsplits', 0):
                            assert particle.nsplits == nsplit
                            particle_iters.append(particle())
                        else:
                            particle_iters.append(itertools.repeat(particle, nsplit))
                    counts, norm = [], 0.
                    for p in zip(*particle_iters):
                        counts.append(count2(*p, **kw)['weight'])
                        norm += compute_norm2(*p, wattrs=wattrs)
                    counts = types.sum(counts)
                else:
                    counts = count2(*particles, **kw)['weight']
                    norm = compute_norm2(*particles, wattrs=wattrs)
                return counts.clone(value=norm_ref / norm * counts.value())

            norm_ref = compute_norm2(*all_particles, wattrs=wattrs)
            counts = []
            resol_limits = zip(zip(limits[:-1], limits[1:]), resols)

            for resol_limit, resol in resol_limits:
                sattrs = SelectionAttrs(s=resol_limit)
                all_particles_resol = list(all_particles)
                digitized = set()
                t0 = time.time()

                if resol is not None:
                    all_particles_resol[1] = _digitize_cartesian(all_particles_resol[1], wattrs=wattrs, cellsize=resol, sharding_mesh=sharding_mesh)
                    digitized.add(1)

                if nsplits > 1:

                    def _get_uniform(size, seed=(84, 'index')):
                        return create_sharded_random(jax.random.uniform, _process_seed(seed), size, out_specs=P(sharding_mesh.axis_names,))

                    def make_particle_splits(particles, x, nsplits, max_nsplits):
                        weights = wattrs(particles)

                        def gen():
                            for isplit in range(max_nsplits):
                                mask = (x >= isplit / nsplits) & (x < (isplit + 1) / nsplits)
                                p = particles.clone(weights=weights * mask)
                                yield _remove_phantom_particles(p, sharding_mesh=sharding_mesh)

                        gen.nsplits = max_nsplits

                        return gen

                    if len(all_particles_resol) - len(digitized) > 1:
                        for ip, particles in enumerate(all_particles_resol):
                            if ip not in digitized:
                                x = _get_uniform(particles.size)
                                all_particles_resol[ip] = make_particle_splits(particles, x, nsplits=nsplits, max_nsplits=max_nsplits)

                    elif max_nsplits < nsplits:
                        for ip, particles in enumerate(all_particles_resol):
                            if ip not in digitized:
                                weights = wattrs(particles)
                                x = _get_uniform(particles.size)
                                mask = x < max_nsplits / nsplits
                                all_particles_resol[ip] = _remove_phantom_particles(particles.clone(weights=weights * mask), sharding_mesh=sharding_mesh)

                counts.append(count2split(*all_particles_resol, sattrs=sattrs, norm_ref=norm_ref))
                if jax.process_index() == 0:
                    logger.info(f'Computed RR counts within {resol_limit} in {time.time() - t0:.1f} s')

            def sum_counts(leaves):
                return leaves[0].clone(counts=sum(leaf.values('counts') for leaf in leaves), norm=leaves[0].values('norm'))

            counts = types.tree_map(sum_counts, counts, level=None, is_leaf=lambda *args: False)
            battrs = BinAttrs(s=edges)
            RR0 = count2_analytic(mattrs=1., battrs=battrs)

            # Divide by volume factor and normalization
            def renormalize(pole):
                return pole.clone(counts=pole.values('counts'), norm=np.mean(norm) * RR0.value())

            counts = counts.map(renormalize)
            correlation = interpolate_window_function(counts, coords=coords, order=3)

        elif method == 'smooth_mesh':
            correlations = []
            # JIT-compile correlation computation for memory efficiency
            # donate_argnums=[0] allows JAX to reuse memory of first argument
            jitted_compute_mesh2_correlation = jax.jit(compute_mesh2_correlation, static_argnames=['los'], donate_argnums=[0])
            # Window computed in configuration space, summing Bessel functions over the Fourier-space mesh
            list_edges = []
            # Loop over scale factors for multigrid window computation (coarse then fine)
            for scale in [1, 4]:
                # Create coarser mesh (larger boxsize) for computational efficiency at coarse scales
                mattrs2 = mattrs.clone(boxsize=scale * mattrs.boxsize)
                if jax.process_index() == 0:
                    logger.info(f'Processing scale x{scale:.0f}, using {mattrs2}')
                all_mesh = []
                # Paint random catalogs on mesh
                for iran, randoms in enumerate(split_particles(all_randoms + [None] * (2 - len(all_randoms)), seed=seed, fields=fields)):
                    # Redistribute particles to mesh and exchange across MPI processes
                    randoms = randoms.clone(attrs=mattrs2).exchange(backend='mpi')
                    # Compute weight normalization (data/random density ratio from input spectrum)
                    alpha = wsum_data[min(iran, len(wsum_data) - 1)] / randoms.weights.sum()
                    # Paint random particles with proper normalization onto mesh
                    all_mesh.append(alpha * randoms.paint(**kw_paint, out='real'))
                # Define radial binning for correlation space window
                # distmax: use 1/4 of boxsize minimum dimension for correlation range
                distmax, cellsize = mattrs2.boxsize.min() / 4., mattrs2.cellsize.min()
                # Create radial bins from 0 to distmax
                edges = np.arange(0., distmax + cellsize, cellsize)
                list_edges.append(edges)
                # Create binning for correlation function (configuration space)
                sbin = BinMesh2CorrelationPoles(mattrs2, edges=edges, **kw_window, basis='bessel')
                # Compute correlation function via FFT (inverse Fourier transform of painted mesh)
                correlation = jitted_compute_mesh2_correlation(all_mesh, bin=sbin, los=los).clone(norm=[np.mean(norm)] * len(sbin.ells))
                # Free mesh memory
                del all_mesh
                #if jax.process_index() == 0: correlation.write(f'_tests/window_correlation2_{scale:.0f}.h5')
                # Interpolate correlation to fine logarithmic k-grid for FFTLog integration
                correlation = interpolate_window_function(correlation, coords=coords, order=3)
                correlations.append(correlation)
            # Create transition masks between coarse and fine scale grids
            # Masks ensure each point is covered by exactly one scale
            masks = [coords < edges[-3] for edges in list_edges[:-1]]
            # Last mask covers remainder
            masks.append((coords < np.inf))
            # Convert masks to exclusive regions (each point weighted by only one scale)
            weights = []
            for mask in masks:
                if len(weights):
                    # Exclude already-weighted regions from previous scales
                    weights.append(mask & (~weights[-1]))
                else:
                    weights.append(mask)
            # Regularize weights to avoid division by zero
            weights = [np.maximum(mask, 1e-6) for mask in weights]
            # Combine correlations from different scales using smooth weights
            correlation = correlations[0].sum(correlations, weights=weights)

        # Wait for window computation to complete
        jax.block_until_ready(correlation)
        if jax.process_index() == 0:
            logger.info('Window functions computed.')

        if kw_zeff is not None:
            correlation.attrs.update(zeff=zeff / norm_zeff, norm_zeff=norm_zeff)

    return correlation


def compute_window_mesh2_spectrum_fm(
    *get_data_randoms: Callable,
    spectra: list[types.Mesh2SpectrumPoles],
    theory: types.Mesh2SpectrumPoles,
    optimal_weights: Callable | None,
    data_to_randoms_ratio: float,
    catalog_split_seed: int,
    geo: bool,
    ric: bool,
    ric_nbins: int | tuple[int, int],
    ric_regions: list[str] | tuple[list[str], list[str]],
    amr: bool,  # is optional
    ellsout: list[int] | None,
    regression_maps: list[str] | tuple[list[str], list[str]] | None,
    templates_paths_kwargs: dict | tuple[dict, dict] | None,
    amr_regions_zranges: list[tuple[str, tuple[float, float]]] | tuple[list[tuple[str, tuple[float, float]]], list[tuple[str, tuple[float, float]]]] | None,
    spectrum_regions_zranges: list[str] | None,
    total_region_zrange: tuple[str, tuple[float, float]],
    unitary_amplitude: bool = True,
    n_realizations: int,
    seeds: list[int] | None,
    batch_size: int = 4,
    rescale_quadrupole: bool = True,
) -> dict[str, dict[str, list[types.WindowMatrix]]]:
    """
    Compute the 2-point spectrum window with :mod:`desiwinds`.

    Parameters
    ----------
    *get_data_randoms : Callable
        Functions that return tuples of (data, randoms) catalogs.
    spectra: list[types.Mesh2SpectrumPoles]
        Measured 2-point spectrum multipoles. Only used for their attributes (binning, norm, wsum...), not their values.
    theory: lsstypes.Mesh2SpectrumPoles
        Input theory power spectrum, used as a fiducial for the derivative. Attributes (e.g. ells) used for mock survey generation; value used for the derivative.
    optimal_weights : Callable or None
        Function taking (ell, catalog) as input and returning total weights to apply to data and randoms.
        It can have an optional attribute 'columns' that specifies which additional columns are needed to compute the optimal weights.
        As a default, ``optimal_weights.columns = ['Z']`` to indicate that redshift information is needed.
        A dictionary ``catalog`` of columns is provided, containing 'INDWEIGHT' and the requested columns.
        If ``None``, no optimal weights are applied.
    data_to_randoms_ratio : float
        Population ratio between "data" and "randoms" to pick in the input randoms catalogs. Must be between 0 and 1.
    catalog_split_seed : int
        Random seed to use for the random split between "data" and "randoms" in the input randoms catalogs.
    geo : bool
        Whether to return the sampled window for the geometry. If False, not computed.
    ric : bool
        Wether to return the sampled window for the geometry + RIC (+/- AMR if amr=True). If False, not computed.
    ric_nbins : int | tuple[int, int]
        Number of radial bins to use for the RIC. Can provide as tuple for cross-spectra.
    ric_regions : list[str] | tuple[list[str], list[str]]
        Regions to use for the RIC, e.g. ``["N", "S"]`` or ``["N", "SnoDES", "DES]``. Can provide as tuple for cross-spectra.
    amr : bool
        Whether to apply the angular mode removal (AMR), i.e. to forward model the power loss due to linear angular systematics weights.
    ellsout : list[int] | None
        For which ells the window is computed. Default None and use ellsout extracted from corresponding power spectra. Useful to split the computation.
    regression_maps : list[str] | tuple[list[str], list[str]] | None
        Names of the systematics templates to use for the AMR. Can be set to ``None`` if ``amr=False``. Can provide as tuple for cross-spectra.
    templates_paths_kwargs : dict | tuple[dict, dict] | None
        Keyword arguments to pass to the function loading the templates maps, e.g. paths to the templates files, EBV map, nside, etc. Not needed if ``amr=False``. Must at least contain the keys ``templates_path_N`` and ``templates_path_S`` with the paths to the templates files for the Northern and Southern regions, respectively. Can (must) provide as tuple for cross-spectra.
    amr_regions_zranges : list[tuple[str, tuple[float, float]]] | tuple[list[tuple[str, tuple[float, float]]], list[tuple[str, tuple[float, float]]]] | None
        Regions where to apply the regressions for the AMR, and corresponding redshift ranges. Can be set to ``None`` if ``amr=False``. Can provide as tuple for cross-spectra.
    spectrum_regions_zranges : list[tuple[str, tuple[float, float]]] | None
        Regions for which to compute the window and power spectrum, along with their corresponding redshift ranges. If ``None``, the whole catalog is used as one region. Typically ``[("NGC", (zmin, zmax)), ("SGC", (zmin, zmax))]``.
    total_region_zrange : tuple[str, tuple[float, float]]
        Total region and redshift range to use for the forward model (but not necessarily the spectra). Should at least encompass all the spectrum regions. Should generally be ("ALL", (zmin, zmax)) with the full redshift range of the tracer, with some exceptions for cross-correlations.
    n_realizations : int
        Number of realizations to compute.
    seeds : list[int] | None
        Seeds to use for each realization. If ``None``, defaults to ``2 * i_realization + 3``.
    unitary_amplitude : bool, optional
        Whether to use unitary amplitude for the mock survey mesh generation, by default True.
    batch_size : int, optional
        Number of window computations to run in parallel, by default 4. Depends on the available memory, number of randoms catalogs, size of the mesh... Lower if needed.
    rescale_P2 : bool, optional
        Whether to allow rescaling of the input theory quadrupole if conditions for Gaussian mock survey generation are not met, by default True. If False, the function will raise an error if the input theory does not satisfy the conditions for Gaussian mock survey generation.

    Returns
    -------
    dict[str, dict[str, list[lsstypes.WindowMatrix]]]
        Dictionary, per effect included (geometry, RIC, RIC+AMR) and per region, of lists of window matrices (one per realization).

    Notes
    -----
    * Particles loaded by ``get_data_randoms`` but not present in any of ``spectrum_regions_zranges`` are used for observational effects (RIC, AMR, data-to-randoms ratio renormalization...) but not taken into account in power spectrum computations. In general, ``get_data_randoms`` should load the full footprint and range of redshifts available, *including outside the overlap for cross-correlations*.
    * Power spectrum regions/redshift ranges should not overlap. Overlapping regions require separate calls to this function.
    """
    assert len(seeds) == n_realizations if seeds is not None else True, "If seeds are provided, their number must match n_realizations."
    # Notes to self:
    # * RIC not optional
    # * n_randoms is effectively set by the length of get_data_randoms

    # Before anything, check that the input theory satisfies the conditions for Gaussian mock survey generation (positive definite covariance matrix). If not, either raise an error or rescale the quadrupole if rescaling is allowed.
    c0v = theory.get(0).value() - 7 / 18 * theory.get(4).value()
    if (c0v <= 0).any():
        raise ValueError("Theory (P_0 - 7/18 * P_4) has negative values and cannot be used to generate gaussian mocks for the window function.")
    c2v = 35 * theory.get(4).value() / 18
    rec0vc2v = 0.5 * theory.get(2).value() - 5 / 18 * theory.get(4).value()
    if (c0v * c2v - rec0vc2v**2 <= 0).any():
        if not rescale_quadrupole:
            raise ValueError(
                "Theory does not satisfy the condition c0 * c2 - Re(c0c2*)^2 > 0 for generating gaussian mocks for the window function. Some P_2 values may be negative. Check input theory or set rescale_quadrupole=True to allow rescaling of P_2."
            )
        else:
            # Rescale c2
            logger.warning("Theory does not satisfy the condition c0 * c2 - Re(c0c2*)^2 > 0 for generating gaussian mocks for the window function. Rescaling P_2 by a global factor to enforce this condition.")
            # Assume P_2 is positive
            rescale = np.min((5 * theory.get(4).value() / 9 + 2 * np.sqrt(c0v * c2v)) / theory.get(2).value()) - 1e-6
            if np.abs(rescale - 1) > 0.1:
                logger.warning(
                    "Rescaling factor for P_2 is %f, which is quite far from 1. Check that the input theory is reasonable.",
                    rescale,
                )
            else:
                logger.info("Rescaling factor for P_2 is %f.", rescale)
            theory = types.Mesh2SpectrumPoles([theory.get(ell).clone(value=theory.get(ell).value() * rescale) if ell == 2 else theory.get(ell) for ell in theory.ells])
            # Check that rescaled theory satisfies the condition
            rec0vc2v = 0.5 * theory.get(2).value() - 5 / 18 * theory.get(4).value()
            if (c0v * c2v - rec0vc2v**2 <= 0).any():
                # Rescaling did not fix the issue, raise an error
                # Rescaling should work well for Kaiser theories, but maybe not for velocileptors...
                raise ValueError("Even after rescaling P_2, theory does not satisfy the condition c0 * c2 - Re(c0c2*)^2 > 0 for generating gaussian mocks for the window function. Some P_2 values may be negative. Check input theory.")

    import mpytools as mpy
    from desiwinds.forward import mock_survey_catalog, prepare_AMR, prepare_RIC
    from desiwinds.window import get_window_spikes
    from jaxpower import BinMesh2SpectrumPoles, FKPField, ParticleField, create_sharding_mesh

    from .tools import add_photometric_template_values, select_region

    def _add_photometric_template_values(catalogs: dict[str, mpy.Catalog], regression_maps, templates_paths_kwargs):
        return {name: add_photometric_template_values(catalogs[name], regression_maps, **templates_paths_kwargs) for name in catalogs}

    def _select_region_zrange(catalogs: dict[str, mpy.Catalog], spectrum_region_zrange: tuple[str, tuple[float, float]]) -> dict[str, mpy.Catalog]:
        spectrum_region, zrange = spectrum_region_zrange
        # Mimic read_clustering_catalog: <= on lower, < on upper
        return {name: cat[select_region(ra=cat["RA"], dec=cat["DEC"], region=spectrum_region) & (cat["Z"] >= zrange[0]) & (cat["Z"] < zrange[1])] for name, cat in catalogs.items()}

    def _select_region_zrange_complement(catalogs: dict[str, mpy.Catalog]) -> dict[str, mpy.Catalog]:
        """Select objects outside the spectrum regions and redshift ranges, but inside the total region and redshift range."""
        total_region, total_zrange = total_region_zrange
        catalogs = dict(catalogs)
        for name, cat in catalogs.items():
            mask = False
            for (spectrum_region, zrange) in spectrum_regions_zranges:
                mask |= (select_region(ra=cat["RA"], dec=cat["DEC"], region=spectrum_region) & (cat["Z"] >= zrange[0]) & (cat["Z"] < zrange[1]))
            mask = (~mask) & (select_region(ra=cat["RA"], dec=cat["DEC"], region=total_region) & (cat["Z"] >= total_zrange[0]) & (cat["Z"] < total_zrange[1]))
            catalogs[name] = cat[mask]
        return catalogs

    def _loadbalance(catalogs: dict[str, mpy.Catalog]) -> dict[str, mpy.Catalog]:
        return {name: catalog.all_to_all() for name, catalog in catalogs.items()}

    def _split_data_randoms(catalogs: dict[str, mpy.Catalog]) -> dict[str, mpy.Catalog]:
        """Split the randoms into "data" and "randoms" based on the provided ratio. Overwrite original "data"."""
        randoms = catalogs["randoms"]
        data_csize = int(data_to_randoms_ratio * randoms.csize)
        randoms_csize = randoms.csize - data_csize
        rng = mpy.random.MPIRandomState(seed=catalog_split_seed, size=randoms.size)  # Use local sizes
        mask_is_data = rng.uniform() < (data_csize / (data_csize + randoms_csize))
        data = randoms[mask_is_data]
        randoms = randoms[~mask_is_data]
        return {"data": data, "randoms": randoms}

    def _safe_divide(a, b):
        return jnp.where(b != 0, a / b, 0.0)

    get_data_randoms = list(get_data_randoms)  # for mutability

    spectrum_regions_zranges = spectrum_regions_zranges or []
    columns_optimal_weights = []
    if optimal_weights is not None:  # FIXME should this be doubled up for cross-spectra?
        columns_optimal_weights += getattr(optimal_weights, "columns", [])  # to compute optimal weights, e.g. for fnl

    ntracers = len(get_data_randoms)
    # Double up parameters for RIC/AMR if not already provided as tuples
    def is_sequence(item):
        return isinstance(item, (list, tuple))

    if not is_sequence(ric_nbins):
        ric_nbins = (ric_nbins,) * ntracers
    if not is_sequence(ric_regions):
        ric_regions = [ric_regions]
    if isinstance(ric_regions[0], str):
        ric_regions = (ric_regions,) * ntracers
    all_regression_maps = None
    if regression_maps is not None:
        if not is_sequence(regression_maps):
            regression_maps = [regression_maps]
        if isinstance(regression_maps[0], str):
            regression_maps = (regression_maps,) * ntracers
        all_regression_maps = {}
        for regression_map in regression_maps:
            all_regression_maps |= dict.fromkeys(regression_map)
        all_regression_maps = list(all_regression_maps)
    if templates_paths_kwargs is not None and not is_sequence(templates_paths_kwargs):
        templates_paths_kwargs = (templates_paths_kwargs,) * ntracers
    if amr_regions_zranges is not None:
        if not is_sequence(amr_regions_zranges[0]):
            amr_regions_zranges = [amr_regions_zranges]
        if isinstance(amr_regions_zranges[0][0], str):
            amr_regions_zranges = (amr_regions_zranges,) * ntracers

    # Recover output and mesh information from the observable spectrum
    ellsout = ellsout or spectra[0].ells
    los = spectra[0].attrs["los"]  # this has to match with theory input
    if los in ["endpoint", "firstpoint"]:
        los = "local"
    # Input power spectrum may be computed in a single region only (NGC), so boxcenter is probably wrong
    mattrs = {name: spectra[0].attrs[name] for name in ["boxsize", "meshsize"]}

    knyq = np.pi / mattrs["boxsize"] * mattrs["cellsize"]
    if theory.get(0).edges("k").max() > knyq:
        logger.info("Limiting theory k_max to mesh k_nyquist.")
        theory = theory.select(k=(0.0, knyq))

    with create_sharding_mesh(meshsize=mattrs.get("meshsize", None)):
        if amr:  # Add photometric template values to the catalogs, if AMR is applied, as they are needed for the regression
            # _templates_paths_kwargs depends on the tracer, so do this before further changing get_data_randoms
            # load all regression maps (i.e. for both tracers in case of cross-correlation) ; this simplifies things greatly and useless ones will be dropped later to save memory
            for itracer, (_get_data_randoms, _templates_paths_kwargs) in enumerate(zip(get_data_randoms, templates_paths_kwargs, strict=True)):

                def wrap(f):
                    return lambda: _add_photometric_template_values(f(), all_regression_maps, _templates_paths_kwargs)

                get_data_randoms[itracer] = wrap(_get_data_randoms)

        # Split into "data" and randoms based on the provided ratio
        def wrap(f):
            return lambda: _loadbalance(_split_data_randoms(f()))

        get_data_randoms = [wrap(_get_data_randoms) for _get_data_randoms in get_data_randoms]

        if len(spectrum_regions_zranges) > 0:
            # Also get particles that might not be in the region selection
            def wrap(f):
                return lambda: _loadbalance(_select_region_zrange_complement(f()))

            get_extra_data_randoms = [wrap(_get_data_randoms) for _get_data_randoms in get_data_randoms]

            # Split catalogs into pk regions, if specified

            def wrap(f, spectrum_region):
                return lambda: _loadbalance(_select_region_zrange(f(), spectrum_region))

            get_data_randoms = [
                wrap(_get_data_randoms, spectrum_region_zrange) for spectrum_region_zrange in spectrum_regions_zranges for _get_data_randoms in get_data_randoms
            ]  # [func1_region1, func2_region1, func3_region1 ... func1_region2, func2_region2, func3_region2 ...]
        else:
            get_extra_data_randoms = []

        all_particles = prepare_jaxpower_particles(
            *get_data_randoms,
            mattrs=mattrs,
            add_randoms=["IDS", "WEIGHT_FKP", "Z", *columns_optimal_weights] + (all_regression_maps if amr else []),
            add_data=["WEIGHT_FKP", "Z", *columns_optimal_weights] + (all_regression_maps if amr else []),
        )
        mattrs = all_particles[0]['randoms'].attrs

        if get_extra_data_randoms:
            extra_particles = prepare_jaxpower_particles(
                *get_extra_data_randoms,
                mattrs=mattrs,
                add_randoms=["IDS", "WEIGHT_FKP", "Z", *columns_optimal_weights] + (all_regression_maps if amr else []),
                add_data=["WEIGHT_FKP", "Z", *columns_optimal_weights] + (all_regression_maps if amr else []),
                check=False,  # these particles will be outside mattrs but it doesn't matter
            )
        else:
            extra_particles = []

        all_randoms = [particles["randoms"] for particles in all_particles]
        all_data = [particles["data"] for particles in all_particles]
        # Extra data and randoms are needed for observational effects but shouldn't be in spectrum computations
        extra_data = [particles["data"] for particles in extra_particles]
        extra_randoms = [particles["randoms"] for particles in extra_particles]
        del all_particles, extra_particles

        # Make into len(spectrum_regions) catalogs if split into spectrum regions, otherwise one catalog
        nregion = len(spectrum_regions_zranges) if len(spectrum_regions_zranges) > 0 else 1
        nrandoms = len(all_randoms)
        ntracers = nrandoms // nregion
        all_randoms = [[all_randoms[ntracers * i + itracer] for itracer in range(ntracers)] for i in range(nregion)]
        all_data = [[all_data[ntracers * i + itracer] for itracer in range(ntracers)] for i in range(nregion)]
        # [[tracer1_region1, tracer2_region1, ...], [tracer1_region2, tracer2_region2, ...], ...]

        for iregion in range(nregion):
            for itracer in range(len(all_randoms[iregion])):
                # Randoms
                extra = all_randoms[iregion][itracer].extra
                if amr:
                    extra.update({"template_values": jnp.stack([extra.pop(map_name) for map_name in regression_maps[itracer]], axis=-1)})
                    for map in set(all_regression_maps) - set(regression_maps[itracer]):
                        del extra[map]  # remove maps not used for this tracer to save memory
                # extra already has weight_FKP, just remove from weights=indweights which contains FKP weights
                all_randoms[iregion][itracer] = all_randoms[iregion][itracer].clone(extra=extra, weights=_safe_divide(all_randoms[iregion][itracer].weights, extra["WEIGHT_FKP"]))

                # Data
                extra = all_data[iregion][itracer].extra
                if amr:
                    extra.update({"template_values": jnp.stack([extra.pop(map_name) for map_name in regression_maps[itracer]], axis=-1)})
                    for map in set(all_regression_maps) - set(regression_maps[itracer]):
                        del extra[map]  # remove maps not used for this tracer to save memory
                all_data[iregion][itracer] = all_data[iregion][itracer].clone(extra=extra, weights=_safe_divide(all_data[iregion][itracer].weights, extra["WEIGHT_FKP"]))
        del extra

        # Add extra data and randoms with WEIGHT_FKP = 0 to avoid them contributing to the window computation, but still have actual weights in RIC/AMR
        for itracer in range(ntracers):
            # Randoms
            extra = extra_randoms[itracer].extra
            if amr:
                extra.update({"template_values": jnp.stack([extra.pop(map_name) for map_name in regression_maps[itracer]], axis=-1)})
                for map in set(all_regression_maps) - set(regression_maps[itracer]):
                    del extra[map]  # remove maps not used for this tracer to save memory
            # Isolate weights and set WEIGHT_FKP to 0
            extra_randoms[itracer] = extra_randoms[itracer].clone(
                extra=extra | {"WEIGHT_FKP": jnp.zeros_like(extra["WEIGHT_FKP"])}, weights=_safe_divide(extra_randoms[itracer].weights, extra["WEIGHT_FKP"])
            )
            # Concatenate to first randoms catalog for this tracer
            all_randoms[0][itracer] = ParticleField.concatenate([all_randoms[0][itracer], extra_randoms[itracer]], local=True)

            # Data
            extra = extra_data[itracer].extra
            if amr:
                extra.update({"template_values": jnp.stack([extra.pop(map_name) for map_name in regression_maps[itracer]], axis=-1)})
                for map in set(all_regression_maps) - set(regression_maps[itracer]):
                    del extra[map]  # remove maps not used for this tracer to save memory
            # Isolate weights and set WEIGHT_FKP to 0
            extra_data[itracer] = extra_data[itracer].clone(
                extra=extra | {"WEIGHT_FKP": jnp.zeros_like(extra["WEIGHT_FKP"])}, weights=_safe_divide(extra_data[itracer].weights, extra["WEIGHT_FKP"])
            )
            # Concatenate to first data catalog for this tracer
            all_data[0][itracer] = ParticleField.concatenate([all_data[0][itracer], extra_data[itracer]], local=True)
        del extra, extra_randoms, extra_data

        if jax.process_index() == 0: logger.info("Catalogs ready, starting preparation...")

        # Prepare arguments for the window computation function
        ric_argss = []
        for itracer, (data_tracer, randoms_tracer) in enumerate(zip(zip(*all_data, strict=True), zip(*all_randoms, strict=True), strict=True)):
            ric_argss.append(prepare_RIC(data=data_tracer, randoms=randoms_tracer, regions=ric_regions[itracer], n_bins=ric_nbins[itracer], apply_to="randoms"))
        ric_argss = tuple(ric_argss)

        if amr:
            extra_effects = "RIC+AMR"
            amr_argss = []
            for itracer, (data_tracer, randoms_tracer) in enumerate(zip(zip(*all_data, strict=True), zip(*all_randoms, strict=True), strict=True)):
                logger.info(f"{itracer} -- {amr_regions_zranges[itracer]}")
                amr_argss.append(prepare_AMR(data=data_tracer, randoms=randoms_tracer, regions_zranges=amr_regions_zranges[itracer], apply_to="randoms"))
            amr_argss = tuple(amr_argss)

            def delete_template_values(particles):
                extra = particles.extra
                del extra["template_values"]
                return particles.clone(extra=extra)

            all_data = [[delete_template_values(particles) for particles in data_region] for data_region in all_data]
            all_randoms = [[delete_template_values(particles) for particles in randoms_region] for randoms_region in all_randoms]
        else:
            extra_effects = "RIC"
            amr_argss = None

        # Turn into FKP fields
        fkp_fields = [
            [FKPField(data=d, randoms=r, attrs=mattrs) for d, r in zip(data_region, randoms_region, strict=True)]
            for data_region, randoms_region in zip(all_data, all_randoms, strict=True)
        ]

        del all_data, all_randoms

        if optimal_weights is None:
            if jax.process_index() == 0:
                logger.info("Using FKP weights, computing window for all ells at once.")
            # Using FKP weights which are symetrical, so this remains an autocorr
            binner = BinMesh2SpectrumPoles(mattrs, edges=spectra[0].get(0).edges("k"), ells=ellsout)  # TODO: check edges are ok
            # get norms from input spectrum
            fkp_norms = [jnp.concatenate([spectrum.get(ell).values("norm") for ell in ellsout], axis=0) for spectrum in spectra]

            # Renormalize data and randoms to input spectrum
            for iregion, (spectrum, fkps) in enumerate(zip(spectra, fkp_fields, strict=True)):
                # autocorrelation and FKP: component [0, 0]
                # autocorrelation and FKP: component [0, itracer]
                for itracer, fkp in enumerate(fkps):
                    # FIXME I think min(itracer, ntracers - 1) always equal to itracer ??
                    alphad = spectrum.get(0).attrs["wsum_data"][0][min(itracer, ntracers - 1)] / (fkp.data.weights * fkp.data.extra["WEIGHT_FKP"]).sum()
                    alphar = spectrum.get(0).attrs["wsum_randoms"][0][min(itracer, ntracers - 1)] / (fkp.randoms.weights * fkp.randoms.extra["WEIGHT_FKP"]).sum()
                    fkp_fields[iregion][itracer] = fkp.clone(
                        data=fkp.data.clone(weights=fkp.data.weights * alphad),
                        randoms=fkp.randoms.clone(weights=fkp.randoms.weights * alphar),
                    )

            # Needed list of lists for mutability before. Now switch to list of tuples to respect FM input signature
            fkp_fields = [tuple(fkp_region) for fkp_region in fkp_fields]

            ## FM based computations
            windows = {}

            # Shared window FM arguments
            window_fm_kw = {
                "mock_survey": mock_survey_catalog,
                "theory": theory,
                "nreal": n_realizations,
                "seeds": seeds,
                "batch_size": batch_size,
                "mock_survey_args": (*fkp_fields,),
                "static_argnames": ["los", "unitary_amplitude", "estimator_weights"],
                "tmpdir": None,  # No temporary output
                "survey_names": [f"{srz[0]}_{srz[1][0]}-{srz[1][1]}" for srz in spectrum_regions_zranges],
            }

            mock_survey_kwargs = {
                "los": los,
                "unitary_amplitude": unitary_amplitude,
                "nam_args": None,
                "fkp_norms": fkp_norms,
                "binner": binner,
                "estimator_weights": "WEIGHT_FKP",
                "data_regions": tuple(ric_arg.data_regions for ric_arg in ric_argss),
                "randoms_regions": tuple(ric_arg.randoms_regions for ric_arg in ric_argss),
            }

            if geo:
                if jax.process_index() == 0: logger.info("Computing geometry window with desiwinds...")
                _, windows["geometry"] = get_window_spikes(
                    **window_fm_kw,
                    mock_survey_kwargs=mock_survey_kwargs | {"ric_args": None, "amr_args": None, "data_regions": None, "randoms_regions": None},
                )

            if ric:
                if jax.process_index() == 0:
                    logger.info("Computing total window (%s) with desiwinds...", extra_effects)
                _, windows[extra_effects] = get_window_spikes(
                    **window_fm_kw,
                    mock_survey_kwargs=mock_survey_kwargs | {"ric_args": ric_argss, "amr_args": amr_argss},
                )

            if jax.process_index() == 0: logger.info("desiwinds window computation finished.")

            for effect in windows:
                windows[effect] = {region_zrange: [windows[effect][ireal][idx] for ireal in range(n_realizations)] for idx, region_zrange in enumerate(spectrum_regions_zranges)}

        else:
            # Optimal weights: non symmetrical, so need to compute "cross-correlation" (same tracer, different weights) + not the same for all ells
            # If actual cross-spectra, need to compute A_w1 x B_w2 and A_w2 x B_w1 -> double the amount of spectra as well (and for each ell)
            # Proceed ell per ell and sum the windows at the end
            if jax.process_index() == 0: logger.info("Using optimal weights, computing windows for each ell separately.")

            # Double up RIC/AMR arguments even for auto-spectra to simplify the logic in the FM call
            if len(ric_argss) == 1:
                ric_argss = ric_argss * 2
                if amr:
                    amr_argss = amr_argss * 2

            # Prepare optimal weigts generators with same structure as fkp_fields
            optimal_weights_data = [
                lambda ell, fkp_region=fkp_region: optimal_weights(  # use default to avoid late binding in the loop
                    ell,
                    [({column: fkp_field.data.extra[column] for column in ["Z", *columns_optimal_weights]}
                        | {"INDWEIGHT": fkp_field.data.weights * fkp_field.data.extra["WEIGHT_FKP"]})
                        for fkp_field in fkp_region],
                )
                for fkp_region in fkp_fields
            ]
            optimal_weights_randoms = [
                lambda ell, fkp_region=fkp_region: optimal_weights(  # use default to avoid late binding in the loop
                    ell,
                    [({column: fkp_field.randoms.extra[column] for column in ["Z", *columns_optimal_weights]}
                     | {"INDWEIGHT": fkp_field.randoms.weights * fkp_field.randoms.extra["WEIGHT_FKP"]})
                        for fkp_field in fkp_region],
                ) for fkp_region in fkp_fields
            ]

            def _attach_weights(fkp_region: tuple[FKPField], data_w1, data_w2, randoms_w1, randoms_w2):
                """
                Attach optimal weights to FKP fields in fields `'optimal_1'` and `'optimal_2'` for data and randoms.

                If the input fkp_region contains one tracer, both optimal weights are attached to the same tracer and the field is duplicated to have a length 2 output.
                If the input fkp_region contains two tracers, the first optimal weight is attached to the first tracer and the second optimal weight to the second tracer.
                """
                # These weights also contain real weights and FKP weights ; need to remove the real weights to isolate the "estimator weights" to apply at computation time in the FM
                if len(fkp_region) == 1:
                    return (
                        fkp_region[0].clone(
                            data=fkp_region[0].data.clone(
                                extra=fkp_region[0].data.extra
                                | {"weight_optimal_1": _safe_divide(data_w1, fkp_region[0].data.weights), "weight_optimal_2": _safe_divide(data_w2, fkp_region[0].data.weights)}
                            ),
                            randoms=fkp_region[0].randoms.clone(
                                extra=fkp_region[0].randoms.extra
                                | {
                                    "weight_optimal_1": _safe_divide(randoms_w1, fkp_region[0].randoms.weights),
                                    "weight_optimal_2": _safe_divide(randoms_w2, fkp_region[0].randoms.weights),
                                }
                            ),
                        ),
                    ) * 2
                elif len(fkp_region) == 2:
                    return (
                        fkp_region[0].clone(
                            data=fkp_region[0].data.clone(extra=fkp_region[0].data.extra | {"weight_optimal_1": _safe_divide(data_w1, fkp_region[0].data.weights)}),
                            randoms=fkp_region[0].randoms.clone(extra=fkp_region[0].randoms.extra | {"weight_optimal_1": _safe_divide(randoms_w1, fkp_region[0].randoms.weights)}),
                        ),
                        fkp_region[1].clone(
                            data=fkp_region[1].data.clone(extra=fkp_region[1].data.extra | {"weight_optimal_2": _safe_divide(data_w2, fkp_region[1].data.weights)}),
                            randoms=fkp_region[1].randoms.clone(extra=fkp_region[1].randoms.extra | {"weight_optimal_2": _safe_divide(randoms_w2, fkp_region[1].randoms.weights)}),
                        ),
                    )
                else:
                    raise ValueError(f"Unexpected number of tracers in fkp_region: {len(fkp_region)}")

            windows = {}
            if geo:
                windows["geometry"] = {ell: [] for ell in ellsout}
            if ric:
                windows[extra_effects] = {ell: [] for ell in ellsout}

            for ell in ellsout:
                binner = BinMesh2SpectrumPoles(fkp_fields[0][0].attrs, edges=spectra[0].get(ell).edges("k"), ells=[ell])  # TODO: check edges are ok
                # Recover FKP normalization for each region and this ell only from input spectrum
                fkp_norms = [jnp.concatenate([spectrum.get(ill).values("norm") for ill in [ell]], axis=0) for spectrum in spectra]

                for iopt, optweights in enumerate(zip(*[owd(ell) for owd in optimal_weights_data], *[owr(ell) for owr in optimal_weights_randoms], strict=True)):
                    # This loop will iterate once for auto and twice for cross (A_w1 x B_w2 and A_w2 x B_w1)
                    # _fkp_fields lists length 2 tuples (that might be one field duplicated for auto) with the optimal weights attached in the extra
                    _fkp_fields = [_attach_weights(fkp_region, *optweights[iregion], *optweights[nregion + iregion]) for iregion, fkp_region in enumerate(fkp_fields)]

                    # Renormalize data and randoms to input spectrum
                    for iregion, (spectrum, (fkp1, fkp2)) in enumerate(zip(spectra, _fkp_fields, strict=True)):
                        # OQE auto AxA: [[w1sum_A, w2sum_A]] | OQE cross AxB: [[w1sum_A, w2sum_B], [w2sum_A, w1sum_B]]
                        alphad1 = spectrum.get(ell).attrs["wsum_data"][iopt][0] / fkp1.data.clone(weights=fkp1.data.weights * fkp1.data.extra["weight_optimal_1"]).weights.sum()
                        alphad2 = spectrum.get(ell).attrs["wsum_data"][iopt][1] / fkp2.data.clone(weights=fkp2.data.weights * fkp2.data.extra["weight_optimal_2"]).weights.sum()
                        alphar1 = (
                            spectrum.get(ell).attrs["wsum_randoms"][iopt][0]
                            / fkp1.randoms.clone(weights=fkp1.randoms.weights * fkp1.randoms.extra["weight_optimal_1"]).weights.sum()
                        )
                        alphar2 = (
                            spectrum.get(ell).attrs["wsum_randoms"][iopt][1]
                            / fkp2.randoms.clone(weights=fkp2.randoms.weights * fkp2.randoms.extra["weight_optimal_2"]).weights.sum()
                        )
                        # Renormalize the optimal weights so that the final mesh is properly normalized
                        _fkp_fields[iregion] = (
                            fkp1.clone(
                                data=fkp1.data.clone(weights=fkp1.data.weights * alphad1),
                                randoms=fkp1.randoms.clone(weights=fkp1.randoms.weights * alphar1),
                            ),
                            fkp2.clone(
                                data=fkp2.data.clone(weights=fkp2.data.weights * alphad2),
                                randoms=fkp2.randoms.clone(weights=fkp2.randoms.weights * alphar2),
                            ),
                        )

                    # Shared window FM arguments
                    window_fm_kw = {
                        "mock_survey": mock_survey_catalog,
                        "theory": theory,
                        "nreal": n_realizations,
                        "seeds": seeds,
                        "batch_size": batch_size,
                        "mock_survey_args": _fkp_fields,  # list of tuples of FKP fields with different norm and optimal weights
                        "static_argnames": ["los", "unitary_amplitude", "estimator_weights"],
                        "tmpdir": None,  # No temporary output
                        "survey_names": [f"{srz[0]}_{srz[1][0]}-{srz[1][1]}" for srz in spectrum_regions_zranges],
                    }

                    mock_survey_kwargs = {
                        "los": los,
                        "unitary_amplitude": unitary_amplitude,
                        "nam_args": None,
                        "fkp_norms": [jnp.ones_like(fkp_norm) for fkp_norm in fkp_norms],  # will renormalize manually when summing cross corr windows
                        "binner": binner,  # one ell only
                        "estimator_weights": ("weight_optimal_1", "weight_optimal_2"),
                        "data_regions": tuple(ric_arg.data_regions for ric_arg in ric_argss),
                        "randoms_regions": tuple(ric_arg.randoms_regions for ric_arg in ric_argss),
                    }

                    if geo:
                        if jax.process_index() == 0:
                            logger.info("Computing geometry window for ell=%i, optimal weights combination %i with desiwinds...", ell, iopt)
                        _, _windows_fm_geo = get_window_spikes(**window_fm_kw, mock_survey_kwargs=mock_survey_kwargs | {"ric_args": None, "amr_args": None, "data_regions": None, "randoms_regions": None})
                        windows["geometry"][ell].append(_windows_fm_geo)

                    if ric:
                        if jax.process_index() == 0:
                            logger.info("Computing %s window for ell=%i, optimal weights combination %i with desiwinds...", extra_effects, ell, iopt)
                        _, _windows_fm = get_window_spikes(**window_fm_kw, mock_survey_kwargs=mock_survey_kwargs | {"ric_args": ric_argss, "amr_args": amr_argss})
                        windows[extra_effects][ell].append(_windows_fm)

                # Sum over weights iteration (ie for cross AxB, sum A_w1 x B_w2 and A_w2 x B_w1) to get the final window for this ell before summing over ells
                # Also renormalize now
                for effect in windows:
                    windows[effect][ell] = jax.tree.map(
                        (lambda *windows_and_norm: windows_and_norm[0].clone(value=np.sum([wd.value() for wd in windows_and_norm[:-1]], axis=0) / windows_and_norm[-1][..., None])),
                        *windows[effect][ell],
                        [fkp_norms] * n_realizations,
                        is_leaf=lambda x: isinstance(x, types.WindowMatrix),
                    )

            if jax.process_index() == 0: logger.info("desiwinds window computation finished.")

            # For each region, sum the windows over ells and apply control variate
            def _combine_ells(windows):
                observables = [window.observable for window in windows]
                observable = types.join(observables)
                value = np.concatenate([window.value() for window in windows], axis=0)
                return windows[0].clone(value=value, observable=observable)  # join multipoles

            for effect in windows:
                windows[effect] = {
                    region_zrange: [_combine_ells([windows[effect][ell][ireal][idx] for ell in ellsout]) for ireal in range(n_realizations)]
                    for idx, region_zrange in enumerate(spectrum_regions_zranges)
                }

        return windows


def run_preliminary_fit_mesh2_spectrum(data: types.Mesh2SpectrumPoles, window: types.WindowMatrix, select: dict=None, theory: str='rept', update_params=None, out: types.Mesh2SpectrumPoles=None):
    """
    Compute a smooth theory spectrum to assume when building the covariance.

    Parameters
    ----------
    data : Mesh2SpectrumPoles or None
        Measured spectrum multipoles used to build the covariance and (optionally)
        to set priors / initialize the fit. If None, the function will still
        construct an analytic covariance from `window` but cannot use data-driven
        priors.
    window : WindowMatrix
        Window matrix describing mode-coupling of the estimator. The window's
        observable axes are matched to the `data` before fitting.
    select : dict, optional
        If provided, a selection is applied to `data` via `data.select(**select)`
        prior to fitting (e.g. to restrict k-ranges or multipoles).
    theory : str, optional
        Theory to use in the fit, one of ['rept', 'kaiser'].
    out : Mesh2SpectrumPoles, optional
        If provided, returns a clone of these power spectrum multipoles with best fit theory values.

    Returns
    -------
    out : Mesh2SpectrumPoles
    """
    # Import spectrum computation and covariance tools from jaxpower
    from jaxpower import MeshAttrs, compute_spectrum2_covariance
    # Select k-range for fitting (avoid very low k)
    smooth = data.select(k=(0.001, 10.))

    # Create mesh attributes from input data
    mattrs = MeshAttrs(**{name: data.attrs[name] for name in ['boxsize', 'boxcenter', 'meshsize']})
    # Compute Gaussian covariance (assuming Gaussian density field)
    covariance = compute_spectrum2_covariance(mattrs, data)  # Gaussian, diagonal covariance

    # Import clustering theory classes from desilike
    from desilike.theories.galaxy_clustering import FixedSpectrum2Template, KaiserTracerSpectrum2Poles, REPTVelocileptorsTracerSpectrum2Poles, DampedBAOWigglesTracerSpectrum2Poles
    from desilike.observables.galaxy_clustering import Spectrum2PolesObservable
    from desilike.likelihoods import ObservablesGaussianLikelihood
    from desilike.profilers import Profiler, Minuit
    from desilike import compile, get_params, Posterior

    # Select theory model (Kaiser or REPT with velocileptors)
    theory_str = theory
    Theory = {'recon': DampedBAOWigglesTracerSpectrum2Poles, 'rept': REPTVelocileptorsTracerSpectrum2Poles, 'kaiser': KaiserTracerSpectrum2Poles}[theory_str]

    # Apply selection to data (restrict to fitting range)
    # start at kmin = 0.1 for BAO, to constrain broadband terms
    select = select or {'k': (0.02, 10.)}
    data = data.select(**select)
    # Match window to data range
    window = window.at.observable.match(data)
    # Restrict window theory to coverage of measurement
    window = window.at.theory.select(k=(0.001, 1.2 * next(iter(data)).coords('k').max()))
    # Match covariance to data range
    covariance = covariance.at.observable.match(data)
    # Extract effective redshift from window
    z = window.observable.get(ells=0).attrs['zeff']

    # Create fiducial theory template at measurement redshift
    template = FixedSpectrum2Template(fiducial='DESI', z=z)
    # Instantiate theory model with template
    theory = Theory(template=template)
    if 'recon' in theory_str:
        theory.update(mode=np.asarray(data.attrs['recon_mode']).flat[0], smoothing_radius=np.asarray(data.attrs['recon_smoothing_radius']).flat[0])
        params = get_params(theory)
        # To avoid spike at low k
        for param in params.select(basename=['al*_[-4:0]']):
            param.update(fixed=True)
        for ell in theory.ells:
            for ik in range(4):
                params.set(params['al0_0'].clone(name=f'al{ell:d}_{ik}'))
        theory.update(params=params)
    # Create observable: combines data, theory, and window function
    observable = Spectrum2PolesObservable(data=data, window=window, theory=theory)
    # Create likelihood: Gaussian likelihood with computed covariance
    likelihood = ObservablesGaussianLikelihood(observable, covariance=covariance.value())
    # Fix specified parameters
    params = get_params(likelihood)
    if update_params is not None:
        for name, update in update_params.items():
            params[name].update(**update)

    # Minimize likelihood to get best-fit theory
    posterior = compile(Posterior(likelihood))    
    profiler = Profiler(posterior, kernel=Minuit(), rng=42)
    profiles = profiler.maximize()
    # Get best-fit parameters
    if profiler.mpicomm.rank == 0:
        print(profiles.to_stats(tablefmt='pretty'))
    params = profiles.choice(index='argmax', squeeze=True).select(input=True).best
    if out is None:
        # Build smooth theory spectrum from best-fit parameters
        poles = []
        for ill, ell in enumerate(theory.ells):
            if ell in smooth.ells:
                # Use original smooth data as template
                pole = smooth.get(ells=ell)
            else:
                # Create new pole with zero shot noise for missing multipoles
                pole = smooth.get(ells=0).clone(meta={"ell": ell})
                if ell != 0:
                    # Zero out shot noise for higher multipoles (only monopole has shot noise)
                    pole = pole.clone(num_shotnoise=np.zeros_like(pole.values("num_shotnoise")))
            # Evaluate theory at k-values from data
            theory.update(k=pole.coords("k"))
            value = compile(theory)(params)[ill]
            pole = pole.clone(value=value)
            poles.append(pole)
        # Build spectrum object from theory poles
        smooth = types.Mesh2SpectrumPoles(poles, attrs=smooth.attrs)
    else:
        # Use provided output structure
        value = []
        for label, pole in out.items(level=1):
            # Evaluate theory at k-values
            theory.update(k=pole.coords('k'))
            value.append(compile(theory)(params)[theory.ells.index(label['ells'])])
        smooth = out.clone(value=value)
    return smooth


def compute_covariance_mesh2_spectrum(*get_data_randoms, theory=None, fields=None, mattrs=None):
    r"""
    Compute the 2-point spectrum covariance with :mod:`jaxpower`.

    Parameters
    ----------
    get_data_randoms : callables
        Functions that return dict of 'data' and 'randoms' catalogs.
        See :func:`prepare_jaxpower_particles` for details.
    theory : Mesh2SpectrumPoles
        Theory 2-point spectrum multipoles.
    fields : tuple, list, optional
        Field names.
    mattrs : dict, optional
        Mesh attributes to define the :class:`jaxpower.ParticleField` objects,
        'boxsize', 'meshsize' or 'cellsize', 'boxcenter'. If ``None``, default attributes are used.

    Returns
    -------
    covariance : CovarianceMatrix
        The computed 2-point spectrum covariance.
    """
    # Import covariance and window computation tools from jaxpower
    from jaxpower import create_sharding_mesh, compute_fkp2_covariance_window, interpolate_window_function, compute_spectrum2_covariance, FKPField
    # Use FFTLog for reliable correlation-to-spectrum conversion
    fftlog = True
    # Use default fields (1, 2, ...) if not provided
    if fields is None:
        fields = list(range(1, 1 + len(get_data_randoms)))

    results = {}
    # Set up distributed computation mesh
    with create_sharding_mesh(meshsize=mattrs.get('meshsize', None)):
        # Load and prepare particles
        all_particles = prepare_jaxpower_particles(*get_data_randoms, mattrs=mattrs, add_randoms=['IDS'])
        # Create FKP fields for covariance window computation
        all_fkp = [FKPField(particles['data'], particles['randoms']) for particles in all_particles]
        mattrs = all_fkp[0].attrs
        # Set correlation binning parameters (finer than spectrum binning)
        kw = dict(edges={'step': mattrs.cellsize.min()}, basis='bessel') if fftlog else dict(edges={})
        # Add fields for cross-covariance and random splitting seed
        kw.update(los='local', fields=fields, split=[(42, fkp.randoms.extra['IDS']) for fkp in all_fkp])
        # Mesh painting parameters: TSC with interlacing
        kw_paint = dict(resampler='tsc', interlacing=3, compensate=True)
        # Compute covariance window function (correlation in configuration space)
        windows = compute_fkp2_covariance_window(all_fkp, **kw, **kw_paint)
        #if jax.process_index() == 0: windows.write(f'_tests/window_correlation.h5')
        if fftlog:
            # Very robust to this choice of FFTLog grid
            # Use logarithmic s-grid for interpolation
            coords = np.logspace(-2, 8, 8 * 1024)
            # Interpolate window functions to fine s-grid
            windows = windows.map(lambda window: interpolate_window_function(window, coords=coords), level=1)
        # Store raw correlation windows for diagnostics
        results['window_covariance_mesh2_correlation'] = windows

    # Convert correlation to power spectrum covariance matrix via FFTLog
    covariance = compute_spectrum2_covariance(windows, theory, flags=['smooth'] + (['fftlog'] if fftlog else []))
    # Store in results dict
    results['raw'] = covariance
    return results


def compute_rotation_mesh2_spectrum(window: types.WindowMatrix, covariance: types.CovarianceMatrix, Minit: str='momt',
                                    data: types.Mesh2SpectrumPoles=None, theory: types.Mesh2SpectrumPoles=None, select: dict=None):
    """
    Compute the rotation to make the window matrix more diagonal.

    Parameters
    ----------
    window : WindowMatrix
        Window matrix.
    covariance : CovarianceMatrix
        Covariance of the measured spectrum.
    Minit : {'momt', ...}, optional
        Initialization method passed to rotation.setup(Minit=...). Defaults to 'momt'.
    data : Mesh2SpectrumPoles or None, optional
        Measured spectrum used to set priors for the rotation (if available).
    theory : Mesh2SpectrumPoles or None, optional
        Theory spectrum used together with `data` when setting priors.

    Returns
    -------
    rotation : WindowRotationSpectrum2
    """
    # Import rotation matrix computation from jaxpower
    from jaxpower import WindowRotationSpectrum2
    # Extract observable from window or data
    observable = window.observable
    if data is not None:
        # Use data as observable instead of window
        if select is not None:
            data = data.select(**select)
        observable = data
    # Match window observable to target observable (reorder/subset as needed)
    window = window.at.observable.match(observable)
    if theory is not None:
        def interpolate_pole(ref, pole):
            # Interpolate theory to match reference k-values
            return ref.clone(value=np.interp(ref.coords('k'), pole.coords('k'), pole.value()))

        # Interpolate theory to window theory k-values
        theory = window.theory.map(lambda pole, label: interpolate_pole(pole, theory.get(ells=label['ells'])), input_label=True, level=1)
    # Match covariance observable to target observable
    covariance = covariance.at.observable.match(observable)
    # Create rotation matrix object
    rotation = WindowRotationSpectrum2(window=window, covariance=covariance, xpivot=0.1)
    # Set up rotation matrix (initialize using 'momt' method: moment-based initialization)
    rotation.setup(Minit=Minit)
    # Fit rotation matrix to data (if provided)
    rotation.fit()
    if rotation.with_momt and data is not None:
        # To set up priors for rotation parameters from data
        rotation.set_prior(data=data, theory=theory)
    return rotation


def compute_box_mesh2_spectrum(*get_data, ells=(0, 2, 4), edges=None, los='z', cache=None, mattrs=None):
    r"""
    Compute the 2-point spectrum multipoles for a cubic box using :mod:`jaxpower`.

    Parameters
    ----------
    get_data : callables
        Functions that return dict of 'data' (optionally 'shifted') catalogs.
        See :func:`prepare_jaxpower_particles` for details.
    mattrs : dict, optional
        Mesh attributes to define the :class:`jaxpower.ParticleField` objects,
        'boxsize', 'meshsize' or 'cellsize', 'boxcenter'.
    ells : list of int, optional
        List of multipole moments to compute. Default is (0, 2, 4).
    edges : dict, optional
        Edges for the binning; array or dictionary with keys 'start' (minimum :math:`k`), 'stop' (maximum :math:`k`), 'step' (:math:`\Delta k`).
        If ``None``, default step of :math:`0.001 h/\mathrm{Mpc}` is used.
        See :class:`jaxpower.BinMesh2SpectrumPoles` for details.
    los : {'x', 'y', 'z', array-like}, optional
        Line-of-sight direction. If 'x', 'y', 'z' use fixed axes, or provide a 3-vector.
    cache : dict, optional
        Cache to store binning class (can be reused if ``meshsize`` and ``boxsize`` are the same).
        If ``None``, a new cache is created.

    Returns
    -------
    spectrum : Mesh2SpectrumPoles
        The computed 2-point spectrum multipoles.
    """
    # Import tools for periodic box power spectrum computation
    from jaxpower import (create_sharding_mesh, FKPField, compute_fkp2_shotnoise, compute_box2_normalization, BinMesh2SpectrumPoles, compute_mesh2_spectrum, compute_fkp2_shotnoise)

    mattrs = mattrs or {}
    # Set up distributed computation across JAX devices
    with create_sharding_mesh(meshsize=mattrs.get('meshsize', None)):
        # Load and prepare particles (data + optional shifted for RSD distortions)
        all_particles = prepare_jaxpower_particles(*get_data, mattrs=mattrs)
        # Initialize or retrieve cached binning object
        if cache is None: cache = {}
        # Set default k-space binning step (0.001 h/Mpc)
        if edges is None: edges = {'step': 0.001}
        # Gather particle attributes (weights, mesh info)
        attrs = _get_jaxpower_attrs(*all_particles)
        # Store line-of-sight direction
        attrs.update(los=los)
        mattrs = all_particles[0]['data'].attrs

        # Define the binner for k-space binning
        key = 'bin_mesh2_spectrum_{}'.format('_'.join(map(str, ells)))
        bin = cache.get(key, None)
        # Create new binning if not cached or mesh changed
        if bin is None or not np.all(bin.mattrs.meshsize == mattrs.meshsize) or not np.allclose(bin.mattrs.boxsize, mattrs.boxsize):
            bin = BinMesh2SpectrumPoles(mattrs, edges=edges, ells=ells)
        # Store binning in cache
        cache.setdefault(key, bin)

        # Computing normalization for periodic box (simpler than survey: no randoms)
        all_data = [particles['data'] for particles in all_particles]
        norm = compute_box2_normalization(*all_data, bin=bin)

        # Computing shot noise from shifted or data catalogs
        # shifted if reconstruction
        all_fkp = [FKPField(particles['data'], particles['shifted']) if particles.get('shifted', None) is not None else particles['data'] for particles in all_particles]
        # Free memory
        del all_particles
        num_shotnoise = compute_fkp2_shotnoise(*all_fkp, bin=bin, fields=None)

        # Paint particles on mesh grid
        kw = dict(resampler='tsc', interlacing=3, compensate=True)
        # out='real' to save memory (store as real arrays)
        all_mesh = []
        for fkp in all_fkp:
            # Paint particle density on mesh
            mesh = fkp.paint(**kw, out='real')
            all_mesh.append(mesh - mesh.mean())
        # Free FKP field memory
        del all_fkp
        # JIT the mesh-based spectrum computation; helps with memory footprint
        jitted_compute_mesh2_spectrum = jax.jit(compute_mesh2_spectrum, static_argnames=['los'])
        #jitted_compute_mesh2_spectrum = compute_mesh2_spectrum
        # Compute power spectrum from painted meshes via FFT
        spectrum = jitted_compute_mesh2_spectrum(*all_mesh, bin=bin, los=los)
        # Attach normalization and shot noise
        spectrum = spectrum.clone(norm=norm, num_shotnoise=num_shotnoise)
        # Propagate attributes to output
        spectrum = spectrum.map(lambda pole: pole.clone(attrs=attrs))
        spectrum = spectrum.clone(attrs=attrs)
        # Wait for computation to complete
        jax.block_until_ready(spectrum)
        if jax.process_index() == 0:
            logger.info('Mesh-based computation finished')
    return spectrum


def compute_window_box_mesh2_spectrum(spectrum: types.Mesh2SpectrumPoles, zsnap: float=None):
    r"""
    Compute the 2-point spectrum window for a box (i.e., binning window) with :mod:`jaxpower`.

    Parameters
    ----------
    spectrum : Mesh2SpectrumPoles
        Measured 2-point spectrum multipoles.

    Returns
    -------
    window : WindowMatrix
        The computed 2-point spectrum window.
    """
    # Compute binning window for periodic box power spectrum
    from jaxpower import create_sharding_mesh, MeshAttrs, BinMesh2SpectrumPoles, compute_mesh2_spectrum_window

    # Extract mesh attributes and multipoles from input spectrum
    mattrs = {name: spectrum.attrs[name] for name in ['boxsize', 'boxcenter', 'meshsize']}
    los = spectrum.attrs['los']
    ells = spectrum.ells
    pole = spectrum.get(0)
    # Set up distributed computation
    with create_sharding_mesh(meshsize=mattrs.get('meshsize', None)):
        mattrs = MeshAttrs(**mattrs)
        # Create binning from spectrum edges
        bin = BinMesh2SpectrumPoles(mattrs, edges=pole.edges('k'), ells=ells)
        #edgesin = np.linspace(bin.edges.min(), bin.edges.max(), 2 * (len(bin.edges) - 1))
        # Use input spectrum bins as theory bins
        edgesin = bin.edges
        # For box, window is just the binning matrix (no mode coupling from survey effects)
        window = compute_mesh2_spectrum_window(mattrs, edgesin=edgesin, ellsin=ells, los=los, bin=bin)
        observable = window.observable
        # Attach redshift/snapshot information if provided
        if zsnap is not None:
            observable = observable.map(lambda pole: pole.clone(attrs=pole.attrs | dict(zeff=zsnap, zsnap=zsnap)))
        window = window.clone(observable=observable)
    return window


def compute_covariance_box_mesh2_spectrum(theory: types.Mesh2SpectrumPoles=None, mattrs=None):
    r"""
    Compute the 2-point spectrum covariance for a box with :mod:`jaxpower`.

    Parameters
    ----------
    theory : Mesh2SpectrumPoles, optional
        Theory spectrum used together with `spectrum` when setting priors.
    mattrs : dict, optional
        Mesh attributes to define the :class:`jaxpower.ParticleField` objects,
        'boxsize', 'meshsize' or 'cellsize', 'boxcenter'. If ``None``, default attributes are used.

    Returns
    -------
    covarance : CovarianceMatrix
        The computed 2-point spectrum covariance.
    """
    # Compute Gaussian covariance for periodic box power spectrum
    from jaxpower import create_sharding_mesh, MeshAttrs, compute_spectrum2_covariance
    # Add zero shot noise to theory for covariance computation
    theory_sn = theory.map(lambda pole: pole.clone(num_shotnoise=pole.values('num_shotnoise') * 0.), level=2)
    mattrs = mattrs or {}
    # Set up distributed computation
    with create_sharding_mesh(meshsize=mattrs.get('meshsize', None)):
        mattrs = MeshAttrs(**mattrs)
        # Compute Gaussian, diagonal covariance
        covariance = compute_spectrum2_covariance(mattrs, theory_sn)  # Gaussian, diagonal covariance

        # Update label names to match observable structure
        fields = covariance.observable.fields
        # Create observable tree with proper labels
        observable = types.ObservableTree(list(covariance.observable), observables=['spectrum2'] * len(fields), tracers=fields)
        covariance = covariance.clone(observable=observable)
    return covariance
