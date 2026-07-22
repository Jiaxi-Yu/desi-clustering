"""
Fourier-space 3-point clustering measurements.

Main functions
--------------
* `compute_mesh3_spectrum`: Main bispectrum measurement backend.
* `compute_window_mesh3_spectrum`: Compute bispectrum window.
* `run_preliminary_fit_mesh3_spectrum`: Fit bias parameters used in the covariance theory.
* `compute_covariance_mesh3_spectrum`: Estimate the joint P + B Fourier-space covariance.
* `compute_box_mesh3_spectrum`: Measure the bispectrum in periodic boxes.
* `compute_covariance_box_mesh3_spectrum`: Estimate the joint P + B covariance for periodic boxes.
"""

import time
import logging
import functools
import operator
import itertools

import numpy as np
import jax
from jax import numpy as jnp
from jax.sharding import PartitionSpec as P
import lsstypes as types

from .tools import compute_fkp_effective_redshift
from .spectrum2_tools import prepare_jaxpower_particles, _get_jaxpower_attrs


logger = logging.getLogger('spectrum3')


def compute_mesh3_spectrum_close_pair_correction(*get_data_randoms, spectrum, auw=None, cut=None, **kwargs):
    """
    Compute and apply close-pair corrections to 3-point spectrum.

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
    Returns
    -------
    spectrum : Mesh3SpectrumPoles
    """

    from cucount.jax import create_sharding_mesh, BinAttrs
    from .correlation2_tools import prepare_cucount_particles

    with create_sharding_mesh() as sharding_mesh:
        if callable(get_data_randoms[0]):
            all_particles = prepare_cucount_particles(*get_data_randoms, concatenate=True)
            if jax.process_index() == 0: logger.info('All particles on the device')

    results = {}
    results['raw'] = spectrum
    corrections = {'auw': auw, 'cut': cut}
    for name in corrections:
        if corrections[name] is not None:
            correction = _compute_mesh3_spectrum_close_pair_correction(all_particles, ells=spectrum.ells, **{name: corrections[name]})
            #types.ObservableTree(list(correction.values()), counts=list(correction.keys())).write('correction.h5')
            results[name] = _apply_mesh3_spectrum_close_pair_correction(spectrum, correction)
    return results


def _compute_mesh3_spectrum_close_pair_correction(all_particles, edges=None, ells: list=None, auw=None, cut=None):
    """Compute and apply close-pair corrections."""

    from cucount.jax import BinAttrs
    from lsstypes.types import convert_ells

    if edges is None:
        # Fine linear bins where the close-pair signal lives, then ~5% log-spaced bins:
        # with the bin-integrated Bessel kernel (edges provided -> method='exact' in
        # jaxpower's _pole_transform), wide bins at large s low-pass the noisy large-s
        # counts (third-particle digitization lumps) instead of transmitting them to high k.
        edges = np.concatenate([np.linspace(1e-3, 200., 76), np.geomspace(200., 8000., 76)[1:]])
        #edges = np.linspace(1e-3, 8000., 3001)
    if ells is None:
        ells = [(0, 0, 0), (2, 0, 2)]
    ells = convert_ells(ells, 'sugiyama', 'slepian')
    battrs = [BinAttrs(s=edges, pole=(tuple(np.unique([ell[idim] for ell in ells])), 'firstpoint')) for idim in range(2)]
    from .correlation3_tools import _compute_particle3_correlation_close_pair_correction
    correction = _compute_particle3_correlation_close_pair_correction(all_particles, battrs, auw=auw, cut=cut, veto23=True, normalize_randoms=True)
    for count_name in correction:
        correction[count_name] = correction[count_name].to_basis('sugiyama')
    #types.ObservableTree(list(correction.values()), labels=list(correction.keys())).write('tmp.h5')
    return correction


def _apply_mesh3_spectrum_close_pair_correction(spectrum, correction):
    """Apply additive corrections to a :class:`Mesh3SpectrumPoles`."""
    from jaxpower.particle3 import Particle3CorrelationPole, Particle3CorrelationPoles
    out = spectrum
    value = 0.
    for count_name in correction:
        sign = (-1)**(3 - count_name.count('D'))  # randoms can be R or S
        poles = correction[count_name].ravel()
        value += sign * jnp.array([pole.values('counts') for pole in poles])
    correlation = []
    for ill, ell in enumerate(spectrum.ells):
        correlation.append(Particle3CorrelationPole(s=poles.coords('s'), s_edges=poles.edges('s'), num_raw=value[ill], norm=jnp.ones_like(value[ill]) * spectrum.get(ell).values('norm').mean(), ell=ell))
    correlation = Particle3CorrelationPoles(correlation)
    value = correlation.to_spectrum(spectrum).value()
    return spectrum.clone(value=spectrum.value() + value)


def compute_mesh3_spectrum(*get_data_randoms, mattrs=None, cut=None, auw=None,
                            basis='sugiyama-diagonal', ells=[(0, 0, 0), (2, 0, 2)], edges=None, los='local',
                            buffer_size=0, norm: dict=None, cache=None):
    r"""
    Compute the 3-point spectrum multipoles using mesh-based FKP fields with :mod:`jaxpower`.

    Parameters
    ----------
    get_data_randoms : callables
        Functions that return dict of 'data', 'randoms' (optionally 'shifted') catalogs.
        See :func:`prepare_jaxpower_particles` for details.
    mattrs : dict, optional
        Mesh attributes to define the :class:`jaxpower.ParticleField` objects. If None, default attributes are used.
        See :func:`prepare_jaxpower_particles` for details.
    basis : str, optional
        Basis for the 3-point spectrum computation. Default is 'sugiyama-diagonal'.
    ells : list of tuples, optional
        List of multipole moments to compute. Default is [(0, 0, 0), (2, 0, 2)] (for the sugiyama basis).
    edges : dict, optional
        Edges for the binning; array or dictionary with keys 'start' (minimum :math:`k`), 'stop' (maximum :math:`k`), 'step' (:math:`\Delta k`).
        If ``None``, default step of :math:`0.005 h/\mathrm{Mpc}` is used for the sugiyama basis, :math:`0.01 h/\mathrm{Mpc}` for the scoccimarro basis.
        See :class:`jaxpower.BinMesh3SpectrumPoles` for details.
    los : {'local', 'x', 'y', 'z', array-like}, optional
        Line-of-sight definition. 'local' uses local LOS, 'x', 'y', 'z' use fixed axes, or provide a 3-vector.
    buffer_size : int, optional
        Buffer size when binning; if the binning is multidimensional, increase for faster computation at the cost of memory.
    norm : dict, optional
        Optional arguments for computing normalization.
        Default is ``{'cellsize': 10.}`` (density computed with ``cellsize = 10.``)
    cache : dict, optional
        Cache to store binning class (can be reused if ``meshsize`` and ``boxsize`` are the same).
        If ``None``, a new cache is created.

    Returns
    -------
    spectrum : Mesh3SpectrumPoles
        The computed 3-point spectrum multipoles.
    """
    from jaxpower import (create_sharding_mesh, FKPField, compute_fkp3_normalization, compute_fkp3_shotnoise, BinMesh3SpectrumPoles, compute_mesh3_spectrum)

    mattrs = mattrs or {}
    # Set up distributed computation mesh across JAX devices
    with create_sharding_mesh(meshsize=mattrs.get('meshsize', None)) as sharding_mesh:
        # Load and prepare particle catalogs (data and randoms) with IDS for reproducibility (random splitting based on object IDs)
        all_particles = prepare_jaxpower_particles(*get_data_randoms, mattrs=mattrs, add_randoms=['IDS'])
        # Attributes about the estimation
        attrs = _get_jaxpower_attrs(*all_particles)
        # Set line-of-sight direction in attributes
        attrs.update(los=los)
        # Get mesh attributes from first particle set (same for all)
        mattrs = all_particles[0]['data'].attrs

        # Initialize or retrieve cached binning object
        if cache is None: cache = {}
        # Set default k-space binning step based on basis
        if edges is None: edges = {'step': 0.02 if 'scoccimarro' in basis else 0.005}
        # Set default normalization parameters
        if norm is None: norm = {'cellsize': 10.}
        kw_norm = dict(norm)

        # Check if binning object is cached and still valid for current mesh
        bin = cache.get(f'bin_mesh3_spectrum_{basis}', None)
        if bin is None or not np.all(bin.mattrs.meshsize == mattrs.meshsize) or not np.allclose(bin.mattrs.boxsize, mattrs.boxsize):
            # Create new binning object if not cached or if mesh changed
            bin = BinMesh3SpectrumPoles(mattrs, edges=edges, basis=basis, ells=ells, buffer_size=buffer_size)
        # Store binning object in cache for future use
        cache.setdefault(f'bin_mesh3_spectrum_{basis}', bin)

        # Create FKP fields from data and random catalogs
        all_fkp = [FKPField(particles['data'], particles['randoms']) for particles in all_particles]
        # Compute FKP normalization: integral of n^3(x)
        # Use IDS for process-invariant random splitting
        norm = compute_fkp3_normalization(*all_fkp, bin=bin, split=[(42, fkp.randoms.extra['IDS']) for fkp in all_fkp],
                                          **kw_norm)

        # Create FKP fields for shot noise computation (using shifted catalogs if available)
        all_fkp = [FKPField(particles['data'], particles['shifted'] if particles.get('shifted', None) is not None else particles['randoms']) for particles in all_particles]
        del all_particles
        # Paint FKP fields onto mesh grid with TSC interpolation and interlacing
        kw = dict(resampler='tsc', interlacing=3, compensate=True)
        # Compute shot noise: variance of Fourier modes from random particles
        num_shotnoise = compute_fkp3_shotnoise(*all_fkp, los=los, bin=bin, **kw)

        # Wait for all computations to finish before proceeding
        jax.block_until_ready((norm, num_shotnoise))
        if jax.process_index() == 0:
            logger.info('Normalization and shotnoise computation finished')

        # Galaxy pairs at small angular separation
        results = {}
        corrections = {'auw': auw, 'cut': cut}
        if any(corrections.values()):
            from jaxpower.particle3 import convert_particles
            all_particles = [{'data': convert_particles(fkp.data, weights=fkp.data.weights, exchange_weights=False),
                             'randoms': convert_particles(fkp.randoms, weights=fkp.randoms.weights, exchange_weights=False)} for fkp in all_fkp]
            for name in corrections:
                if corrections[name] is not None:
                    results[name] = _compute_mesh3_spectrum_close_pair_correction(all_particles, ells=bin.ells, **{name: corrections[name]})

        # Paint FKP fields onto mesh grids (stored as real-valued arrays to save memory)
        meshes = [fkp.paint(**kw, out='real') for fkp in all_fkp]
        # Clear FKP fields from memory
        del all_fkp

        # JIT-compile spectrum computation for performance (static_argnames for non-JAX arguments)
        jitted_compute_mesh3_spectrum = jax.jit(compute_mesh3_spectrum, static_argnames=['los'])

        # Compute bispectrum (3-point spectrum) from mesh grids
        spectrum = jitted_compute_mesh3_spectrum(*meshes, los=los, bin=bin)
        # Attach normalization and shot noise to spectrum object
        spectrum = spectrum.clone(norm=norm, num_shotnoise=num_shotnoise)
        # Propagate coordinate/attribute information to each multipole pole
        spectrum = spectrum.map(lambda pole: pole.clone(attrs=attrs))
        # Attach global attributes to spectrum (z, los, weights, etc.)
        spectrum = spectrum.clone(attrs=attrs)

        # Wait for spectrum computation to complete on all devices
        jax.block_until_ready(spectrum)
        if jax.process_index() == 0:
            logger.info('Mesh-based computation finished')

        for name in results:
            results[name] = _apply_mesh3_spectrum_close_pair_correction(spectrum, results[name])
        results['raw'] = spectrum

    return results


def _get_window_edges(mattrs, scales: tuple=(1, 4)):
    """Return window edges."""
    # Get maximum separation (1/4 of smallest box dimension)
    distmax, cellmin = mattrs.boxsize.min() / 4., mattrs.cellsize.min()
    # Define cell size progressions: 6 doublings then stop (None)
    nsizes, cellsizes = [6] * 5 + [None], [cellmin * 2**i for i in range(6)]
    edges = []
    # Create edge bins for each scale factor (e.g., 1x and 4x)
    for scale in scales:
        edges_scale = []
        start = 0.
        # Progressively increase cell size with doublings
        for nsize, cellsize in zip(nsizes, cellsizes):
            cellsize = cellsize * scale
            # Use regular spacing up to fixed number or until max distance
            if nsize is None:
                tmp = np.arange(start, distmax * scale / scales[0] + cellsize, cellsize)
            else:
                tmp = start + np.arange(nsize) * cellsize
            # Keep only valid edges and update start point
            if tmp.size:
                start = tmp[-1] + cellsize
                edges_scale.append(tmp)
        # Concatenate all edge segments
        edges_scale = np.concatenate(edges_scale, axis=0)
        # Filter to maximum distance
        edges_scale = edges_scale[edges_scale < distmax * scale / scales[0] + cellsize]
        edges.append(edges_scale)
    return edges


def compute_window_mesh3_spectrum(*get_data_randoms, spectrum, zeff: dict=None, ibatch: tuple=None,
                                  computed_batches: list=None, buffer_size: int=0, method: str='smooth_mesh', split_randoms: int=None):
    r"""
    Compute the 3-point spectrum window with :mod:`jaxpower`.

    Parameters
    ----------
    get_data_randoms : callables
        Functions that return tuples of (data, randoms) catalogs.
        See :func:`prepare_jaxpower_particles` for details.
    spectrum : Mesh3SpectrumPoles
        Measured 3-point spectrum multipoles.
    zeff : dict, optional
        Optional arguments for computing effective redshift.
        Default is ``{'cellsize': 10.}`` (density computed with ``cellsize = 10.``)
    ibatch : tuple, optional
        To split the window function multipoles to compute in batches, provide (0, nbatches) for the first batch,
        (1, nbatches) for the second, etc; up to (nbatches - 1, nbatches).
        ``None`` to compute the final window matrix.
    method : string, optional
        ``'smooth_mesh'`` to use the "smooth" method with 2D window correlation computed with FFTs on the mesh,
        ``'smooth_particle'`` for particle counts.
    computed_batches : list, optional
        The window function multipoles that have been computed thus far.

    Returns
    -------
    spectrum : WindowMatrix or dict of WindowMatrix
        The computed 3-point spectrum window.
    """
    from jaxpower import create_sharding_mesh, BinMesh3SpectrumPoles, compute_smooth3_spectrum_window, get_smooth3_window_bin_attrs, MeshAttrs

    # Extract mesh attributes from spectrum
    mattrs = {name: spectrum.attrs[name] for name in ['boxsize', 'boxcenter', 'meshsize']}

    # Extract first multipole pole
    pole = next(iter(spectrum))
    ells, edges, basis = spectrum.ells, pole.edges('k'), pole.basis
    # Gather normalization from all multipoles
    norm = jnp.concatenate([spectrum.get(ell).values('norm') for ell in spectrum.ells])

    # Build 1D k-bin edges
    k, index = np.unique(pole.coords('k', center='mid_if_edges')[..., 0], return_index=True)
    edges = edges[index, 0]
    edges = np.insert(edges[:, 1], 0, edges[0, 0])

    # Set up distributed computation mesh
    with create_sharding_mesh(meshsize=mattrs.get('meshsize', None)) as sharding_mesh:

        if zeff is None: zeff = {'cellsize': 10.}
        kw_zeff = dict(zeff)
        # Compute raw mesh3 correlation window
        all_particles = prepare_jaxpower_particles(*get_data_randoms, mattrs=mattrs, add_randoms=['IDS'])
        all_randoms = [particles['randoms'] for particles in all_particles]
        mattrs = all_randoms[0].attrs
        # Map particle indices to catalog indices (cycle if fewer catalogs than 3 fields)
        fields = list(range(len(all_randoms)))
        fields += [fields[-1]] * (3 - len(fields))

        # Use object IDs for process-invariant random splitting
        seed = [(42, randoms.extra['IDS']) for randoms in all_randoms]
        zeff, norm_zeff = compute_fkp_effective_redshift(*all_randoms, order=3, split=seed, fields=fields, return_fraction=True, **kw_zeff)

        kw, ellsin = get_smooth3_window_bin_attrs(ells, ellsin=2, fields=fields, return_ellsin=True)
        ellsw = kw['ells']
        if 'mesh' in method:
            # Filter to low multipoles only (reduce computational cost)
            ellsw = [ell for ell in ellsw if all(ell <= 2 for ell in ell)]
            # For now, keep only (0, 0, 0) multipole
            ellsw = ellsw[:1]
        # If batching, join with previously computed batches
        if computed_batches:
            correlation = types.join(computed_batches)
            # Reorder to match original ells sequence
            correlation = types.join([correlation.get(ells=[ell]) for ell in ellsw])
        else:
            # If batching, select only subset of multipoles for this batch
            if ibatch is not None:
                start, stop = ibatch[0] * len(ellsw) // ibatch[1], (ibatch[0] + 1) * len(ellsw) // ibatch[1]
                ellsw = ellsw[start:stop]
            correlation = compute_smooth3_spectrum_window_correlation(*all_randoms, spectrum=spectrum, zeff=kw_zeff,
                                                                      buffer_size=buffer_size, method=method, ells=ellsw, split_randoms=split_randoms, fields=fields)

        #def zero(pole, label):
        #    if label['ells'] in [(1, 1, 0), (1, 1, 2), (1, 3, 2)]:
        #        mask = (pole.coords('s1') > 20.)[:, None] & (pole.coords('s2') > 20.)
        #        return pole.clone(value=mask * pole.value())
        #    return pole
        #correlation = correlation.map(zero, input_label=True)

        # Create spectrum binning
        bin = BinMesh3SpectrumPoles(mattrs, edges=edges, ells=ells, basis=basis, mask_edges='')
        # Create finer input correlation binning
        stop = bin.edges1d[0].max()
        step = np.diff(bin.edges1d[0], axis=-1).min()
        edgesin = np.arange(0., 1.5 * stop, step / 2.)
        edgesin = jnp.column_stack([edgesin[:-1], edgesin[1:]])

    results = {}
    results[f'window_{method}3_correlation_raw'] = correlation

    # When batching, only return intermediate raw correlations
    if ibatch is not None:
        return results

    if jax.process_index() == 0:
        logger.info('Building window matrix.')

    # Convert correlation window into spectrum window
    window = compute_smooth3_spectrum_window(correlation, edgesin=edgesin, ellsin=ellsin, bin=bin,
                                             flags=('fftlog',), batch_size=4)

    # Update observable metadata
    observable = window.observable
    observable = observable.map(lambda pole, label: pole.clone(norm=spectrum.get(**label).values('norm'),
                                                               attrs=pole.attrs | dict(zeff=zeff / norm_zeff, norm_zeff=norm_zeff)),
                                                               input_label=True)

    # Renormalize final window
    window = window.clone(observable=observable, value=window.value() / (norm[..., None] / np.mean(norm)))
    results['raw'] = window
    return results


def compute_smooth3_spectrum_window_correlation(*get_data_randoms, spectrum=None, zeff: dict=None, ells: int | list=None, buffer_size: int=0, method: str='smooth_mesh', split_randoms: int=None, fields: list=None):
    r"""
    Compute the 3-point window correlation with :mod:`jaxpower` or :mod:`cucount`.

    Parameters
    ----------
    get_data_randoms : callables
        Functions that return tuples of (data, randoms) catalogs.
        See :func:`prepare_jaxpower_particles` for details.
    spectrum : Mesh3SpectrumPoles
        Measured 3-point spectrum multipoles.
    zeff : dict, optional
        Optional arguments for computing effective redshift.
        Default is ``{'cellsize': 10.}`` (density computed with ``cellsize = 10.``)
    ells : list, optional
        The window function multipoles to compute.
    method : string, optional
        ``'smooth_mesh'`` to use the "smooth" method with 2D window correlation computed with FFTs on the mesh,
        ``'smooth_particle'`` for particle counts.
    split_randoms : float, tuple
        If provided, number of subsets to split the random catalogs into.
        If a tuple, (number of splits, used number of splits).
        (e.g. (10, 5) will just use the first 5 splits out of 10).

    Returns
    -------
    window : ObservableTree
        The computed 3-point window correlation.
    """
    # Import window and correlation functions from jaxpower
    from jaxpower import (create_sharding_mesh, BinMesh3CorrelationPoles, compute_mesh3_correlation,
                        get_smooth3_window_bin_attrs, interpolate_window_function, split_particles)

    assert method in ['smooth_particle', 'smooth_mesh']
    # Extract mesh attributes from spectrum (boxsize, gridsize, etc.)
    mattrs = {name: spectrum.attrs[name] for name in ['boxsize', 'boxcenter', 'meshsize']}
    los = spectrum.attrs['los']
    # Set default effective redshift computation parameters
    if zeff is None: kw_zeff = None
    else: kw_zeff = dict(zeff)

    # Set up distributed computation
    with create_sharding_mesh(meshsize=mattrs.get('meshsize', None)) as sharding_mesh:
        # Load random catalogs and prepare particles
        if callable(get_data_randoms[0]):
            all_particles = prepare_jaxpower_particles(*get_data_randoms, mattrs=mattrs, add_randoms=['IDS'])
            all_randoms = [particles['randoms'] for particles in all_particles]
            del all_particles
        else:
            all_randoms = list(get_data_randoms)

        # Update mesh attributes from random catalog attributes
        mattrs = all_randoms[0].attrs

        # Extract first multipole pole to get coordinate information
        pole = next(iter(spectrum))
        edges, basis = pole.edges('k'), pole.basis
        # Gather normalization from all multipoles
        norm = jnp.concatenate([pole.values('norm') for pole in spectrum])

        # Map particle indices to catalog indices (cycle if fewer catalogs than 3 fields)
        if fields is None:
            fields = list(range(len(all_randoms)))
        fields = list(fields)
        fields += [fields[-1]] * (3 - len(fields))
        # Use object IDs for process-invariant random splitting
        seed = [(42, randoms.extra['IDS']) for randoms in all_randoms]
        # Compute effective redshift for window computation (third-order expansion)
        if kw_zeff is not None:
            zeff, norm_zeff = compute_fkp_effective_redshift(*all_randoms, order=3, split=seed, fields=fields, return_fraction=True, **kw_zeff)

        correlations = []
        # Get window basis attributes (e.g., which multipoles to compute)
        kw = get_smooth3_window_bin_attrs(spectrum.ells, ellsin=2, fields=fields)
        if ells is not None:
            kw['ells'] = ells
        ells = kw['ells']

        # Create logarithmic s-grid for window interpolation (and FFTlog)
        coords = jnp.logspace(-3, 5, 1024)

        assert len(pole.attrs['wsum_data']) == 1
        wsum_data = pole.attrs['wsum_data'][0]

        if method == 'smooth_particle':
            from jaxpower.particle2 import convert_particles
            from cucount.jax import BinAttrs, SelectionAttrs, WeightAttrs
            from cucount.types import count3, count3_analytic, compute_norm3
            from lsstypes.types import convert_ells
            from jaxpower.mesh import create_sharded_random, _process_seed
            from .correlation3_tools import _digitize_cartesian, _remove_phantom_particles

            all_particles = []
            # Paint random catalogs on coarse mesh
            for iran, randoms in enumerate(split_particles(all_randoms + [None] * (3 - len(all_randoms)),
                                                            seed=seed, fields=fields)):
                # Normalize by data/random weight ratio
                alpha = wsum_data[min(iran, len(wsum_data) - 1)] / randoms.weights.sum()
                # Paint random on mesh scaled by alpha
                randoms = randoms.clone(weights=alpha * randoms.weights)
                all_particles.append(convert_particles(randoms))
            del all_randoms

            ells_slepian = convert_ells(ells, 'sugiyama', 'slepian')
            ells12, ells13 = [tuple(np.unique([ell[idim] for ell in ells_slepian])) for idim in range(2)]

            sepmax = jnp.sqrt(jnp.sum(mattrs.boxsize**2))
            limits = [mattrs.cellsize.min(), 200., 500.]  # drop bin at 0., which is noisy
            limits = [lim for lim in limits if lim < sepmax] + [sepmax]
            resols = [None, 40., 100.]
            steps = [step * mattrs.cellsize.min() for step in [1., 4., 10.]]
            list_edges = [np.linspace(*limit, np.ceil((limit[1] - limit[0]) / step).astype(int)) for limit, step in zip(zip(limits[:-1], limits[1:]), steps)]
            edges = np.concatenate([np.column_stack([edges[:-1], edges[1:]]) for edges in list_edges], axis=0)
            mid = np.mean(edges, axis=-1)
            counts = np.zeros((len(mid),) * 2, dtype='f8')
            count3pole = types.Count3Pole(counts=counts, norm=np.ones_like(counts), s1=mid, s2=mid, s1_edges=edges, s2_edges=edges, coords=['s1', 's2'], ell=(0, 0, 0), basis='slepian')
            nsplits, max_nsplits = 1, 1
            if split_randoms is not None:
                if isinstance(split_randoms, tuple):
                    nsplits, max_nsplits = split_randoms
                else:
                    nsplits = max_nsplits = split_randoms

            wattrs = WeightAttrs()
            prod = functools.partial(functools.reduce, operator.mul)

            def count3split(*particles, battrs12=None, battrs13=None, sattrs12=None, sattrs13=None, norm_ref=1.):
                kw = dict(wattrs=wattrs, battrs12=battrs12, battrs13=battrs13,
                          sattrs12=sattrs12, sattrs13=sattrs13, norm=1.)

                nsplits_ = [getattr(p, 'nsplits', 0) for p in particles]

                if any(nsplits_):
                    nsplit = next(n for n in nsplits_ if n)
                    particles = list(particles)
                    particle_iters = []
                    for particle in particles:
                        if getattr(particle, 'nsplits', 0):
                            assert particle.nsplits == nsplit
                            particle_iters.append(iter(particle()))
                        else:
                            particle_iters.append(itertools.repeat(particle, nsplit))
                    counts, norm = [], 0.
                    for p in zip(*particle_iters):
                        counts.append(count3(*p, **kw)['weight'])
                        norm += compute_norm3(*p, wattrs=wattrs)
                    counts = types.sum(counts)
                else:
                    counts = count3(*particles, **kw)['weight']
                    norm = compute_norm3(*particles, wattrs=wattrs)
                return counts.clone(value=norm_ref / norm * counts.value())

            norm_ref = compute_norm3(*all_particles, wattrs=wattrs)
            counts = []
            resol_limits = list(zip(zip(limits[:-1], limits[1:]), list_edges, resols))

            for (resol_limit12, edges12, resol12), (resol_limit13, edges13, resol13) in itertools.product(resol_limits, repeat=2):
                battrs12 = BinAttrs(s=edges12, pole=(ells12, 'firstpoint'))
                battrs13 = BinAttrs(s=edges13, pole=(ells13, 'firstpoint'))
                sattrs12 = SelectionAttrs(s=resol_limit12)
                sattrs13 = SelectionAttrs(s=resol_limit13)
                all_particles_resol = list(all_particles)
                digitized = set()
                t0 = time.time()

                if resol12 is not None:
                    all_particles_resol[1] = _digitize_cartesian(all_particles_resol[1], wattrs=wattrs, cellsize=resol12, sharding_mesh=sharding_mesh)
                    digitized.add(1)

                if resol13 is not None:
                    all_particles_resol[2] = _digitize_cartesian(all_particles_resol[2], wattrs=wattrs, cellsize=resol13, sharding_mesh=sharding_mesh)
                    digitized.add(2)

                if nsplits > 1:

                    def _get_uniform(size, seed=(84, 'index')):
                        return create_sharded_random(jax.random.uniform, _process_seed(seed), size, out_specs=P(sharding_mesh.axis_names,))

                    def make_particle_splits(particles, x, nsplits, max_nsplits):
                        weights = wattrs(particles)
                        def gen():
                            for isplit in range(max_nsplits):
                                mask = ((x >= isplit / nsplits) & (x < (isplit + 1) / nsplits))
                                #yield particles.clone(weights=weights * mask)
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
                                x = _get_uniform(len(weights))
                                mask = x < max_nsplits * 1. / nsplits
                                all_particles_resol[ip] = _remove_phantom_particles(particles.clone(weights=weights * mask), sharding_mesh=sharding_mesh)

                counts.append(count3split(*all_particles_resol, battrs12=battrs12, battrs13=battrs13, sattrs12=sattrs12, sattrs13=sattrs13, norm_ref=norm_ref))
                if jax.process_index() == 0:
                    logger.info(f'Computed RRR counts within {resol_limit12} x {resol_limit13} in {time.time() - t0:.1f} s')

            def sum_counts(leaves):
                pole = count3pole.clone(meta=leaves[0].meta)
                counts = np.zeros_like(pole.values('counts'))
                # leaves have been computed with different bin slices, fix that here
                for leaf in leaves:
                    indices = []
                    for edges_, self_edges_ in zip(pole.edges().values(), leaf.edges().values()):
                        width = np.abs(edges_[..., 1] - edges_[..., 0])
                        tol = 1e-5 * width
                        mask = ((self_edges_[None, :, 0] >= edges_[:, None, 0] - tol[:, None]) &
                                (self_edges_[None, :, 1] <= edges_[:, None, 1] + tol[:, None]))
                        assert np.all(mask.sum(axis=0) == 1)
                        index, index_self = np.nonzero(mask)
                        index = index[np.argsort(index_self)]
                        assert np.all(edges_[index] == self_edges_)
                        indices.append(index)
                    counts[np.ix_(*indices)] += leaf.values('counts')
                return pole.clone(counts=counts)

            counts = types.tree_map(sum_counts, counts, level=None, is_leaf=lambda *args: False)
            counts = counts.to_basis('sugiyama', ells=ells)
            battrs12 = battrs13 = BinAttrs(s=np.append(edges[:, 0], edges[-1, 1]))
            RRR0 = count3_analytic(mattrs=1., battrs12=battrs12, battrs13=battrs13)

            # Divide by volume factor and normalization
            def renormalize(pole):
                return pole.clone(counts=pole.values('counts'), norm=np.mean(norm) * RRR0.value())

            counts = counts.map(renormalize)

            def pad_value(value, label=None):
                value1 = value
                value = jnp.pad(value, ((0, 1),) * value.ndim, mode='constant', constant_values=0.)
                if label['ells'] == (0, 0, 0):
                    value = jnp.pad(value, ((1, 0),) * value.ndim, mode='edge')
                else:
                    value = jnp.pad(value, ((1, 0),) * value.ndim, mode='constant', constant_values=0.)
                return value

            correlation = interpolate_window_function(counts, coords=coords, order=3, pad_value=pad_value)

        elif method == 'smooth_mesh':
            # JIT-compile 3-point correlation computation
            jitted_compute_mesh3_correlation = jax.jit(compute_mesh3_correlation, static_argnames=['los'], donate_argnums=[0])

            # List of scale factors for multigrid computation (coarse and fine)
            list_scales = [1, 4]
            # Get binning edges for each scale
            list_edges = _get_window_edges(mattrs, scales=list_scales)

            # Compute window using multigrid approach
            # Painting parameters
            kw_paint = dict(resampler='tsc', interlacing=3, compensate=True)
            # Loop over scale factors (coarse to fine)
            for scale, edges in zip(list_scales, list_edges):
                # Create coarser mesh (larger boxsize)
                mattrs2 = mattrs.clone(boxsize=scale * mattrs.boxsize)
                if jax.process_index() == 0:
                    logger.info(f'Processing scale x{scale:.0f}, using {mattrs2}')
                # Create binning for coarse mesh
                sbin = BinMesh3CorrelationPoles(mattrs2, edges=edges, **kw, buffer_size=buffer_size)

                meshes = []
                # Paint random catalogs on coarse mesh
                for iran, randoms in enumerate(split_particles(all_randoms + [None] * (3 - len(all_randoms)),
                                                               seed=seed, fields=fields)):
                    # Adapt random catalog to coarse mesh and exchange across processes
                    randoms = randoms.clone(attrs=mattrs2).exchange(backend='mpi')
                    # Normalize by data/random weight ratio
                    alpha = wsum_data[min(iran, len(wsum_data) - 1)] / randoms.weights.sum()
                    # Paint random on mesh scaled by alpha
                    meshes.append(alpha * randoms.paint(**kw_paint, out='real'))

                # Compute 3-point correlation on coarse mesh
                t0 = time.time()
                correlation = jitted_compute_mesh3_correlation(meshes, bin=sbin, los=los)
                # Normalize correlation by average normalization factor
                correlation = correlation.clone(norm=[np.mean(norm)] * len(sbin.ells))
                jax.block_until_ready(correlation)
                if jax.process_index() == 0:
                    logger.info(f"Computed windows {kw['ells']}, scale {scale}, in {time.time() - t0:.2f} s.")
                # Interpolate correlation to fine s-grid
                correlation = interpolate_window_function(correlation.unravel(), coords=coords, order=3)
                correlations.append(correlation)

            # Extract coordinate grids from correlations
            coords = list(next(iter(correlations[0])).coords().values())
            # Create masks for smooth transition between scales (using -3 offset for cubic spline)
            masks = [(coords[0] < edges[-3])[:, None] * (coords[1] < edges[-3])[None, :] for edges in list_edges[:-1]]
            # Add final mask (all points)
            masks.append((coords[0] < np.inf)[:, None] * (coords[1] < np.inf)[None, :])
            weights = []
            for mask in masks:
                if len(weights):
                    # Exclude already-weighted regions
                    weights.append(mask & (~weights[-1]))
                else:
                    weights.append(mask)
            # Regularize weights to avoid zero division
            weights = [np.maximum(mask, 1e-6) for mask in weights]
            # Combine correlations from different scales using weights
            correlation = correlations[0].sum(correlations, weights=weights)

        # Wait for window computation to complete
        jax.block_until_ready(correlation)
        if jax.process_index() == 0:
            logger.info('Window functions computed.')

        if kw_zeff is not None:
            correlation.attrs.update(zeff=zeff / norm_zeff, norm_zeff=norm_zeff)

    return correlation



def run_preliminary_fit_mesh3_spectrum(spectrum2, spectrum3, window2=None, window3=None, mattrs=None,
                                       free=None, select2={'k': slice(0, None, 10)}, select3={'k': slice(0, None, 2)}):
    r"""
    Fit bias parameters on the joint (P, B) data vector to build the theory assumed in the covariance.

    Following ``jax-power/scripts/example_fit_bias_covariance.py``: the tree-level
    :mod:`jaxpower.pt` theory (P / B / T kernels) is evaluated at the measured binning and the
    free bias parameters are fitted using the periodic-box P + B covariance,
    ``compute_spectrum3_covariance(mattrs, mattrs, ...)`` (analytic Sugiyama et al. formulas;
    P-only, see below).

    Parameters
    ----------
    spectrum2 : Mesh2SpectrumPoles
        Measured 2-point spectrum multipoles.
    spectrum3 : Mesh3SpectrumPoles
        Measured 3-point spectrum multipoles (Sugiyama basis).
    window2 : WindowMatrix, optional
        2-point spectrum window matrix; if provided (together with ``window3``), the P model is
        convolved with it before being compared to ``spectrum2`` in the fit, as in
        :func:`run_preliminary_fit_mesh2_spectrum`. Also used as the source for the effective
        redshift ``z``.
    window3 : WindowMatrix, optional
        3-point spectrum window matrix; if provided, its theory axis is first compacted via
        :func:`full_shape.tools.rebin_spectrum3_window`, then the B model is convolved with it
        before being compared to ``spectrum3``. If ``None``, the window effect is ignored
        entirely (``window2`` is then ignored too) and the model is compared directly to the
        data at their own measured binning, as before.
    mattrs : MeshAttrs, optional
        Mesh attributes setting the volume of the periodic covariance used in the fit.
        If ``None``, taken from ``spectrum2.attrs`` --- that is the embedding box, which
        overestimates the survey volume; pass e.g. ``mattrs.clone(boxsize=...)`` for a more
        realistic effective volume (this only reweights the preliminary fit).
    free : tuple, optional
        Names of the bias parameters to fit. If ``None`` (default), all bias parameters are free:
        ``('b1', 'b2', 'bs', 'b3nl', 'c1', 'c2', 'X_FoG', 'snb0', 'sn0')``.
    select2, select3 : dict, optional
        If provided, selections applied to ``spectrum2`` / ``spectrum3`` prior to fitting
        (e.g. ``{'k': (0.02, 0.15)}`` to restrict to scales where the tree-level theory holds).

    Returns
    -------
    theory : callable
        ``theory(fields)`` returning the 3D P / B / T callables consumed by
        :func:`compute_covariance_mesh3_spectrum`, evaluated at the best-fit bias.
        Best-fit values are stored as ``theory.bias`` (also ``theory.z``, ``theory.f``, ``theory.chi2``).
        The best-fit model multipoles, at the (selected, and windowed if ``window3`` is provided)
        data binning, are stored as ``theory.spectrum2`` / ``theory.spectrum3``, and the data they
        are to be compared to (after ``select2`` / ``select3``) as ``theory.data2`` / ``theory.data3``.
    """
    from jaxpower import MeshAttrs
    from jaxpower.cov3 import compute_spectrum3_covariance
    from jaxpower.pt import (prepare_spectrum2_redshift_tracer, spectrum2_redshift_tracer,
                             spectrum3_redshift_tracer, spectrum4_redshift_tracer,
                             ProjectToPoles, ProjectToSell)
    from jaxpower.utils import get_legendre

    # window3 gates windowing altogether: with no window3, ignore window2 too and fall back to
    # comparing the raw (unwindowed) model to the data at their own measured binning.
    with_window = window3 is not None

    # Restrict data to the fitting range if requested
    if select2 is not None: spectrum2 = spectrum2.select(**select2)
    if select3 is not None: spectrum3 = spectrum3.select(**select3)

    if with_window:
        # Match window observable axes to the (possibly restricted) data before fitting
        window2 = window2.at.observable.match(spectrum2)
        from full_shape.tools import rebin_spectrum3_window
        window3 = rebin_spectrum3_window(window3, data=spectrum3)
        window3 = window3.at.observable.match(spectrum3)

    # Volume assumed for the periodic covariance used in the fit
    if mattrs is None:
        mattrs = MeshAttrs(**{name: spectrum2.attrs[name] for name in ['boxsize', 'boxcenter', 'meshsize']})

    # Fiducial linear power spectrum at the effective redshift
    from cosmoprimo.fiducial import DESI
    cosmo = DESI(engine='camb')
    if window2 is not None:
        # As in run_preliminary_fit_mesh2_spectrum: read zeff directly from the window
        # (measured Mesh2SpectrumPoles do not carry 'zeff', only window matrices do).
        z = window2.observable.get(ells=0).attrs['zeff']
    elif 'zeff' in spectrum2.attrs:
        z = np.asarray(spectrum2.attrs['zeff']).flat[0]
    else:
        raise ValueError("z could not be determined: provide window2, or attach 'zeff' to spectrum2.attrs")
    f = float(cosmo.growth_rate(z))
    # Shot noise (1 + alpha) / nbar, read off the FKP monopole shot noise level.
    shotnoise = float(np.mean(spectrum2.get(0).values('shotnoise')))
    kt = np.logspace(-4, 1, 512)
    pkt = cosmo.get_fourier().pk_interpolator().to_1d(z=z)(kt)

    def pk_callable(q):
        return jnp.interp(q, kt, pkt)

    pknow_callable = pk_callable  # no-wiggle spectrum; plug a BAO-filtered one if desired

    # Tabulate the 1-loop P(k) integrals once
    k_table = jnp.logspace(-3, np.log10(mattrs.knyq.max()), 80)
    table, table_now = prepare_spectrum2_redshift_tracer(k_table, pk_callable, pknow_callable)

    fid_bias = {'b1': 2.0, 'b2': 0.5, 'bs': -0.3, 'b3nl': 0.1,
                'c1': 0.1, 'c2': 0.2, 'X_FoG': 2., 'snb0': 0.1, 'sn0': 0.1}

    def make_theory(bias):
        # 3D P / B / T callables consumed by compute_spectrum3_covariance;
        # the key of bias_params must match the observable / window field label (0)
        bias_params = {0: dict(bias)}

        # P(kvec): precompute the P_ell(k) multipoles once, on table['matter']['k']
        # (the same k grid spectrum2_redshift_tracer's own kvec branch already
        # interpolates from -- see its 'is_kvec' branch in jaxpower.pt), instead of
        # letting it redo that (nk_table, nmu) EFT/TNS/IR-resummation evaluation on
        # every call. compute_spectrum3_covariance calls P(kvec) many times while
        # assembling the covariance, so hoisting this out is exact (same table, same
        # linear interpolation in k), just no longer redundantly recomputed.
        ells_P = list(range(0, 8, 2))
        to_poles_P = ProjectToPoles(mu=10, ells=ells_P)
        poles_P = to_poles_P(spectrum2_redshift_tracer(to_poles_P.mu, table, table_now, f, bias_params))
        k_table_P = table['matter']['k']

        def P(kvec):
            kvec = jnp.asarray(kvec)
            knorm = jnp.sqrt(jnp.sum(kvec**2, axis=-1))
            mu = jnp.where(knorm > 0., kvec[..., 2] / knorm, 0.)
            pole_at_k = jax.vmap(lambda pole: jnp.interp(knorm.ravel(), k_table_P, pole))(poles_P)
            return sum(pole_at_k[ill].reshape(knorm.shape) * get_legendre(ell)(mu)
                      for ill, ell in enumerate(ells_P))

        def B(k1vec, k2vec, k3vec):
            return spectrum3_redshift_tracer(k1vec, k2vec, pk_callable, pknow_callable, f=f, bias_params=bias_params)

        def T(k1vec, k2vec, k3vec, k4vec):
            return spectrum4_redshift_tracer(k1vec, k2vec, k3vec, pk_callable, pknow_callable, f=f, bias_params=bias_params)

        def theory(fields):
            return {2: P, 3: B, 4: T}.get(len(fields), None)

        theory.bias, theory.z, theory.f = dict(bias), z, f
        return theory

    # Joint (P, B) data vector
    observable = types.ObservableTree([spectrum2, spectrum3], fields=[(0, 0), (0, 0, 0)])
    data = np.concatenate([np.asarray(obs.value()).ravel() for _, obs in observable.items(level=None)])

    # Periodic-box covariance evaluated at the fiducial bias: P-only (Gaussian PP term), since
    # this preliminary covariance is only used to weight the bias fit's chi2, not the final one
    # (B and T theory are set to None so the BB/PB/PT terms don't contribute).
    def theory_cov(fields):
        return make_theory(fid_bias)(fields) if len(fields) == 2 else None

    if jax.process_index() == 0:
        logger.info('Computing preliminary P + B covariance')
    covariance = compute_spectrum3_covariance(mattrs, mattrs, observable, theory=theory_cov,
                                              shotnoise=shotnoise, cache={})
    Cinv = jnp.asarray(np.linalg.inv(np.asarray(covariance.value())))

    names = list(free) if free is not None else list(fid_bias.keys())

    if with_window:
        # Precompute, once, the projectors and kvec grids (independent of the bias parameters).
        # The k (P) / (k1, k2) (B) grid is assumed shared across all window theory ells, so P/B
        # need only be evaluated once (not once per ell) and projected to all ells at once --
        # mirroring the unwindowed branch below, which does the same for spectrum2/spectrum3's
        # own (shared) k grid.
        ells2 = list(window2.theory.ells)
        k2d = np.asarray(window2.theory.get(ells=ells2[0]).coords('k'))  # (nk,)
        proj2 = ProjectToPoles(ells=ells2, mu=10)
        mu = np.asarray(proj2.mu)
        kvec2 = jnp.asarray(k2d[:, None, None] * np.stack([np.sqrt(1. - mu**2), np.zeros_like(mu), mu], axis=-1))

        ells3 = list(window3.theory.ells)
        k3d = np.asarray(window3.theory.get(ells=ells3[0]).coords('k'))  # (nbins, 2) paired (k1, k2)
        proj3 = ProjectToSell(ells=ells3, size=6)
        k1vec3 = jnp.asarray(k3d[:, 0, None, None] * np.asarray(proj3.k1hat)[None, ...])
        k2vec3 = jnp.asarray(k3d[:, 1, None, None] * np.asarray(proj3.k2hat)[None, ...])

        # Precompute the (concrete) window matrices once; window2.theory.ells / window3.theory.ells
        # (each ell's own k values raveled, in this order) fix the flattening order that both
        # these and model_vector's own concatenation must follow.
        window2_value = jnp.asarray(window2.value())
        window3_value = jnp.asarray(window3.value())

        def model_vector(x):
            bias_params = {0: fid_bias | dict(zip(names, x))}
            P3d = spectrum2_redshift_tracer(kvec2, table, table_now, f, bias_params)  # (nk, nmu)
            p_flat = proj2(P3d).ravel()  # (nells2, nk) -> flat, ell-major
            B3d = spectrum3_redshift_tracer(k1vec3, k2vec3, pk_callable, pknow_callable, f=f, bias_params=bias_params)
            b_flat = proj3(B3d).ravel()  # (nells3, nbins) -> flat, ell-major
            p_poles = window2_value.dot(p_flat)
            b_poles = window3_value.dot(b_flat)
            return jnp.concatenate([p_poles, b_poles])

    else:
        # Binned model multipoles, evaluated directly at the measured k coordinates (no window)
        to_poles = ProjectToPoles(ells=spectrum2.ells, mu=10)
        k2d = np.asarray(next(iter(spectrum2)).coords('k'))  # (nk,)
        mu = np.asarray(to_poles.mu)
        kvec_P = k2d[:, None, None] * np.stack([np.sqrt(1. - mu**2), np.zeros_like(mu), mu], axis=-1)  # (nk, nmu, 3)

        to_Sell = ProjectToSell(ells=spectrum3.ells, size=6)
        k3d = np.asarray(next(iter(spectrum3)).coords('k'))  # (nbins, 2) paired (k1, k2)
        k1vec_B = k3d[:, 0, None, None] * np.asarray(to_Sell.k1hat)[None, ...]  # (nbins, nnodes, 3)
        k2vec_B = k3d[:, 1, None, None] * np.asarray(to_Sell.k2hat)[None, ...]

        def model_vector(x):
            bias_params = {0: fid_bias | dict(zip(names, x))}
            P3d = spectrum2_redshift_tracer(jnp.asarray(kvec_P), table, table_now, f, bias_params)  # (nk, nmu)
            p_poles = to_poles(P3d)  # (nells2, nk)
            B3d = spectrum3_redshift_tracer(jnp.asarray(k1vec_B), jnp.asarray(k2vec_B),
                                            pk_callable, pknow_callable, f=f, bias_params=bias_params)
            b_poles = to_Sell(B3d)  # (nells3, nbins)
            return jnp.concatenate([p_poles.ravel(), b_poles.ravel()])

    def chi2(x):
        r = jnp.asarray(data) - model_vector(x)
        return r @ Cinv @ r

    if jax.process_index() == 0:
        logger.info(f'Starting preliminary fit')
    value_and_grad = jax.jit(jax.value_and_grad(chi2))

    from scipy import optimize
    res = optimize.minimize(value_and_grad, x0=np.array([fid_bias[name] for name in names]),
                            jac=True, method='L-BFGS-B')
    best = dict(zip(names, np.asarray(res.x)))
    if jax.process_index() == 0:
        logger.info(f'Preliminary fit: {best}, chi2 = {res.fun:.1f} ({data.size} data points)')

    theory = make_theory(fid_bias | best)
    theory.chi2 = float(res.fun)
    # Best-fit model multipoles at the data binning, to inspect the quality of the fit
    model = np.asarray(model_vector(jnp.asarray(res.x)))
    size2 = np.asarray(spectrum2.value()).size
    theory.spectrum2 = spectrum2.clone(value=model[:size2])
    theory.spectrum3 = spectrum3.clone(value=model[size2:])
    theory.data2, theory.data3 = spectrum2, spectrum3
    return theory


def get_post_recon_spectrum_kernels(cosmo, z, f, b1, smoothing_radius, shotnoise=0.):
    r"""
    Analytic reconstruction-damped post-recon auto and pre-recon x post-recon cross power
    spectrum kernels, following the Zel'dovich-reconstruction formalism (e.g. Padmanabhan,
    White & Seo), ported from ``cai-mock-benchmark/dr2/data_pip.py::get_post_recon_spectrum``.

    Unlike that reference (which projects to multipoles via a fixed mu quadrature), this
    returns raw ``P(kvec)`` callables (``kvec``: ``(..., 3)`` array, with :math:`\mu` extracted
    as ``kvec[..., 2] / |kvec|``), matching the calling convention of :mod:`jaxpower.pt`'s
    tree-level kernels, so they can be used directly as part of the ``theory(fields)`` callable
    consumed by :func:`jaxpower.cov3.compute_spectrum3_covariance`.

    Shot noise convention: :func:`jaxpower.cov3.compute_spectrum3_covariance` automatically adds
    its scalar ``shotnoise`` argument to any AUTO field pair (fields[0] == fields[1]); the
    returned ``'post_post'`` kernel therefore does NOT bake in shot noise (mirroring how the
    pre-recon ``P`` kernel from :func:`run_preliminary_fit_mesh3_spectrum` doesn't either). The
    ``'pre_post'`` cross kernel DOES bake in a damped shot-noise term, because pre- and
    post-recon are the same particles (just displaced), so their shot noise is correlated --
    unlike the covariance assembly's default assumption of zero cross-field shot noise.

    Parameters
    ----------
    cosmo : Cosmology
        Fiducial cosmology (linear power spectrum, drag scale, growth rate).
    z : float
        Effective redshift.
    f : float
        Growth rate at ``z`` (e.g. already computed by :func:`run_preliminary_fit_mesh3_spectrum`).
    b1 : float
        Linear bias, shared between the pre- and post-recon fields (same tracer).
    smoothing_radius : float
        Reconstruction Gaussian smoothing radius.
    shotnoise : float, optional
        Shot noise :math:`(1 + \alpha) / \bar{n}`, entering only the pre-post cross kernel.

    Returns
    -------
    kernels : dict
        ``{'post_post': callable(kvec), 'pre_post': callable(kvec)}``.
    """
    from scipy import special, integrate
    from cosmoprimo import PowerSpectrumBAOFilter

    klin = np.logspace(-3., 2., 1000)
    pklin_interp = cosmo.get_fourier().pk_interpolator(of='delta_cb').to_1d(z=z)
    pknow_interp = PowerSpectrumBAOFilter(pklin_interp, engine='wallish2018').smooth_pk_interpolator()
    pklin_klin, pknow_klin = pklin_interp(klin), pknow_interp(klin)
    q = cosmo.rs_drag

    def pklin(k):
        return jnp.interp(k, klin, pklin_klin)

    def pknow(k):
        return jnp.interp(k, klin, pknow_klin)

    sk_klin = np.exp(-0.5 * (klin * smoothing_radius)**2)
    j0_klin = special.jn(0, q * klin)
    # Post x post damping: only the BAO wiggles are resummed/damped, broadband is untouched.
    sigma_pp = 1. / (3. * np.pi**2) * integrate.simpson((1. - j0_klin) * (1. - sk_klin)**2 * pklin_klin, klin)
    # Pre x post damping: the whole (wiggly) signal is damped as one piece.
    sigma_pp_cross = 1. / (6. * np.pi**2) * integrate.simpson(sk_klin * pklin_klin, klin)

    def _k_mu(kvec):
        kvec = jnp.asarray(kvec)
        k = jnp.sqrt(jnp.sum(kvec**2, axis=-1))
        # Same los-z convention as jaxpower.pt.spectrum2_redshift_tracer.
        mu = jnp.where(k > 0., kvec[..., 2] / k, 0.)
        return k, mu

    def post_post(kvec):
        k, mu = _k_mu(kvec)
        ksq = (1. + f * (f + 2.) * mu**2) * k**2
        wiggles = pklin(k) - pknow(k)
        return (b1 + f * mu**2)**2 * (pknow(k) + jnp.exp(-0.5 * ksq * sigma_pp) * wiggles)

    def pre_post(kvec):
        k, mu = _k_mu(kvec)
        ksq = (1. + (1. + f)**2 * mu**2) * k**2
        return jnp.exp(-0.5 * ksq * sigma_pp_cross) * ((b1 + f * mu**2)**2 * pklin(k) + shotnoise)

    return {'post_post': post_post, 'pre_post': pre_post}


def _compute_cov3_windows(get_data_randoms, fields=None, mattrs=None, edges=None, ells2=(0, 2, 4),
                          ells3=((0, 0, 0),), buffer_size=0):
    """
    Compute the cov3-style (grouped-field-label) two-anchor and three-anchor covariance windows
    from randoms. Shared by :func:`compute_covariance_mesh3_spectrum` and :func:`compute_covariance`.
    """
    # NOTE: compute_fkp2_covariance_window must come from cov3 (grouped field labels),
    # not the flat-label cov2 variant used by compute_covariance_mesh2_spectrum
    from jaxpower import create_sharding_mesh, interpolate_window_function, FKPField
    from jaxpower.cov3 import compute_fkp2_covariance_window, compute_fkp3_covariance_window

    if fields is None:
        fields = list(range(len(get_data_randoms)))

    mattrs = mattrs or {}
    # Set up distributed computation mesh
    with create_sharding_mesh(meshsize=mattrs.get('meshsize', None)):
        # Load and prepare particles
        all_particles = prepare_jaxpower_particles(*get_data_randoms, mattrs=mattrs, add_randoms=['IDS'])
        # Create FKP fields for covariance window computation
        all_fkp = [FKPField(particles['data'], particles['randoms']) for particles in all_particles]
        mattrs = all_fkp[0].attrs
        # Set correlation binning parameters
        if edges is None:
            edges = {'step': mattrs.cellsize.min()}
        # Split randoms (process-invariant, based on object IDs) to avoid common noise between window factors
        split = [(42, fkp.randoms.extra['IDS']) for fkp in all_fkp]
        # Mesh painting parameters: TSC with interlacing
        kw_paint = dict(resampler='tsc', interlacing=3, compensate=True)

        # Two-anchor window Q_W^{(A)(B)}(s), with grouped field labels of sizes (2, 3, 4):
        # required by the mixed PB / BB / PT covariance terms
        window2 = compute_fkp2_covariance_window(all_fkp, edges=edges, los='local', fields=fields, split=split,
                                                 group_sizes=(2, 3, 4), max_total_size=6, ells=list(ells2), **kw_paint)
        jax.block_until_ready(window2)
        if jax.process_index() == 0:
            logger.info('2-point covariance window computed')

        # Three-anchor window Q_W^{ABC}(s1, s2) in the Sugiyama basis (WWW piece)
        window3 = compute_fkp3_covariance_window(all_fkp, edges=edges, los='local', fields=fields, split=split,
                                                 ells=[tuple(ell) for ell in ells3], buffer_size=buffer_size, **kw_paint)
        jax.block_until_ready(window3)
        if jax.process_index() == 0:
            logger.info('3-point covariance window computed')

        # Interpolate the windows on a log-spaced grid for the Hankel transforms in the assembly
        coords = jnp.logspace(-3, 4, 1024)
        window2 = interpolate_window_function(window2, coords=coords, order=3)
        window3 = window3.map(lambda pole: pole.unravel())
        window3 = interpolate_window_function(window3, coords=coords, order=3)

    return window2, window3


def _restrict_theory(theory, terms='PBT'):
    """Restrict the P / B / T callables provided by *theory* to the requested *terms*
    ('P', 'PB' or 'PBT'): n-point functions not in *terms* are served as ``None``,
    dropping the corresponding covariance contributions (the connected BB and P x B
    cross terms for 'B'; the PP- and BB-block trispectrum terms for 'T')."""
    sizes = {'P': 2, 'B': 3, 'T': 4}
    terms = str(terms).upper()
    unknown = set(terms) - set(sizes)
    if unknown or 'P' not in terms:
        raise ValueError(f"terms must be 'P', 'PB' or 'PBT', got {terms!r}")
    keep = {sizes[term] for term in terms}
    if keep == set(sizes.values()):
        return theory

    def restricted(fields):
        return theory(fields) if len(fields) in keep else None

    return restricted


def compute_covariance_mesh3_spectrum(*get_data_randoms, spectrum2=None, spectrum3=None, theory=None, shotnoise=0.,
                                      fields=None, mattrs=None, edges=None, ells2=(0, 2, 4), ells3=((0, 0, 0),),
                                      buffer_size=0, batch_size=16, window2=None, window3=None, terms='PBT'):
    r"""
    Compute the joint 2-point + 3-point spectrum covariance with :mod:`jaxpower`.

    The two-anchor and three-anchor covariance window multipoles are estimated from the randoms,
    then assembled into the (P, B) covariance following the Sugiyama et al. formulas
    (PP / PB / PPP + BB + PT terms), cf. ``jax-power/scripts/example_fit_bias_covariance.py``.

    Parameters
    ----------
    get_data_randoms : callables
        Functions that return dict of 'data' and 'randoms' catalogs.
        See :func:`prepare_jaxpower_particles` for details.
    spectrum2 : Mesh2SpectrumPoles
        Measured 2-point spectrum multipoles, defining the covariance binning.
    spectrum3 : Mesh3SpectrumPoles
        Measured 3-point spectrum multipoles (Sugiyama basis), defining the covariance binning.
    theory : callable
        ``theory(fields)`` returning the 3D P / B / T callables, with bias parameters keyed by
        the field labels (0-based); see :func:`run_preliminary_fit_mesh3_spectrum`.
    shotnoise : float, optional
        Shot noise :math:`(1 + \alpha) / \bar{n}`.
    fields : tuple, list, optional
        Field names. Default is ``(0, 1, ...)``, matching the bias parameter keys of
        :func:`run_preliminary_fit_mesh3_spectrum`.
    mattrs : dict, optional
        Mesh attributes to define the :class:`jaxpower.ParticleField` objects,
        'boxsize', 'meshsize' or 'cellsize', 'boxcenter'. If ``None``, default attributes are used.
    edges : dict, optional
        Separation binning for the covariance windows. Default is ``{'step': cellsize.min()}``;
        coarsen (e.g. ``{'step': 40.}``) to speed up the 3-point window computation.
    ells2 : tuple, optional
        Multipoles of the two-anchor covariance window. Default is ``(0, 2, 4)``.
    ells3 : list of tuples, optional
        Sugiyama multipoles of the three-anchor covariance window. Default is ``[(0, 0, 0)]``.
    buffer_size : int, optional
        Buffer size for the 3-point window binning; increase for faster computation at the cost of memory.
    batch_size : int, optional
        Batch size for the covariance assembly.
    window2, window3 : optional
        Precomputed (interpolated) covariance windows, as returned in the results dictionary
        ('window_covariance_mesh2_correlation', 'window_covariance_mesh3_correlation') of a
        previous call; file paths are read with :func:`lsstypes.read`. Both must be provided
        to skip the window computation from the randoms (``mattrs``, ``edges``, ``ells2``,
        ``ells3``, ``buffer_size`` are then ignored); the field labels must match *fields*.
    terms : str, optional
        Theory ingredients contributing to the covariance: 'P' keeps only the power spectrum
        (Gaussian PP / PPP terms), 'PB' adds the bispectrum (connected BB and P x B cross
        terms), 'PBT' (default) adds the trispectrum (PP- and BB-block trispectrum terms).

    Returns
    -------
    results : dict
        Dictionary containing the covariance matrix ('raw') and the interpolated covariance window
        correlations ('window_covariance_mesh2_correlation', 'window_covariance_mesh3_correlation').
    """
    from pathlib import Path
    from jaxpower.cov3 import compute_spectrum3_covariance

    theory = _restrict_theory(theory, terms=terms)

    # Use default fields (0, 1, ...) if not provided
    if fields is None:
        fields = list(range(len(get_data_randoms)))

    # The measured observables define the covariance binning; field labels must match the windows
    observable = types.ObservableTree([spectrum2, spectrum3], fields=[(fields[0],) * 2, (fields[0],) * 3])

    if False: #window2 is not None and window3 is not None:
        if isinstance(window2, (str, Path)): window2 = types.read(window2)
        if isinstance(window3, (str, Path)): window3 = types.read(window3)
        if jax.process_index() == 0:
            logger.info('Using precomputed covariance windows')
    else:
        window2, window3 = _compute_cov3_windows(get_data_randoms, fields=fields, mattrs=mattrs, edges=edges,
                                                 ells2=ells2, ells3=ells3, buffer_size=buffer_size)
    # Store interpolated correlation windows for diagnostics / reuse
    results = {'window_covariance_mesh2_correlation': window2, 'window_covariance_mesh3_correlation': window3}

    # Assemble the windowed (P, B) covariance at the input theory
    covariance = compute_spectrum3_covariance(window2, window3, observable, theory=theory,
                                              shotnoise=shotnoise, cache={}, batch_size=batch_size)
    results['raw'] = covariance
    return results


def compute_covariance(*get_data_randoms, spectrum2=None, spectrum3=None, recon_spectrum2=None, theory=None,
                       RR=None, shotnoise=0., fields=None, mattrs=None, edges=None, ells2=(0, 2, 4),
                       ells3=((0, 0, 0),), buffer_size=50, batch_size=16, smoothing_radius=None,
                       prelim_mattrs=None, split_SS=True, window2=None, window3=None, terms='PBT'):
    r"""
    Compute the joint covariance of the pre-recon 2-point + 3-point spectra and the
    post-reconstruction 2-point correlation function, i.e. of
    (``mesh2_spectrum``, ``mesh3_spectrum``, ``recon_particle2_correlation``).

    The (P_pre, B_pre, P_post) block is first assembled with :mod:`jaxpower.cov3` (see
    :func:`compute_covariance_mesh3_spectrum`), *aliasing* the post-recon field's covariance
    window from the pre-recon one: reconstruction displaces particles but does not change the
    survey selection function, so the same randoms-only window applies to both. The post-recon
    power spectrum block is then projected to configuration space with
    :func:`.correlation2_tools.project_spectrum2_covariance_to_correlation`, reusing the exact
    rotation / RR-window deconvolution / shot-noise machinery used by
    :func:`.correlation2_tools.compute_covariance_particle2_correlation`; the pre-recon P and B
    blocks (and their cross-covariance with the post-recon block) pass through unprojected.

    Parameters
    ----------
    get_data_randoms : callables
        Functions that return dict of 'data' and 'randoms' catalogs for the (single) tracer,
        used to build the covariance windows. See :func:`prepare_jaxpower_particles` for details.
        Note the post-recon field's window is *aliased* from this one (same survey randoms);
        no separate post-recon particle input is required.
    spectrum2 : Mesh2SpectrumPoles
        Measured pre-recon 2-point spectrum multipoles, defining the covariance binning.
    spectrum3 : Mesh3SpectrumPoles
        Measured pre-recon 3-point spectrum multipoles (Sugiyama basis), defining the covariance binning.
    recon_spectrum2 : Mesh2SpectrumPoles
        Measured post-recon 2-point spectrum multipoles, defining the covariance binning of the
        (post, post) block prior to projection. Its ``attrs['recon_smoothing_radius']`` is used
        as the default reconstruction smoothing radius.
    theory : callable, optional
        ``theory(fields)`` returning the 3D P / B / T callables. If ``None``, fit automatically:
        pre-recon P/B via :func:`run_preliminary_fit_mesh3_spectrum`, and post-recon / cross
        kernels via :func:`get_post_recon_spectrum_kernels` (reusing the fitted ``b1``, ``z``, ``f``).
    RR : Count2 or ObservableTree
        Post-recon RR pair counts (e.g. the 'RR' branch of a ``recon_particle2_correlation``
        measurement), consumed by the correlation-space projection.
    shotnoise : float, optional
        Shot noise :math:`(1 + \alpha) / \bar{n}` (shared between pre- and post-recon fields;
        reconstruction only displaces particles, it does not change their discreteness noise).
    fields : tuple, list, optional
        Field name for the pre-recon tracer. Default is ``(0,)``; the post-recon field is
        always labeled ``1`` internally (used only to distinguish theory/window aliasing, not
        exposed as an actual second tracer -- this function is single-tracer only).
    mattrs : dict, optional
        Mesh attributes to define the :class:`jaxpower.ParticleField` objects,
        'boxsize', 'meshsize' or 'cellsize', 'boxcenter'. If ``None``, default attributes are used.
    edges : dict, optional
        Separation binning for the covariance windows. Default is ``{'step': cellsize.min()}``.
    ells2 : tuple, optional
        Multipoles of the two-anchor covariance window. Default is ``(0, 2, 4)``.
    ells3 : list of tuples, optional
        Sugiyama multipoles of the three-anchor covariance window. Default is ``[(0, 0, 0)]``.
    buffer_size : int, optional
        Buffer size for the 3-point window binning; increase for faster computation at the cost of memory.
    batch_size : int, optional
        Batch size for the covariance assembly.
    smoothing_radius : float, optional
        Reconstruction Gaussian smoothing radius. If ``None``, read from
        ``recon_spectrum2.attrs['recon_smoothing_radius']``.
    prelim_mattrs : MeshAttrs, optional
        Box volume assumed in the pre-recon bias preliminary fit; see
        :func:`run_preliminary_fit_mesh3_spectrum`. If ``None``, the measurement's embedding box is used.
    split_SS : bool, optional
        If ``True``, split the calculation of the post-recon shot-noise-shot-noise term;
        ensures much better convergence for low-density samples such as QSO.
    window2, window3 : optional
        Precomputed (interpolated) covariance windows, as returned in the results dictionary
        ('window_covariance_mesh2_correlation', 'window_covariance_mesh3_correlation') of a
        previous call; file paths are read with :func:`lsstypes.read`. Both must be provided
        to skip the window computation from the randoms (``mattrs``, ``edges``, ``ells2``,
        ``ells3``, ``buffer_size`` are then ignored); the field labels must match *fields*.
    terms : str, optional
        Theory ingredients contributing to the covariance: 'P' keeps only the power spectra
        (Gaussian terms, including the pre/post-recon kernels), 'PB' adds the bispectrum
        (connected BB and P x B cross terms), 'PBT' (default) adds the trispectrum.

    Returns
    -------
    results : dict
        Dictionary containing the final covariance ('raw', with blocks (``mesh2_spectrum``,
        ``mesh3_spectrum``, ``recon_particle2_correlation``)) and the interpolated covariance
        window correlations used to build the (P_pre, B_pre, P_post) block
        ('window_covariance_mesh2_correlation', 'window_covariance_mesh3_correlation').
    """
    from jaxpower import create_sharding_mesh, interpolate_window_function, compute_fkp2_covariance_window, compute_fkp2_normalization, FKPField
    from jaxpower.cov3 import compute_spectrum3_covariance
    from .correlation2_tools import project_spectrum2_covariance_to_correlation

    if fields is None:
        fields = [0]
    pre, post = fields[0], 1

    if smoothing_radius is None:
        smoothing_radius = float(np.asarray(recon_spectrum2.attrs['recon_smoothing_radius']).flat[0])

    if theory is None:
        theory_pre = run_preliminary_fit_mesh3_spectrum(spectrum2, spectrum3, mattrs=prelim_mattrs)
        from cosmoprimo.fiducial import DESI
        kernels = get_post_recon_spectrum_kernels(DESI(), z=theory_pre.z, f=theory_pre.f, b1=theory_pre.bias['b1'],
                                                  smoothing_radius=smoothing_radius, shotnoise=shotnoise)

        def theory(flds):
            flds = tuple(flds)
            # Any all-pre combination (P, B, and the PP block's T) is handled exactly as in the
            # pre-recon-only covariance; field identity doesn't matter to theory_pre itself.
            if all(fl == pre for fl in flds):
                return theory_pre(flds)
            if len(flds) == 2 and set(flds) == {pre, post}:
                return kernels['pre_post']
            if flds == (post, post):
                return kernels['post_post']
            # No bispectrum/trispectrum theory involving the post-recon field.
            return None

    # (P_pre, B_pre, P_post) covariance windows: the post-recon field's window is aliased from
    # the pre-recon one (reconstruction does not change the survey selection function, only the
    # theory). compute_spectrum3_covariance queries window2/window3 with many different
    # field-group combinations of {pre, post} -- not just the (pre,pre)/(pre,post)/(post,post)
    # pairs of the top-level blocks, but also e.g. mixed-size groups pulled in by cov3's
    # shot-noise-driven reducible bispectrum/trispectrum terms (fields (pre,pre,post) etc, since
    # pre/post share correlated shot noise). Rather than enumerate every such combination,
    # collapse any field-group touching 'post' to the identical 'pre' one at lookup time.
    if window2 is not None and window3 is not None:
        from pathlib import Path
        if isinstance(window2, (str, Path)): window2 = types.read(window2)
        if isinstance(window3, (str, Path)): window3 = types.read(window3)
        if jax.process_index() == 0:
            logger.info('Using precomputed covariance windows')
    else:
        window2, window3 = _compute_cov3_windows(get_data_randoms, fields=[pre], mattrs=mattrs, edges=edges,
                                                 ells2=ells2, ells3=ells3, buffer_size=buffer_size)
    # Store the real (unwrapped, serializable) windows for diagnostics / reuse
    results = {'window_covariance_mesh2_correlation': window2, 'window_covariance_mesh3_correlation': window3}

    class _AliasFieldWindow:
        """
        Thin wrapper around a cov3-style window2/window3 ObservableTree (or a symmetrization
        pair thereof) that collapses the virtual 'post' field to the real 'pre' one for any
        ``fields``/``fields1``/``fields2``/``fields3`` lookup, before delegating to the
        underlying window's own ``.get(...)``.
        """

        def __init__(self, window):
            self._window = window

        def _dealias(self, value):
            if isinstance(value, tuple):
                return tuple(pre if v == post else v for v in value)
            return pre if value == post else value

        def get(self, **labels):
            labels = {key: (self._dealias(value) if key.startswith('fields') else value) for key, value in labels.items()}
            return self._window.get(**labels)

        def __getattr__(self, name):
            return getattr(self._window, name)

    observable = types.ObservableTree([spectrum2, spectrum3, recon_spectrum2],
                                      fields=[(pre, pre), (pre, pre, pre), (post, post)])

    covariance = compute_spectrum3_covariance(_AliasFieldWindow(window2), _AliasFieldWindow(window3),
                                              observable, theory=_restrict_theory(theory, terms=terms),
                                              shotnoise=shotnoise, cache={}, batch_size=batch_size)

    # Build the cov2-style (flat-label) WW/WS/SW/SS window + fkp_norm needed by the
    # correlation-space projection of the post-recon block, again aliased from the pre-recon
    # field (same survey randoms).
    mattrs_ = mattrs or {}
    with create_sharding_mesh(meshsize=mattrs_.get('meshsize', None)):
        all_particles = prepare_jaxpower_particles(*get_data_randoms, mattrs=mattrs_, add_randoms=['IDS'])
        all_fkp = [FKPField(particles['data'], particles['randoms']) for particles in all_particles]
        mattrs_ = all_fkp[0].attrs
        edges_ = edges if edges is not None else {'step': mattrs_.cellsize.min()}
        split = [(42, fkp.randoms.extra['IDS']) for fkp in all_fkp]
        kw_paint = dict(resampler='tsc', interlacing=3, compensate=True)
        cov2_windows = compute_fkp2_covariance_window(all_fkp, edges=edges_, basis='bessel', los='local',
                                                      fields=[pre], split=split, **kw_paint)
        coords = np.logspace(-2, 8, 8 * 1024)
        cov2_windows = cov2_windows.map(lambda window: interpolate_window_function(window, coords=coords), level=1)

        # Alias the post-recon field's WW/WS/SW/SS windows from the pre-recon one.
        aliased = []
        for label, branch in cov2_windows.items(level=1):
            base_branch = branch.get(fields=(pre,) * 4)
            branch = branch.insert([base_branch], fields=[(post,) * 4])
            aliased.append(branch)
        cov2_windows = types.ObservableTree(aliased, types=cov2_windows.types)

        def get_fkp_norm(*all_randoms):
            fkp_norm = compute_fkp2_normalization(*all_randoms, cellsize=None, split=(42, 'index'))
            for randoms in all_randoms + (all_randoms[-1],) * (2 - len(all_randoms)):
                fkp_norm /= randoms.sum()
            return fkp_norm

        fkp_norm_pre = get_fkp_norm(*[particles['randoms'] for particles in all_particles])

    fkp_norm = {(pre, pre): fkp_norm_pre, (post, post): fkp_norm_pre}

    covariance = project_spectrum2_covariance_to_correlation(covariance, RR, fkp_norm, windows=cov2_windows,
                                                              fields=[(post, post)], split_SS=split_SS)
    results['raw'] = covariance
    return results


def compute_box_mesh3_spectrum(*get_data, mattrs=None,
                                basis='sugiyama-diagonal', ells=[(0, 0, 0), (2, 0, 2)], edges=None, los='z',
                                buffer_size=0, mask_edges=None, cache=None):
    r"""
    Compute the 3-point spectrum multipoles for a cubic box using :mod:`jaxpower`.

    Parameters
    ----------
    get_data : callables
        Functions that return tuples of (data, [shifted]) catalogs.
        See :func:`prepare_jaxpower_particles` for details.
    mattrs : dict, optional
        Mesh attributes to define the :class:`jaxpower.ParticleField` objects.
        See :func:`prepare_jaxpower_particles` for details.
    ells : list of int, optional
        List of multipole moments to compute. Default is (0, 2, 4).
    edges : dict, optional
        Edges for the binning; array or dictionary with keys 'start' (minimum :math:`k`), 'stop' (maximum :math:`k`), 'step' (:math:`\Delta k`).
        If ``None``, default step of :math:`0.001 h/\mathrm{Mpc}` is used.
        See :class:`jaxpower.BinMesh3SpectrumPoles` for details.
    los : {'x', 'y', 'z', array-like}, optional
        Line-of-sight direction. If 'x', 'y', 'z' use fixed axes, or provide a 3-vector.
    cache : dict, optional
        Cache to store binning class (can be reused if ``meshsize`` and ``boxsize`` are the same).
        If ``None``, a new cache is created.

    Returns
    -------
    spectrum : Mesh3SpectrumPoles
        The computed 3-point spectrum multipoles.
    """
    from jaxpower import (create_sharding_mesh, FKPField, compute_fkp3_shotnoise, compute_box3_normalization, BinMesh3SpectrumPoles, compute_mesh3_spectrum, compute_fkp3_shotnoise)

    # Use provided mesh attributes or empty dict
    mattrs = mattrs or {}
    # Set up distributed computation mesh
    with create_sharding_mesh(meshsize=mattrs.get('meshsize', None)):
        # Load and prepare particle catalogs (data and optional shifted)
        all_particles = prepare_jaxpower_particles(*get_data, mattrs=mattrs)
        # Extract JAX power attributes from particles
        attrs = _get_jaxpower_attrs(*all_particles)
        # Set line-of-sight direction
        attrs.update(los=los)
        # Get mesh attributes from first particle set
        mattrs = all_particles[0]['data'].attrs

        # Initialize or retrieve cached binning object
        if cache is None: cache = {}
        # Retrieve cached binning or None if not cached
        bin = cache.get(f'bin_mesh3_spectrum_{basis}', None)
        # Set default k-space binning (finer for sugiyama)
        if edges is None: edges = {'step': 0.02 if 'scoccimarro' in basis else 0.005}

        # Create binning if not cached or if mesh parameters changed
        if bin is None or not np.all(bin.mattrs.meshsize == mattrs.meshsize) or not np.allclose(bin.mattrs.boxsize, mattrs.boxsize):
            bin = BinMesh3SpectrumPoles(mattrs, edges=edges, basis=basis, ells=ells, buffer_size=buffer_size, mask_edges=mask_edges)
        # Cache the binning object
        cache.setdefault(f'bin_mesh3_spectrum_{basis}', bin)

        # Extract data catalogs (periodic box has no randoms)
        all_data = [particles['data'] for particles in all_particles]
        # Compute normalization for periodic box (simpler than survey case)
        norm = compute_box3_normalization(*all_data, bin=bin)

        # Create FKP fields for shot noise (use shifted if available, else use data)
        all_fkp = [FKPField(particles['data'], particles['shifted']) if particles.get('shifted', None) is not None else particles['data'] for particles in all_particles]
        # Clear particle data from memory
        del all_particles
        # Compute shot noise for periodic box
        num_shotnoise = compute_fkp3_shotnoise(*all_fkp, bin=bin, fields=None)

        # Painting parameters: TSC with interlacing
        kw = dict(resampler='tsc', interlacing=3, compensate=True)
        # Paint all FKP fields onto mesh (store as real arrays to save memory)
        all_mesh = []
        for fkp in all_fkp:
            mesh = fkp.paint(**kw, out='real')
            # Subtract mean
            all_mesh.append(mesh - mesh.mean())
        # Clear FKP fields from memory
        del all_fkp

        # JIT-compile spectrum computation for performance
        jitted_compute_mesh3_spectrum = jax.jit(compute_mesh3_spectrum, static_argnames=['los'])
        # Compute bispectrum from mesh grids
        spectrum = jitted_compute_mesh3_spectrum(*all_mesh, bin=bin, los=los)
        # Attach normalization and shot noise
        spectrum = spectrum.clone(norm=norm, num_shotnoise=num_shotnoise)
        # Propagate attributes to each multipole pole
        spectrum = spectrum.map(lambda pole: pole.clone(attrs=attrs))
        # Attach global attributes to spectrum
        spectrum = spectrum.clone(attrs=attrs)
        # Wait for computation to complete on all devices
        jax.block_until_ready(spectrum)
        if jax.process_index() == 0:
            logger.info('Mesh-based computation finished')
    return spectrum


def compute_covariance_box_mesh3_spectrum(spectrum2: types.Mesh2SpectrumPoles, spectrum3: types.Mesh3SpectrumPoles,
                                          theory, shotnoise: float=0., mattrs=None, terms='PBT'):
    r"""
    Compute the joint 2-point + 3-point spectrum covariance for a box with :mod:`jaxpower`.

    Periodic-box analog of :func:`compute_covariance_mesh3_spectrum`: the covariance windows are
    replaced by the box volume, i.e. ``compute_spectrum3_covariance(mattrs, mattrs, ...)`` (analytic
    Sugiyama et al. formulas: Gaussian PPP, single-delta BB and PT terms), cf.
    ``jax-power/scripts/example_fit_bias_covariance.py``.

    Parameters
    ----------
    spectrum2 : Mesh2SpectrumPoles
        Measured (or smooth theory) 2-point spectrum multipoles, defining the covariance binning.
    spectrum3 : Mesh3SpectrumPoles
        Measured (or smooth theory) 3-point spectrum multipoles (Sugiyama basis), defining the
        covariance binning.
    theory : callable
        ``theory(fields)`` returning the 3D P / B / T callables, with bias parameters keyed by
        the field labels (0-based); see :func:`run_preliminary_fit_mesh3_spectrum`.
    shotnoise : float, optional
        Shot noise :math:`(1 + \alpha) / \bar{n}`.
    mattrs : dict, optional
        Mesh attributes setting the box volume, 'boxsize', 'meshsize' or 'cellsize', 'boxcenter'.
        If ``None``, taken from ``spectrum2.attrs`` (the embedding box).
    terms : str, optional
        Theory ingredients contributing to the covariance: 'P' keeps only the power spectrum
        (Gaussian PP / PPP terms), 'PB' adds the bispectrum (connected BB and P x B cross
        terms), 'PBT' (default) adds the trispectrum (PP- and BB-block trispectrum terms).

    Returns
    -------
    covariance : CovarianceMatrix
        The computed joint (P, B) periodic-box covariance.
    """
    # Compute the analytic (Gaussian PPP + single-delta BB/PT) covariance for a periodic box
    from jaxpower import create_sharding_mesh, MeshAttrs
    from jaxpower.cov3 import compute_spectrum3_covariance

    # Use the embedding box of the measurement if no mesh attributes are provided
    if mattrs is None:
        mattrs = {name: spectrum2.attrs[name] for name in ['boxsize', 'boxcenter', 'meshsize']}

    # The measured observables only set the covariance binning; theory (not their values) is used
    observable = types.ObservableTree([spectrum2, spectrum3], fields=[(0, 0), (0, 0, 0)])

    with create_sharding_mesh(meshsize=mattrs.get('meshsize', None)):
        mattrs = MeshAttrs(**mattrs)
        # Passing mattrs in place of (window2, window3) selects the periodic (no-window) branch
        covariance = compute_spectrum3_covariance(mattrs, mattrs, observable, theory=_restrict_theory(theory, terms=terms),
                                                  shotnoise=shotnoise, cache={})
    return covariance