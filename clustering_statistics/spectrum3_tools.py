"""
Fourier-space 3-point clustering measurements.

Main functions
--------------
* `compute_mesh3_spectrum`: Main bispectrum measurement backend.
* `compute_window_mesh3_spectrum`: Compute bispectrum window.
* `compute_box_mesh3_spectrum`: Measure the bispectrum in periodic boxes.
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
            results[name] = _apply_mesh3_spectrum_close_pair_correction(spectrum, correction)
    return results


def _compute_mesh3_spectrum_close_pair_correction(all_particles, edges=None, ells: list=None, auw=None, cut=None):
    """Compute and apply close-pair corrections."""

    from cucount.jax import BinAttrs
    from lsstypes.types import convert_ells

    if edges is None:
        edges = np.linspace(1e-3, 8000., 3001)
        #edges = np.linspace(1e-3, 8000., 4001)
        #edges = np.linspace(1e-3, 400., 401)
        #edges = np.geomspace(1e-3, 8000., 3001)
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