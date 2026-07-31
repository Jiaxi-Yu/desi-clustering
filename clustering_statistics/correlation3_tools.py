"""
Configuration-space 3-point clustering measurements.

Main functions
--------------
* `compute_particle3_angular_upweights`: Derive angular upweights for fiber-collision mitigation.
"""

import time
import logging
import itertools
from functools import partial

import numpy as np
import jax
from jax import numpy as jnp
from jax.sharding import PartitionSpec as P

import lsstypes as types
from .tools import _format_bitweights
from .correlation2_tools import prepare_cucount_particles, _get_particle_combinations, _remove_phantom_particles


logger = logging.getLogger('correlation3')


_identity_fn = lambda x: x


def _digitize_angular(particles, wattrs, nside=128, sharding_mesh=None):
    import healpy as hp
    if sharding_mesh.axis_names:
        particles = jax.jit(_identity_fn, out_shardings=jax.sharding.NamedSharding(sharding_mesh, spec=P(None)))(particles)
    pix = hp.vec2pix(nside, *particles.get('positions').T, nest=False)
    weights = wattrs(particles)
    npix = hp.nside2npix(nside)
    pix_weights = np.bincount(pix, weights=weights, minlength=npix)
    mask = pix_weights > 0
    pix_weights = pix_weights[mask]
    pix_positions = np.column_stack(hp.pix2vec(nside, np.flatnonzero(mask), nest=False))
    particles = particles.clone(positions=pix_positions, weights=pix_weights, exchange=False)
    #print(time.time() - t0, flush=True)
    return particles


def _digitize_cartesian(particles, wattrs, cellsize=40., sharding_mesh=None):
    if sharding_mesh.axis_names:
        particles = jax.jit(_identity_fn, out_shardings=jax.sharding.NamedSharding(sharding_mesh, spec=P(None)))(particles)
    positions = particles.get('positions')
    weights = wattrs(particles)
    extent = positions.min(axis=0), positions.max(axis=0)
    boxsize = extent[1] - extent[0]
    meshsize = jnp.ceil(boxsize / cellsize).astype('i4')
    boxsize = meshsize * cellsize
    index = (positions - extent[0]) / boxsize * meshsize
    index = jnp.rint(index).astype('i4')
    index_weights = jnp.zeros(tuple(meshsize))
    index_weights = index_weights.at[tuple(jnp.unstack(index, axis=-1))].add(weights).ravel()
    index = jnp.meshgrid(*(jnp.arange(size) for size in meshsize), indexing='ij')
    index = jnp.column_stack([idx.ravel() for idx in index])
    index_positions = index * cellsize + extent[0]
    mask = index_weights != 0.
    return particles.clone(positions=index_positions[mask], weights=index_weights[mask], index_value=None, exchange=False)


def compute_particle3_angular_upweights(*get_data_randoms):
    """
    Compute angular upweights (AUW) from fibered and parent data catalogs.

    Parameters
    ----------
    get_data_randoms : callables
        Functions returning dicts with 'fibered_data', 'parent_data', and optionally
        'fibered_randoms', 'parent_randoms'. Catalogs must contain 'RA', 'DEC',
        'INDWEIGHT', optionally 'BITWEIGHT'.

    Returns
    -------
    auw : ObservableTree
        Angular upweights as an ObservableTree.
    """
    import itertools, logging
    import numpy as np
    import jax
    from cucount.jax import Particles, BinAttrs, create_sharding_mesh
    from lsstypes import ObservableLeaf, ObservableTree

    with create_sharding_mesh() as sharding_mesh:
        all_particles = prepare_cucount_particles(*get_data_randoms, positions_type='rd')
        all_particles_fibered, all_particles_parent = [], []
        for particles in all_particles:
            all_particles_fibered.append({'data': particles['fibered_data'], 'randoms': particles['fibered_randoms']})
            all_particles_parent.append({'data': particles['parent_data'], 'randoms': particles['parent_randoms']})

        # 0.5-dex bins everywhere (the upweights are flat below 0.003 deg and vary on
        # >~ 1-dex scales beyond 0.1 deg), refined to 0.1 dex across the collision kernel
        # (0.003 - 0.1 deg, steep + double-close transition at ~0.05 deg; exact counting
        # there, so fine bins are cheap). Wide bins at wide angles let
        # compute_particle3_resol digitize coarsely while keeping the healpix pixel
        # below ~1/4 of the bin width.
        theta = np.concatenate([
            10**np.arange(-4., -2.5, 0.5),
            10**np.arange(-2.5, -1., 0.1),
            10**np.arange(-1., np.log10(180.), 0.5),
            [180.],
        ])
        battrs = BinAttrs(theta=theta)

        counts_fibered = _compute_particle3_correlation_close_pair_correction(all_particles_fibered, [battrs] * 3, auw=None, cut=None, veto23=None, normalize_randoms=False)
        counts_parent = _compute_particle3_correlation_close_pair_correction(all_particles_parent, [battrs] * 3, auw=None, cut=None, veto23=None, normalize_randoms=False)

    coords = ['theta1', 'theta2', 'theta3']
    kw = dict(coords=coords)
    for coord in coords:
        kw[coord] = battrs.coords('theta')
        kw[f'{coord}_edges'] = battrs.edges('theta')

    auw = {}
    for combinations in counts_fibered:
        value_fibered = counts_fibered[combinations].value()
        value_parent = counts_parent[combinations].value()
        auw[combinations] = ObservableLeaf(value=np.where(value_fibered == 0., 1., value_parent / value_fibered), **kw)
        auw[combinations + 'parent'] = counts_parent[combinations]
        auw[combinations + 'fibered'] = counts_fibered[combinations]

    return ObservableTree(list(auw.values()), triplets=list(auw.keys()))


def compute_particle3_correlation(*get_data_randoms, battrs: dict=None, zeff: dict=None, auw=None, cut=None, split_randoms: float | tuple=False):
    """
    Compute 3-point correlation function using :mod:`cucount.jax`.

    Parameters
    ----------
    get_data_randoms : callables
        Functions returning dicts with 'data', 'randoms' (optionally 'shifted').
        Catalogs must contain 'POSITION', 'INDWEIGHT', optionally 'BITWEIGHT'.
    battrs : dict, optional
        Bin attributes for cucount.jax.BinAttrs.
    zeff : dict, optional
        Options to estimate the effective redshift, e.g. ``{'cellsize': 10.}``.
    auw : ObservableTree, optional
        Angular upweights to apply.
    cut : bool, optional
        If provided, apply a theta-cut of (0, 0.05) degrees.
    split_randoms : float, tuple
        If provided, ratio of randoms / data to split the (concatenated) randoms or shifted catalogs into.
        If a tuple, (ratio of randoms / data, number of random splits).

    Returns
    -------
    Count3Correlation
    """
    from cucount.jax import BinAttrs, WeightAttrs
    from cucount.types import count3
    from lsstypes import Count3Correlation
    from jaxpower import create_sharding_mesh
    from .spectrum2_tools import prepare_jaxpower_particles, compute_fkp_effective_redshift

    if zeff is None: zeff = {'boxpad': 1.1, 'cellsize': 10.}
    kw_zeff = dict(zeff)

    def merge_randoms(catalog):
        if not isinstance(catalog, (tuple, list)): return catalog
        return catalog[0].concatenate(catalog)

    get_randoms = [lambda f=f: {'randoms': merge_randoms(f()['randoms'])} for f in get_data_randoms]

    with create_sharding_mesh(meshsize=kw_zeff.get('meshsize', None)):
        all_particles = prepare_jaxpower_particles(*get_randoms, mattrs=kw_zeff, add_randoms=['IDS'])
        all_randoms = [particles['randoms'] for particles in all_particles]
        seed = [(42, randoms.extra['IDS']) for randoms in all_randoms]
        zeff, norm_zeff = compute_fkp_effective_redshift(*all_randoms, order=3, split=seed,
                                                         resampler='cic', return_fraction=True)

    with create_sharding_mesh():
        if battrs is None:
            battrs = dict(s=np.linspace(0., 160., 161), pole=(list(range(6)), 'firstpoint'))
        battrs = BinAttrs(**battrs)
        mattrs = None

        all_particles = prepare_cucount_particles(*get_data_randoms, split_randoms=split_randoms)
        if jax.process_index() == 0: logger.info('All particles on the device')

        def count3split(*particles, wattrs=None):
            kw = dict(battrs12=battrs, battrs13=battrs, mattrs1=mattrs, mattrs2=mattrs, mattrs3=mattrs, wattrs=wattrs)
            nsplits = [len(p) if isinstance(p, list) else 0 for p in particles]
            if any(nsplits):
                nsplit = next(n for n in nsplits if n)
                particles = list(particles)
                for ip, particle in enumerate(particles):
                    if isinstance(particle, list):
                        assert len(particle) == nsplit
                    else:
                        particles[ip] = [particle] * nsplit
                counts = [count3(*p, **kw)['weight'] for p in zip(*particles)]
                return types.sum(counts)
            return count3(*particles, **kw)['weight']

        counts = {}
        for combinations in itertools.product(['D', 'S'], repeat=3):
            combinations = ''.join(combinations)
            if jax.process_index() == 0: logger.info(f'Computing {combinations} term.')
            particles, _combinations = _get_particle_combinations(combinations, all_particles)
            wattrs = WeightAttrs()
            counts[combinations] = count3split(*particles, wattrs=wattrs)

        if 'RRR' not in counts:
            if all(particles.get('shifted', None) is None for particles in all_particles):
                counts['RRR'] = counts['SSS']
            else:
                counts['RRR'] = count3split(*[particles['randoms'] for particles in all_particles])

    correlation = Count3Correlation(estimator='landyszalay', **counts)
    correlation.attrs.update(zeff=zeff / norm_zeff, norm_zeff=norm_zeff)

    # Galaxy pairs at small angular separation
    results = compute_particle3_correlation_close_pair_correction(*all_particles, correlation=correlation, battrs=battrs, auw=auw, cut=cut)
    return results


def compute_particle3_correlation_close_pair_correction(*get_data_randoms, correlation, battrs=None, auw=None, cut=None, split_randoms: bool | float=False):
    """
    Compute and apply close-pair corrections to 3-point correlation function.

    Parameters
    ----------
    get_data_randoms : callables
        Functions returning dicts with 'data', 'randoms' (optionally 'shifted').
        Catalogs must contain 'POSITION', 'INDWEIGHT', optionally 'BITWEIGHT'.
    correlation : ObservableTree
        Input correlation function to add close pair correction to.
    battrs : dict, optional
        Bin attributes for :class:`cucount.jax.BinAttrs`.
    auw : ObservableTree, optional
        Angular upweights to apply.
    cut : bool, optional
        If provided, apply a theta-cut of (0, 0.05) degrees.
    split_randoms : float, tuple
        If provided, ratio of randoms / data to split the (concatenated) randoms or shifted catalogs into.
        If a tuple, (ratio of randoms / data, number of random splits).

    Returns
    -------
    correlation : Count3Correlation
    """
    from cucount.jax import create_sharding_mesh, BinAttrs

    with create_sharding_mesh() as sharding_mesh:
        if callable(get_data_randoms[0]):
            all_particles = prepare_cucount_particles(*get_data_randoms, split_randoms=split_randoms)
            if jax.process_index() == 0: logger.info('All particles on the device')
        else:
            all_particles = list(get_data_randoms)

        if battrs is None:
            edges = correlation.edges()
            ells = getattr(correlation.get('DDD'), 'ells', [])
            ells = [[ell[idim] for ell in ells] for idim in [0, 1]]

            battrs = []
            for idim in range(3):
                for name in ['s', 'theta']:
                    coord_name = f'{name}{idim + 1}'
                    if coord_name in edges:
                        edge = edges[coord_name]
                        d = {name: np.append(edge[:, 0], edge[-1, 1])}
                        if idim < len(ells) and ells[idim]:
                            d['pole'] = (ells[idim], 'firstpoint')
                        battrs.append(BinAttrs(**d))

        results = {}
        results['raw'] = correlation
        corrections = {'auw': auw, 'cut': cut}
        for correction_name in corrections:
            if corrections[correction_name] is not None:
                correction = _compute_particle3_correlation_close_pair_correction(all_particles, battrs, **{correction_name: corrections[correction_name]}, normalize_randoms=False, veto23=None)
                results[correction_name] = _apply_particle3_correlation_close_pair_correction(correlation, correction)

    return results


def _apply_particle3_correlation_close_pair_correction(correlation, correction):
    """Apply additive corrections to a :class:`Count3Correlation`."""
    from .correlation2_tools import _apply_particle2_correlation_close_pair_correction
    return _apply_particle2_correlation_close_pair_correction(correlation, correction)



def _compute_particle3_correlation_close_pair_correction(all_particles, battrs, auw=None, cut=None, veto23: bool=None, normalize_randoms: bool=True):
    """
    Compute close-pair corrections to 3-point counts.
    Returns additive correction counts keyed by count name.
    """
    from cucount.jax import BinAttrs, SelectionAttrs, WeightAttrs, get_sharding_mesh
    from cucount.types import count3close

    sharding_mesh = get_sharding_mesh()

    input_all_particles = []
    for particles in all_particles:
        particles = dict(particles)
        if 'shifted' in particles:
            particles.pop('randoms')
        input_all_particles.append(particles)
    all_particles = input_all_particles

    _identity_fn = lambda x: x
    if veto23:
        veto23 = SelectionAttrs(s=(0., 1e-3))
    else:
        veto23 = None

    if not isinstance(battrs, (tuple, list)):
        battrs = [battrs] * 2
    battrs = [battr if isinstance(battr, BinAttrs) else BinAttrs(**battr) for battr in battrs]
    battrs12, battrs13, *battrs23 = battrs
    battrs23 = battrs23[0] if battrs23 else None
    kw_battrs = dict(battrs12=battrs12, battrs13=battrs13, battrs23=battrs23)

    sattrs = SelectionAttrs(theta=(0., 0.05))
    wattrs = WeightAttrs()

    def _map_catalogs(obj, fn):
        if isinstance(obj, list):
            return [fn(o) for o in obj]
        return fn(obj)

    if normalize_randoms:
        def normalize_randoms(data, randoms, wattrs=WeightAttrs()):
            data_weights, randoms_weights = wattrs(data), wattrs(randoms)
            return randoms.clone(weights=data_weights.sum() / randoms_weights.sum() * randoms_weights)

    for particles in all_particles:
        #particles['data'] = _remove_phantom_particles(particles['data'], sharding_mesh=sharding_mesh)
        if normalize_randoms:
            for name in ['shifted', 'randoms']:
                if name in particles:
                    particles[name] = _map_catalogs(particles[name], lambda randoms, data=particles['data']: normalize_randoms(data, randoms, wattrs=wattrs))

    def count3close_split(*particles, **kw):
        nsplits = [len(p) if isinstance(p, list) else 0 for p in particles]
        if any(nsplits):
            nsplit = next(n for n in nsplits if n)
            particles = list(particles)
            for ip, particle in enumerate(particles):
                if isinstance(particle, list):
                    assert len(particle) == nsplit
                else:
                    particles[ip] = [particle] * nsplit
            counts = [count3close(*p, **kw)['weight'] for p in zip(*particles)]
            return types.sum(counts)
        return count3close(*particles, **kw)['weight']

    def compute_particle3_resol(*all_particles, sattrs, wattrs, close_pairs=[(1, 2), (1, 3), (2, 3)], **kw):
        """Compute close-pair counts, optionally digitizing the non-close third particle."""

        resol_coord = 's' if 's' in battrs12.coords() else 'theta'
        battrs_resol = battrs12 if resol_coord in battrs12.coords() else battrs13
        sepmax = battrs_resol.coords()[resol_coord].max()

        if resol_coord == 's':
            limits = [0., 100., 500., 2000.]
            limits = [lim for lim in limits if lim < sepmax] + [sepmax]
            resols = [None, 50., 100., 500.]
        else:  # theta
            import healpy as hp
            # Digitization snaps the digitized particle to healpix pixel centers, which perturbs
            # theta by up to ~1 pixel: start each resolution shell at the first bin edge whose bin
            # is >= 4 pixels wide, so the perturbation stays below ~1/4 bin width and shell seams
            # sit on bin edges (a seam cutting through a bin mixes two pixelizations in one bin).
            edges = np.unique(battrs_resol.edges(resol_coord))
            widths = np.diff(edges)
            # nside candidates spaced by x8: each shell is a separate count3close launch
            # with sizable fixed cost, so few coarse steps beat many fine ones.
            limits, resols = [0.], [None]
            for nside in [512, 64, 16]:
                pixsize = np.degrees(hp.nside2resol(nside))
                wide_enough = np.flatnonzero(widths >= 4. * pixsize)
                if not wide_enough.size:
                    break
                lim = edges[wide_enough[0]]
                if lim > limits[-1]:
                    limits.append(lim)
                    resols.append(nside)
            limits = [lim for lim in limits if lim < sepmax] + [sepmax]

        with_veto = True
        if len(close_pairs) == 1:
            if len(limits) > 2:
                limits = [0.] + limits[2:]
            resols = resols[1:]
            with_veto = False

        all_particles = list(all_particles) + [all_particles[-1]] * (3 - len(all_particles))
        results = []

        for resol_limit, resol in zip(zip(limits[:-1], limits[1:]), resols):
            t0 = time.time()
            sattrs_limit = SelectionAttrs(**{resol_coord: resol_limit})

            def digitize(particles, concatenate=True):
                if isinstance(particles, list):
                    if concatenate:
                        return digitize(particles[0].concatenate(particles, local=True) if len(particles) > 1 else particles[0])
                    return [digitize(particle) for particle in particles]
                if resol is None:
                    return particles
                if resol_coord == 's':
                    return _digitize_cartesian(particles, wattrs, cellsize=resol, sharding_mesh=sharding_mesh)
                return _digitize_angular(particles, wattrs, nside=resol, sharding_mesh=sharding_mesh)

            if (1, 2) in close_pairs:
                all_particles_resol = list(all_particles)
                all_particles_resol[2] = digitize(all_particles_resol[2])
                result12 = count3close_split(*all_particles_resol, **kw_battrs, sattrs12=sattrs,
                                             sattrs13=sattrs_limit, veto23=veto23,
                                             wattrs=wattrs, close_pair=(1, 2))
                results += [result12]

            if (1, 3) in close_pairs:
                all_particles_resol = list(all_particles)
                all_particles_resol[1] = digitize(all_particles_resol[1])
                result13 = count3close_split(*all_particles_resol, **kw_battrs, sattrs13=sattrs,
                                             veto12=sattrs if with_veto else None,
                                             sattrs12=sattrs_limit, veto23=veto23,
                                             wattrs=wattrs, close_pair=(1, 3))
                results += [result13]

            if (2, 3) in close_pairs:
                all_particles_resol = list(all_particles)
                all_particles_resol[0] = digitize(all_particles_resol[0])
                result23 = count3close_split(*all_particles_resol, **kw_battrs, sattrs23=sattrs,
                                             veto12=sattrs if with_veto else None,
                                             veto13=sattrs if with_veto else None,
                                             sattrs12=sattrs_limit, veto23=veto23,
                                             wattrs=wattrs, close_pair=(2, 3), shard_particle=2)
                results += [result23]

            if jax.process_index() == 0:
                logger.info(f'Computed correction for {close_pairs}, {resol_limit} in {time.time() - t0:.3f} s')

        def sum_counts(leaves):
            return leaves[0].clone(counts=sum(leaf.values('counts') for leaf in leaves), norm=leaves[0].values('norm'))

        return types.tree_map(sum_counts, results, level=None, is_leaf=lambda *args: False)

    correction = {}
    if cut is not None:
        wattrs = WeightAttrs()
        all_particles_cut = []
        for particles in all_particles:
            data = particles['data'].clone(weights=wattrs(particles['data']))
            for name in ['shifted', 'randoms']:
                if name in particles:
                    if isinstance(particles[name], list):
                        for randoms in particles[name]:
                            data = data.concatenate([data, randoms.clone(weights=-randoms.get('individual_weight')[0])])
                    else:
                        data = data.concatenate([data, particles[name].clone(weights=-particles[name].get('individual_weight')[0])])
            all_particles_cut.append(data)
        counts = compute_particle3_resol(*all_particles_cut, sattrs=sattrs, wattrs=wattrs)
        correction['DDD'] = counts.clone(value=-counts.value())
        return correction

    bitwise = angular = None
    with_bitweights = all_particles[0]['data'].get('bitwise_weight')
    if with_bitweights:
        raise NotImplementedError('bitweights not supported')

    all_particles = list(all_particles) + [all_particles[-1]] * (3 - len(all_particles))

    for combinations in itertools.product(['D', 'S'], repeat=3):
        combinations = ''.join(combinations)
        if combinations.count('D') < 2:
            continue
        particles, _combinations = _get_particle_combinations(combinations, all_particles)
        if jax.process_index() == 0:
            logger.info(f'Computing contribution of close pairs to {_combinations}.')

        angular = None
        if auw is not None:
            if _combinations not in auw.triplets:
                continue
            angular = dict(edges=[np.append(edge[:, 0], edge[-1, 1]) for edge in auw.get(_combinations).edges().values()],
                           weight=auw.get(_combinations).value())
            if not with_bitweights:
                angular['weight'] = angular['weight'] - 1.
            if jax.process_index() == 0:
                logger.info('Applying AUW.')

        wattrs = WeightAttrs(bitwise=bitwise, angular=angular)
        close_pairs = [(1, 2), (1, 3), (2, 3)]
        if combinations.count('S'):
            close_pairs = [close_pair for close_pair in close_pairs if 1 + combinations.index('S') not in close_pair]
        correction[_combinations] = compute_particle3_resol(*particles, sattrs=sattrs, wattrs=wattrs, close_pairs=close_pairs)

    return correction


def compute_box_particle3_correlation(*get_data, battrs: dict=None, mattrs: dict=None, nran: int=10, split_randoms: bool | float=False):
    """
    Compute 3-point correlation function using :mod:`cucount.jax`.

    Parameters
    ----------
    get_data : callables
        Functions returning dicts with 'data' and optionally 'shifted'.
    battrs : dict, optional
        Bin attributes for cucount.jax.BinAttrs.
    mattrs : dict, optional
        Mesh attributes, typically with 'boxsize' and 'boxcenter'.
    split_randoms : float, tuple
        If provided, ratio of shifted randoms / data to split the (concatenated) shifted catalogs into.
        If a tuple, (ratio of randoms / data, number of random splits).

    Returns
    -------
    Count3Correlation
    """
    from cucount.jax import BinAttrs, WeightAttrs, MeshAttrs
    from cucount.types import count3, count3_analytic
    from lsstypes import Count3Correlation
    from jaxpower import create_sharding_mesh, generate_uniform_particles

    with create_sharding_mesh():

        if battrs is None:
            battrs = dict(s=np.linspace(0., 180., 181), pole=(list(range(6)), 'z'))
        battrs = BinAttrs(**battrs)

        mattrs = mattrs or {}
        wattrs = WeightAttrs()

        all_particles = prepare_cucount_particles(*get_data, split_randoms=split_randoms)

        if jax.process_index() == 0:
            logger.info('All particles on the device')

        mattrs = MeshAttrs(*[particles['data'] for particles in all_particles], battrs=battrs, **mattrs)

        # Decide S vs R mode
        use_shifted = all((particles.get('shifted', None) is not None) for particles in all_particles)

        # Generate randoms if needed
        if not use_shifted:
            for ip, particles in enumerate(all_particles):
                data = particles['data']
                if jax.process_index() == 0:
                    logger.info(f'Generating randoms for catalogue {ip}')

                particles['randoms'] = [generate_uniform_particles(mattrs, size=data.size, seed=(42 + ip * nran + iran, 'index'), exchange=True, backend='mpi', return_inverse=True) for iran in range(nran)]

        # Helper
        def count3split(*particles, wattrs=None):
            kw = dict(battrs12=battrs, battrs13=battrs, mattrs1=mattrs, mattrs2=mattrs, mattrs3=mattrs, wattrs=wattrs, norm=1.)

            nsplits = [len(p) if isinstance(p, list) else 0 for p in particles]

            if any(nsplits):
                nsplit = next(n for n in nsplits if n)
                particles = list(particles)

                for ip, particle in enumerate(particles):
                    if isinstance(particle, list):
                        assert len(particle) == nsplit
                    else:
                        particles[ip] = [particle] * nsplit

                counts = [count3(*p, **kw)['weight'] for p in zip(*particles)]
                return types.sum(counts)

            return count3(*particles, **kw)['weight']

        counts = {}
        # Analytic RRR
        counts['RRR'] = count3_analytic(battrs12=battrs, battrs13=battrs, mattrs1=mattrs, mattrs2=mattrs, mattrs3=mattrs)

        for combinations in itertools.product(['D', 'S'], repeat=3):
            combinations = ''.join(combinations)
            if jax.process_index() == 0: logger.info(f'Computing {combinations} term.')
            particles, combinations = _get_particle_combinations(combinations, all_particles)
            wattrs = WeightAttrs()
            counts[combinations] = count3split(*particles, wattrs=wattrs)

    return Count3Correlation(estimator='landyszalay', **counts)