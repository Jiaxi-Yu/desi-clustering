"""
Configuration-space 2-point clustering measurements.

Main functions
--------------
* `compute_particle2_correlation`: Measure the cutsky 2PCF from pair counts (includes jackknife utility).
* `compute_particle2_angular_upweights`: Derive angular upweights for fiber-collision mitigation.
* `compute_box_particle2_correlation`: Measure the 2PCF in cubic boxes.
"""

import logging
import itertools
from functools import partial
import warnings

import numpy as np
import jax
from jax import numpy as jnp
from jax.sharding import PartitionSpec as P

import lsstypes as types
from .tools import _format_bitweights


logger = logging.getLogger('particle2_correlation')


def compute_particle2_angular_upweights(*get_data, battrs=None):
    """
    Compute angular upweights (AUW) from fibered and parent data catalogs.

    Parameters
    ----------
    get_data : callables
        Functions returning dicts with 'fibered_data' and 'parent_data'.
        Catalogs must contain 'RA', 'DEC', 'INDWEIGHT', optionally 'BITWEIGHT'.

    Returns
    -------
    auw : ObservableTree
        Angular upweights as an ObservableTree with 'DD' leaf.
    """
    from cucount.jax import BinAttrs, create_sharding_mesh
    from lsstypes import ObservableLeaf, ObservableTree

    with create_sharding_mesh():
        all_particles = prepare_cucount_particles(*get_data, positions_type='rd')
        all_fibered_particles = [{'data': particles['fibered_data']} for particles in all_particles]
        all_parent_particles = [{'data': particles['parent_data']} for particles in all_particles]
        if jax.process_index() == 0:
            logger.info('All particles on the device')

        if battrs is None:
            battrs = {'theta': 10**np.arange(-5, -1 + 0.1, 0.1)}
            #battrs = {'theta': 1e-5 + np.arange(0., 0.1 + 0.001, 0.005)}
        battrs = BinAttrs(**battrs)

        #theta_limit = (0., battrs.edges('theta').max())
        counts_fibered = _compute_particle2_correlation_close_pair_correction(all_fibered_particles, battrs, auw=None, cut=None, normalize_randoms=False) #, theta_limit=theta_limit)
        counts_parent = _compute_particle2_correlation_close_pair_correction(all_parent_particles, battrs, auw=None, cut=None, normalize_randoms=False) #, theta_limit=theta_limit)

    kw = dict(theta=battrs.coords('theta'), theta_edges=battrs.edges('theta'), coords=['theta'])
    auw = {}
    for combinations in counts_fibered:
        auw[combinations] = ObservableLeaf(value=np.where(counts_fibered[combinations].value() == 0., 1., counts_parent[combinations].value() / counts_fibered[combinations].value()), **kw)
        auw[combinations + 'parent'] = counts_parent[combinations]
        auw[combinations + 'fibered'] = counts_fibered[combinations]

    return ObservableTree(list(auw.values()), pairs=list(auw.keys()))


def prepare_cucount_particles(*get_data_randoms, positions_type='pos', subsampler=None, jackknife=None, split_randoms=False, concatenate=False, wattrs=None):
    """
    Convert catalogs to :class:`cucount.Particles`.

    Parameters
    ----------
    get_data_randoms : callables
    positions_type : str, optional
        'pos' (default, uses 'POSITION') or 'rd' (uses 'RA', 'DEC').
    subsampler : optional
        Optional object with a ``label(positions)`` method.
    jackknife : dict, optional
        Jackknife configuration, e.g. ``{'mode': 'angular', 'nsplits': 60, 'nside': 512, 'random_state': 42}``.
        If provided, build a :class:`KMeansSubsampler`.
    split_randoms : float, tuple
        If provided, ratio of randoms / data to split the (concatenated) randoms or shifted catalogs into.
        If a tuple, (ratio of randoms / data per split, number of random splits).
    concatenate : bool
        If ``True``, return concatenated randoms or shifted catalogs.
    wattrs : WeightAttrs, optional
        Weight attributes passed to :class:`KMeansSubsampler`.

    Returns
    -------
    all_particles : list of dict
    subsampler : optional
    """
    import numpy as np
    from cucount.jax import Particles

    if jackknife is None: jackknife = {}
    kw_jackknife = dict(jackknife)
    if kw_jackknife: kw_jackknife = {'mode': 'angular', 'nsplits': 60, 'nside': 512, 'random_state': 42} | kw_jackknife

    def get_pw(catalog):
        catalog = catalog.all_to_all()  # loadbalance
        if positions_type == 'rd':
            positions = (catalog['RA'], catalog['DEC'])
        else:
            positions = catalog['POSITION']
        weights = [catalog['INDWEIGHT']] + _format_bitweights(catalog['BITWEIGHT'] if 'BITWEIGHT' in catalog else None)
        return positions, weights

    def _is_list(catalog): return isinstance(catalog, (tuple, list))

    if kw_jackknife and subsampler is None:
        import cucount
        from cucount.jax import SplitAttrs, WeightAttrs
        from cucount.utils import KMeansSubsampler
        if wattrs is None: wattrs = WeightAttrs()

        jackknife_particles = []
        for _get_data_randoms in get_data_randoms:
            data = cucount.numpy.Particles(*get_pw(_get_data_randoms()['data'].gather(mpiroot=None)))
            jackknife_particles.append(data)

        jackknife_particles = cucount.numpy.Particles.concatenate(jackknife_particles)
        subsampler = KMeansSubsampler(jackknife_particles, wattrs=wattrs, **kw_jackknife)

    def _concatenate_catalog(catalog, collective=False):
        if _is_list(catalog):
            if len(catalog) == 1:
                catalog = catalog[0]
            else:
                if collective:
                    catalog = catalog[0].cconcatenate(catalog)
                else:
                    catalog = catalog[0].concatenate(catalog)
        return catalog

    def _split_catalog(catalog, split_randoms, data_size):
        from mpytools.random import MPIRandomState
        if data_size is None:
            raise ValueError('split_randoms is in terms of data size, so provide data')
        catalog = _concatenate_catalog(catalog, collective=True)
        csize = catalog.csize
        if isinstance(split_randoms, tuple):
            split_size, nsplits = split_randoms[0] * data_size, split_randoms[1]
            frac = (split_size * nsplits) / csize
            if frac > 1.:
                warnings.warn(f'catalog of randoms is {1. / frac:.2f} too small to perform {nsplits:d} nsplits with {split_randoms[0]:.2f} the data size')
            frac = min(frac, 1.)
        else:
            split_size = split_randoms * data_size
            nsplits = max(int(csize / split_size), 1)
            frac = 1.
        rng = MPIRandomState(seed=42, size=catalog.size)
        x = rng.uniform(0., 1.)
        toret = []
        for isplit in range(nsplits):
            mask = (x >= isplit * frac / nsplits) & (x < (isplit + 1) * frac / nsplits)
            toret.append(catalog[mask])
        return toret

    def get_all_particles(catalog, as_list=False, data_size=None):
        if as_list and not _is_list(catalog):
            catalog = [catalog]
        if as_list and split_randoms:
            catalog = _split_catalog(catalog, split_randoms, data_size)
        if _is_list(catalog):
            if concatenate:
                catalog = _concatenate_catalog(catalog)
            else:
                return [get_all_particles(c) for c in catalog]
        positions, weights = get_pw(catalog)
        splits = None if subsampler is None else subsampler.label(positions).astype('i8')
        return Particles(positions, weights=weights, splits=splits, positions_type=positions_type, exchange=True)

    all_particles = []
    for _get_data_randoms in get_data_randoms:
        catalogs = _get_data_randoms()
        data_size = catalogs['data'].csize if 'data' in catalogs else None
        particles = {}
        for name, catalog in catalogs.items():
            particles[name] = get_all_particles(catalog, as_list=name in ['randoms', 'shifted'], data_size=data_size)
        all_particles.append(particles)

    if subsampler is not None:
        return all_particles, subsampler
    return all_particles


def _guess_wattrs(get_data_randoms, auw=None):
    """Guess WeightAttrs from input catalogs (callable or Particles)."""
    from cucount.jax import WeightAttrs
    bitwise = angular = None
    if callable(get_data_randoms):
        catalogs = get_data_randoms()
        if 'BITWEIGHT' in catalogs['data']:
            bitwise = dict(weights=_format_bitweights(catalogs['data']['BITWEIGHT']))
            if jax.process_index() == 0:
                logger.info(f'Applying PIP weights {bitwise}.')
    else:  # Particles
        bitwise = get_data_randoms.get('bitwise_weight')

    if auw is not None:
        angular = dict(sep=auw.get('DD').coords('theta'), weight=auw.get('DD').value())
        if jax.process_index() == 0:
            logger.info(f'Applying AUW {angular}.')

    wattrs = WeightAttrs(bitwise=bitwise, angular=angular)
    return wattrs


def compute_particle2_correlation(*get_data_randoms, auw=None, cut=None, battrs: dict=None, zeff: dict=None, jackknife: dict=None, split_randoms: float | tuple=False):
    """
    Compute 2-point correlation function using :mod:`cucount.jax`.

    Parameters
    ----------
    get_data_randoms : callables
        Functions returning dicts with 'data', 'randoms' (optionally 'shifted').
        Catalogs must contain 'POSITION', 'INDWEIGHT', optionally 'BITWEIGHT'.
    auw : ObservableTree, optional
        Angular upweights to apply.
    cut : bool, optional
        If provided, apply a theta-cut of (0, 0.05) degrees.
    battrs : dict, optional
        Bin attributes for cucount.jax.BinAttrs.
    zeff : dict, optional
        Options to estimate the effective redshift, e.g. ``{'cellsize': 10.}``.
    jackknife : dict, optional
        Jackknife configuration, e.g. ``{'mode': 'angular', 'nsplits': 60, 'nside': 512, 'random_state': 42}``.
    split_randoms : float, tuple
        If provided, ratio of randoms / data to split the (concatenated) randoms or shifted catalogs into.
        If a tuple, (ratio of randoms / data, number of random splits).

    Returns
    -------
    correlation : Count2Correlation or Count2JackknifeCorrelation
    """
    from cucount.jax import BinAttrs, WeightAttrs, SelectionAttrs, SplitAttrs
    from cucount.types import count2
    from lsstypes import Count2Correlation, Count2JackknifeCorrelation
    from jaxpower import create_sharding_mesh
    from .spectrum2_tools import prepare_jaxpower_particles, compute_fkp_effective_redshift

    if zeff is None: zeff = {'boxpad': 1.1, 'cellsize': 10.}
    kw_zeff = dict(zeff)

    if jackknife is None: jackknife = {}
    kw_jackknife = dict(jackknife)

    def merge_randoms(catalog):
        if not isinstance(catalog, (tuple, list)): return catalog
        return catalog[0].concatenate(catalog)

    get_randoms = [lambda f=f: {'randoms': merge_randoms(f()['randoms'])} for f in get_data_randoms]

    with create_sharding_mesh(meshsize=kw_zeff.get('meshsize', None)):
        all_particles = prepare_jaxpower_particles(*get_randoms, mattrs=kw_zeff, add_randoms=['IDS'])
        all_randoms = [particles['randoms'] for particles in all_particles]
        seed = [(42, randoms.extra['IDS']) for randoms in all_randoms]
        zeff, norm_zeff = compute_fkp_effective_redshift(*all_randoms, order=2, split=seed,
                                                         resampler='cic', return_fraction=True)

    key = 'raw'
    with create_sharding_mesh():
        if battrs is None:
            battrs = dict(s=np.linspace(0., 180., 181), mu=(np.linspace(-1., 1., 201), 'midpoint'))
        battrs = BinAttrs(**battrs)

        sattrs = None
        if cut is not None:
            sattrs = SelectionAttrs(theta=(0.05, 180.))
            if jax.process_index() == 0: logger.info(f'Applying theta-cut {sattrs}.')
            key = 'cut'
        if auw is not None:
            key = 'auw'

        wattrs = _guess_wattrs(get_data_randoms[0], auw=auw)
        mattrs = None

        spattrs = None
        all_particles = prepare_cucount_particles(*get_data_randoms, jackknife=kw_jackknife, split_randoms=split_randoms, wattrs=wattrs)
        if isinstance(all_particles, tuple):
            all_particles, subsampler = all_particles
            spattrs = SplitAttrs(mode='jackknife', nsplits=subsampler.nsplits)

        if jax.process_index() == 0: logger.info('All particles on the device')

        def count2split(*particles, wattrs=None):
            kw = dict(battrs=battrs, mattrs=mattrs, sattrs=sattrs, spattrs=spattrs, wattrs=wattrs)
            nsplits = [len(p) if isinstance(p, list) else 0 for p in particles]
            if any(nsplits):
                nsplit = next(n for n in nsplits if n)
                particles = list(particles)
                for ip, particle in enumerate(particles):
                    if isinstance(particle, list):
                        assert len(particle) == nsplit
                    else:
                        particles[ip] = [particle] * nsplit
                counts = [count2(*p, **kw)['weight'] for p in zip(*particles)]
                return types.sum(counts)
            return count2(*particles, **kw)['weight']

        counts = {}

        if jax.process_index() == 0: logger.info('Computing DD term.')
        counts['DD'] = count2split(*[particles['data'] for particles in all_particles], wattrs=wattrs)

        for particles in all_particles:
            particles['data'] = particles['data'].clone(weights=wattrs(particles['data']))

        for combinations in ['DS', 'SD', 'SS']:
            particles, _combinations = _get_particle_combinations(combinations, all_particles, with_repeats=False)
            if jax.process_index() == 0:
                logger.info(f'Computing {_combinations} term.')
            counts[combinations] = count2split(*particles)

        if 'RR' not in counts:
            if all(particles.get('shifted', None) is None for particles in all_particles):
                counts['RR'] = counts['SS']
            else:
                if jax.process_index() == 0: logger.info('Computing RR term.')
                counts['RR'] = count2split(*[particles['randoms'] for particles in all_particles])

    correlation = (Count2JackknifeCorrelation if kw_jackknife else Count2Correlation)(estimator='landyszalay', **counts)
    correlation.attrs.update(zeff=zeff / norm_zeff, norm_zeff=norm_zeff)
    return {key: correlation}



def _get_particle_combinations(combinations, all_particles, with_repeats=True):
    particles, _combinations, keys = [], '', []
    for icomb, comb in enumerate(combinations):
        key = {'D': 'data', 'S': 'shifted'}[comb]
        icomb = min(len(all_particles) - 1, icomb)
        particle = all_particles[icomb].get(key, None)
        if key == 'shifted':
            if particle is None:
                key, particle = 'randoms', all_particles[icomb]['randoms']
                _combinations += 'R'
            else:
                _combinations += 'S'
        else:
            _combinations += 'D'
        key = (icomb, key)
        if not with_repeats and key in keys:
            continue
        keys.append(key)
        particles.append(particle)
    return particles, _combinations


def compute_box_particle2_correlation(*get_data, battrs: dict=None, mattrs: dict=None, split_randoms: bool | float=False):
    """
    Compute periodic-box 2-point correlation function using :mod:`cucount.jax`.

    Parameters
    ----------
    get_data : callables
        Functions returning dicts with 'data' and optionally 'shifted'.
    battrs : dict, optional
        Bin attributes for cucount.jax.BinAttrs.
    mattrs : dict, optional
        Mesh attributes, typically with 'boxsize', 'meshsize'.
    split_randoms : float, tuple
        If provided, ratio of shifted randoms / data to split the (concatenated) shifted catalogs into.
        If a tuple, (ratio of randoms / data, number of random splits).

    Returns
    -------
    correlation : Count2Correlation
    """
    from cucount.jax import BinAttrs, WeightAttrs, MeshAttrs, create_sharding_mesh
    from cucount.types import count2, count2_analytic
    from lsstypes import Count2Correlation

    with create_sharding_mesh():
        if battrs is None:
            battrs = dict(s=np.linspace(0., 180., 181), mu=(np.linspace(-1., 1., 201), 'z'))
        battrs = BinAttrs(**battrs)

        mattrs = mattrs or {}
        wattrs = WeightAttrs()

        all_particles = prepare_cucount_particles(*get_data, split_randoms=split_randoms)
        if jax.process_index() == 0: logger.info('All particles on the device')

        mattrs = MeshAttrs(*[particles['data'] for particles in all_particles], battrs=battrs, **mattrs)

        def count2split(*particles, wattrs=None):
            kw = dict(battrs=battrs, mattrs=mattrs, wattrs=wattrs)
            nsplits = [len(p) if isinstance(p, list) else 0 for p in particles]
            if any(nsplits):
                nsplit = next(n for n in nsplits if n)
                particles = list(particles)
                for ip, particle in enumerate(particles):
                    if isinstance(particle, list):
                        assert len(particle) == nsplit
                    else:
                        particles[ip] = [particle] * nsplit
                counts = [count2(*p, **kw)['weight'] for p in zip(*particles)]
                return types.sum(counts)
            return count2(*particles, **kw)['weight']

        if jax.process_index() == 0:
            logger.info('Computing DD term.')
        DD = count2split(*[particles['data'] for particles in all_particles], wattrs=wattrs)

        for particles in all_particles:
            particles['data'] = particles['data'].clone(weights=wattrs(particles['data']))

        if jax.process_index() == 0: logger.info('Computing analytic RR term.')
        RR = count2_analytic(battrs=battrs, mattrs=mattrs)

        counts = dict(DD=DD, RR=RR)

        for combinations in ['DS', 'SD', 'SS']:
            particles, _combinations = _get_particle_combinations(combinations, all_particles)
            if not particles:
                continue
            if jax.process_index() == 0:
                logger.info(f'Computing {combinations} term.')
            counts[combinations] = count2split(*particles)

    if all(name in counts for name in ['DS', 'SD', 'SS']):
        return Count2Correlation(estimator='landyszalay', **counts)
    return Count2Correlation(estimator='natural', DD=counts['DD'], RR=counts['RR'])


def compute_particle2_correlation_close_pair_correction(*get_data_randoms, correlation, battrs=None, auw=None, cut=None, jackknife=None,  split_randoms: float | tuple=False, **kwargs):
    """
    Compute and apply close-pair corrections to 2-point correlation function.

    Parameters
    ----------
    get_data_randoms : callables
        Functions returning dicts with 'data', 'randoms' (optionally 'shifted').
        Catalogs must contain 'POSITION', 'INDWEIGHT', optionally 'BITWEIGHT'.
    correlation : Count2Correlation, Count2JackknifeCorrelation
        Input correlation function to add close pair correction to.
    battrs : dict, optional
        Bin attributes for :class:`cucount.jax.BinAttrs`.
    auw : ObservableTree, optional
        Angular upweights to apply.
    cut : bool, optional
        If provided, apply a theta-cut of (0, 0.05) degrees.
    jackknife : dict, optional
        Jackknife configuration, e.g. ``{'mode': 'angular', 'nsplits': 60, 'nside': 512, 'random_state': 42}``.
    split_randoms : float, tuple
        If provided, ratio of randoms / data to split the (concatenated) randoms or shifted catalogs into.
        If a tuple, (ratio of randoms / data, number of random splits).

    Returns
    -------
    correlation : Count2Correlation or Count2JackknifeCorrelation
    """
    from cucount.jax import create_sharding_mesh, BinAttrs, SplitAttrs

    if jackknife is None: jackknife = {}
    kw_jackknife = dict(jackknife)

    with create_sharding_mesh() as sharding_mesh:
        if callable(get_data_randoms[0]):
            spattrs = None
            wattrs = _guess_wattrs(get_data_randoms[0], auw=auw)
            all_particles = prepare_cucount_particles(*get_data_randoms, jackknife=kw_jackknife, wattrs=wattrs, split_randoms=split_randoms)
            if isinstance(all_particles, tuple):
                all_particles, subsampler = all_particles
                spattrs = SplitAttrs(mode='jackknife', nsplits=subsampler.nsplits)
            if jax.process_index() == 0: logger.info('All particles on the device')

        if battrs is None:
            # Try to find battrs from lsstypes object
            edges = correlation.edges()
            ells = getattr(correlation.get('DD'), 'ells', [])
            d = {}
            los = 'midpoint'  # guessing line-of-sight
            for coord_name in ['s', 'theta']:
                if coord_name in edges:
                    edge = edges[coord_name]
                    edge = np.append(edge[:, 0], edge[-1, 1])
                    if coord_name == 'mu':
                        edge = (edge, los)
                    d[coord_name] = edge
            if ells:
                d['pole'] = (ells, los)
            battrs = BinAttrs(**d)
        else:
            battrs = BinAttrs(**battrs)
        results = {}
        results['raw'] = correlation
        corrections = {'auw': auw, 'cut': cut}
        for correction_name in corrections:
            if corrections[correction_name] is not None:
                correction = _compute_particle2_correlation_close_pair_correction(all_particles, battrs, **{correction_name: corrections[correction_name]}, spattrs=spattrs, normalize_randoms=False)
                results[correction_name] = _apply_particle2_correlation_close_pair_correction(correlation, correction)

    return results


def _apply_particle2_correlation_close_pair_correction(correlation, correction):
    """Apply additive corrections to a :class:`Count2Correlation`."""
    out = correlation.copy()

    def sum_counts(leaves):
        return leaves[0].clone(counts=sum(leaf.values('counts') for leaf in leaves), norm=leaves[0].values('norm'))

    _branches = []
    for label, branch in correlation.items(level=1):
        correction_count_name = label['count_names'].replace('S', 'R')
        if correction_count_name in correction:
            branch = types.tree_map(sum_counts, [branch, correction[correction_count_name]], level=None, is_leaf=lambda *args: False)
        _branches.append(branch)

    out._branches = _branches
    return out


_identity_fn = lambda x: x


def _remove_phantom_particles(particles, sharding_mesh=None):
    if sharding_mesh.axis_names:
        particles = jax.jit(_identity_fn, out_shardings=jax.sharding.NamedSharding(sharding_mesh, spec=P(None)))(particles)
    weights = particles.get('individual_weight')[0]
    if weights is None:
        return particles
    return particles[weights != 0].clone(exchange=True)


def _compute_particle2_correlation_close_pair_correction(all_particles, battrs, spattrs=None, auw=None, cut=None, normalize_randoms: bool=True, theta_limit: tuple=(0., 0.05)):
    """
    Compute close-pair corrections to 2-point counts.
    Returns additive correction counts keyed by count name.
    """
    from cucount.jax import BinAttrs, SelectionAttrs, WeightAttrs, get_sharding_mesh
    from cucount.types import count2

    sharding_mesh = get_sharding_mesh()

    input_all_particles = []
    for particles in all_particles:
        particles = dict(particles)
        if 'shifted' in particles:
            particles.pop('randoms')
        input_all_particles.append(particles)
    all_particles = input_all_particles

    kw_battrs = dict(battrs=battrs)

    sattrs = SelectionAttrs(theta=(0., 0.05) if theta_limit is None else theta_limit)
    wattrs = WeightAttrs()

    if normalize_randoms:
        def normalize_randoms(data, randoms, wattrs=WeightAttrs()):
            data_weights, randoms_weights = wattrs(data), wattrs(randoms)
            return randoms.clone(weights=data_weights.sum() / randoms_weights.sum() * randoms_weights)

    for particles in all_particles:
        #particles['data'] = _remove_phantom_particles(particles['data'], sharding_mesh=sharding_mesh)
        if normalize_randoms:
            for name in ['shifted', 'randoms']:
                if name in particles:
                    particles[name] = normalize_randoms(particles['data'], particles[name], wattrs=wattrs)

    def count2split(*particles, wattrs=None, mattrs=None):
        kw = dict(battrs=battrs, wattrs=wattrs, sattrs=sattrs, mattrs=mattrs, spattrs=spattrs) #, norm=1.)
        nsplits = [len(p) if isinstance(p, list) else 0 for p in particles]
        if any(nsplits):
            for nsplit in nsplits:
                if nsplit: break
            particles = list(particles)
            for ip, particle in enumerate(particles):
                if isinstance(particle, list):
                    assert len(particle) == nsplit
                else:
                    particles[ip] = [particle] * nsplit
            counts = [count2(*p, **kw)['weight'] for p in zip(*particles)]
            return types.sum(counts)
        return count2(*particles, **kw)['weight']

    correction = {}
    if cut is not None:
        wattrs = WeightAttrs()
        with_randoms = any(name in all_particles[0] for name in ['randoms', 'shifted'])
        correction = {}
        for combinations in itertools.product(['D'] + (['S'] if with_randoms else []), repeat=2):
            particles, combinations = _get_particle_combinations(combinations, all_particles, with_repeats=False)
            counts = count2split(*particles, wattrs=wattrs)
            correction[combinations] = counts.clone(value=-counts.value())
        return correction

    bitwise = angular = None
    bitweights = all_particles[0]['data'].get('bitwise_weight')
    if bitweights:
        bitwise = dict(weights=bitweights)

    angular = None
    if auw is not None:
        combinations = 'DD'
        angular = dict(edges=[np.append(edge[:, 0], edge[-1, 1]) for edge in auw.get(combinations).edges().values()],
                       weight=auw.get(combinations).value())
        if jax.process_index() == 0:
            logger.info('Applying AUW.')

    if bitwise is not None or auw is not None:
        wattrs = WeightAttrs(bitwise=bitwise)
        # Set default negative_weight = individual_weight
        for i, particles in enumerate(all_particles):
            particles = particles['data']
            if not particles.get('negative_weight'):
                negative_weight = wattrs(particles)
                all_particles[i]['data'] = particles.clone(weights=particles.get('weights') + [negative_weight], index_value=particles.index_value.clone(negative_weight=1))

    wattrs = WeightAttrs(bitwise=bitwise, angular=angular)
    particles, combinations = _get_particle_combinations('DD', all_particles, with_repeats=False)
    correction['DD'] = count2split(*particles, wattrs=wattrs)
    return correction


def project_spectrum2_covariance_to_correlation(covariance, RR, fkp_norm, windows=None, fields=None, split_SS=True):
    r"""
    Project (a subset of) the field-pair blocks of a :class:`Mesh2SpectrumPoles` covariance to
    :class:`Count2CorrelationPoles`, deconvolving the RR pair-count window and (if ``split_SS``)
    adding back the shot-noise x shot-noise term directly in configuration space.

    This is the "spectrum covariance -> correlation covariance" transform used by
    :func:`compute_covariance_particle2_correlation`, factored out so it can also be reused on
    top of an externally-computed covariance (e.g. a joint power spectrum + bispectrum
    covariance where only one field's power spectrum block should be projected to a
    post-reconstruction correlation function).

    Parameters
    ----------
    covariance : CovarianceMatrix
        Covariance whose ``observable`` is an :class:`~lsstypes.ObservableTree` with one block
        per field-pair label ``'fields'`` (as built by e.g. :func:`jaxpower.compute_spectrum2_covariance`
        or :func:`jaxpower.cov3.compute_spectrum3_covariance`).
    RR : ObservableTree or Count2-like
        RR pair counts, one per projected field pair (labeled by ``'fields'``), or a single
        unlabeled object broadcast to all projected field pairs.
    fkp_norm : dict
        FKP normalization for each projected field pair (see
        :func:`compute_covariance_particle2_correlation`), used to put the RR normalization on
        the same footing as the FKP-based spectrum normalization.
    windows : ObservableTree, optional
        Configuration-space ``WW``/``WS``/``SW``/``SS`` covariance windows, as returned by
        :func:`jaxpower.compute_fkp2_covariance_window`. Required if ``split_SS``.
    fields : list, optional
        Field-pair tuples to project to correlation-function space. If ``None`` (default),
        every field pair in ``covariance.observable`` is projected (matching
        :func:`compute_covariance_particle2_correlation`'s original, unrestricted behavior).
        Field pairs *not* in this list are passed through unchanged (their block keeps its
        original type, e.g. ``Mesh2SpectrumPoles`` or ``Mesh3SpectrumPoles``, via an identity
        sub-block in the rotation) -- their cross-covariance with a projected field is still
        correctly carried over by the same block-diagonal congruence transform.
    split_SS : bool, optional
        If ``True``, split the calculation of the shot-noise-shot-noise term; ensures much
        better convergence for low-density samples such as QSO.

    Returns
    -------
    covariance : CovarianceMatrix
        Covariance with the requested field-pair blocks projected to ``Count2CorrelationPoles``
        (other blocks left untouched).
    """
    from scipy.linalg import block_diag
    from jaxpower.cov2 import matrix_project_to_correlation
    from jaxpower.utils import legendre_product
    from lsstypes.types import compute_RR2_window

    def compute_SS_contribution(observable, QS):

        def get_wj(ww, sedges1, sedges2, q1, q2):
            s1, s2 = np.mean(sedges1, axis=-1), np.mean(sedges2, axis=1)
            w = sum(legendre_product(q1, q2, q) * ww.get(q).value().real if q in ww.ells else jnp.zeros(()) for q in list(range(abs(q1 - q2), q1 + q2 + 1)))
            s = ww.get(0).coords('s')
            w = np.interp(s1, s, w)

            def get_volume(*edges):
                volume = 4. / 3. * np.pi * (edges[1]**3 - edges[0]**3)
                return jnp.where(volume < 0., 0., volume)

            sedges_inter = jnp.maximum(sedges1[:, 0], sedges2[None, :, 0]), jnp.minimum(sedges1[:, 1], sedges2[:, 1])
            volume_inter = get_volume(*sedges_inter)
            volume_joint = get_volume(*sedges1.T)[:, None] * get_volume(*sedges2.T)
            return volume_inter / volume_joint * jnp.diag(w)  # FIXME

        pole1 = pole2 = observable
        ills1 = list(range(len(pole1.ells)))
        ills2 = list(range(len(pole2.ells)))

        def init():
            return [[np.zeros((len(pole1.get(pole1.ells[ill1]).coords('s')), len(pole2.get(pole2.ells[ill2]).coords('s')))) for ill2 in ills2] for ill1 in ills1]

        cov_SS = init()
        for ill1, ill2 in itertools.product(ills1, ills2):
            ell1, ell2 = pole1.ells[ill1], pole2.ells[ill2]
            sedges1, sedges2 = pole1.get(ell1).edges('s'), pole2.get(ell2).edges('s')
            cov_SS[ill1][ill2] += 2 * (2 * ell1 + 1) * (2 * ell2 + 1) * get_wj(QS, sedges1, sedges2, ell1, ell2)
        return np.block(cov_SS)

    all_fields = covariance.observable.fields
    project_fields = all_fields if fields is None else list(fields)
    if RR is not None and 'fields' not in RR.labels(return_type='keys'):
        RR = types.ObservableTree([RR] * len(project_fields), fields=project_fields)

    rotation, observable, covariance_SS = [], [], []
    for label, spectrum in covariance.observable.items(level=1):
        field = label['fields']
        if field not in project_fields:
            # Pass through unchanged: identity rotation block, keep the block as-is
            # (whatever its type, e.g. Mesh2SpectrumPoles or Mesh3SpectrumPoles).
            size = len(spectrum.value())
            rotation.append(np.eye(size))
            observable.append(spectrum)
            if split_SS:
                covariance_SS.append(np.zeros((size, size)))
            continue

        RRfield = RR.get(fields=field)
        if hasattr(RRfield, 'realization'):  # jackknife
            RRfield = RRfield.value(return_type=None)
        RRfield = RRfield.clone(norm=fkp_norm[field] * RRfield.values('norm'))
        s, s_edges = RRfield.coords('s'), RRfield.edges('s')

        # already integrates analytically spectrum to s_edges
        projector = matrix_project_to_correlation(s_edges, spectrum)
        ells = spectrum.ells
        window = compute_RR2_window(RRfield, edges=s_edges, ells=ells, ellsin=ells, kind='RR', resolution=1)
        projector = np.linalg.solve(window, projector)  # inv(window).dot(projector), deconvolve from the window

        correlation = []
        for ell in ells:
            RR0 = RRfield.select(s=s_edges).value().sum(axis=-1)
            norm = RRfield.select(s=s_edges).values('norm').sum(axis=-1)
            correlation.append(types.Count2CorrelationPole(s=s, s_edges=s_edges, value=np.zeros_like(s),
                                                           RR0=RR0, norm=norm, ell=ell))
        correlation = types.Count2CorrelationPoles(correlation)
        correlation = correlation.clone(value=projector.dot(spectrum.value()))
        rotation.append(projector)
        observable.append(correlation)
        if split_SS:
            cov_SS = compute_SS_contribution(correlation, windows.get(types='SS', fields=field * 2))
            inv_window = np.linalg.inv(window)
            cov_SS = inv_window.dot(cov_SS).dot(inv_window.T)
            covariance_SS.append(cov_SS)

    rotation = block_diag(*rotation)
    observable = types.ObservableTree(observable, fields=all_fields)
    covariance = covariance.clone(value=rotation.dot(covariance.value()).dot(rotation.T), observable=observable)
    if split_SS:
        covariance = covariance.clone(value=covariance.value() + block_diag(*covariance_SS))
    return covariance


def compute_covariance_particle2_correlation(*get_data_randoms, theory=None, RR=None, fields=None,
                                             mattrs=None, split_SS=True):
    r"""
    Compute the 2-point correlation covariance with :mod:`jaxpower`.

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
    split_SS : bool, optional
        If ``True``, split the calculation of the shotnoise-shotnoise term;
        ensures much better convergence for low-density samples such as QSO.

    Returns
    -------
    covariance : CovarianceMatrix
        The computed 2-point correlation covariance.
    """
    from .spectrum2_tools import prepare_jaxpower_particles
    # Import covariance and window computation tools from jaxpower
    from jaxpower import create_sharding_mesh, compute_fkp2_covariance_window, interpolate_window_function, compute_spectrum2_covariance, FKPField, compute_fkp2_normalization
    # Use FFTLog for reliable correlation-to-spectrum conversion
    fftlog = True
    # Use default fields (1, 2, ...) if not provided
    if fields is None:
        fields = list(range(1, 1 + len(get_data_randoms)))
    fields = tuple(fields)

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
        all_fkp = [particles['randoms'] for particles in all_particles]
        # Computing normalization: integral of density^2, splitting randoms ('split') to avoid common noise

        def get_fkp_norm(*all_randoms):
            kw_norm = {'cellsize': None, 'split': (42, 'index')}
            fkp_norm = compute_fkp2_normalization(*all_randoms, **kw_norm)
            for randoms in all_randoms + (all_randoms[-1],) * (2 - len(all_randoms)):
                fkp_norm /= randoms.sum()
            return fkp_norm

        fkp_norm = get_fkp_norm(*[particles['randoms'] for particles in all_particles])
        if len(fields) == 1:
            fkp_norm = {fields[:1] * 2: fkp_norm}
        else:
            fkp_norm = {fields[:1] * 2: get_fkp_norm(all_particles[0]['randoms']),
                        fields[1:] * 2: get_fkp_norm(all_particles[1]['randoms']),
                        fields: fkp_norm}

    def interp_log(spectrum, nmu=6):
        from scipy.special import eval_legendre

        #k_fftlog = np.logspace(-2., 2., 1024)
        k_fftlog = np.logspace(-3., 1.5, 1024)

        k_edges = np.empty(len(k_fftlog) + 1)
        k_edges[1:-1] = np.sqrt(k_fftlog[:-1] * k_fftlog[1:])
        k_edges[0] = k_fftlog[0]**2 / k_edges[1]
        k_edges[-1] = k_fftlog[-1]**2 / k_edges[-2]
        k_edges = np.column_stack([k_edges[:-1], k_edges[1:]])

        # collect input multipoles
        labels, ells, poles = [], [], []
        for label, pole in spectrum.items():
            labels.append(label)
            ells.append(label['ells'])
            kin = np.asarray(pole.coords('k'))
            poles.append(np.asarray(pole.value()))

        ells = np.asarray(ells)
        poles = np.asarray(poles)  # (nell, nk)

        # Gauss-Legendre mu bins / quadrature nodes
        mu, wmu = np.polynomial.legendre.leggauss(nmu)

        # Legendre matrix: L[a, imu] = L_ell_a(mu)
        leg = np.asarray([eval_legendre(ell, mu) for ell in ells])

        # multipoles -> P(k, mu)
        # convention: P(k, mu) = sum_ell P_ell(k) L_ell(mu)
        pkmu = np.einsum('lk,lm->km', poles, leg)

        logkin = np.log(kin)
        logkout = np.log(k_fftlog)

        def interp_signed_loglog(y):
            """
            Log-log interpolation/extrapolation in |y| with sign preserved.
            Falls back to linear-log if sign changes.
            """
            y = np.asarray(y)

            if np.all(y > 0.) or np.all(y < 0.):
                sign = np.sign(y[0])
                logy = np.log(np.abs(y))

                out = np.interp(logkout, logkin, logy)

                # left extrapolation
                slope_low = (logy[1] - logy[0]) / (logkin[1] - logkin[0])
                mask_low = logkout < logkin[0]
                out[mask_low] = logy[0] + slope_low * (logkout[mask_low] - logkin[0])

                # right extrapolation
                slope_high = (logy[-1] - logy[-2]) / (logkin[-1] - logkin[-2])
                mask_high = logkout > logkin[-1]
                out[mask_high] = logy[-1] + slope_high * (logkout[mask_high] - logkin[-1])

                return sign * np.exp(out)

            # fallback if P(k, mu) changes sign
            out = np.interp(logkout, logkin, y)

            slope_low = (y[1] - y[0]) / (logkin[1] - logkin[0])
            mask_low = logkout < logkin[0]
            out[mask_low] = y[0] + slope_low * (logkout[mask_low] - logkin[0])

            slope_high = (y[-1] - y[-2]) / (logkin[-1] - logkin[-2])
            mask_high = logkout > logkin[-1]
            out[mask_high] = y[-1] + slope_high * (logkout[mask_high] - logkin[-1])

            return out

        # interpolate/extrapolate each mu bin
        pkmu_out = np.stack([interp_signed_loglog(pkmu[:, imu]) for imu in range(nmu)], axis=1)  # (nkout, nmu)

        # P(k, mu) -> multipoles
        # P_ell(k) = (2ell + 1)/2 int_{-1}^{1} dmu P(k,mu) L_ell(mu)
        poles_out = []
        for label, ell in zip(labels, ells):
            leg = eval_legendre(ell, mu)
            value = (2 * ell + 1) / 2. * np.einsum('m,km,m->k', wmu, pkmu_out, leg)
            poles_out.append(types.Mesh2SpectrumPole(num_raw=jnp.asarray(value), k=k_fftlog, k_edges=k_edges, ell=ell))
        return types.Mesh2SpectrumPoles(poles_out)

    theory = theory.map(interp_log, level=1)  # apply to all tracers
    #if jax.process_index() == 0: theory.write('theory.h5')
    covariances = compute_spectrum2_covariance(windows, theory, flags=['smooth'] + (['fftlog'] if fftlog else []), return_type='list')
    if split_SS:
        covariances = covariances[:2]  # leave out SS, added later
    covariance = covariances[0].clone(value=sum(cov.value() for cov in covariances))
    #covariance.write('covariance_pk.h5')

    covariance = project_spectrum2_covariance_to_correlation(covariance, RR, fkp_norm, windows=windows, split_SS=split_SS)
    # Store in results dict
    results['raw'] = covariance
    return results