"""
Run on the GPU, with
```
srun -n 1 python toy_fiber_assignment.py
```
(single process only!)
"""
from pathlib import Path
import os
import numpy as np
import healpy as hp
import jax
from jax import config
from lsstypes import ObservableTree

from mpytools import Catalog
from clustering_statistics import tools
from clustering_statistics.tools import setup_logging, _compute_binned_weight
from clustering_statistics.correlation2_tools import prepare_cucount_particles


dirname = Path('./toy_fiber_assignment/')
dirname.mkdir(exist_ok=True)


def mask_fiber_collisions(ra, dec, theta_f_deg=0.05,
                          alpha=0.5, seed=42):
    from scipy.spatial import cKDTree
    import numpy as np

    rng = np.random.default_rng(seed)
    n = ra.size

    # Cartesian unit vectors
    theta = np.radians(90. - dec)
    phi = np.radians(ra)

    xyz = np.column_stack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])

    theta_f = np.radians(theta_f_deg)
    rmax = 2.0 * np.sin(theta_f / 2.)

    tree = cKDTree(xyz)

    keep = np.ones(n, dtype=bool)
    weight = np.ones(n, dtype=float)

    for i in range(n):

        if not keep[i]:
            continue

        neighbors = tree.query_ball_point(xyz[i], rmax)

        n_removed = 0

        for j in neighbors:

            if j <= i or not keep[j]:
                continue

            if rng.random() < alpha:
                keep[j] = False
                n_removed += 1

        weight[i] += n_removed

    weight[~keep] = 0.

    return weight


def generate_sample_fiber_collisions(n, theta_f_deg=0.05, alpha=0.5, seed=None):
    """
    Generate random points on the sphere and apply fiber collisions.

    Parameters
    ----------
    n : int
        Number of initial points.
    theta_f_deg : float
        Collision scale in degrees.
    alpha : float
        Probability of removing a colliding point.
    seed : int, optional

    Returns
    -------
    ra, dec : arrays
        Original catalog in degrees.
    keep : boolean array
        Surviving ("fibered") objects.
    """
    rng = np.random.default_rng(seed)

    # Uniform sphere
    ra = rng.uniform(0., 360., n)
    z = rng.uniform(-1., 1., n)
    dec = np.degrees(np.arcsin(z))
    return ra, dec, mask_fiber_collisions(ra, dec, theta_f_deg=theta_f_deg, alpha=alpha, seed=seed)



def get_output_fn(basename, imock):
    return dirname / f'{basename}_{imock:04d}.h5'


def compute_auw(imock):

    ra, dec, weight = generate_sample_fiber_collisions(5_000_000, theta_f_deg=0.05, alpha=0.5, seed=42)
    parent = Catalog({'RA': ra, 'DEC': dec, 'INDWEIGHT': np.ones_like(ra), 'assigned': weight})
    fibered = parent[parent['assigned'] > 0]
    fibered['INDWEIGHT'] = fibered['assigned'].astype('f8')

    from cucount.jax import BinAttrs, SelectionAttrs, WeightAttrs, get_sharding_mesh
    from cucount.types import count2

    def get_counts(*get_data, battrs=None, norm=None):
        all_particles = prepare_cucount_particles(*get_data, positions_type='rd')
        all_particles = [particles['data'] for particles in all_particles]
        if battrs is None:
            battrs = {'theta': np.arange(0.001, 0.5, 0.001)}
        battrs = BinAttrs(**battrs)
        return count2(*all_particles, battrs=battrs, norm=norm)['weight']

    result = {}
    result['GfGf'] = get_counts(lambda: {'data': fibered})
    result['GpGp'] = get_counts(lambda: {'data': parent})
    result = ObservableTree(list(result.values()), pairs=list(result.keys()))
    result.write(get_output_fn('all_counts', imock=imock))


if __name__ == '__main__':

    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'
    config.update('jax_enable_x64', True)

    try: jax.distributed.initialize()
    except RuntimeError: print('Distributed environment already initialized')
    else: print('Initializing distributed environment')

    setup_logging()

    imock = 0
    compute_auw(imock)