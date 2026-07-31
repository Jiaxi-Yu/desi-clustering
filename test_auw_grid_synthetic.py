"""
Synthetic validation of the close-pair triple counting against a brute-force reference.

Uniform random points on a spherical cap; triplets with at least one pair within
theta_c = 0.05 deg are counted exactly with a KD-tree (with exact multiple-close-pair
corrections), then compared to `_compute_particle3_correlation_close_pair_correction`
run with the old (uniform 0.2-dex) and new (non-uniform 26-bin) theta grids.

Context: production 2026-07-07 showed a ~30% DDD triplet-count deficit with the new
grid (data-dr2-v2-2, FKP2) vs the old one (FKP3 control, bit-reproducible), with the
far-leg distribution reshaped at wide angles. Totals must be independent of the
binning grid, so one of the two is mis-counting; this script decides which.

Run at NERSC (needs cucount.jax + GPU), from the desi-clustering directory:
    python test_auw_grid_synthetic.py
The verdict is in the "ratio to brute force" column: the correct grid gives the same
constant for every theta2 range; the broken one shows range-dependent ratios.
"""

import numpy as np

from jax import config
config.update('jax_enable_x64', True)


def make_cap(n, cap_deg=60., seed=42):
    """Uniform points on a polar cap; return (ra, dec) in degrees."""
    rng = np.random.default_rng(seed)
    cth = rng.uniform(np.cos(np.radians(cap_deg)), 1., n)
    phi = rng.uniform(0., 2. * np.pi, n)
    dec = np.degrees(np.arcsin(cth))  # cap around the pole -> dec in (90-cap_deg, 90)
    ra = np.degrees(phi)
    return ra, dec


def unit_vectors(ra, dec):
    ra, dec = np.radians(ra), np.radians(dec)
    return np.column_stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


def brute_force_reference(pos, theta_c_deg=0.05, ranges=None, nmc=10**7, seed=1):
    """
    Exact expected triple counts: each unordered triplet with >= 1 close pair counted once.
    Also returns the expected distribution of the (far) pair separation across `ranges`,
    from a Monte Carlo of independent point pairs on the same cap (the far leg of a
    close-pair triplet is an independent point to within O(theta_c)).
    """
    from scipy.spatial import cKDTree
    from itertools import combinations

    n = len(pos)
    chord = 2. * np.sin(np.radians(theta_c_deg) / 2.)
    tree = cKDTree(pos)
    pairs = tree.query_pairs(chord, output_type='ndarray')  # unordered close pairs
    npairs = len(pairs)
    k = np.bincount(pairs.ravel(), minlength=n)
    n_multi = int((k * (k - 1) // 2).sum())
    # close triangles (all three pairs close)
    neighbors = [[] for _ in range(n)]
    for a, b in pairs:
        neighbors[a].append(b)
        neighbors[b].append(a)
    n_tri = 0
    for i in range(n):
        for a, b in combinations(neighbors[i], 2):
            d = pos[a] - pos[b]
            if d @ d < chord**2:
                n_tri += 1
    n_tri //= 3
    # naive = sum over close pairs of (n-2) counts each triplet once per close pair it holds;
    # subtract multiplicities so each qualifying triplet counts exactly once
    total = npairs * (n - 2) - (n_multi - n_tri)

    frac = None
    if ranges is not None:
        rng = np.random.default_rng(seed)
        i = rng.integers(0, n, nmc)
        j = rng.integers(0, n, nmc)
        good = i != j
        cth = np.clip((pos[i[good]] * pos[j[good]]).sum(axis=1), -1., 1.)
        th = np.degrees(np.arccos(cth))
        frac = np.array([np.mean((th >= lo) & (th < hi)) for lo, hi in ranges])
    return total, npairs, n_multi, n_tri, frac


def run_counting(ra, dec, theta_grid, ra_r=None, dec_r=None):
    """Run the pipeline close-pair counting with the given theta binning."""
    from cucount.jax import Particles, BinAttrs, create_sharding_mesh
    from clustering_statistics.correlation3_tools import _compute_particle3_correlation_close_pair_correction

    with create_sharding_mesh():
        data = Particles((ra, dec), [np.ones(len(ra))], positions_type='rd')
        particles = {'data': data}
        if ra_r is not None:
            particles['randoms'] = Particles((ra_r, dec_r), [np.ones(len(ra_r))], positions_type='rd')
        battrs = BinAttrs(theta=theta_grid)
        counts = _compute_particle3_correlation_close_pair_correction(
            [particles], [battrs] * 3, auw=None, cut=None, veto23=None, normalize_randoms=False)
    return counts


def main():
    n = 200000
    cap_deg = 60.
    ra, dec = make_cap(n, cap_deg=cap_deg, seed=42)
    ra_r, dec_r = make_cap(n, cap_deg=cap_deg, seed=43)
    pos = unit_vectors(ra, dec)

    # shared bin edges of the old (0.2-dex) and new grids, used for the range comparison
    ranges = [(1., 2.511886), (2.511886, 6.309573), (6.309573, 15.848932),
              (15.848932, 39.810717), (39.810717, 100.), (100., 120.)]

    print('building brute-force reference (KD-tree)...')
    total_bf, npairs, n_multi, n_tri, frac = brute_force_reference(pos, ranges=ranges)
    print(f'  close pairs: {npairs}, multi-close corrections: {n_multi} (triangles {n_tri})')
    print(f'  expected total (each triplet once): {total_bf:.6e}')
    print(f'  far-leg range fractions: ' + ' '.join(f'{f:.4f}' for f in frac))

    grids = {
        'old (0.2 dex)': 10**np.arange(-4, np.log10(180.), 0.2),
        'new (26 bins)': np.concatenate([10**np.arange(-4., -2.5, 0.5),
                                         10**np.arange(-2.5, -1., 0.1),
                                         10**np.arange(-1., 0., 0.5),
                                         10**np.arange(0., np.log10(180.), 0.4),
                                         [180.]]),
        'new, last edge 158.489': np.concatenate([10**np.arange(-4., -2.5, 0.5),
                                                  10**np.arange(-2.5, -1., 0.1),
                                                  10**np.arange(-1., 0., 0.5),
                                                  10**np.arange(0., np.log10(180.), 0.4),
                                                  [158.489]]),
        'old, last edge 180': np.concatenate([10**np.arange(-4, np.log10(180.), 0.2)[:-1], [180.]]),
    }

    for lab, grid in grids.items():
        counts = run_counting(ra, dec, grid, ra_r=ra_r, dec_r=dec_r)
        for name in counts:
            leaf = counts[name]
            c = np.asarray(leaf.value() if not hasattr(leaf, 'values') else leaf.values('counts'))
            tot = c.sum()
            ref = total_bf if name == 'DDD' else np.nan
            line = f'{lab:24s} {name}: total = {tot:.6e}'
            if name == 'DDD':
                line += f'   ratio to brute force = {tot / total_bf:.4f}'
            print(line)
            if name != 'DDD':
                continue
            # theta2-axis marginal per shared range, vs brute-force prediction
            edges = grid
            for (lo, hi), f in zip(ranges, frac):
                sel = (edges[:-1] >= lo * 0.9999) & (edges[1:] <= hi * 1.0001)
                v = c[:, sel, :].sum()
                pred = npairs * (n - 2) * f  # per-range prediction (multi-close corr. negligible)
                print(f'    theta2 in ({lo:7.3f},{hi:7.3f}): counts = {v:.4e}   pred = {pred:.4e}   ratio = {v / pred if pred > 0 else np.nan:.4f}')
        print()


if __name__ == '__main__':
    main()
