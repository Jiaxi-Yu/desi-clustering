"""Export the non-parametric redshift-error (velocity smearing) PDFs to lsstypes format.

Files: /pscratch/sd/s/shengyu/repeats/DA2/loa-v1/verr_mode/CDF_verr_nonparam_<TRACER>_z<zmin>-<zmax>.npz
Each holds 'grid' (dv in km/s), 'pdf' (P(dv), normalised on grid) and 'cdf'.

The PDF depends on the tracer and redshift range only -- not on the region or the weighting --
but the region/weight tags are kept in the output name to match the surrounding convention.
"""
from pathlib import Path

import numpy as np
from lsstypes import ObservableLeaf

from clustering_statistics import tools


def _convert(fn, tracer, zrange):
    """Read one CDF_verr_nonparam npz and return it as an ``lsstypes.ObservableLeaf``.

    The leaf carries ``dv`` [km/s] as its coordinate and ``pdf`` / ``cdf`` as values, so
    ``leaf.coords('dv')``, ``leaf.values('pdf')`` and ``leaf.value()`` (the pdf) work, and
    ``leaf.select(dv=(-1000., 1000.))`` trims the catastrophic tail. Tracer and redshift
    range go to ``meta``; the source path to ``attrs``.
    """
    with np.load(fn) as f:
        dv, pdf, cdf = f['grid'], f['pdf'], f['cdf']
    return ObservableLeaf(dv=dv, pdf=pdf, cdf=cdf, coords=['dv'],
                          meta={'tracer': tracer, 'zmin': float(zrange[0]), 'zmax': float(zrange[1])},
                          attrs={'source': str(fn)})


def convert_smearing(in_fn, out_fn, tracer, zrange):
    leaf = _convert(in_fn, tracer=tracer, zrange=zrange)
    leaf.write(out_fn)
    return leaf


if __name__ == '__main__':

    in_dir = Path('/pscratch/sd/s/shengyu/repeats/DA2/loa-v1/verr_mode')
    list_zrange = [('BGS_BRIGHT-21.35', (0.1, 0.4)),
                   ('LRG', (0.4, 0.6)),
                   ('LRG', (0.6, 0.8)),
                   ('LRG', (0.8, 1.1)),
                   ('ELG_LOPnotqso', (0.8, 1.1)),
                   ('ELG_LOPnotqso', (1.1, 1.6)),
                   ('QSO', (0.8, 2.1))]

    for tracer, zrange in list_zrange:
        simple_tracer = tools.get_simple_tracer(tracer)
        in_smearing_fn = in_dir / f'CDF_verr_nonparam_{simple_tracer}_z{zrange[0]:.1f}-{zrange[1]:.1f}.npz'
        for region in ['GCcomb']:
            kw = dict(stats_dir=tools.base_stats_dir / 'auxiliary_data/redshift_smearing', weight='default-FKP',
                      version='data-dr2-v2', tracer=tracer, region=region, zrange=zrange)
            smearing_fn = tools.get_stats_fn(kind='redshift_smearing', **kw)
            convert_smearing(in_smearing_fn, smearing_fn, tracer=tracer, zrange=zrange)
            print(f'wrote {smearing_fn}')
