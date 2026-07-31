"""
Run with:
salloc -N 1 -C gpu -t 02:00:00 --gpus 4 --qos interactive --account desi_g
cosmodesienv main
srun -n 4 python correlation_data_c3.py
"""

import os
import numpy as np
import functools
from pathlib import Path

# Import clustering statistics and logging utilities
from clustering_statistics import tools, setup_logging
setup_logging()

from mpi4py import MPI
mpicomm = MPI.COMM_WORLD


def run_stats(version='data-dr1-v1.5', tracer='LRG', weight='default-FKP', zranges=None, stats_dir=Path(os.getenv('SCRATCH')) / 'measurements', stats=['mesh2_spectrum'], ibatch=None, **kwargs):
    # Everything inside this function will be executed on the compute nodes;
    # This function must be self-contained; and cannot rely on imports from the outer scope.
    import os
    import sys
    import functools
    from pathlib import Path
    # Import JAX for GPU-accelerated array operations
    import jax
    from jax import config
    # Enable 64-bit precision for accurate clustering calculations
    config.update('jax_enable_x64', True)
    # Allocate 90% of available GPU memory to JAX arrays
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.95'
    # Initialize JAX distributed computing across MPI processes
    try: jax.distributed.initialize()
    except RuntimeError: print('Distributed environment already initialized')
    else: print('Initializing distributed environment')
    # Import clustering statistics computation functions and logging
    from clustering_statistics import tools, setup_logging, compute_stats_from_options, fill_fiducial_options
    setup_logging()
    # Initialize cache dictionary to store intermediate results across regions
    cache = {}
    # If redshift ranges not provided, use fiducial values from tools
    if zranges is None:
        zranges = tools.propose_fiducial('zranges', tracer)
    # Create partial function with stats directory preset for cleaner calls
    get_stats_fn = functools.partial(tools.get_stats_fn, stats_dir=stats_dir)
    # Loop over Northern and Southern galactic caps
    for region in ['NGC', 'SGC']:
        options = dict(catalog=dict(version=version, tracer=tracer, zrange=zranges, region=region, weight=weight), **kwargs)
        # Fill in missing options with default/fiducial values from tools
        options = fill_fiducial_options(options)
        # Compute all requested statistics
        compute_stats_from_options(stats, get_stats_fn=get_stats_fn, cache=cache, prepare_catalog=prepare_catalog **options)


def postprocess_stats(version='data-dr2-v1.1', tracer='LRG', weight='default-FKP', stats_dir=Path(os.getenv('SCRATCH')) / 'measurements', postprocess=['combine_regions'], **kwargs):
    # Post-processing step: combine measurements from NGC and SGC regions
    from clustering_statistics import postprocess_stats_from_options
    # Get fiducial redshift ranges for tracer
    zranges = tools.propose_fiducial('zranges', tracer)
    # Create partial function with stats directory preset
    get_stats_fn = functools.partial(tools.get_stats_fn, stats_dir=stats_dir)
    # Build post-processing options: combine statistics across regions
    # List statistics to combine: power spectrum, bispectrum, windows, covariance, pair correlations
    options = dict(catalog=dict(version=version, tracer=tracer, zrange=zranges, weight=weight), combine_regions={'stats': ['particle2_correlation']}, **kwargs)
    # Execute post-processing: combines NGC+SGC, computes weighted averages, propagates covariance
    postprocess_stats_from_options(postprocess, get_stats_fn=get_stats_fn, **options)


if __name__ == '__main__':

    stats = ['particle2_correlation'][:0]
    postprocess = ['combine_regions'][:1]

    # Output directory for measurement results (SCRATCH filesystem for performance)
    stats_dir = Path(os.environ['SCRATCH']) / 'correlation_data_c3'
    # Catalog version
    #version = 'data-dr2-v1.1'
    version = 'data-dr1-v1.5'

    # Loop over tracer types: BGS (Bright Galaxy Survey), LRG (Luminous Red Galaxy), ELG (Emission Line Galaxy), QSO (Quasar)
    # [1:2] selects only LRG; change to [:] to process all tracers
    for tracer in ['BGS', 'LRG', 'ELG', 'QSO'][1:2]:
        # Get full tracer name including version suffix (e.g., 'LRG_0' for redshift bin 0)
        tracer = tools.get_full_tracer(tracer, version=version)
        # Get fiducial redshift ranges for this tracer; [:1] takes only first bin
        zranges = tools.propose_fiducial('zranges', tracer)

        for battrs in [{'rp': (np.geomspace(0.01, 100., 49), 'midpoint'), 'pi': (np.linspace(-40., 40., 81), 'midpoint')}]:
            kwargs = {'particle2_correlation': {'jackknife': {'nsplits': 60}, 'battrs': battrs}}
            # Execute statistics computation if requested (stats list not empty)
            if stats:
                run_stats(version=version, tracer=tracer, zranges=zranges, stats_dir=stats_dir, stats=stats, **kwargs)
            # Execute post-processing if requested (combine regions, etc.)
            if postprocess:
                postprocess_stats(version=version, tracer=tracer, zranges=zranges, stats_dir=stats_dir, postprocess=postprocess, **kwargs)