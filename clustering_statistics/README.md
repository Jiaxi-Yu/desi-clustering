# DESI Clustering Statistics Pipeline and Products

## Overview

The package follows a simple structure:

* **`compute_stats.py`**: high-level orchestration and command-line-interface entry point
* **`tools.py`**: shared utilities for default options, catalog I/O, cutsky catalog and measurement paths
* **`box_tools.py`**: same for periodic boxes
* **`*_tools.py`**: statistic-specific measurement backends: `correlation2_tools.py` (2-point correlation function), `spectrum2_tools.py` (2-point power spectrum, window, covariance), `spectrum3_tools.py` (bispectrum, window), `recon_tools.py` (BAO reconstruction)
* **`job_scripts/`**: production launch examples
* **`nb/`**: validation and usage notebooks
* **`tests/`**: lightweight tests
* **`catalog_blinding/`**: `desiblind`-backed catalog-level blinding adapters and saved-catalog preparation

A good mental model is:

> `compute_stats.py` = *pipeline driver*
> `tools.py` = *plumbing and conventions*
> `*_tools.py` = *actual measurement kernels*
> `catalog_blinding/` = *catalog-level blinding adapters and saved-catalog preparation*

## How to run a measurement

Typical example for a 2-point power spectrum, with the command line interface:

```bash
clustering-stats \
    --stats mesh2_spectrum \
    --analysis full_shape \
    --version holi-v1-altmtl \
    --tracer LRG \
    --zrange 0.4 0.6 \
    --region NGC SGC \
    --weight default-FKP
```

Typical example for a 2PCF:

```bash
clustering-stats \
    --stats particle2_correlation \
    --analysis bao \
    --version holi-v1-altmtl \
    --tracer LRG \
    --zrange 0.4 0.6
```

To discover all options:

```bash
clustering-stats --help
```

---

## For more options / larger runs

For production runs, the easiest way to get started is to copy from:

```text
clustering_statistics/job_scripts/
```

Useful examples include:

* `desipipe_data_bao.py`
* `desipipe_data_png.py`
* `desipipe_holi_mocks.py`
* `desipipe_abacus_mocks.py`
* `desipipe_box_abacus_mocks.py`

These scripts provide **real production configurations** and are often the best starting point for new analyses.


---

## Quick “where do I look?” map

* **Run from CLI** → `compute_stats.py:main(...)`
* **Understand default options** → `tools.py:fill_fiducial_options(...)`
* **Find catalog loading** → `tools.py:read_clustering_catalog(...)`
* **Find output paths** → `tools.py:get_stats_fn(...)`
* **Measure `P(k)`** → `spectrum2_tools.py:compute_mesh2_spectrum(...)`
* **Measure `ξ(s)`** → `correlation2_tools.py:compute_particle2_correlation(...)`
* **Measure `B(k)`** → `spectrum3_tools.py:compute_mesh3_spectrum(...)`
* **Run reconstruction** → `recon_tools.py:compute_reconstruction(...)`
* **Run box measurements** → `compute_box_stats.py`
* **Read clustering catalogs** → `../nb/read_clustering_catalog.ipnyb`
* **Read clustering measurements** → `nb/example_read_stats.ipnyb`
* **Latest clustering measurements** → `nb/check_latest_measurements.ipnyb`
* **Catalog blinding** → `catalog_blinding/`, `catalog_blinding/cli.py`, and the section below


## Data Access

The base directory is
```/global/cfs/cdirs/desi/science/cai/desi-clustering/dr2/summary_statistics/```

Within the base directory, there is a corresponding key project (KP) directory:
* ```bao```
* ```full_shape```
* ```png_local```
* ```merged_catalogs```: Merged mock data catalogs used for reshuffling the randoms to estimate the radial integral constraint (RIC).
* ```auxiliary_data```: Note: If you add a file to this directory, please log it in the `README.txt` file within the directory.

Furthermore, within each KP directory, there are sub-directories to seperate different measurements variations. Below we list some of them:
* ```base```: Fiducial measurements for the KP.
* ```data_splits```: Variations in data splits, e.g., region splits beyond the ones considered in ```base```.
* ```systematic_weights```: Variations in systematic weights, beyond what is available in ```base```.
* ```...```

Finally, within each, sub-directories correspond to the data and mocks clustering products. Below we list some of them:
* ```abacus-2ndgen-complete```
* ```holi-v1-altmtl```
* ```glam-uchuu-v1-altmtl```
* ```...```

Directory tree:
```
/global/cfs/cdirs/desi/science/cai/desi-clustering/dr2/summary_statistics/
├── auxiliary_data
├── bao
│   └── base
├── full_shape
│   ├── base
│   │   ├── abacus-2ndgen-altmtl
│   │   ├── abacus-2ndgen-complete
│   │   ├── glam-uchuu-v1-altmtl
│   │   └── holi-v1-altmtl
│   └── data_splits
├── local_png
│   └── base
│       └── glam-uchuu-v1-altmtl
└── merged_catalogs
    └── glam-uchuu-v1-altmtl
```

## Documentation
### Reading clustering statistics

All clustering products follow a `base_filename` structure such that `base_filename = {tracer}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{region}_weight-{weight_type}{extra}`, with:
* tracer ```tracer```: 'LRG', 'ELG_LOPnotqso', 'QSO'.

* region ```region```: 'NGC', 'SGC', or 'GCcomb'. Combined power spectrum measurements 'GCcomb' are the average of 'NGC' and 'SGC' power spectra, weighted by their normalization factor.

* redshift range ```zrange```:
  * For `full_shape`: (0.4, 0.6), (0.6, 0.8), (0.8, 1.1), (1.1, 1.6), (0.8, 2.1)
  * For `png_local`:  (0.4, 1.1), (0.8, 1.6), (0.8, 3.5)

* `weight_type`: identifies how the tracers were weighted. This can be any combination of weights, but the default choices are dependent on the KP and are `default-FKP` ('full shape') and `default-fkp-oqe` ('local png').

* `extra` is a suffix that can be any combination of extra processing done before, during, or after the measurement, separated by an underscore (`_`). Some default choices below:
    *  `_thetacut`: $\theta$-cut removes all pairs with angular separation < 0.05°, to mitigate fiber assignment effects.
    *  `_auw`: angular upweighting scheme [Bianchi et al. 2025](https://arxiv.org/pdf/2411.12025)...
    *  `_noric`: The redshifts of the randoms catalogs were reshuffled to remove the nulling of radial modes due to the 'shuffling' method. The 'shuffling' method subsamples the redshifts of the randoms from the data. NOTE: These are only used for the estimation of the radial integral constraint (RIC).

Therefore, for each statistic:
* ```pk```: `mesh2_spectrum_poles_{base_filename}.h5`
* ```bk```: `mesh3_spectrum_{basis}_poles_{base_filename}.h5`
    * `basis`: `sugiyama-diagonal`...
* ```xi```: `particle2_correlation_{base_filename}.h5`


An example of how the full path of a mock measurement would look:
```
$BASEDIR/full_shape/base/glam-uchuu-v1-altmtl/mock100/mesh2_spectrum_poles_LRG_z0.4-0.6_GCcomb_weight-default-FKP_thetacut.h5
```

Please refer to the `nb/example_read_stats.ipynb` for an example on how to load clustering statistics.

---

## desiblind catalog-level blinding adapters

Catalog-level blinding transformations are implemented and validated in the
separate `desiblind` package. `desi-clustering` should provide convenient
measurement-side wrappers/drivers where useful, but should not reimplement the
blinding physics.

The on-the-fly path is deliberately narrow: it measures Pk/xi after applying
`desiblind.catalog_bao.CatalogBAOBlinder` in memory. The code for this BAO/AP-only
path is `catalog_blinding/bao.py`. Use the explicit catalog option
`catalog_bao_blinding`; the removed older generic `catalog['blinding']` workflow
should not be used.

This BAO/AP on-the-fly path is separate from the existing statistic/data-vector blinding
in `tools.apply_blinding`; the two are not stacked by default. The BAO/AP data
redshifts are remapped with `desiblind`, and random catalogs are kept matched to
the shifted data with the LSS-style redshift-column resampling implemented in
`catalog_blinding/lss_catalogs.py` rather than by directly BAO-shifting random
redshifts. Heavier catalog-level workflows such as RSD and future fNL should be
explicit saved-catalog drivers that call `desiblind` for the blinding transforms,
then pass the final blinded catalogs to `compute_stats_from_options` as a normal
`cat_dir`. RSD additionally needs a reconstruction step.

A lightweight executable example of the BAO/AP-only on-the-fly path is in
`clustering_statistics/nb/example_catalog_bao_blinding.ipynb`. The matching
single-bin script is
`clustering_statistics/job_scripts/run_catalog_bao_blinding_dr1_example.py`.
For an all-DR1-tracer production-like driver, see
`clustering_statistics/job_scripts/run_catalog_bao_blinding_dr1.py`. It can
run either through desipipe/Slurm, or directly inside the current allocation for
fast turnaround. Start with:

```bash
python clustering_statistics/job_scripts/run_catalog_bao_blinding_dr1.py --dry-run

# One-task smoke test inside an allocation.
srun -n 4 python clustering_statistics/job_scripts/run_catalog_bao_blinding_dr1.py \
  --mode interactive --tracers LRG --regions NGC --zrange-index 0

# Full selected grid, sequentially in the current allocation; bypasses desipipe queue overhead.
srun -n 4 python clustering_statistics/job_scripts/run_catalog_bao_blinding_dr1.py \
  --mode direct-grid
```

The open demo default uses explicit parameters (`w0=-0.95`, `wa=0.10`) that pass
the Andrade et al. 3% BAO/AP alpha-shift mask. Closed workflows should use
a private `desiblind` hashed parameter bank, consistent with summary-statistic
blinding. Create the shared catalog-level `w0`/`wa` bank with
`desiblind/scripts/create_catalog_w0wa_blinding_bank.py`. For BAO/AP-only
measurements, write a BAO bank:

```bash
python /global/homes/u/uendert/repos/desi/desiblind/scripts/create_catalog_w0wa_blinding_bank.py \
  --output /private/path/catalog_bao_blinding_parameters.npy \
  --bid <blind-id> \
  --generate --seed <private-seed> \
  --record-fn /private/path/catalog_w0wa_blinding_record.json \
  --chmod 600
```

Then pass that private bank to the measurement driver:

```bash
python clustering_statistics/job_scripts/run_catalog_bao_blinding_dr1.py \
  --mode desipipe \
  --parameter-source desiblind \
  --parameters-fn /private/path/catalog_bao_blinding_parameters.npy \
  --bid <blind-id> \
  --metadata closed
```

The same hidden `w0`/`wa` draw can also write an RSD bank. In that case
`fgrowth_blind` is derived from `w0`, `wa`, `zeff`, and `bias`; it is not drawn
independently. Following the Andrade et al. (2025) catalog-blinding prescription, the
chosen `fgrowth_blind` compensates the BAO/AP volume rescaling in the linear
Kaiser monopole factor, up to the configured fractional cap:

```bash
python /global/homes/u/uendert/repos/desi/desiblind/scripts/create_catalog_w0wa_blinding_bank.py \
  --effects bao rsd \
  --bao-output /private/path/catalog_bao_blinding_parameters.npy \
  --rsd-output /private/path/catalog_rsd_blinding_parameters.npy \
  --bid <blind-id> \
  --generate --seed <private-seed> \
  --rsd-bin LRG1:0.50:2.0 \
  --record-fn /private/path/catalog_w0wa_blinding_record.json \
  --chmod 600
```

This writes two effect-specific bank files because the current `desiblind` BAO/AP
and RSD blinders have separate hashed-key namespaces and different parameter
payloads. The BAO/AP bank stores the shared `w0`/`wa`; the RSD bank stores the
same `w0`/`wa` plus the tracer-bin metadata and derived `fgrowth_blind`. The
single private `catalog_w0wa_blinding_record.json` ties those banks to one hidden
cosmology, so **BAO/AP and RSD use the same blind cosmology**. A future cleanup could
move this to one combined shared bank file once both blinders agree on a common
multi-effect bank schema.

Historical LSS `w0wa` bank rows are also supported for compatibility:

```bash
python clustering_statistics/job_scripts/run_catalog_bao_blinding_dr1.py \
  --parameter-source lss \
  --lss-w0wa-bank /global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/w0wa_initvalues_zeffcombined_1000realisations.txt \
  --lss-filerow /path/to/filerow.txt
```

By default the BAO/AP path validates `w0 + wa <= 0` and `|alpha_parallel - 1|`,
`|alpha_perp - 1| < 0.03` over `0.4 < z < 2.1`, matching the Andrade et al.
selection.

A first explicit saved-catalog driver is available as
`clustering-catalog-blinding`. The examples below are explicit/open examples:
the current saved-catalog CLI takes `--w0`/`--wa` directly and does not yet load
from the private hashed `w0`/`wa` banks described above.
The example output paths keep standard LSS filenames inside a private blinded
catalog-version directory so downstream measurement tools can treat them as a
normal `LSScats` catalog tree. LSS saved the RSD reconstructed-realspace
intermediate as `*_clustering.IFFTrsd.dat.fits`, then wrote the final RSD-blinded
catalog back to the standard `*_clustering.dat.fits` name in the blinded output
directory. Random catalogs are matched to the current data catalog by resampling
redshift-dependent columns (`Z`, weights, and related columns) from the shifted
data, following the LSS `mkclusran`/`clusran_resamp` pattern. The examples below
put optional saved reconstructed-realspace and reconstruction-random
intermediates under an `intermediate/` subdirectory.

```bash
clustering-catalog-blinding \
  --modes bao rsd \
  --input-catalog LRG_NGC_clustering.dat.fits \
  --realspace-catalog LRG_NGC_clustering.IFFTrsd.dat.fits \
  --random-catalog LRG_NGC_0_clustering.ran.fits \
  --output-catalog /private/path/LSScats/vX-desiblind-w0wa-closed/LRG_NGC_clustering.dat.fits \
  --output-random-catalog /private/path/LSScats/vX-desiblind-w0wa-closed/LRG_NGC_0_clustering.ran.fits \
  --diagnostic-plot-dir /private/path/LSScats/vX-desiblind-w0wa-closed/diagnostics/LRG_NGC \
  --tracer-name LRG3 \
  --w0 -0.95 --wa 0.10 --zeff 0.8 --bias 2.0 --fiducial-f 0.8
```

When `--diagnostic-plot-dir` is supplied, the driver writes real-run diagnostics
from the catalogs being processed: data redshift distributions through the
blinding steps, BAO/AP-blinded-input and final random matching plots, weight diagnostics, and
a JSON summary with matching metrics. These are generated artifacts and should
usually live next to private blinded outputs, not in git.

For `rsd`, the reconstructed-realspace catalog can be supplied explicitly as
above. Alternatively, compute the realspace catalog directly from this
repository, without importing LSS. The reference-compatible backend is direct
`pyrecon`:

```bash
clustering-catalog-blinding \
  --modes bao rsd \
  --run-pyrecon \
  --input-catalog LRG_NGC_clustering.dat.fits \
  --random-catalog LRG_NGC_0_clustering.ran.fits \
  --save-realspace-catalog /private/path/LSScats/vX-desiblind-w0wa-closed/intermediate/LRG_NGC_clustering.IFFTrsd.dat.fits \
  --save-reconstruction-random-catalog /private/path/LSScats/vX-desiblind-w0wa-closed/intermediate/LRG_NGC_0_clustering.bao-matched.ran.fits \
  --output-catalog /private/path/LSScats/vX-desiblind-w0wa-closed/LRG_NGC_clustering.dat.fits \
  --output-random-catalog /private/path/LSScats/vX-desiblind-w0wa-closed/LRG_NGC_0_clustering.ran.fits \
  --tracer-name LRG3 \
  --w0 -0.95 --wa 0.10 --zeff 0.8 --bias 2.0 --fiducial-f 0.8 \
  --recon-method iterative_fft \
  --recon-smoothing-radius 15 --recon-growth-rate 0.8 \
  --recon-boxsize 6000 --recon-meshsize 64 --recon-cellsize 7
```

There is also a JAX-native backend for speed/on-the-fly studies. The driver
matches pyrecon's data-derived box center and random-threshold convention before
running JAX reconstruction. The optional saved `JAXrsd` intermediate is the
JAX-native analog of the LSS `IFFTrsd` realspace catalog:

```bash
clustering-catalog-blinding \
  --modes bao rsd \
  --run-jaxrecon \
  --input-catalog LRG_NGC_clustering.dat.fits \
  --random-catalog LRG_NGC_0_clustering.ran.fits \
  --save-realspace-catalog /private/path/LSScats/vX-desiblind-w0wa-closed/intermediate/LRG_NGC_clustering.JAXrsd.dat.fits \
  --save-reconstruction-random-catalog /private/path/LSScats/vX-desiblind-w0wa-closed/intermediate/LRG_NGC_0_clustering.bao-matched.ran.fits \
  --output-catalog /private/path/LSScats/vX-desiblind-w0wa-closed/LRG_NGC_clustering.dat.fits \
  --output-random-catalog /private/path/LSScats/vX-desiblind-w0wa-closed/LRG_NGC_0_clustering.ran.fits \
  --tracer-name LRG3 \
  --w0 -0.95 --wa 0.10 --zeff 0.8 --bias 2.0 --fiducial-f 0.8 \
  --recon-smoothing-radius 15 --recon-growth-rate 0.8 \
  --recon-boxsize 6000 --recon-meshsize 64
```

With `--run-pyrecon`, the driver uses
`clustering_statistics.recon_tools.compute_pyrecon_rsd_realspace_positions`, a
direct pyrecon wrapper that mirrors the LSS `convention='rsd'` reconstruction
logic but does not import LSS. With `--run-jaxrecon`, the driver uses
`clustering_statistics.recon_tools.compute_rsd_realspace_positions`, which wraps
`jaxrecon` and reads the RSD-only shifted data positions (`field='rsd'`). If both
`bao` and `rsd` are requested, the driver follows the LSS ordering from the
clustering-catalog stage onward: BAO/AP-remap the data with `desiblind`, store the
`n(z)_in/n(z)_out` BAO correction as an internal factor folded into the final
`WEIGHT` (leaving `WEIGHT_SYS` available for imaging/systematics weights or
later `TARGETID` matching, and without writing extra blinding-specific columns),
resample random redshift-dependent columns from the BAO/AP-blinded data for
reconstruction, run reconstruction, apply the RSD redshift shift to the data, then
resample final randoms again from the BAO/AP-blinded, RSD-shifted data and recompute
simple `NZ`/`NX`/`WEIGHT_FKP` columns. LSS is only used
by separate validation scripts as the benchmark/reference workflow. The intended
workflow remains two-stage: prepare saved blinded catalogs, then measure Pk/xi
from those catalogs with `compute_stats_from_options`. A future closed-production
version of this driver should load the same private `w0`/`wa` banks produced by
`create_catalog_w0wa_blinding_bank.py`, rather than putting `w0`/`wa` on the
command line.

For mesh-parity diagnostics or externally fixed meshes, pass
`--recon-boxcenter X Y Z` to force an exact Cartesian reconstruction box center.
If omitted, the JAX RSD backend uses pyrecon's data-derived bounding-box center.

Validation note: direct `pyrecon` and matched-convention `jaxrecon` both matched
the LSS/pyrecon 2k-row NGC/SGC RSD reference products at `~3e-9` max final-Z
delta. The key conventions were: pyrecon-style data box center and
`--recon-threshold-randoms-method mean`. Remaining work for making JAX the
default on-the-fly choice is performance/scaling validation on larger catalogs.
