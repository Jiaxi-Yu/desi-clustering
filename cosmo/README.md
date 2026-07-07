# Cosmology inference

This directory contains cosmological inference helpers that run downstream of
`clustering_statistics/` measurements and `bao/` BAO fitting/compression.

DESI cosmology fits are run with Cobaya using lightweight `desi-clustering`
helpers: small runtime functions, explicit option dictionaries, injectable
paths, registry-driven likelihood metadata, and thin desipipe job scripts.

## Environment

Use Arnaud's current cosmodesi environment on Perlmutter:

```bash
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
cd /global/homes/u/uendert/repos/desi/desi-clustering-cosmo
export PYTHONPATH=$PWD:$PYTHONPATH
```

The module environment includes Cobaya, desipipe, desilike, and the standard
DESI cosmology stack. During development, the environment may also load a
packaged `desi-clustering` module. If Python imports that packaged copy instead
of this checkout, unload it before setting `PYTHONPATH`:

```bash
module unload desi-clustering || true
export PYTHONPATH=$PWD:$PYTHONPATH
```

This is a development safeguard while working from a checkout, not a separate
cosmology-runtime requirement.

## Current scope

BAO likelihoods use native lightweight Cobaya wrapper classes in
`cosmo.cobaya.external_likelihoods.bao.desi_dr2`. Shared BAO dataset metadata is kept in
`cosmo.cobaya.mapping_likelihoods.bao`, registered in `cosmo.cobaya.mapping_likelihoods.registry`, and translated to
Cobaya likelihood dictionaries by `cosmo.cobaya.mapping_likelihoods.cobaya`. The registry uses a
general `likelihoods` interface so likelihood families can be combined without
changing the runtime helpers.

DESI BAO DR2 mean/covariance text files are read from the DESI collaboration CFS
area by default:

```text
/global/cfs/cdirs/desicollab/science/cpe/y3_bao_cosmo/bao_v1p2/bao/cobaya_data
```

Override this path with `likelihood_path=` or:

```bash
export DESI_CLUSTERING_COSMO_BAO_DATA_PATH=/path/to/cobaya_data
```

Registered likelihoods include DESI BAO DR2, BAO GQC/Lya-FS variants, SN, BBN, compressed CMB, full CMB,
and CMB-lensing entries. BBN, SN zmin variants, compressed-CMB priors, and the
Momento/SROLL wrapper are native `desi-clustering` likelihoods under
`cosmo.cobaya.external_likelihoods`. The Cobaya helper supports background-only
combinations, such as BAO+SN+BBN+compressed-CMB priors, and full-CMB Cobaya
configs for standard external Planck/ACT/lensing likelihoods. The `CMB-SP4A`
preset is the June 2026 CPE full-CMB baseline: PR4 CamSpec-NPIPE-lite, ACT
DR6 CMB-only, SPT-3G lite, low-ell TT/EE, and joint Planck+ACT+SPT lensing.
Full-CMB initialization and production runs depend on the external Cobaya
likelihood packages/data available in the active environment. DESI DR1 BAO
would require a separate data mapping.

- likelihoods: `desi-dr2-bao-all`, SN names such as `pantheonplus`, BBN names such
  as `schoneberg2024-bbn`, compressed-CMB names such as `CMB-compressed-theta`,
  and full-CMB names such as `planck-NPIPE-highl-CamSpec-TTTEEE`
  and `CMB-SP4A`;
- models: `base`, `base_w`, `base_w_wa`;
- theory: `camb`;
- samplers: `evaluate`, `cobaya` MCMC, `iminuit` minimization.

Use explicit likelihood names for BAO products, following the desilike
mapping. In particular, `desi-dr2-bao-all` is the standard DESI DR2 BAO
product also used by the standard desilike DR2 BAO mapping. For CPE SP4A
reference-chain comparisons, use `desi-dr2-bao-gqc-lya-fs`, the GQC + updated
Lyman-alpha full-shape compressed product. Pass combinations as comma-separated
explicit likelihood names, e.g. `desi-dr2-bao-gqc-lya-fs,desdovekie,CMB-SP4A`.

Compressed BAO products from `bao/` can be connected by writing compatible
mean/covariance text files and pointing `likelihood_path` to their directory.

## Direct Python usage

Use `cosmo.cobaya` for Cobaya configuration and runtime helpers.

```python
from cosmo.cobaya import get_cobaya_info, sample_cobaya

info = get_cobaya_info(
    model='base',
    likelihoods='desi-dr2-bao-all',
    sampler='evaluate',
    output=False,
)

sample_cobaya(model='base', likelihoods='desi-dr2-bao-all', sampler='evaluate', output=False)
```

For BAO-only calls, `dataset=` accepts explicit BAO dataset names:

```python
info = get_cobaya_info(model='base', dataset='desi-dr2-bao-all', sampler='evaluate', output=False)
```

Full-CMB configs can be generated with the `likelihoods=` interface:

```python
info = get_cobaya_info(
    model='base',
    likelihoods='desi-dr2-bao-all,planck-NPIPE-highl-CamSpec-TTTEEE',
    sampler='evaluate',
    output=False,
)
```

To write a Cobaya YAML:

```python
from cosmo.cobaya import get_cobaya_info, write_cobaya_yaml

info = get_cobaya_info(model='base', likelihoods='desi-dr2-bao-all', sampler='cobaya')
write_cobaya_yaml(info, 'configs/base_desi-dr2-bao-all.yaml')
```

## Cobaya run launcher

The Cobaya launcher entry point is:

```bash
python cosmo/job_scripts/run_cobaya.py \
    --todo evaluate \
    --models base \
    --likelihoods desi-dr2-bao-all,pantheonplus,CMB-compressed-theta
```

The launcher uses explicit likelihood names. Pass comma-separated names for
combined configurations. Examples:

```text
desi-dr2-bao-all
desi-dr2-bao-gqc
desi-dr2-bao-lya-fs
desi-dr2-bao-gqc-lya-fs
desi-dr2-bao-gqc-lya-fs,desdovekie
desi-dr2-bao-gqc-lya-fs,desdovekie,CMB-SP4A
CMB-SP4A
CMB-SPA
```

Comma-separated explicit likelihood combinations are supported. Pass multiple
`--likelihoods` values to create multiple configurations:

```bash
python cosmo/job_scripts/run_cobaya.py \
    --todo evaluate \
    --models base \
    --likelihoods desi-dr2-bao-all \
                  desi-dr2-bao-all,pantheonplus \
                  desi-dr2-bao-all,CMB-compressed-theta
```

Inspect and spawn:

```bash
desipipe tasks -q desi_clustering_cobaya
desipipe spawn -q desi_clustering_cobaya --spawn
desipipe queues -q desi_clustering_cobaya
```

### Run modes

The job script supports three runtime modes. All modes use the same likelihood
preset expansion and Cobaya configuration builder.

#### 1. Desipipe batch mode

By default, the script creates desipipe tasks and stores them in the requested
queue. This is the preferred mode for production matrices.

```bash
python cosmo/job_scripts/run_cobaya.py \
    --todo sample \
    --models base \
    --likelihoods desi-dr2-bao-all desi-dr2-bao-all,desdovekie desi-dr2-bao-all,schoneberg2024-bbn \
    --run run1 \
    --output_dir $SCRATCH/desi-clustering-cosmo-dev \
    --queue-name desi_clustering_cobaya

desipipe tasks -q desi_clustering_cobaya
desipipe spawn -q desi_clustering_cobaya --spawn
desipipe queues -q desi_clustering_cobaya
```

#### 2. Direct mode in the current environment

Use `--interactive` to run immediately in the current shell/allocation without
creating a desipipe queue. This is useful for `--test`, debugging, or when you
already have a compute allocation.

```bash
python cosmo/job_scripts/run_cobaya.py \
    --interactive \
    --todo evaluate \
    --models base \
    --likelihoods desi-dr2-bao-all,CMB-compressed-theta \
    --test
```

#### 3. Direct mode with a new interactive node

Use `--interactive-node` from a login node to request a NERSC interactive node
and then re-run this same script in `--interactive` mode inside that allocation.
This avoids writing a separate wrapper script for quick direct runs.

```bash
python cosmo/job_scripts/run_cobaya.py \
    --interactive-node \
    --todo sample \
    --models base \
    --likelihoods desi-dr2-bao-all,schoneberg2024-bbn desi-dr2-bao-all,CMB-compressed-theta \
    --run interactive-test \
    --output_dir $SCRATCH/desi-clustering-cosmo-dev \
    --time 04:00:00
```

The default interactive-node Slurm options are `-A desi -C cpu -q interactive
-N 1 -n 4 -c 32`. They can be changed with `--interactive-account`,
`--interactive-qos`, `--interactive-constraint`, `--interactive-nodes`,
`--mpiprocs-per-worker`, and `--cpus-per-task`.

To resume an interrupted direct run, add `--resume` and keep the same `--run`
label and `--output_dir`.

To export Cobaya YAML files for review without launching jobs:

```bash
python cosmo/job_scripts/run_cobaya.py \
    --todo export \
    --models base base_w base_w_wa \
    --likelihoods desi-dr2-bao-all desi-dr2-bao-all,schoneberg2024-bbn desi-dr2-bao-all,CMB-compressed-theta desi-dr2-bao-all,CMB-SPA \
    --config-dir configs/cobaya
```

This writes files such as:

```text
configs/cobaya/base/bao.yaml
configs/cobaya/base/bao-bbn.yaml
configs/cobaya/base/desi-dr2-bao-all,CMB-compressed-theta.yaml
configs/cobaya/base/bao-cmb-spa.yaml
```


## Validation notebooks

`cosmo/nb/simple_chain_comparison.ipynb` validates the simple/background
Cobaya chains from `desi-clustering-cosmo` against archived DESI KP/Y3 reference
chains. It includes BAO-only, SN-only, and BAO+DESY5-Dovekie diagnostics to
separate the standard v1p2 BAO product from Cristhian's `bao.desi_dr2_updated`
BAO product.

## Validation status

Generated Cobaya info dictionaries can be compared against the Y3 reference with:

```bash
python cosmo/job_scripts/compare_y3_cobaya_configs.py
```

The comparison canonicalizes native `desi-clustering` likelihood class paths and
checks canonical likelihoods, sampled parameters, and CAMB `extra_args`. The
default comparison set covers BAO-only, BAO+SN, BAO+BBN, BAO+compressed-CMB,
BAO+Planck NPIPE, BAO+SN+Planck NPIPE, and BAO+Planck NPIPE+lensing. For these
tested `base` configurations, sampled parameters and CAMB `extra_args` match the
reference comparison, except where runtime-complete native configs intentionally
provide parameters required by native likelihood initialization. The comparison
utility imports the Y3 repository only as a reference; runtime likelihood
configuration is native to this package.
