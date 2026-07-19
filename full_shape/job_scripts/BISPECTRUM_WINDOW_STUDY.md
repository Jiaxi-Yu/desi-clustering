# Bispectrum-window grid convergence study

`study_bispectrum_window.py` measures the numerical impact of the bispectrum
window grid without requiring new converged chains. Comparisons at different
theory grids always use identical observable coordinates, data, covariance,
scale cuts, and parameter values.

The cached LRG2 bispectrum window has native spacing
`0.0025 h/Mpc`. The production compactification has two stages, so the usual
first-stage choices lead to:

| Observable `dk` | First-stage theory `dk` | Final theory `dk` |
|---:|---:|---:|
| 0.005 | dynamic = 0.005 | 0.010 |
| 0.010 | dynamic = 0.010 | 0.020 |
| either | fixed = 0.005 | 0.010 |
| either | fixed = 0.0025 | 0.005 |

The exact values are logged at runtime from the loaded window.

## LRG2 evaluation

From `full_shape/job_scripts` on `enrique-dev-dk`:

```bash
OUT=/pscratch/sd/e/epaillas/bk_window_convergence/lrg2

srun -n 1 python study_bispectrum_window.py \
  --observable-dk 0.005 0.01 \
  --theory-dk 0.0025 0.005 0.01 \
  --reference-theory-dk 0.0025 \
  --stages evaluate direct native fisher \
  --chain dk005_dynamic=/pscratch/sd/e/epaillas/fits_abacus_mocks/jul17_dk005/abacus-hf-dr2-v2-altmtl/cosmo-base_LRG2-S2+LRG2-S3-11dcbf5a \
  --chain dk01_dynamic=/pscratch/sd/e/epaillas/fits_abacus_mocks/jul17/abacus-hf-dr2-v2-altmtl/cosmo-base_LRG2-S2+LRG2-S3-05e9bf55 \
  --chain dk01_fixed005=/pscratch/sd/e/epaillas/fits_abacus_mocks/jul18/abacus-hf-dr2-v2-altmtl/cosmo-base_LRG2-S2+LRG2-S3-63740048 \
  --output-dir "$OUT"
```

The default dense sample is 2,000 points per matching source chain. For a fast
smoke run, add `--nposterior 8 --ndirect 2 --native-anchors 1`.

Native-window evaluations are deliberately sparse because they bypass both
window compactification stages. Direct evaluations disable Taylor emulators;
the resulting `emulator_validation.csv` keeps emulator and window errors
separate.

## Profiles

Profiles are independent products and can be launched after the prediction
study:

```bash
srun -n 1 python study_bispectrum_window.py \
  --observable-dk 0.005 0.01 \
  --theory-dk 0.0025 0.005 0.01 \
  --stages profile \
  --output-dir "$OUT"
```

Each observable-grid directory contains its own fit outputs and `profiles.csv`.
No MCMC sampling is run.

## Tracer-wide confirmation

After reviewing LRG2, repeat the prediction and profile stages for each tracer
with available joint P+B inputs. Use a separate output directory, for example:

```bash
srun -n 1 python study_bispectrum_window.py \
  --tracer LRG1 \
  --observable-dk 0.005 0.01 \
  --theory-dk 0.0025 0.005 0.01 \
  --stages evaluate profile \
  --nposterior 1 \
  --output-dir /pscratch/sd/e/epaillas/bk_window_convergence/lrg1
```

With no `--chain`, the evaluation contains the fiducial/default point and the
profiles provide the inference-level check.

## Products

Each `observable_dk_*` directory contains:

- `metrics.csv`: covariance-weighted model error and same-point Δχ²;
- `convergence.csv`: successive-grid convergence ratios and order estimates;
- `reweighting.csv`: chain importance-reweighting shifts and weight ESS;
- `emulator_validation.csv`: direct-versus-emulated prediction errors;
- `native_validation.csv`: finest compact grid versus the uncompressed window;
- `fisher_bias.csv`: local parameter-bias projections including Gaussian priors;
- `profiles.csv`: optional grid-specific Minuit results;
- `predictions.npz` and `manifest.json`: reproducibility inputs.

Open `full_shape/nb/explore_bispectrum_window_convergence.ipynb` to summarize
these products. The notebook performs no likelihood evaluations.
