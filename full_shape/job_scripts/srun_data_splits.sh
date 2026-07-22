#!/bin/bash
set -euo pipefail

set +u
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
set -u

srun -N 1 -n 4 -c 32 -C gpu --gpus-per-node=4 --gpu-bind=single:1 \
    -t 04:00:00 --qos interactive --account desi_g \
    python test_data_splits.py \
    --fit_tracers LRG1 \
    --fit_regions GCcomb_noN GCcomb_noDES NGC SGC N NGCnoN SGCnoDES \
    --fits_dir /global/cfs/cdirs/desi/users/shengyu/Y3/full-shape/data_splits/fits \
    --todo profile \
    --cosmo_params base \
    --sampler pocomc \
    --auw \
    --use_rsf

# salloc -N 1 -n 4 -c 32 -C cpu --qos interactive -t 04:00:00 --account desi
