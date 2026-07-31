#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -c 32
#SBATCH -t 09:00:00
#SBATCH -C cpu
#SBATCH --array=0-20
#SBATCH -q regular
#SBATCH -A desi
#SBATCH --output=./logs/mesh3_ELG_QSO_%A_%a.out

set -euo pipefail
# set +u
# source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
# unset JAX_PLATFORMS
# unset JAX_PLATFORM_NAME
# export MPICH_GPU_SUPPORT_ENABLED=1
# export MPICH_MPIIO_DVS_MAXNODES=1
# set -u

source /global/homes/s/shengyu/env.sh fit_env

# TRACERS=('BGS1' 'LRG1' 'LRG2' 'LRG3')
TRACERS=('ELG1' 'ELG2' 'QSO1')
# REGIONS=('NGC' 'SGC')
# REGIONS=('GCcomb_noN' 'GCcomb_noDES' 'NGC' 'SGC' 'N' 'NGCnoN' 'SGCnoDES')
REGIONS=('GCcomb')
# REGIONS=('ACT_DR6' 'PLANCK_PR4' 'GAL040' 'GAL060')

NTRACERS=${#TRACERS[@]}
NREGIONS=${#REGIONS[@]}
NTASKS=$((NTRACERS * NREGIONS))
TASK_ID=${SLURM_ARRAY_TASK_ID}
if (( TASK_ID < 0 || TASK_ID >= NTASKS )); then
    echo "SLURM_ARRAY_TASK_ID=${TASK_ID} is outside task range 0-$((NTASKS - 1))" >&2
    exit 1
fi
TRACER_INDEX=$((TASK_ID / NREGIONS))
REGION_INDEX=$((TASK_ID % NREGIONS))
tracer=${TRACERS[$TRACER_INDEX]}
region=${REGIONS[$REGION_INDEX]}

REPO_DIR="${HOME}/Y3/blinded_data_splits/desi-clustering"
SCRIPT_DIR="${REPO_DIR}/full_shape/job_scripts"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

cd "${SCRIPT_DIR}"

echo "Running task ${TASK_ID}/${NTASKS}: tracer=${tracer}, region=${region}"

srun -N 1 -n 4 -C cpu -c 32 --cpu-bind=cores \
    python test_data_splits.py \
    --fit_tracers "${tracer}" \
    --fit_regions "${region}" \
    --fits_dir /global/cfs/cdirs/desi/users/shengyu/Y3/full-shape/data_splits/fits \
    --todo sample \
    --cosmo_params base \
    --sampler pocomc \
    --auw \
    --use_rsf

# salloc -N 1 -n 4 -c 32 -C cpu --qos interactive -t 04:00:00 --account desi
