#!/bin/bash
#SBATCH --job-name=ctol_noa_fno_b02
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100-80G
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:15:00
#SBATCH --signal=B:USR1@180
#SBATCH --output=/cluster/tufts/paralab/tawal01/no_audit_20260914/fno_burgers02/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/no_audit_20260914/fno_burgers02/logs/%j.err
set -euo pipefail
cd /cluster/tufts/paralab/tawal01/no_audit_20260914/fno_burgers02
export JAX_ENABLE_X64=true
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=8
export WANDB_MODE=offline
export WANDB_DISABLED=true
export TMPDIR=/cluster/tufts/paralab/tawal01/no_audit_20260914/fno_burgers02/tmp
export XDG_CACHE_HOME=/cluster/tufts/paralab/tawal01/no_audit_20260914/fno_burgers02/cache
export TORCH_HOME=/cluster/tufts/paralab/tawal01/no_audit_20260914/fno_burgers02/cache/torch
export MPLCONFIGDIR=/cluster/tufts/paralab/tawal01/no_audit_20260914/fno_burgers02/cache/matplotlib
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
sha256sum -c MANIFEST.sha256
( cd data && sha256sum -c DATA.sha256 )
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
$PY -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}'); sys.exit(0 if b=='gpu' else 42)"
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv
exec "$PY" code/worker_burgers_long.py
