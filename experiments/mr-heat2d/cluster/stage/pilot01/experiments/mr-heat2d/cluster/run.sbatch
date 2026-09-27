#!/usr/bin/env bash
#SBATCH --job-name=ctol_mr_heat_p01
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=job-%j.log
set -euo pipefail
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_DEFAULT_MATMUL_PRECISION=highest
export JAX_ENABLE_X64=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
export OMP_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="$PWD/matplotlib-cache"
export JAX_COMPILATION_CACHE_DIR="$PWD/jax-cache"
df -h "$PWD"
nvidia-smi
$PY -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}'); sys.exit(0 if b=='gpu' else 42)"
$PY experiments/mr-heat2d/verify_heat.py
$PY experiments/mr-heat2d/run_pilot.py --out outputs
