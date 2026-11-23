#!/usr/bin/env bash
#SBATCH --job-name=ctol_jwd_f512
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=job.log
#SBATCH --error=job.log
set -euo pipefail
export JAX_ENABLE_X64=true
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export SOURCE_COMMIT=3276526f8731c54ae41020eab26b353a092049a2
export TMPDIR="$SLURM_SUBMIT_DIR/tmp"
export XDG_CACHE_HOME="$SLURM_SUBMIT_DIR/cache"
export CUDA_CACHE_PATH="$SLURM_SUBMIT_DIR/cache/cuda"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$CUDA_CACHE_PATH"
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
cd "$SLURM_SUBMIT_DIR/code"
$PY -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}'); sys.exit(0 if b=='gpu' else 42)"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
mkdir -p checks
$PY -u fixture.py
$PY -u run.py --config configs/fresh_r512.json --output ../output
