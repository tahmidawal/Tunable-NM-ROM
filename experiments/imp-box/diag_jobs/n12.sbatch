#!/bin/bash
#SBATCH --job-name=ibox-diag-n12
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/imp-box-diag/slurm-%j.out
set -euo pipefail
cd /cluster/tufts/paralab/tawal01/imp-box-diag/code
export JAX_DEFAULT_MATMUL_PRECISION=highest XLA_PYTHON_CLIENT_PREALLOCATE=false
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
nvidia-smi --query-gpu=name --format=csv,noheader
$PY -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}'); sys.exit(0 if b == 'gpu' else 42)" || exit 42
C=/cluster/tufts/paralab/tawal01/imp-box-diag/cache-$SLURM_JOB_ID
$PY diag_jobs/n12.py nocache
$PY diag_jobs/n12.py cache $C
$PY diag_jobs/n12.py cache $C
