#!/bin/bash
#SBATCH --job-name=mr_b2d_historical
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100-80G
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=04:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/mr_burgers2d_20260907/historical_20260910_r1/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/mr_burgers2d_20260907/historical_20260910_r1/logs/%j.err
set -euo pipefail
TASK_ROOT=/cluster/tufts/paralab/tawal01/mr_burgers2d_20260907/historical_20260910_r1
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_DEFAULT_MATMUL_PRECISION=highest
export JAX_ENABLE_X64=true
export PYTHONUNBUFFERED=1
cd "$TASK_ROOT"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}', flush=True); sys.exit(0 if b=='gpu' else 42)"
sha256sum -c MANIFEST.sha256 --quiet
"$PY" run_ladder.py
echo ALL-DONE
