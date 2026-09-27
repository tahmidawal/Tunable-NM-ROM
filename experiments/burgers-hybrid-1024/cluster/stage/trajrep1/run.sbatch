#!/bin/bash
#SBATCH -J hybb_trajrep1
#SBATCH -p gpu
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:30:00
#SBATCH -o /cluster/tufts/paralab/tawal01/hybb1024/trajrep1/logs/%j.out
#SBATCH -e /cluster/tufts/paralab/tawal01/hybb1024/trajrep1/logs/%j.err
set -euo pipefail
cd "/cluster/tufts/paralab/tawal01/hybb1024/trajrep1/code"
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export BH_COMMIT=b002506ce976c2b114b1503a76fc550e2e07115c
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "commit=b002506ce976c2b114b1503a76fc550e2e07115c dirty=e3b0c44298fc cell=trajrep1"
$PY - <<'PRE' || { echo "GPU PREFLIGHT FAILED"; exit 42; }
import jax, sys
backend = jax.default_backend()
print(f"jax_backend={backend}", jax.devices()[0], flush=True)
sys.exit(0 if backend == "gpu" else 42)
PRE
export NS=64,256 COARSE_NS=48,64 N_VAL=4 VAL_START=4 VAL_SEED=20260818 VAL_DRAW_COUNT=16
$PY bh_trajectory_quality_probe.py ../out/trajectory_quality.json
echo ALL-DONE
