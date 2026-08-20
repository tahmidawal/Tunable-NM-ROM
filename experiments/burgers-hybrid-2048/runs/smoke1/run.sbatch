#!/bin/bash
#SBATCH -J ctol_hybb2048_smoke1
#SBATCH -p gpu
#SBATCH --gres=gpu:h200:1
#SBATCH -c 8
#SBATCH --mem=256G
#SBATCH -t 01:00:00
#SBATCH -o /cluster/tufts/paralab/tawal01/hybb2048/ctol_hybb2048_smoke1/logs/%j.out
#SBATCH -e /cluster/tufts/paralab/tawal01/hybb2048/ctol_hybb2048_smoke1/logs/%j.err
set -euo pipefail
cd "/cluster/tufts/paralab/tawal01/hybb2048/ctol_hybb2048_smoke1/code"
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export BH_COMMIT=3f95afa2d178dbe6ef93483e425680cdf48a500e
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "commit=3f95afa2d178dbe6ef93483e425680cdf48a500e dirty=e3b0c44298fc cell=ctol_hybb2048_smoke1"
$PY - <<'PRE' || { echo "GPU PREFLIGHT FAILED"; exit 42; }
import jax, sys
backend = jax.default_backend()
print(f"jax_backend={backend}", jax.devices()[0], flush=True)
sys.exit(0 if backend == "gpu" else 42)
PRE
export NS=2048 FOM_TAUS=1e-6 LINEAR_TOLS=1e-2 TEST_SEED=20260818 DRAW_COUNT=16 TEST_START=12 N_CASES=1 PAIR_BLOCKS=1 BURN_S=0.1 REFERENCE_RESIDUAL_GATE=1e-11 SELECTION_JSON=selection/dynamic_selection_choice.json PAIR_AUDIT_JSON=selection/dynamic_pair_audit.json BH_2048_SMOKE=1
$PY bh_2048_final.py ../out/smoke.json
echo ALL-DONE
