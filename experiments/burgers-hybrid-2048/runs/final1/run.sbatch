#!/bin/bash
#SBATCH -J ctol_hybb2048_final1
#SBATCH -p gpu
#SBATCH --gres=gpu:h200:1
#SBATCH -c 8
#SBATCH --mem=256G
#SBATCH -t 04:00:00
#SBATCH -o /cluster/tufts/paralab/tawal01/hybb2048/ctol_hybb2048_final1/logs/%j.out
#SBATCH -e /cluster/tufts/paralab/tawal01/hybb2048/ctol_hybb2048_final1/logs/%j.err
set -euo pipefail
cd "/cluster/tufts/paralab/tawal01/hybb2048/ctol_hybb2048_final1/code"
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export BH_COMMIT=b1131b1e4501258d153a2a519e8c0a42d9063842
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "commit=b1131b1e4501258d153a2a519e8c0a42d9063842 dirty=e3b0c44298fc cell=ctol_hybb2048_final1"
$PY - <<'PRE' || { echo "GPU PREFLIGHT FAILED"; exit 42; }
import jax, sys
backend = jax.default_backend()
print(f"jax_backend={backend}", jax.devices()[0], flush=True)
sys.exit(0 if backend == "gpu" else 42)
PRE
export NS=2048 FOM_TAUS=1e-6,1e-8,1e-10 LINEAR_TOLS=1e-2,1e-4,1e-5 TEST_SEED=20260827 DRAW_COUNT=16 TEST_START=0 N_CASES=4 PAIR_BLOCKS=6 BURN_S=3 REFERENCE_RESIDUAL_GATE=1e-11 SELECTION_JSON=selection/dynamic_selection_choice.json PAIR_AUDIT_JSON=selection/dynamic_pair_audit.json BH_2048_SMOKE=0
$PY bh_2048_final.py ../out/final.json
echo ALL-DONE
