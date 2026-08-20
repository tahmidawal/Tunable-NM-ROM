#!/bin/bash
#SBATCH -J ctol_hybb_dynfinal1
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 02:00:00
#SBATCH -o /cluster/tufts/paralab/tawal01/hybb1024/ctol_hybb_dynfinal1/logs/%j.out
#SBATCH -e /cluster/tufts/paralab/tawal01/hybb1024/ctol_hybb_dynfinal1/logs/%j.err
set -euo pipefail
cd "/cluster/tufts/paralab/tawal01/hybb1024/ctol_hybb_dynfinal1/code"
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export BH_COMMIT=9cdeca5aabc9f911f3d6057e371813e11d3e6c89
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "commit=9cdeca5aabc9f911f3d6057e371813e11d3e6c89 dirty=e3b0c44298fc cell=ctol_hybb_dynfinal1"
$PY - <<'PRE' || { echo "GPU PREFLIGHT FAILED"; exit 42; }
import jax, sys
backend = jax.default_backend()
print(f"jax_backend={backend}", jax.devices()[0], flush=True)
sys.exit(0 if backend == "gpu" else 42)
PRE
export NS=32,64,128,256,512,1024 FOM_TAUS=1e-6,1e-8,1e-10 LINEAR_TOLS=1e-2,1e-4,1e-5 TEST_SEED=20260825 DRAW_COUNT=16 TEST_START=0 N_CASES=4 PAIR_BLOCKS=6 BURN_S=3 REFERENCE_RESIDUAL_GATE=1e-11 SELECTION_JSON=selection/dynamic_selection_choice.json PAIR_AUDIT_JSON=selection/dynamic_pair_audit.json
$PY bh_dynamic_final.py ../out/dynamic_final.json
echo ALL-DONE
