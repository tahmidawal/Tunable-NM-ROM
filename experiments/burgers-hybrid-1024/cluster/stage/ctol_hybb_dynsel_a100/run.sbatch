#!/bin/bash
#SBATCH -J ctol_hybb_dynsel_a100
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:30:00
#SBATCH -o /cluster/tufts/paralab/tawal01/hybb1024/ctol_hybb_dynsel_a100/logs/%j.out
#SBATCH -e /cluster/tufts/paralab/tawal01/hybb1024/ctol_hybb_dynsel_a100/logs/%j.err
set -euo pipefail
cd "/cluster/tufts/paralab/tawal01/hybb1024/ctol_hybb_dynsel_a100/code"
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export BH_COMMIT=21008d9baceffc7f99dd561302cc21c033783ceb
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "commit=21008d9baceffc7f99dd561302cc21c033783ceb dirty=e3b0c44298fc cell=ctol_hybb_dynsel_a100"
$PY - <<'PRE' || { echo "GPU PREFLIGHT FAILED"; exit 42; }
import jax, sys
backend = jax.default_backend()
print(f"jax_backend={backend}", jax.devices()[0], flush=True)
sys.exit(0 if backend == "gpu" else 42)
PRE
export NS=256 COARSE_NS=64,128,256 RELAXATIONS=1.0 FOM_TAU=1e-6 LIN_TOL=1e-2 N_CASES=4 CASE_START=0 DEV_SEED=20260818 DRAW_COUNT=16 TIME_REPS=7 BURN_S=3
$PY bh_dynamic_correction_gate.py ../out/dynamic_selection.json
echo ALL-DONE
