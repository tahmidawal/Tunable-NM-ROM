#!/bin/bash
#SBATCH -J ctol_hybb_dynpair1
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:30:00
#SBATCH -o /cluster/tufts/paralab/tawal01/hybb1024/ctol_hybb_dynpair1/logs/%j.out
#SBATCH -e /cluster/tufts/paralab/tawal01/hybb1024/ctol_hybb_dynpair1/logs/%j.err
set -euo pipefail
cd "/cluster/tufts/paralab/tawal01/hybb1024/ctol_hybb_dynpair1/code"
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export BH_COMMIT=d9543a1304e2bf13add463dddccf025dd348b9c1
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "commit=d9543a1304e2bf13add463dddccf025dd348b9c1 dirty=e3b0c44298fc cell=ctol_hybb_dynpair1"
$PY - <<'PRE' || { echo "GPU PREFLIGHT FAILED"; exit 42; }
import jax, sys
backend = jax.default_backend()
print(f"jax_backend={backend}", jax.devices()[0], flush=True)
sys.exit(0 if backend == "gpu" else 42)
PRE
export N=256 COARSE_N=256 RELAXATION=1.0 FOM_TAU=1e-6 LIN_TOL=1e-2 DEV_SEED=20260818 DRAW_COUNT=16 CASE_START=8 N_CASES=4 PAIR_BLOCKS=6 BURN_S=3 SELECTION_JSON=selection/dynamic_selection_choice.json
$PY bh_dynamic_pair.py ../out/dynamic_pair.json
echo ALL-DONE
