#!/bin/bash
#SBATCH -J hybb_solver_lock
#SBATCH -p gpu
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 00:35:00
#SBATCH -o /cluster/tufts/paralab/tawal01/hybb1024/solver_lock/logs/%j.out
#SBATCH -e /cluster/tufts/paralab/tawal01/hybb1024/solver_lock/logs/%j.err
set -euo pipefail
cd "/cluster/tufts/paralab/tawal01/hybb1024/solver_lock/code"
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export BH_COMMIT=2c00e93b2a983405d0820d9438b68c15ca6c7e4f
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "commit=2c00e93b2a983405d0820d9438b68c15ca6c7e4f dirty=e3b0c44298fc cell=solver_lock"
$PY - <<'PRE' || { echo "GPU PREFLIGHT FAILED"; exit 42; }
import jax, sys
backend = jax.default_backend()
print(f"jax_backend={backend}", jax.devices()[0], flush=True)
sys.exit(0 if backend == "gpu" else 42)
PRE
export PRECONDITIONERS=none,helmholtz LINEAR_TOLS=1e-2,1e-3,1e-4,1e-5 NS=64,256 FOM_TAUS=1e-6,1e-8,1e-10 CALIB_SEED=2 CALIB_DRAW_COUNT=8 N_CALIB_TRAJ=4 TIME_REPS=7 TIME_WARM=2 BURN_S=3
$PY bh_linear_tolerance.py ../out/solver_lock.json
echo ALL-DONE
