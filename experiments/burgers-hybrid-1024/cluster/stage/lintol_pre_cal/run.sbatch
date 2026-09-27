#!/bin/bash
#SBATCH -J hybb_lintol_pre_cal
#SBATCH -p gpu
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 00:35:00
#SBATCH -o /cluster/tufts/paralab/tawal01/hybb1024/lintol_pre_cal/logs/%j.out
#SBATCH -e /cluster/tufts/paralab/tawal01/hybb1024/lintol_pre_cal/logs/%j.err
set -euo pipefail
cd "/cluster/tufts/paralab/tawal01/hybb1024/lintol_pre_cal/code"
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export BH_COMMIT=a44b459ce5b2ad9225108640a4b3f92c0a4d5d5d
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "commit=a44b459ce5b2ad9225108640a4b3f92c0a4d5d5d dirty=e3b0c44298fc cell=lintol_pre_cal"
$PY - <<'PRE' || { echo "GPU PREFLIGHT FAILED"; exit 42; }
import jax, sys
backend = jax.default_backend()
print(f"jax_backend={backend}", jax.devices()[0], flush=True)
sys.exit(0 if backend == "gpu" else 42)
PRE
export PRECONDITIONERS=none,jacobi LINEAR_TOLS=1e-3,1e-4,1e-5,1e-6 NS=64,256 FOM_TAUS=1e-6,1e-8,1e-10 CALIB_SEED=2 CALIB_DRAW_COUNT=8 N_CALIB_TRAJ=4 TIME_REPS=7 TIME_WARM=2 BURN_S=3
$PY bh_linear_tolerance.py out/lintol_pre_cal.json
echo ALL-DONE
