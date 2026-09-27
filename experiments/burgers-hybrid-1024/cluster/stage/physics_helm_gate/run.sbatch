#!/bin/bash
#SBATCH -J hybb_physics_helm_gate
#SBATCH -p gpu
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 00:20:00
#SBATCH -o /cluster/tufts/paralab/tawal01/hybb1024/physics_helm_gate/logs/%j.out
#SBATCH -e /cluster/tufts/paralab/tawal01/hybb1024/physics_helm_gate/logs/%j.err
set -euo pipefail
cd "/cluster/tufts/paralab/tawal01/hybb1024/physics_helm_gate/code"
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export BH_COMMIT=8743f9650c5295fb72cf69d76f1ce8e7d220a2a6
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "commit=8743f9650c5295fb72cf69d76f1ce8e7d220a2a6 dirty=e3b0c44298fc cell=physics_helm_gate"
$PY - <<'PRE' || { echo "GPU PREFLIGHT FAILED"; exit 42; }
import jax, sys
backend = jax.default_backend()
print(f"jax_backend={backend}", jax.devices()[0], flush=True)
sys.exit(0 if backend == "gpu" else 42)
PRE
export NS=64,256 FOM_TAUS=1e-6 LINEAR_TOL=1e-2 PRECONDITIONER=helmholtz RUN_ROLE=calibration DATA_SEED=2 DRAW_COUNT=8 N_TRAJ=4 ARMS=linear,quadratic,cubic,imex_euler,imex_ab2 TIME_REPS=7 TIME_WARM=2 BURN_S=3 REFERENCE_RESIDUAL_GATE=1e-11
$PY bh_physics_controls.py ../out/physics_helm_gate.json
echo ALL-DONE
