#!/bin/bash
#SBATCH -J hybb_history_blend_gate
#SBATCH -p gpu
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 00:15:00
#SBATCH -o /cluster/tufts/paralab/tawal01/hybb1024/history_blend_gate/logs/%j.out
#SBATCH -e /cluster/tufts/paralab/tawal01/hybb1024/history_blend_gate/logs/%j.err
set -euo pipefail
cd "/cluster/tufts/paralab/tawal01/hybb1024/history_blend_gate/code"
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export BH_COMMIT=46c5ea892f71cad94db843c83e1c2ed69d608739
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "commit=46c5ea892f71cad94db843c83e1c2ed69d608739 dirty=e3b0c44298fc cell=history_blend_gate"
$PY - <<'PRE' || { echo "GPU PREFLIGHT FAILED"; exit 42; }
import jax, sys
backend = jax.default_backend()
print(f"jax_backend={backend}", jax.devices()[0], flush=True)
sys.exit(0 if backend == "gpu" else 42)
PRE
export NS=64,256 FOM_TAUS=1e-6 LINEAR_TOL=1e-2 PRECONDITIONER=helmholtz RUN_ROLE=calibration DATA_SEED=2 DRAW_COUNT=8 N_TRAJ=4 ARMS=linear,lq25,lq50,lq75,quadratic,qc25,qc50,qc75,cubic TIME_REPS=7 TIME_WARM=2 BURN_S=3 REFERENCE_RESIDUAL_GATE=1e-11
$PY bh_physics_controls.py ../out/history_blend_gate.json
echo ALL-DONE
