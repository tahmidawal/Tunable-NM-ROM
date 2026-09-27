#!/bin/bash
#SBATCH -J hybb_oracle_confirm
#SBATCH -p gpu
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 00:20:00
#SBATCH -o /cluster/tufts/paralab/tawal01/hybb1024/oracle_confirm/logs/%j.out
#SBATCH -e /cluster/tufts/paralab/tawal01/hybb1024/oracle_confirm/logs/%j.err
set -euo pipefail
cd "/cluster/tufts/paralab/tawal01/hybb1024/oracle_confirm/code"
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export BH_COMMIT=46c5ea892f71cad94db843c83e1c2ed69d608739
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "commit=46c5ea892f71cad94db843c83e1c2ed69d608739 dirty=e3b0c44298fc cell=oracle_confirm"
$PY - <<'PRE' || { echo "GPU PREFLIGHT FAILED"; exit 42; }
import jax, sys
backend = jax.default_backend()
print(f"jax_backend={backend}", jax.devices()[0], flush=True)
sys.exit(0 if backend == "gpu" else 42)
PRE
export NS=256 FOM_TAUS=1e-6 QUALITIES=.5,.75,.9,.99,1 LINEAR_TOL=1e-2 PRECONDITIONER=helmholtz TEST_SEED=1 TEST_DRAW_COUNT=16 TEST_START=0 N_TEST_TRAJ=4 TIME_REPS=21 TIME_WARM=2 BURN_S=3 TIGHTEST_FINAL_TAU=1e-10
$PY bh_oracle_confirm.py ../out/oracle_confirm.json
echo ALL-DONE
