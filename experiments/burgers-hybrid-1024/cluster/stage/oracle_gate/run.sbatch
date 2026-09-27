#!/bin/bash
#SBATCH -J hybb_oracle_gate
#SBATCH -p gpu
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH -t 00:35:00
#SBATCH -o /cluster/tufts/paralab/tawal01/hybb1024/oracle_gate/logs/%j.out
#SBATCH -e /cluster/tufts/paralab/tawal01/hybb1024/oracle_gate/logs/%j.err
set -euo pipefail
cd "/cluster/tufts/paralab/tawal01/hybb1024/oracle_gate/code"
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export BH_COMMIT=c79c485cd58daf5ce5c852063fbe95377bc6f30b
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "commit=c79c485cd58daf5ce5c852063fbe95377bc6f30b dirty=e3b0c44298fc cell=oracle_gate"
$PY - <<'PRE' || { echo "GPU PREFLIGHT FAILED"; exit 42; }
import jax, sys
backend = jax.default_backend()
print(f"jax_backend={backend}", jax.devices()[0], flush=True)
sys.exit(0 if backend == "gpu" else 42)
PRE
export NS=64,256 FOM_TAUS=1e-6 QUALITY=0,0.1,0.25,0.5,0.75,0.9,0.99,1 N_TEST_TRAJ=4 TIME_REPS=7 TIME_WARM=2 BURN_S=3 RUN_ROLE=gate
$PY bh_oracle.py ../out/oracle_gate.json
echo ALL-DONE
