#!/bin/bash
#SBATCH -J ctol_hybb2048_diagnostic2
#SBATCH -p gpu
#SBATCH --gres=gpu:h200:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 02:00:00
#SBATCH -o /cluster/tufts/paralab/tawal01/hybb2048/ctol_hybb2048_diagnostic2/logs/%j.out
#SBATCH -e /cluster/tufts/paralab/tawal01/hybb2048/ctol_hybb2048_diagnostic2/logs/%j.err
set -euo pipefail
cd "/cluster/tufts/paralab/tawal01/hybb2048/ctol_hybb2048_diagnostic2/code"
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export BH_COMMIT=1f81e7cbcf48475fc018737e5590ab9cffeca512
export NEWTON_ITERS=25
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "commit=1f81e7cbcf48475fc018737e5590ab9cffeca512 dirty=e3b0c44298fc cell=ctol_hybb2048_diagnostic2"
$PY - <<'PRE' || { echo "GPU PREFLIGHT FAILED"; exit 42; }
import jax, sys
backend = jax.default_backend()
print(f"jax_backend={backend}", jax.devices()[0], flush=True)
sys.exit(0 if backend == "gpu" else 42)
PRE
export REFERENCE_DIAGNOSTIC=1
$PY bh_2048_reference_diag.py ../out/diagnostic.json
$PY bh_2048_reference_diag_check.py ../out/diagnostic.json ../out/producer-check.json
test -s ../out/producer-check.json
echo PRODUCER-CHECK-DONE
echo ALL-DONE
