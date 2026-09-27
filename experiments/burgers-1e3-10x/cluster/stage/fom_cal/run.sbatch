#!/bin/bash
#SBATCH -J ctol_b10_fom_cal
#SBATCH -p gpu
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 01:30:00
#SBATCH -o /cluster/tufts/paralab/tawal01/burgers_nmrom_1e3_10x/fom_cal/logs/%j.out
#SBATCH -e /cluster/tufts/paralab/tawal01/burgers_nmrom_1e3_10x/fom_cal/logs/%j.err
set -euo pipefail
cd "/cluster/tufts/paralab/tawal01/burgers_nmrom_1e3_10x/fom_cal/code"
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export B10_COMMIT=048e83f1a75e8019bd610eaebcbd2b56c1fa9d23
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "commit=048e83f1a75e8019bd610eaebcbd2b56c1fa9d23 dirty=e3b0c44298fc cell=fom_cal"
$PY - <<'PRE' || { echo "GPU PREFLIGHT FAILED"; exit 42; }
import jax, sys
backend = jax.default_backend()
print(f"jax_backend={backend}", jax.devices()[0], flush=True)
sys.exit(0 if backend == "gpu" else 42)
PRE
export NS=256,512,1024 CAL_SEED=20260822 CAL_DRAW_COUNT=32 CAL_CASES=4 TIME_REPS=14 TIME_WARM=2 BURN_SECONDS=3
$PY b10_fom_cal.py ../out/fom_cal.json
echo ALL-DONE
