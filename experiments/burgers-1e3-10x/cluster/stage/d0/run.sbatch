#!/bin/bash
#SBATCH -J ctol_b10_d0
#SBATCH -p gpu
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 00:45:00
#SBATCH -o /cluster/tufts/paralab/tawal01/burgers_nmrom_1e3_10x/d0/logs/%j.out
#SBATCH -e /cluster/tufts/paralab/tawal01/burgers_nmrom_1e3_10x/d0/logs/%j.err
set -euo pipefail
cd "/cluster/tufts/paralab/tawal01/burgers_nmrom_1e3_10x/d0/code"
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export B10_COMMIT=e378215c68b5742db5c9efc1eacabbac27bbb376
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "commit=e378215c68b5742db5c9efc1eacabbac27bbb376 dirty=e3b0c44298fc cell=d0"
$PY - <<'PRE' || { echo "GPU PREFLIGHT FAILED"; exit 42; }
import jax, sys
backend = jax.default_backend()
print(f"jax_backend={backend}", jax.devices()[0], flush=True)
sys.exit(0 if backend == "gpu" else 42)
PRE
export N=64 DATA_SEED=0 DRAW_COUNT=704 D0_TRAIN_COUNT=512 SELECTION_START=512 SELECTION_COUNT=64 HG_DEGREES=4,5
$PY b10_d0.py ../out/d0.json ../out/d0.npz deps/inherited_h160.json
echo ALL-DONE
