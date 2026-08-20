#!/bin/bash
#SBATCH -J ctol_hybb2048_learned1
#SBATCH -p gpu
#SBATCH --gres=gpu:h200:1
#SBATCH -c 8
#SBATCH --mem=256G
#SBATCH -t 08:00:00
#SBATCH -o /cluster/tufts/paralab/tawal01/hybb2048/ctol_hybb2048_learned1/logs/%j.out
#SBATCH -e /cluster/tufts/paralab/tawal01/hybb2048/ctol_hybb2048_learned1/logs/%j.err
set -euo pipefail
cd "/cluster/tufts/paralab/tawal01/hybb2048/ctol_hybb2048_learned1/code"
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export BH_COMMIT=b1131b1e4501258d153a2a519e8c0a42d9063842
export FILM_CHECKPOINT=deps/burgers2d-rom-latent-stepping/blat_ad_N64_K8.pkl
export SMOKE_AUDIT=smoke/smoke_audit.json
export N=64 N_TRAIN=512 N_VAL=64 SEED=0 BC_MODE=poly K_LAT=8
export AD_HIDDEN=256 AD_LAYERS=5 GN_BUDGET=30 GN_TOL=1e-9 IC_BUDGET=100
export EQ_SNAPS=64 EQ_GRID_POOL=4096
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "commit=b1131b1e4501258d153a2a519e8c0a42d9063842 dirty=e3b0c44298fc cell=ctol_hybb2048_learned1"
$PY - <<'PRE' || { echo "GPU PREFLIGHT FAILED"; exit 42; }
import jax, sys
backend = jax.default_backend()
print(f"jax_backend={backend}", jax.devices()[0], flush=True)
sys.exit(0 if backend == "gpu" else 42)
PRE
$PY bh_2048_learned.py ../out/learned.json
echo ALL-DONE
