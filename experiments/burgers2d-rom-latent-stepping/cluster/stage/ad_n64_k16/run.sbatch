#!/bin/bash
#SBATCH -J ad_n64_k16
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 16:00:00
#SBATCH -o /cluster/tufts/paralab/tawal01/blat/ad_n64_k16/logs/%j.out
#SBATCH -e /cluster/tufts/paralab/tawal01/blat/ad_n64_k16/logs/%j.err
set -euo pipefail
cd "/cluster/tufts/paralab/tawal01/blat/ad_n64_k16/code"
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
echo "host=$(hostname)  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "git_commit=3efd6dedde23c2747ff88816d675d5cb1976ecd6"
$PY - <<'PRE' || { echo "GPU PREFLIGHT FAILED"; exit 42; }
import sys, jax
d = jax.devices()[0]
print("jax_backend=" + d.platform, d, flush=True)
sys.exit(0 if d.platform == "gpu" else 42)
PRE
export N=64 K_LAT=16 AD_STEPS=60000 AD_BATCH=128 P_SUB=2048 N_TEST=16 FLOOR_BUDGET=40 GN_BUDGET=30 POD_KS=8,16,32,64
$PY -u blat_train_ad.py ../out && $PY -u blat_rom.py ../out/blat_ad_N64_K16.pkl ../out
echo "ALL-DONE"
