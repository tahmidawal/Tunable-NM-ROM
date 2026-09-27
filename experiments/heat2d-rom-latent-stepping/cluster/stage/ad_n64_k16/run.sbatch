#!/bin/bash
#SBATCH -J ad_n64_k16
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 16:00:00
#SBATCH -o /cluster/tufts/paralab/tawal01/hlat/ad_n64_k16/logs/%j.out
#SBATCH -e /cluster/tufts/paralab/tawal01/hlat/ad_n64_k16/logs/%j.err
set -euo pipefail
cd "/cluster/tufts/paralab/tawal01/hlat/ad_n64_k16/code"
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
echo "host=$(hostname)  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "git_commit=90171caef6800d6decc71aab627b126c6b8db80d dirty=0"
$PY - <<'PRE' || { echo "GPU PREFLIGHT FAILED"; exit 42; }
import sys, jax
d = jax.devices()[0]
print("jax_backend=" + d.platform, d, flush=True)
sys.exit(0 if d.platform == "gpu" else 42)
PRE
export N=64 K_LAT=16 AD_STEPS=60000 AD_BATCH=128 P_SUB=2048 N_TEST=16 FLOOR_BUDGET=40 GN_BUDGET=30 POD_KMAX=64 POD_KS=6,8,16,32,64
$PY -u hlat_train_ad.py ../out && $PY -u hlat_rom.py ../out/hlat_ad_N64_K16.pkl ../out
rc=$?
[[ $rc -eq 0 ]] && echo "ALL-DONE" || { echo "FAILED rc=$rc"; exit $rc; }
