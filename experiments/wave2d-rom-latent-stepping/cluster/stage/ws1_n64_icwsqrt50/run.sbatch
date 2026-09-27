#!/bin/bash
#SBATCH -J ws1_n64_icwsqrt50
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH -o /cluster/tufts/paralab/tawal01/wlat/ws1_n64_icwsqrt50/logs/%j.out
#SBATCH -e /cluster/tufts/paralab/tawal01/wlat/ws1_n64_icwsqrt50/logs/%j.err
set -euo pipefail
cd "/cluster/tufts/paralab/tawal01/wlat/ws1_n64_icwsqrt50/code"
export PYTHONPATH="/cluster/tufts/paralab/tawal01/wlat/ws1_n64_icwsqrt50/code"
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
echo "host=$(hostname)  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "git_commit=ad575708e8cb064d6997d5527a7979673a3479b4"
echo "cell=ws1_n64_icwsqrt50"
$PY - <<'PRE' || { echo "GPU PREFLIGHT FAILED"; exit 42; }
import sys, jax
d = jax.devices()[0]
print("jax_backend=" + d.platform, d, flush=True)
sys.exit(0 if d.platform == "gpu" else 42)
PRE
export N=64 N_TEST=16 GN_BUDGET=30 IC_BUDGET=100 FLOOR_BUDGET=60 EQ_SNAPS=64 EQ_POOL=4096 S1_RS=10 S1_BUDGET=60 IC_W=7.0710678118654755
$PY -u wlat_stage1.py in/wave2d_film_N64.pkl ../out
rc=$?
[[ $rc -eq 0 ]] && echo "ALL-DONE" || { echo "FAILED rc=$rc"; exit $rc; }
