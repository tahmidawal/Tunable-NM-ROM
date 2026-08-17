#!/bin/bash
#SBATCH -J s1_n64
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 8:00:00
#SBATCH -o /cluster/tufts/paralab/tawal01/blat/s1_n64/logs/%j.out
#SBATCH -e /cluster/tufts/paralab/tawal01/blat/s1_n64/logs/%j.err
set -euo pipefail
cd "/cluster/tufts/paralab/tawal01/blat/s1_n64/code"
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
export N=64 N_TEST=16 S1_BUDGET=100
mkdir -p ../out/icw_sqrt50 ../out/icw1 && $PY -u blat_stage1.py in/burgers2d_film_N64.pkl ../out/icw_sqrt50 && IC_W=1 $PY -u blat_stage1.py in/burgers2d_film_N64.pkl ../out/icw1
echo "ALL-DONE"
