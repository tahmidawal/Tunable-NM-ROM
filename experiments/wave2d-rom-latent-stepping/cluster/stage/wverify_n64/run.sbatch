#!/bin/bash
#SBATCH -J wverify_n64
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 2:00:00
#SBATCH -o /cluster/tufts/paralab/tawal01/wlat/wverify_n64/logs/%j.out
#SBATCH -e /cluster/tufts/paralab/tawal01/wlat/wverify_n64/logs/%j.err
set -euo pipefail
cd "/cluster/tufts/paralab/tawal01/wlat/wverify_n64/code"
export PYTHONPATH="/cluster/tufts/paralab/tawal01/wlat/wverify_n64/code"
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
echo "host=$(hostname)  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "git_commit=0934a3224433388f11e4280e29cd8bd5551a27b9"
echo "cell=wverify_n64"
$PY - <<'PRE' || { echo "GPU PREFLIGHT FAILED"; exit 42; }
import sys, jax
d = jax.devices()[0]
print("jax_backend=" + d.platform, d, flush=True)
sys.exit(0 if d.platform == "gpu" else 42)
PRE
export N=64 ROM_SUBSTEPS=20 N_TEST=16 RS_LIST=1,2,4,8,16,20,40,80 N_VERIFY=6
$PY wlat_verify.py ../out
rc=$?
[[ $rc -eq 0 ]] && echo "ALL-DONE" || { echo "FAILED rc=$rc"; exit $rc; }
