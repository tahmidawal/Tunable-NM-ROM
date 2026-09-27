#!/bin/bash
#SBATCH -J pt_n
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 06:00:00
#SBATCH -o /cluster/tufts/paralab/tawal01/pobj2/pt_n/logs/%j.out
#SBATCH -e /cluster/tufts/paralab/tawal01/pobj2/pt_n/logs/%j.err
set -euo pipefail
cd "/cluster/tufts/paralab/tawal01/pobj2/pt_n"
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
echo "host=$(hostname)  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)  commit=5547a96  dirty=e3b0c44298fc  cell=pt_n"
$PY - <<'PRE' || { echo "GPU PREFLIGHT FAILED"; exit 42; }
import sys, jax
d = jax.devices()[0]
print("jax_backend=" + d.platform, d, flush=True)
sys.exit(0 if d.platform == "gpu" else 42)
PRE
cd code
MODE=n PKL=../ckpt/autodec_K8_N64_hbc_stages.pkl NS=32,64,128,256,512 M=64 MQ=256 GN_ITERS=60 TIME_REPS=7 $PY -u followup/fu_timing.py ../out/timing_n.json
rc=$?
[[ $rc -eq 0 ]] && echo "ALL-DONE" || { echo "FAILED rc=$rc"; exit $rc; }
