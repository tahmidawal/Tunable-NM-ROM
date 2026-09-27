#!/bin/bash
#SBATCH -J pt_k
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -o /cluster/tufts/paralab/tawal01/pobj2/pt_k/logs/%j.out
#SBATCH -e /cluster/tufts/paralab/tawal01/pobj2/pt_k/logs/%j.err
set -euo pipefail
cd "/cluster/tufts/paralab/tawal01/pobj2/pt_k"
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
echo "host=$(hostname)  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)  commit=5547a96  dirty=7ffde622ea1f  cell=pt_k"
$PY - <<'PRE' || { echo "GPU PREFLIGHT FAILED"; exit 42; }
import sys, jax
d = jax.devices()[0]
print("jax_backend=" + d.platform, d, flush=True)
sys.exit(0 if d.platform == "gpu" else 42)
PRE
cd code
MODE=k PKLS=../ckpt/autodec_K2_N64_hbc_stages.pkl,../ckpt/autodec_K4_N64_hbc_stages.pkl,../ckpt/autodec_K6_N64_hbc_stages.pkl,../ckpt/autodec_K8_N64_hbc_stages.pkl,../ckpt/autodec_K12_N64_hbc_stages.pkl,../ckpt/autodec_K16_N64_hbc_stages.pkl,../ckpt/autodec_K24_N64_hbc_stages.pkl,../ckpt/autodec_K32_N64_hbc_stages.pkl PKL=../ckpt/autodec_K8_N64_hbc_stages.pkl M=64 MQ=256 GN_ITERS=60 TIME_REPS=7 POD_KS=2,4,6,8,12,16,24,32,48,64 $PY -u followup/fu_timing.py ../out/timing_k.json
rc=$?
[[ $rc -eq 0 ]] && echo "ALL-DONE" || { echo "FAILED rc=$rc"; exit $rc; }
