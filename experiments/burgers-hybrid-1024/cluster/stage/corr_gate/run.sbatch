#!/bin/bash
#SBATCH -J hybb_corr_gate
#SBATCH -p gpu
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH -t 00:25:00
#SBATCH -o /cluster/tufts/paralab/tawal01/hybb1024/corr_gate/logs/%j.out
#SBATCH -e /cluster/tufts/paralab/tawal01/hybb1024/corr_gate/logs/%j.err
set -euo pipefail
cd "/cluster/tufts/paralab/tawal01/hybb1024/corr_gate/code"
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export BH_COMMIT=5da366e0045b27cf8754fbc2c482b7896964ee12
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "commit=5da366e0045b27cf8754fbc2c482b7896964ee12 dirty=e3b0c44298fc cell=corr_gate"
$PY - <<'PRE' || { echo "GPU PREFLIGHT FAILED"; exit 42; }
import jax, sys
backend = jax.default_backend()
print(f"jax_backend={backend}", jax.devices()[0], flush=True)
sys.exit(0 if backend == "gpu" else 42)
PRE
export N=64 CORR_N_TRAIN=128 CORR_N_VAL=32 VAL_START=512 K_MAX=32 RANKS=4,8,16,32 RFF_DIM=256 BANDWIDTHS=0.5,1,2,4 RIDGES=1e-8,1e-6,1e-4 SVD_POWER_ITERS=2 MODEL_OUT=../out/corr_gate.npz
$PY bh_correction_spectrum.py ../out/corr_gate.json
echo ALL-DONE
