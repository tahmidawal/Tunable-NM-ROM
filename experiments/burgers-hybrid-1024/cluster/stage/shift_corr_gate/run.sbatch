#!/bin/bash
#SBATCH -J hybb_shift_corr_gate
#SBATCH -p gpu
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 00:12:00
#SBATCH -o /cluster/tufts/paralab/tawal01/hybb1024/shift_corr_gate/logs/%j.out
#SBATCH -e /cluster/tufts/paralab/tawal01/hybb1024/shift_corr_gate/logs/%j.err
set -euo pipefail
cd "/cluster/tufts/paralab/tawal01/hybb1024/shift_corr_gate/code"
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export BH_COMMIT=9cffc38a3dace3a5aae808fa1dbb54082415c8b5
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "commit=9cffc38a3dace3a5aae808fa1dbb54082415c8b5 dirty=e3b0c44298fc cell=shift_corr_gate"
$PY - <<'PRE' || { echo "GPU PREFLIGHT FAILED"; exit 42; }
import jax, sys
backend = jax.default_backend()
print(f"jax_backend={backend}", jax.devices()[0], flush=True)
sys.exit(0 if backend == "gpu" else 42)
PRE
export N=64 CORR_N_TRAIN=128 CORR_N_VAL=32 VAL_START=512 RANKS=4,8,16,32 ALIGNMENTS=translate,translate_scale RIDGES=1e-8,1e-6,1e-4,1e-2 MODEL_OUT=../out/shift_corr_gate.npz
$PY bh_shifted_correction_gate.py ../out/shift_corr_gate.json
echo ALL-DONE
