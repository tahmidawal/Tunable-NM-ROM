#!/bin/bash
#SBATCH -J ctol_b10_s0_spline_r1
#SBATCH -p gpu
#SBATCH --gres=gpu:h200:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -o /cluster/tufts/paralab/tawal01/burgers_nmrom_1e3_10x/s0_spline_r1/logs/%j.out
#SBATCH -e /cluster/tufts/paralab/tawal01/burgers_nmrom_1e3_10x/s0_spline_r1/logs/%j.err
set -euo pipefail
cd "/cluster/tufts/paralab/tawal01/burgers_nmrom_1e3_10x/s0_spline_r1/code"
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export B10_COMMIT=3d9adb769c1534f507ae15d86ad2f38506e77954
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "commit=3d9adb769c1534f507ae15d86ad2f38506e77954 dirty=e3b0c44298fc cell=s0_spline_r1"
$PY - <<'PRE' || { echo "GPU PREFLIGHT FAILED"; exit 42; }
import jax, sys
backend = jax.default_backend()
print(f"jax_backend={backend}", jax.devices()[0], flush=True)
sys.exit(0 if backend == "gpu" else 42)
PRE
export SMOKE=0 TIME_REPS=10 TIME_WARM=1 BURN_SECONDS=3 ORACLE_WORKERS=8 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
$PY b10_s0_spline.py ../out/s0.json ../out/s0.npz
echo ALL-DONE
