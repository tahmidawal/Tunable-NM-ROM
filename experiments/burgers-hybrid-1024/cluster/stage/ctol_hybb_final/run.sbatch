#!/bin/bash
#SBATCH -J ctol_hybb_final
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 1-00:00:00
#SBATCH -o /cluster/tufts/paralab/tawal01/hybb1024/ctol_hybb_final/logs/%j.out
#SBATCH -e /cluster/tufts/paralab/tawal01/hybb1024/ctol_hybb_final/logs/%j.err
set -euo pipefail
cd "/cluster/tufts/paralab/tawal01/hybb1024/ctol_hybb_final/code"
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export BH_COMMIT=5cfd399a475d2c2dc8be31e0ad1a45a6e0cf9502
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "commit=5cfd399a475d2c2dc8be31e0ad1a45a6e0cf9502 dirty=e3b0c44298fc cell=ctol_hybb_final"
$PY - <<'PRE' || { echo "GPU PREFLIGHT FAILED"; exit 42; }
import jax, sys
backend = jax.default_backend()
print(f"jax_backend={backend}", jax.devices()[0], flush=True)
sys.exit(0 if backend == "gpu" else 42)
PRE
export N=64 NS=32,64,128,256,512,1024 FOM_TAUS=1e-6,1e-8,1e-10 LINEAR_TOLS=1e-2,1e-4,1e-5 PRECONDITIONER=helmholtz DECODE_CHUNK=2 DECODE_RESOLUTION=64 LATENT_HISTORY_MODES=extrapolation MAX_STEP_JACOBIANS=2 EQ_GRID_POOL=4096 FILM_CHECKPOINT=deps/burgers2d-rom-latent-stepping/blat_ad_N64_K8.pkl TEST_SEED=1 TEST_DRAW_COUNT=16 TEST_START=0 N_TEST_TRAJ=4 TIME_REPS=21 TIME_WARM=2 BURN_S=3 REFERENCE_RESIDUAL_GATE=1e-11
$PY bh_film_confirmation.py ../out/final_confirmation.json
echo ALL-DONE
