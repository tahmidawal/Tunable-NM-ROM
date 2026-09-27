#!/bin/bash
#SBATCH -J hybb_film_smoke
#SBATCH -p gpu
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:15:00
#SBATCH -o /cluster/tufts/paralab/tawal01/hybb1024/film_smoke/logs/%j.out
#SBATCH -e /cluster/tufts/paralab/tawal01/hybb1024/film_smoke/logs/%j.err
set -euo pipefail
cd "/cluster/tufts/paralab/tawal01/hybb1024/film_smoke/code"
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export BH_COMMIT=3cbde3056436f497a6c6b94960f3d126aa46fa08
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "commit=3cbde3056436f497a6c6b94960f3d126aa46fa08 dirty=e3b0c44298fc cell=film_smoke"
$PY - <<'PRE' || { echo "GPU PREFLIGHT FAILED"; exit 42; }
import jax, sys
backend = jax.default_backend()
print(f"jax_backend={backend}", jax.devices()[0], flush=True)
sys.exit(0 if backend == "gpu" else 42)
PRE
export N=64 NS=32 FOM_TAU=1e-6 LINEAR_TOL=1e-2 PRECONDITIONER=helmholtz DECODE_CHUNK=2 FILM_CHECKPOINT=deps/burgers2d-rom-latent-stepping/blat_ad_N64_K8.pkl TEST_SEED=1 TEST_DRAW_COUNT=16 N_TEST_TRAJ=1 TIME_REPS=1 TIME_WARM=0 BURN_S=.1 REFERENCE_RESIDUAL_GATE=1e-11
$PY bh_film_confirmation.py ../out/film_smoke.json
echo ALL-DONE
