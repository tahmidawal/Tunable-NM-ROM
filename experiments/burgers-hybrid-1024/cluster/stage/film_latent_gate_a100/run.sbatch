#!/bin/bash
#SBATCH -J hybb_film_latent_gate_a100
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 01:30:00
#SBATCH -o /cluster/tufts/paralab/tawal01/hybb1024/film_latent_gate_a100/logs/%j.out
#SBATCH -e /cluster/tufts/paralab/tawal01/hybb1024/film_latent_gate_a100/logs/%j.err
set -euo pipefail
cd "/cluster/tufts/paralab/tawal01/hybb1024/film_latent_gate_a100/code"
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export BH_COMMIT=2c14e8b2b7bc251aeb652e8bc867c39304e7994e
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "commit=2c14e8b2b7bc251aeb652e8bc867c39304e7994e dirty=e3b0c44298fc cell=film_latent_gate_a100"
$PY - <<'PRE' || { echo "GPU PREFLIGHT FAILED"; exit 42; }
import jax, sys
backend = jax.default_backend()
print(f"jax_backend={backend}", jax.devices()[0], flush=True)
sys.exit(0 if backend == "gpu" else 42)
PRE
export N=64 NS=64 FOM_TAU=1e-6 LINEAR_TOL=1e-2 PRECONDITIONER=helmholtz DECODE_CHUNK=2 DECODE_RESOLUTION=64 LATENT_HISTORY_MODES=previous,extrapolation EQ_GRID_POOL=4096 FILM_CHECKPOINT=deps/burgers2d-rom-latent-stepping/blat_ad_N64_K8.pkl TEST_SEED=1 TEST_DRAW_COUNT=16 TEST_START=4 N_TEST_TRAJ=4 TIME_REPS=9 TIME_WARM=2 BURN_S=3 REFERENCE_RESIDUAL_GATE=1e-11
$PY bh_film_confirmation.py ../out/film_latent_gate.json
echo ALL-DONE
