#!/bin/bash
#SBATCH --job-name=n256push_r4s1024
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=02:30:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/n256_push/r4s1024/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/n256_push/r4s1024/logs/%j.err

# ROUND 4 PROFILE + SPEED at N=1024.  Trains nothing: loads the committed
# round-3 checkpoint and varies only the solve path.
set -euo pipefail
ROOT=/cluster/tufts/paralab/tawal01/n256_push/r4s1024
VENV=/cluster/tufts/paralab/tawal01/ae-research/venv
PY="$VENV/bin/python"
export JAX_DEFAULT_MATMUL_PRECISION=highest

echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}', flush=True); sys.exit(0 if b=='gpu' else 42)"
( cd "$ROOT" && sha256sum -c MANIFEST.sha256 --quiet && echo "stage MANIFEST OK" )
sha256sum "$ROOT/ckpt/sep_burgers_r3_N1024_K16_R512.pkl"

cd "$ROOT/code"
env N=1024 CKPT="$ROOT/ckpt/sep_burgers_r3_N1024_K16_R512.pkl" N_TEST=16 EQ_MS=64 EQ_CAND_CAP=65536 \
  SEED0=0 REPS=7 WARM=2 PAIR_REPS=3 TR_FACTOR=0.01 STEP_TOL=1e-9 EXTRAP=1.0 \
  IC_ENC_BUDGET=50 ENC_STEPS=12000 \
  STALLS=1e-12,1e-6,1e-4,1e-3,1e-2 BUDGETS=30,8,5,3,2,1 EXTRAPS=0,1.0,1.5 \
  BATCHES=1,2,4,8,16 \
  NEWTON_TOLS=3e-1,1e-1,3e-2,1e-2,3e-3,1e-3,1e-4 LIN_FRACS=0.05,0.5 \
  MAX_NEWTON=20 FEAT_CHUNK=131072 \
  OUT_PREFIX="$ROOT/out/" \
  "$PY" sep_speed_r4.py

cd "$ROOT/out" && sha256sum *.json > RESULTS.sha256
echo ALL-DONE
