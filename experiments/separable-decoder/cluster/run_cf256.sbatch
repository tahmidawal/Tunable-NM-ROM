#!/bin/bash
#SBATCH --job-name=burgacc_cf256
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=01:40:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/burgacc/cf256/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/burgacc/cf256/logs/%j.err

# ROUND 5 CONFIRMATION at N=256: does a better ORACLE become a better ROLLOUT,
# and at what speed?  Refits h on the FROZEN r3a bank at K=48 -- the best arm
# measured (oracle 3.96e-3 against 7.81e-3 for the incumbent, 18.6x the span
# floor against 36.8x) -- emits it as a drop-in decoder, and then runs the
# round-4 speed/accuracy protocol on it UNCHANGED, so the row is directly
# comparable with the committed N=256 round-4 row (ROM 36.54 ms @ 2.268e-2 vs
# matched classical 13.99 ms @ 8.377e-4, 0.40x).
#
# EQ set: the CONTROL set (M=64, m=256) only, which is exactly what the round-4
# N=256 row used.  A finer-EQ accuracy grade is a separate job; a coarse-EQ
# speed number is never to be paired with a fine-EQ accuracy number.
set -uo pipefail
ROOT=/cluster/tufts/paralab/tawal01/burgacc/cf256
VENV=/cluster/tufts/paralab/tawal01/ae-research/venv
PY="$VENV/bin/python"
export JAX_DEFAULT_MATMUL_PRECISION=highest

echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}', flush=True); sys.exit(0 if b=='gpu' else 42)" || exit 42
( cd "$ROOT" && sha256sum -c MANIFEST.sha256 --quiet && echo "stage MANIFEST OK" )

cd "$ROOT/code"
env NPZ="$ROOT/in/sep_coeff_N256_K16_R512.npz" \
  CKPT="$ROOT/in/sep_burgers_r3_N256_K16_R512.pkl" \
  OUT="$ROOT/out/hfit_cf256.json" \
  ARMS=k48 STEPS=120000 BATCH=4096 LR=1e-3 TIME_CAP=1200 \
  ORACLE_ITERS=150 CODEDIAG_N=2048 CODEDIAG_ITERS=60 ENC_STEPS=12000 SEED0=0 \
  EMIT=k48 EMIT_PATH="$ROOT/out/sep_hfit_k48_N256.pkl" \
  "$PY" sep_hfit_run.py || { echo "HFIT FAILED"; exit 1; }

env CKPT="$ROOT/out/sep_hfit_k48_N256.pkl" N=256 N_TEST=8 EQ_MS=64 \
  EQ_CAND_CAP=65536 SEED0=0 REPS=5 WARM=2 PAIR_REPS=3 TR_FACTOR=0.01 \
  STEP_TOL=1e-9 EXTRAP=1.0 IC_ENC_BUDGET=50 ENC_STEPS=12000 \
  STALLS=1e-12,1e-6,1e-4,1e-3,1e-2 BUDGETS=30,8,5,3,2,1 EXTRAPS=0,1.0,1.5 \
  BATCHES=1,2,4,8,16 NEWTON_TOLS=3e-1,1e-1,3e-2,1e-2,3e-3,1e-3,1e-4 \
  LIN_FRACS=0.05,0.5 OUT_TAG=k48_N256 OUT_PREFIX="$ROOT/out/" \
  "$PY" sep_speed_r5.py || echo "SPEED FAILED"

cd "$ROOT/out" && sha256sum * > RESULTS.sha256
echo ALL-DONE
