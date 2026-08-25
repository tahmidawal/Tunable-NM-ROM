#!/bin/bash
#SBATCH --job-name=burgacc_dr256
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=01:30:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/burgacc/dr256/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/burgacc/dr256/logs/%j.err

# ROUND 5 WAVE 7b -- COST-MATCHED DENSITY.
#
# cfd256 produced the most accurate Burgers decoder of the campaign (rollout
# 5.02e-3 against the round-4 incumbent's 2.75e-2 at N=256) but at 73.99 ms
# against 35.60 ms.  It changed TWO things at once: the data (576 -> 4608
# trajectories, free at solve time) and h's width (256x2 -> 1024x3, NOT free --
# the h-track Jacobian is a large share of an LM iteration).  These arms keep
# h at or near the incumbent's width and spend the whole gain on density, then
# measure the result on the round-4 protocol.
set -uo pipefail
ROOT=/cluster/tufts/paralab/tawal01/burgacc/dr256
SRC=/cluster/tufts/paralab/tawal01/burgacc/dn256/out
VENV=/cluster/tufts/paralab/tawal01/ae-research/venv
PY="$VENV/bin/python"
export JAX_DEFAULT_MATMUL_PRECISION=highest

echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}', flush=True); sys.exit(0 if b=='gpu' else 42)" || exit 42
( cd "$ROOT" && sha256sum -c MANIFEST.sha256 --quiet && echo "stage MANIFEST OK" )
cp "$SRC/sep_coeff_N256_K16_R512.npz" "$SRC/sep_coeff_N256_K16_R512.json" "$ROOT/in/"
echo "copied dense coefficients into this job's own dir"

cd "$ROOT/code"
env NPZ="$ROOT/in/sep_coeff_N256_K16_R512.npz" \
  CKPT="$ROOT/in/sep_burgers_r3_N256_K16_R512.pkl" \
  OUT="$ROOT/out/hfit_dr256.json" \
  ARMS=mid,ctl_cold STEPS=200000 BATCH=4096 LR=1e-3 TIME_CAP=700 \
  ORACLE_ITERS=150 CODEDIAG_N=2048 CODEDIAG_ITERS=60 ENC_STEPS=12000 SEED0=0 \
  EMIT=mid EMIT_PATH="$ROOT/out/sep_hfit_dense_mid_N256.pkl" \
  "$PY" sep_hfit_run.py || { echo "HFIT FAILED"; exit 1; }

env CKPT="$ROOT/out/sep_hfit_dense_mid_N256.pkl" N=256 N_TEST=8 EQ_MS=64 \
  EQ_CAND_CAP=65536 SEED0=0 REPS=5 WARM=2 PAIR_REPS=3 TR_FACTOR=0.01 \
  STEP_TOL=1e-9 EXTRAP=1.0 IC_ENC_BUDGET=50 ENC_STEPS=12000 \
  STALLS=1e-12,1e-6,1e-4,1e-3,1e-2 BUDGETS=30,8,5,3,2,1 EXTRAPS=0,1.0,1.5 \
  BATCHES=1,2,4,8,16 NEWTON_TOLS=3e-1,1e-1,3e-2,1e-2,3e-3,1e-3,1e-4 \
  LIN_FRACS=0.05,0.5 OUT_TAG=dense_mid_N256 OUT_PREFIX="$ROOT/out/" \
  "$PY" sep_speed_r5.py || echo "SPEED FAILED"

cd "$ROOT/out" && sha256sum *.json *.pkl > RESULTS.sha256
echo ALL-DONE
