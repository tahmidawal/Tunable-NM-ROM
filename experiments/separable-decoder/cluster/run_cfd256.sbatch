#!/bin/bash
#SBATCH --job-name=burgacc_cfd256
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=02:15:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/burgacc/cfd256/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/burgacc/cfd256/logs/%j.err

# ROUND 5 -- THE ACCURACY-AT-PRESERVED-SPEED CONFIRMATION.
#
# Round 5 found two levers that move the oracle and nothing else that does:
#   K            oracle/span-floor 29.8x -> 18.6x -> 16.7x -> 13.0x -> 9.7x for
#                K = 16/48/64/96/128 (h=1024x3, 576 trajectories, N=256)
#   mu-density   the same 1.7x per doubling of training trajectories
# and measured, in job cf256, that the K lever is NOT free online: K=48 at
# N=256 costs 135.75 ms against the round-4 K=16 row's 36.54 ms, a 3.7x
# slowdown for a 2.4x error improvement.  Density costs NOTHING online.
#
# So this job holds K at 16 -- byte-for-byte the round-4 online cost -- and
# spends the whole budget on mu-density: 4608 trajectories (canonical 576 plus
# 4032 appended from seed 1000; the canonical draw and the cohort definitions
# are untouched).  It then runs the round-4 speed/accuracy protocol on the
# CONTROL EQ set, the same set the committed round-4 N=256 row used, so the
# comparison is like-for-like on both halves.
set -uo pipefail
ROOT=/cluster/tufts/paralab/tawal01/burgacc/cfd256
VENV=/cluster/tufts/paralab/tawal01/ae-research/venv
PY="$VENV/bin/python"
export JAX_DEFAULT_MATMUL_PRECISION=highest

echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}', flush=True); sys.exit(0 if b=='gpu' else 42)" || exit 42
( cd "$ROOT" && sha256sum -c MANIFEST.sha256 --quiet && echo "stage MANIFEST OK" )
CK="$ROOT/in/sep_burgers_r3_N256_K16_R512.pkl"

cd "$ROOT/code"
env N=256 K=16 R=512 MAX_SNAPS=131072 T_EARLY=5 N_TEST=8 SEED0=0 LOOSE=1 \
  EXTRA_SEED=1000 EXTRA_TRAJ=4032 GEN_CHUNK=64 PROJ_CHUNK=256 IDENT_ROWS=64 \
  CKPT="$CK" OUT_PREFIX="$ROOT/out/" \
  "$PY" sep_coeff_extract.py || { echo "EXTRACT FAILED"; exit 1; }

env NPZ="$ROOT/out/sep_coeff_N256_K16_R512.npz" CKPT="$CK" \
  OUT="$ROOT/out/hfit_cfd256.json" \
  ARMS=wide STEPS=200000 BATCH=4096 LR=1e-3 TIME_CAP=1500 \
  ORACLE_ITERS=150 CODEDIAG_N=2048 CODEDIAG_ITERS=60 ENC_STEPS=12000 SEED0=0 \
  EMIT=wide EMIT_PATH="$ROOT/out/sep_hfit_dense_K16_N256.pkl" \
  "$PY" sep_hfit_run.py || { echo "HFIT FAILED"; exit 1; }

env CKPT="$ROOT/out/sep_hfit_dense_K16_N256.pkl" N=256 N_TEST=8 EQ_MS=64 \
  EQ_CAND_CAP=65536 SEED0=0 REPS=5 WARM=2 PAIR_REPS=3 TR_FACTOR=0.01 \
  STEP_TOL=1e-9 EXTRAP=1.0 IC_ENC_BUDGET=50 ENC_STEPS=12000 \
  STALLS=1e-12,1e-6,1e-4,1e-3,1e-2 BUDGETS=30,8,5,3,2,1 EXTRAPS=0,1.0,1.5 \
  BATCHES=1,2,4,8,16 NEWTON_TOLS=3e-1,1e-1,3e-2,1e-2,3e-3,1e-3,1e-4 \
  LIN_FRACS=0.05,0.5 OUT_TAG=dense_K16_N256 OUT_PREFIX="$ROOT/out/" \
  "$PY" sep_speed_r5.py || echo "SPEED FAILED"

cd "$ROOT/out" && sha256sum *.json *.pkl > RESULTS.sha256
echo ALL-DONE
