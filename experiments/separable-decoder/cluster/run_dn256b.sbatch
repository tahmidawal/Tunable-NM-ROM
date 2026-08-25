#!/bin/bash
#SBATCH --job-name=burgacc_dn256b
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --begin=2026-08-25T12:10:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/burgacc/dn256b/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/burgacc/dn256b/logs/%j.err

# ROUND 5 -- the decisive N=256 run: DENSE mu-sampling + the best K, on the
# frozen r3a bank, then the round-4 speed/accuracy protocol on the result.
#
# RECIPE CORRECTED 2026-08-25 04:35 from the N=256 Pareto measurement: the arm
# to confirm is dense mu-sampling with a MODEST h (512x2) at the incumbent K,
# not a large K.  At N=256 that gives rollout error 2.751e-2 -> 8.961e-3 (3.07x)
# at 35.60 -> 45.15 ms with the matched-accuracy paired ratio UNCHANGED (0.39x
# -> 0.40x), whereas K=48 gives a worse error (9.232e-3) at 135.75 ms (0.12x)
# and a wide h at the same density gives 5.021e-3 but at 73.99 ms (0.23x).
#
# Round 5 established, at both resolutions, that the sole remaining rung is h's
# generalisation in mu, and that exactly two levers move it: the latent
# dimension K (oracle/span-floor 58.1x -> 36.8x -> 23.9x -> 18.6x -> 16.7x for
# K = 8/16/32/48/64 at N=256) and the number of training trajectories (a clean
# power law, held-out oracle falling by 1.74x per doubling from 72 to 432).
# Everything else measured is neutral or harmful: capacity past h=1024x3, latent
# Fourier features, code jitter, code polish, early-time weighting, better
# latent initialisation, and more span.
#
# --begin is set after the 2026-08-25 06:00-12:00 maintenance reservation.
set -uo pipefail
ROOT=/cluster/tufts/paralab/tawal01/burgacc/dn256b
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

NPZ="$ROOT/out/sep_coeff_N256_K16_R512.npz"
# learning curve on the reserved cohort, then the emitted decoder at full density
for F in 576 4096; do
  env NPZ="$NPZ" CKPT="$CK" OUT="$ROOT/out/hfit_dn256b_f$F.json" \
    ARMS=mid,wide,k32 STEPS=200000 BATCH=4096 LR=1e-3 TIME_CAP=1500 \
    ORACLE_ITERS=150 CODEDIAG_N=2048 CODEDIAG_ITERS=60 ENC_STEPS=12000 \
    SEED0=0 TRAJ_FIT=$F HOLD_FROM=4096 HOLD_N=512 \
    "$PY" sep_hfit_run.py || echo "hfit F=$F FAILED"
done
env NPZ="$NPZ" CKPT="$CK" OUT="$ROOT/out/hfit_dn256b_full.json" \
  ARMS=mid,wide,k32 STEPS=200000 BATCH=4096 LR=1e-3 TIME_CAP=1500 \
  ORACLE_ITERS=150 CODEDIAG_N=2048 CODEDIAG_ITERS=60 ENC_STEPS=12000 SEED0=0 \
  EMIT=mid EMIT_PATH="$ROOT/out/sep_hfit_dense_mid_N256_dense.pkl" \
  "$PY" sep_hfit_run.py || { echo "HFIT-FULL FAILED"; exit 1; }

env CKPT="$ROOT/out/sep_hfit_dense_mid_N256_dense.pkl" N=256 N_TEST=8 EQ_MS=64,256 \
  EQ_CAND_CAP=65536 SEED0=0 REPS=5 WARM=2 PAIR_REPS=3 TR_FACTOR=0.01 \
  STEP_TOL=1e-9 EXTRAP=1.0 IC_ENC_BUDGET=50 ENC_STEPS=12000 \
  STALLS=1e-12,1e-6,1e-4,1e-3,1e-2 BUDGETS=30,8,5,3,2,1 EXTRAPS=0,1.0,1.5 \
  BATCHES=1,2,4,8,16 NEWTON_TOLS=3e-1,1e-1,3e-2,1e-2,3e-3,1e-3,1e-4 \
  LIN_FRACS=0.05,0.5 OUT_TAG=dense_mid_N256_dense \
  OUT_PREFIX="$ROOT/out/" \
  "$PY" sep_speed_r5.py || echo "SPEED FAILED"

cd "$ROOT/out" && sha256sum *.json *.pkl > RESULTS.sha256
echo ALL-DONE
