#!/bin/bash
#SBATCH --job-name=n256push_r4x128
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=05:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/n256_push/r4x128/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/n256_push/r4x128/logs/%j.err

# CROSS-RESOLUTION point at N=128: the SAME architecture and recipe as r3a
# (N=256) and r3d (N=1024) -- K=16, R=512, G_HIDDEN=1024, single-scale
# Fourier features, point-pool trainer, 300k steps -- so the optimized-solver
# curve has four comparable points.  The old scaling-study checkpoints at
# N=128/512 are the RANK-CAPPED R=64/R=96 decoders and are NOT usable here.
# Stage 1 trains (sep_burgers_r3.py); stage 2 runs the identical round-4 speed
# sweep on the checkpoint stage 1 just produced.
set -euo pipefail
ROOT=/cluster/tufts/paralab/tawal01/n256_push/r4x128
VENV=/cluster/tufts/paralab/tawal01/ae-research/venv
PY="$VENV/bin/python"
export JAX_DEFAULT_MATMUL_PRECISION=highest

echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}', flush=True); sys.exit(0 if b=='gpu' else 42)"
( cd "$ROOT" && sha256sum -c MANIFEST.sha256 --quiet && echo "stage MANIFEST OK" )

cd "$ROOT/code"
echo "===== STAGE 1: train (r3 recipe) ====="
env ROUND=3 N=128 K=16 R=512 G_HIDDEN=1024 H_HIDDEN=256 N_FF=128 FF_SCALE=4.0 \
  SNAP_NORM=0 STEPS=300000 LR=1e-3 P_SUB=4096 WD=1e-5 EMA_DECAY=0.999 \
  LAM_ORTH=1e-4 FULL_LAST=0 TIME_CAP=7200 MAX_SNAPS=16384 T_EARLY=5 \
  FULLROWS=64 N_TRAJ=576 \
  EQ_MS=64,256 EQ_CAND_CAP=65536 STEP_TOLS=1e-9,1e-6 \
  N_TEST=8 REPS=5 WARM=2 PAIR_REPS=3 TR_FACTOR=0.01 SEED0=0 \
  IC_TOP=12 IC_ENC_BUDGET=50 ENC_STEPS=12000 EXTRAP=1.0 \
  ORACLE_BUDGET=150 SSTEP_TS=1,2,3,5,10,25,50 SSTEP_BUDGET=120 \
  NEWTON_TOLS=3e-1,1e-1,3e-2,1e-2,3e-3,1e-3,1e-4 LIN_FRACS=0.05,0.5 \
  MAX_NEWTON=20 BATCHED=1 POOL=0 MESH_ARM=1 GEN_CHUNK=64 \
  OUT_PREFIX="$ROOT/out/" \
  "$PY" sep_burgers_r3.py

echo "===== STAGE 2: round-4 speed sweep on that checkpoint ====="
env N=128 CKPT="$ROOT/out/sep_burgers_r3_N128_K16_R512.pkl" \
  N_TEST=16 EQ_MS=64 EQ_CAND_CAP=65536 SEED0=0 REPS=7 WARM=2 PAIR_REPS=3 \
  TR_FACTOR=0.01 STEP_TOL=1e-9 EXTRAP=1.0 IC_ENC_BUDGET=50 ENC_STEPS=12000 \
  STALLS=1e-12,1e-6,1e-4,1e-3,1e-2 BUDGETS=30,8,5,3,2,1 EXTRAPS=0,1.0,1.5 \
  BATCHES=1,2,4,8,16 \
  NEWTON_TOLS=3e-1,1e-1,3e-2,1e-2,3e-3,1e-3,1e-4 LIN_FRACS=0.05,0.5 \
  MAX_NEWTON=20  \
  OUT_PREFIX="$ROOT/out/" \
  "$PY" sep_speed_r4.py

cd "$ROOT/out" && sha256sum *.json *.pkl > RESULTS.sha256
echo ALL-DONE
