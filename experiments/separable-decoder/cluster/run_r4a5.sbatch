#!/bin/bash
#SBATCH --job-name=n256push_r4a5
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=05:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/n256_push/r4a5/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/n256_push/r4a5/logs/%j.err

# ROUND 4 ACCURACY.  EVERYTHING at K=16: widest+deepest h, early-time weighting and code polish together.
# Sized to finish before the 2026-08-25T06:00 cluster maintenance window;
# TIME_CAP truncates training cleanly and the driver saves after every stage.
set -euo pipefail
ROOT=/cluster/tufts/paralab/tawal01/n256_push/r4a5
VENV=/cluster/tufts/paralab/tawal01/ae-research/venv
PY="$VENV/bin/python"
export JAX_DEFAULT_MATMUL_PRECISION=highest

echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}', flush=True); sys.exit(0 if b=='gpu' else 42)"
( cd "$ROOT" && sha256sum -c MANIFEST.sha256 --quiet && echo "stage MANIFEST OK" )

cd "$ROOT/code"
env ROUND=4 R=512 G_HIDDEN=1024 N_FF=128 FF_SCALE=4.0 \
  LR=1e-3 P_SUB=4096 WD=1e-5 EMA_DECAY=0.999 LAM_ORTH=1e-4 FULL_LAST=0 \
  MAX_SNAPS=16384 T_EARLY=5 FULLROWS=64 \
  EQ_MS=64,256 EQ_CAND_CAP=65536 EQ_M_FACTOR=4 EQ_TAIL=1 EQ_TAIL_ROUNDS=2 \
  STEP_TOLS=1e-9,1e-6 N_TEST=8 REPS=5 WARM=2 PAIR_REPS=3 TR_FACTOR=0.01 \
  SEED0=0 IC_TOP=12 IC_ENC_BUDGET=50 ENC_STEPS=12000 EXTRAP=1.0 \
  ORACLE_BUDGET=150 SSTEP_TS=1,2,3,5,10,25,50 SSTEP_BUDGET=120 \
  NEWTON_TOLS=3e-1,1e-1,3e-2,1e-2,3e-3,1e-3,1e-4 LIN_FRACS=0.05,0.5 \
  MAX_NEWTON=20 BATCHED=1 \
  N=256 K=16 H_HIDDEN=1024 H_LAYERS=4 STEPS=300000 TIME_CAP=9000 POOL=0 MESH_ARM=1 W_EARLY=4 Z_POLISH=20000 \
  OUT_PREFIX="$ROOT/out/" \
  "$PY" sep_burgers_r4.py

cd "$ROOT/out" && sha256sum *.json *.pkl > RESULTS.sha256
echo ALL-DONE
