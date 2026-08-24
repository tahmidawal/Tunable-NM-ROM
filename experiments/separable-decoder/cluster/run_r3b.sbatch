#!/bin/bash
#SBATCH --job-name=n256push_r3b
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/n256_push/r3b/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/n256_push/r3b/logs/%j.err

# ROUND 3, Burgers.  TRACK A control -- identical to r3a except SNAP_NORM=1 (per-snapshot loss normalisation, i.e. minimise the metric that is actually reported).  Isolates that lever.
set -euo pipefail
ROOT=/cluster/tufts/paralab/tawal01/n256_push/r3b
VENV=/cluster/tufts/paralab/tawal01/ae-research/venv
PY="$VENV/bin/python"
export JAX_DEFAULT_MATMUL_PRECISION=highest

echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}', flush=True); sys.exit(0 if b=='gpu' else 42)"
( cd "$ROOT" && sha256sum -c MANIFEST.sha256 --quiet && echo "stage MANIFEST OK" )

cd "$ROOT/code"
env ROUND=3 K=16 LR=1e-3 P_SUB=4096 WD=1e-5 EMA_DECAY=0.999 LAM_ORTH=1e-4 \
  MAX_SNAPS=16384 T_EARLY=5 FULLROWS=64 \
  N_FF=128 FF_SCALE=4.0 H_HIDDEN=256 \
  EQ_MS=64,256 EQ_CAND_CAP=65536 STEP_TOLS=1e-9,1e-6 \
  N_TEST=8 REPS=5 WARM=2 PAIR_REPS=3 TR_FACTOR=0.01 SEED0=0 \
  IC_TOP=12 IC_ENC_BUDGET=50 ENC_STEPS=12000 EXTRAP=1.0 \
  ORACLE_BUDGET=150 SSTEP_TS=1,2,3,5,10,25,50 SSTEP_BUDGET=120 \
  NEWTON_TOLS=3e-1,1e-1,3e-2,1e-2,3e-3,1e-3,1e-4 LIN_FRACS=0.05,0.5 \
  MAX_NEWTON=20 BATCHED=1 \
  N=256 R=512 G_HIDDEN=1024 SNAP_NORM=1 STEPS=300000 TIME_CAP=12600 FULL_LAST=10000 POOL=0 MESH_ARM=1 \
  OUT_PREFIX="$ROOT/out/" \
  "$PY" sep_burgers_r3.py

cd "$ROOT/out" && sha256sum *.json *.pkl > RESULTS.sha256
echo ALL-DONE
