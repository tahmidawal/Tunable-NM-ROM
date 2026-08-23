#!/bin/bash
#SBATCH --job-name=n256push_r2bg
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/n256_push/r2_burgers/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/n256_push/r2_burgers/logs/%j.err

# N=256 push ROUND 2, Burgers (PRIMARY 1e-3 target): R=512 per the POD-floor
# verdict (fresh test-state floor 7.0e-5 mean / 2.3e-3 max at R=512), 16384
# training states with the early-time window fully included, multi-scale FF,
# v2 trainer; STEP_TOL ladder; EQ certification M in {64,128,256}; per-step
# error tables; per-set single-step weak-opt vs per-state oracle.
set -euo pipefail
ROOT=/cluster/tufts/paralab/tawal01/n256_push/r2_burgers
VENV=/cluster/tufts/paralab/tawal01/ae-research/venv
PY="$VENV/bin/python"
export JAX_DEFAULT_MATMUL_PRECISION=highest

echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}', flush=True); sys.exit(0 if b=='gpu' else 42)"

cd "$ROOT/code"
env N=256 K=16 R=512 STEPS=300000 LR=1e-3 P_SUB=4096 WD=1e-5 \
  EMA_DECAY=0.999 FULL_LAST=10000 TIME_CAP=10800 LAM_ORTH=1e-4 \
  MAX_SNAPS=16384 T_EARLY=5 \
  N_FF=128 FF_SCALES=1,4,16 G_HIDDEN=256 H_HIDDEN=256 \
  EQ_MS=64,128,256 STEP_TOLS=1e-9,1e-6,1e-5 \
  N_TEST=8 REPS=5 WARM=2 TR_FACTOR=0.01 SEED0=0 \
  IC_TOP=12 IC_ENC_BUDGET=50 ENC_STEPS=12000 EXTRAP=1.0 \
  ORACLE_BUDGET=150 SSTEP_TS=1,2,3,5,10,25,50 SSTEP_BUDGET=120 \
  NEWTON_TOLS=1e-2,1e-3,1e-5,1e-8 LIN_FACTOR=1e-2 MAX_NEWTON=20 \
  OUT_PREFIX="$ROOT/out/" \
  "$PY" sep_burgers_r2.py

cd "$ROOT/out" && sha256sum *.json *.pkl > RESULTS.sha256
echo ALL-DONE
