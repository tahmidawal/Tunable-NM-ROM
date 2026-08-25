#!/bin/bash
#SBATCH --job-name=eqladder_lad256k16
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/eqladder/lad256k16/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/eqladder/lad256k16/logs/%j.err

# EQ FIDELITY LADDER at N=256 on the committed checkpoint sep_burgers_r3_N256_K16_R512.pkl.
# Trains nothing.  Sampled (EQ) vs full-grid residual / gradient / Hessian /
# LM step at every solver iterate, at oracle states, at training snapshots.
set -euo pipefail
ROOT=/cluster/tufts/paralab/tawal01/eqladder/lad256k16
VENV=/cluster/tufts/paralab/tawal01/ae-research/venv
PY="$VENV/bin/python"
export JAX_DEFAULT_MATMUL_PRECISION=highest

echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}', flush=True); sys.exit(0 if b=='gpu' else 42)"
( cd "$ROOT" && sha256sum -c MANIFEST.sha256 --quiet && echo "stage MANIFEST OK" )
sha256sum "$ROOT/in/sep_burgers_r3_N256_K16_R512.pkl"

cd "$ROOT/code"
env N=256 CKPT="$ROOT/in/sep_burgers_r3_N256_K16_R512.pkl" N_TEST=4 EQ_MS=64,256 EQ_M_FACTOR=4 EQ_CAND_CAP=65536 \
  SEED0=0 TR_FACTOR=0.01 STEP_TOL=1e-9 STALL=1e-3 EXTRAP=1.0 \
  ORACLE_TS=0,1,2,3,5,10,25,50 N_TRAIN_SNAP=16 \
  OUT_TAG=N256_K16_r3a OUT_PREFIX="$ROOT/out/" \
  "$PY" sep_eq_ladder.py

cd "$ROOT/out" && sha256sum *.json > RESULTS.sha256
echo ALL-DONE
