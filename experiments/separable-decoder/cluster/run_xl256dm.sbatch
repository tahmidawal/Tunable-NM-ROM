#!/bin/bash
#SBATCH --job-name=exlin_xl256dm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/exlin/xl256dm/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/exlin/xl256dm/logs/%j.err

# EQ FIDELITY LADDER with EXACT LINEAR TERMS (EXLIN=1) and an advection-only
# NNLS node fit (EQ_ADV_ONLY=1), N=256, checkpoint sep_hfit_dense_mid_N256.pkl.
# Trains nothing.  gate L (exlin linear vs full-grid linear) and gate A
# (advection unchanged) replace gate 0 for the exlin ops; gate 0 still runs
# on the incumbent-form ops; gate F unchanged.
set -euo pipefail
ROOT=/cluster/tufts/paralab/tawal01/exlin/xl256dm
VENV=/cluster/tufts/paralab/tawal01/ae-research/venv
PY="$VENV/bin/python"
export JAX_DEFAULT_MATMUL_PRECISION=highest

echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}', flush=True); sys.exit(0 if b=='gpu' else 42)"
( cd "$ROOT" && sha256sum -c MANIFEST.sha256 --quiet && echo "stage MANIFEST OK" )
sha256sum "$ROOT/in/sep_hfit_dense_mid_N256.pkl"

cd "$ROOT/code"
env N=256 CKPT="$ROOT/in/sep_hfit_dense_mid_N256.pkl" N_TEST=4 EQ_MS=64,256 EQ_M_FACTOR=4 EQ_CAND_CAP=65536 \
  EXLIN=1 EQ_ADV_ONLY=1 \
  SEED0=0 TR_FACTOR=0.01 STEP_TOL=1e-9 STALL=1e-3 EXTRAP=1.0 \
  ORACLE_TS=0,1,2,3,5,10,25,50 N_TRAIN_SNAP=16 \
  OUT_TAG=N256_K16_dense_mid_exlin OUT_PREFIX="$ROOT/out/" \
  "$PY" sep_eq_ladder.py

cd "$ROOT/out" && sha256sum *.json > RESULTS.sha256
echo ALL-DONE
