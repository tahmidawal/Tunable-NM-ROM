#!/bin/bash
#SBATCH --job-name=burgacc_hf8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/burgacc/hf8/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/burgacc/hf8/logs/%j.err

# ROUND 5 h-REFINEMENT ARMS at N=256, in the frozen bank's coefficient space.
# Trains only h and the codes against the EXACT whitened objective (identity (*)
# in sep_hfit.py), so an arm is minutes rather than the 1.5-4 h a full decoder
# retrain costs.  The bank is frozen and identical across all arms, so the span
# floor is one constant and oracle/span-floor is comparable arm to arm.
# Sized to finish before the 2026-08-25T06:00 maintenance window.
set -euo pipefail
ROOT=/cluster/tufts/paralab/tawal01/burgacc/hf8
VENV=/cluster/tufts/paralab/tawal01/ae-research/venv
PY="$VENV/bin/python"
export JAX_DEFAULT_MATMUL_PRECISION=highest

echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}', flush=True); sys.exit(0 if b=='gpu' else 42)"
( cd "$ROOT" && sha256sum -c MANIFEST.sha256 --quiet && echo "stage MANIFEST OK" )

cd "$ROOT/code"
env NPZ="$ROOT/in/sep_coeff_N256_K16_R512.npz" \
  CKPT="$ROOT/in/sep_burgers_r3_N256_K16_R512.pkl" \
  OUT="$ROOT/out/hfit_hf8.json" \
  ARMS=k48_wd4,k48_wd3,k48_wd2,k48_short STEPS=120000 BATCH=4096 LR=1e-3 TIME_CAP=900 \
  ORACLE_ITERS=150 CODEDIAG_N=2048 CODEDIAG_ITERS=60 ENC_STEPS=12000 \
  SEED0=0  \
  "$PY" sep_hfit_run.py

cd "$ROOT/out" && sha256sum * > RESULTS.sha256
echo ALL-DONE
