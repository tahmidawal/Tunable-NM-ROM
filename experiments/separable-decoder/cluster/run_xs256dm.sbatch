#!/bin/bash
#SBATCH --job-name=exlin_xs256dm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/exlin/xs256dm/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/exlin/xs256dm/logs/%j.err

# ROUND-4/5 SPEED PROTOCOL on the EXACT-LINEAR residual (sep_burgers_exlin.py,
# EXLIN=1 EQ_ADV_ONLY=1), N=256, dense_mid full-density checkpoint -- the
# A/B partner of the incumbent run (dn256b job 2837431 / dn1024 job 2837430):
# same checkpoint, same protocol, same env, only the residual + node fit
# differ.  Trains nothing.
set -euo pipefail
ROOT=/cluster/tufts/paralab/tawal01/exlin/xs256dm
VENV=/cluster/tufts/paralab/tawal01/ae-research/venv
PY="$VENV/bin/python"
export JAX_DEFAULT_MATMUL_PRECISION=highest

echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}', flush=True); sys.exit(0 if b=='gpu' else 42)"
( cd "$ROOT" && sha256sum -c MANIFEST.sha256 --quiet && echo "stage MANIFEST OK" )
sha256sum "$ROOT/in/sep_hfit_dense_mid_N256_dense.pkl"

cd "$ROOT/code"
env CKPT="$ROOT/in/sep_hfit_dense_mid_N256_dense.pkl" N=256 N_TEST=8 EQ_MS=64,256 \
  EXLIN=1 EQ_ADV_ONLY=1 \
  EQ_CAND_CAP=65536 SEED0=0 REPS=5 WARM=2 PAIR_REPS=3 TR_FACTOR=0.01 \
  STEP_TOL=1e-9 EXTRAP=1.0 IC_ENC_BUDGET=50 ENC_STEPS=12000 \
  STALLS=1e-12,1e-6,1e-4,1e-3,1e-2 BUDGETS=30,8,5,3,2,1 EXTRAPS=0,1.0,1.5 \
  BATCHES=1,2,4,8,16 NEWTON_TOLS=3e-1,1e-1,3e-2,1e-2,3e-3,1e-3,1e-4 \
  LIN_FRACS=0.05,0.5 OUT_TAG=dense_mid_N256_dense_exlin \
  OUT_PREFIX="$ROOT/out/" \
  "$PY" sep_burgers_exlin.py

cd "$ROOT/out" && sha256sum *.json > RESULTS.sha256
echo ALL-DONE
