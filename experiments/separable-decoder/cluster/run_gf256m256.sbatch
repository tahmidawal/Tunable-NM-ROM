#!/bin/bash
#SBATCH --job-name=exlin_gf256m256
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/exlin/gf256m256/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/exlin/gf256m256/logs/%j.err

# STAGE 2 -- same-target NNLS for advection: 4 quadratures (inc/adv/path/grad)
# at fixed M=256, m=1024, all online with the exact-linear residual;
# certified by the ladder on held-out iterates + test solver path + oracle.
# N=256, checkpoint sep_hfit_dense_mid_N256.pkl.  Trains nothing.
set -euo pipefail
ROOT=/cluster/tufts/paralab/tawal01/exlin/gf256m256
VENV=/cluster/tufts/paralab/tawal01/ae-research/venv
PY="$VENV/bin/python"
export JAX_DEFAULT_MATMUL_PRECISION=highest

echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}', flush=True); sys.exit(0 if b=='gpu' else 42)"
( cd "$ROOT" && sha256sum -c MANIFEST.sha256 --quiet && echo "stage MANIFEST OK" )
sha256sum "$ROOT/in/sep_hfit_dense_mid_N256.pkl"

cd "$ROOT/code"
env N=256 CKPT="$ROOT/in/sep_hfit_dense_mid_N256.pkl" N_TEST=4 EQ_M=256 EQ_M_FACTOR=4 EQ_CAND_CAP=65536 \
  VARIANTS=inc,adv,path,grad N_FIT_TRAJ=8 N_HELDOUT=64 \
  SEED0=0 TR_FACTOR=0.01 STEP_TOL=1e-9 STALL=1e-3 EXTRAP=1.0 \
  ORACLE_TS=0,1,2,3,5,10,25,50 N_TRAIN_SNAP=16 \
  OUT_TAG=N256_K16_dense_mid_M256 OUT_PREFIX="$ROOT/out/" \
  "$PY" sep_eq_gradfit.py

cd "$ROOT/out" && sha256sum *.json > RESULTS.sha256
echo ALL-DONE
