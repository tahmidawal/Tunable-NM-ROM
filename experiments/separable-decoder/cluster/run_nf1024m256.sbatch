#!/bin/bash
#SBATCH --job-name=exlin_nf1024m256
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=08:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/exlin/nf1024m256/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/exlin/nf1024m256/logs/%j.err

# STAGE 3 -- learned continuous quadrature nodes (sep_eq_nodefit.py), M=256,
# m=1024, N=1024, checkpoint sep_burgers_r4_N1024_K32_R512_h512x3.pkl.  Frozen decoder + frozen full-grid
# teacher; inner NNLS variable projection; ladder-shaped loss; certified
# against the adv (stage-1) and grad (stage-2) sets on held-out iterates,
# test solver path, oracle states, and 4 test rollouts.
set -euo pipefail
ROOT=/cluster/tufts/paralab/tawal01/exlin/nf1024m256
VENV=/cluster/tufts/paralab/tawal01/ae-research/venv
PY="$VENV/bin/python"
export JAX_DEFAULT_MATMUL_PRECISION=highest

echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}', flush=True); sys.exit(0 if b=='gpu' else 42)"
( cd "$ROOT" && sha256sum -c MANIFEST.sha256 --quiet && echo "stage MANIFEST OK" )
sha256sum "$ROOT/in/sep_burgers_r4_N1024_K32_R512_h512x3.pkl"

cd "$ROOT/code"
env N=1024 CKPT="$ROOT/in/sep_burgers_r4_N1024_K32_R512_h512x3.pkl" N_TEST=4 EQ_M=256 EQ_M_FACTOR=4 EQ_CAND_CAP=65536 \
  VARIANTS=adv,grad,node N_FIT_TRAJ=8 N_HELDOUT=64 \
  STEPS=2000 REFIT_EVERY=400 LR=3e-3 \
  SEED0=0 TR_FACTOR=0.01 STEP_TOL=1e-9 STALL=1e-3 EXTRAP=1.0 \
  ORACLE_TS=0,1,2,3,5,10,25,50 N_TRAIN_SNAP=16 \
  OUT_TAG=N1024_K32_r4a6_M256 OUT_PREFIX="$ROOT/out/" \
  "$PY" sep_eq_nodefit.py

cd "$ROOT/out" && sha256sum *.json *.npz > RESULTS.sha256
echo ALL-DONE
