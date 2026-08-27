#!/bin/bash
#SBATCH --job-name=cdmm_n1024m256
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=08:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/codesign-mm/n1024m256/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/codesign-mm/n1024m256/logs/%j.err

# NODES-ONLY AT m=4M, N=1024 (CONTROL).  The expect-a-tie budget at the fine
# mesh; identical to n1024m64 except EQ_M_FACTOR=4.
set -euo pipefail
ROOT=/cluster/tufts/paralab/tawal01/codesign-mm/n1024m256
VENV=/cluster/tufts/paralab/tawal01/ae-research/venv
PY="$VENV/bin/python"
export JAX_DEFAULT_MATMUL_PRECISION=highest

echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}', flush=True); sys.exit(0 if b=='gpu' else 42)"
( cd "$ROOT" && sha256sum -c MANIFEST.sha256 --quiet && echo "stage MANIFEST OK" )
sha256sum "$ROOT/in/sep_burgers_N1024_K16_R64.pkl"

cd "$ROOT/code"
env TRAIN_H=0 TRAIN_NODES=1 SAMP_REL=1 JAC_REL=1 SOB_REL=0 REC_W=10 \
  CKPT="$ROOT/in/sep_burgers_N1024_K16_R64.pkl" N=1024 N_TRAIN=96 N_VAL=8 \
  TEACHER_CHUNK=8 EQ_M=64 EQ_M_FACTOR=4 \
  STEPS=2000 LR=3e-5 LR_NODES=3e-3 REFIT_EVERY=500 REFIT_JAC_STATES=16 \
  EVAL_EVERY=200 N_TEST=4 SEED0=0 \
  OUT_TAG=n_N1024_m256 OUT_PREFIX="$ROOT/out/" \
  "$PY" sep_codesign.py

cd "$ROOT/out" && sha256sum *.json *.npz *.pkl > RESULTS.sha256
echo ALL-DONE
