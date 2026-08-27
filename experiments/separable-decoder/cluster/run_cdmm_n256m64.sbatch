#!/bin/bash
#SBATCH --job-name=cdmm_n256m64
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/codesign-mm/n256m64/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/codesign-mm/n256m64/logs/%j.err

# NODES-ONLY AT m=M, N=256 (PRIMARY).  Does the mini-pilot's F3 result
# (-17.3% rollout at N=64, m=M=64, decoder frozen) transfer to the real mesh?
# sep_codesign.py arm n: TRAIN_H=0, node positions trained against the frozen
# full-grid advection teacher, weights NNLS-re-solved every 500 steps.  Same
# driver commit, hyperparameters, and seed as the pilot; in-driver base
# certification (NNLS grid nodes at the same m) is the control.  Checkpoint
# sep_burgers_N256_K16_R64.pkl (sepdec n256_j2, canonical default data draw,
# max_snaps 8192).
set -euo pipefail
ROOT=/cluster/tufts/paralab/tawal01/codesign-mm/n256m64
VENV=/cluster/tufts/paralab/tawal01/ae-research/venv
PY="$VENV/bin/python"
export JAX_DEFAULT_MATMUL_PRECISION=highest

echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}', flush=True); sys.exit(0 if b=='gpu' else 42)"
( cd "$ROOT" && sha256sum -c MANIFEST.sha256 --quiet && echo "stage MANIFEST OK" )
sha256sum "$ROOT/in/sep_burgers_N256_K16_R64.pkl"

cd "$ROOT/code"
env TRAIN_H=0 TRAIN_NODES=1 SAMP_REL=1 JAC_REL=1 SOB_REL=0 REC_W=10 \
  CKPT="$ROOT/in/sep_burgers_N256_K16_R64.pkl" N=256 EQ_M=64 EQ_M_FACTOR=1 \
  STEPS=2000 LR=3e-5 LR_NODES=3e-3 REFIT_EVERY=500 REFIT_JAC_STATES=16 \
  EVAL_EVERY=200 N_TEST=4 SEED0=0 \
  OUT_TAG=n_N256_m64 OUT_PREFIX="$ROOT/out/" \
  "$PY" sep_codesign.py

cd "$ROOT/out" && sha256sum *.json *.npz *.pkl > RESULTS.sha256
echo ALL-DONE
