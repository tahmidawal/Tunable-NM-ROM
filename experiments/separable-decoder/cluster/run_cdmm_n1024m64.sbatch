#!/bin/bash
#SBATCH --job-name=cdmm_n1024m64
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=08:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/codesign-mm/n1024m64/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/codesign-mm/n1024m64/logs/%j.err

# NODES-ONLY AT m=M, N=1024 (PRIMARY).  Same question as n256m64 at the fine
# mesh.  Checkpoint sep_burgers_N1024_K16_R64.pkl (sepdec_n1024 j2): its data
# draw was 96+8 trajectories, max_snaps 2048 -- N_TRAIN/N_VAL below reproduce
# it exactly (gate R enforces).  H200 + 240G per the sepdec_n1024 j2 sizing
# (~44 GB host per full f64 state array).  Truth generation uses the staged
# Helmholtz-preconditioned BiCGStab (patch_bf_precond.py; unpreconditioned
# stalls at N=1024).
set -euo pipefail
ROOT=/cluster/tufts/paralab/tawal01/codesign-mm/n1024m64
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
  TEACHER_CHUNK=8 EQ_M=64 EQ_M_FACTOR=1 \
  STEPS=2000 LR=3e-5 LR_NODES=3e-3 REFIT_EVERY=500 REFIT_JAC_STATES=16 \
  EVAL_EVERY=200 N_TEST=4 SEED0=0 \
  OUT_TAG=n_N1024_m64 OUT_PREFIX="$ROOT/out/" \
  "$PY" sep_codesign.py

cd "$ROOT/out" && sha256sum *.json *.npz *.pkl > RESULTS.sha256
echo ALL-DONE
