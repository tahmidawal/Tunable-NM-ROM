#!/bin/bash
#SBATCH --job-name=b1d_n256
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/b1dqf/b1d_n256/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/b1dqf/b1d_n256/logs/%j.err

# 1D Burgers frozen-decoder node-learning screening at N=256 (one seed).
# Everything regenerates in-job from seeds: data (residual gate 1e-8),
# decoder training (then frozen), NNLS node fits, learned nodes, and the
# four-arm certification (oracle / NNLS m=M / learned m=M / NNLS generous).
set -euo pipefail
ROOT=/cluster/tufts/paralab/tawal01/b1dqf/b1d_n256
VENV=/cluster/tufts/paralab/tawal01/ae-research/venv
PY="$VENV/bin/python"
export JAX_DEFAULT_MATMUL_PRECISION=highest

echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}', flush=True); sys.exit(0 if b=='gpu' else 42)"
( cd "$ROOT" && sha256sum -c MANIFEST.sha256 --quiet && echo "stage MANIFEST OK" )

cd "$ROOT/code"
env N=256 K=8 R=32 M=32 N_TRAIN=512 N_TEST=8 TRAIN_STEPS=40000 \
  NODE_STEPS=2000 LR_NODES=3e-3 REFIT_EVERY=500 REFIT_JAC_STATES=16 \
  EVAL_EVERY=200 SAMP_REL=1 JAC_REL=1 SEED0=0 DATA_SEED=0 TEST_SEED=1 \
  TIME_REPS=3 OUT_TAG=n256 OUT_PREFIX="$ROOT/out/" \
  "$PY" sep_b1d_screen.py

cd "$ROOT/out" && sha256sum *.json *.npz *.pkl > RESULTS.sha256
echo ALL-DONE
