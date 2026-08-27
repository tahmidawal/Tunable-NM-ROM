#!/bin/bash
#SBATCH --job-name=b1ds_n128
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/b1dqf/b1ds_n128/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/b1dqf/b1ds_n128/logs/%j.err

# 1D Burgers scaling + cost study at N=128: six ROM arms with the on-device
# rollout instrument (cost+accuracy same invocation, ic/solve/decode split,
# persisted reps) and the honest tridiagonal tolerance-Newton FOM baseline,
# timed in the same job on the same GPU.  Gates J/T2/E/F/C/D/G/V.
set -euo pipefail
ROOT=/cluster/tufts/paralab/tawal01/b1dqf/b1ds_n128
VENV=/cluster/tufts/paralab/tawal01/ae-research/venv
PY="$VENV/bin/python"
export JAX_DEFAULT_MATMUL_PRECISION=highest

echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}', flush=True); sys.exit(0 if b=='gpu' else 42)"
( cd "$ROOT" && sha256sum -c MANIFEST.sha256 --quiet && echo "stage MANIFEST OK" )

cd "$ROOT/code"
env N=128 K=8 R=32 M=32 N_TRAIN=512 N_TEST=8 TRAIN_STEPS=40000 \
  NODE_STEPS=2000 LR_NODES=3e-3 REFIT_EVERY=500 REFIT_JAC_STATES=16 \
  EVAL_EVERY=200 SAMP_REL=1 JAC_REL=1 SEED0=0 DATA_SEED=0 TEST_SEED=1 \
  TIME_REPS=5 BURN=2 FOM_TOLS=1e-3,1e-8 \
  OUT_TAG=n128 OUT_PREFIX="$ROOT/out/" \
  "$PY" sep_b1d_scale.py

cd "$ROOT/out" && sha256sum *.json *.npz *.pkl > RESULTS.sha256
echo ALL-DONE
