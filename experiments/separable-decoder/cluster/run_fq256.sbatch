#!/bin/bash
#SBATCH --job-name=burgacc_fq256
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=01:10:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/burgacc/fq256/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/burgacc/fq256/logs/%j.err

# ROUND 5 -- FINE-QUADRATURE GRADE of the cost-matched dense decoder.
#
# The Pareto row (dense mu-sampling, h=512x2, K=16) was measured on the CONTROL
# EQ set, M=64 / m=256, because that is the set the round-4 row used and a
# speed number must not be paired with a different quadrature than the accuracy
# number beside it.  This job re-measures the SAME decoder on M=256 / m=1024,
# where the r4a6 single-step diagnostic puts objective truncation at 1.004 by
# t=5 and 1.001 from t=10 on, so the reported error is the decoder's and not
# the quadrature's.  Its own timing and matched-accuracy pairing are re-run
# here too, so the fine-EQ accuracy and the fine-EQ speed are one job.
set -uo pipefail
ROOT=/cluster/tufts/paralab/tawal01/burgacc/fq256
VENV=/cluster/tufts/paralab/tawal01/ae-research/venv
PY="$VENV/bin/python"
export JAX_DEFAULT_MATMUL_PRECISION=highest

echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}', flush=True); sys.exit(0 if b=='gpu' else 42)" || exit 42
( cd "$ROOT" && sha256sum -c MANIFEST.sha256 --quiet && echo "stage MANIFEST OK" )

cd "$ROOT/code"
env CKPT="$ROOT/in/sep_hfit_dense_mid_N256.pkl" N=256 N_TEST=8 EQ_MS=256 \
  EQ_CAND_CAP=65536 SEED0=0 REPS=5 WARM=2 PAIR_REPS=3 TR_FACTOR=0.01 \
  STEP_TOL=1e-9 EXTRAP=1.0 IC_ENC_BUDGET=50 ENC_STEPS=12000 \
  STALLS=1e-12,1e-6,1e-4,1e-3,1e-2 BUDGETS=30,8,5,3,2,1 EXTRAPS=0,1.0,1.5 \
  BATCHES=1,2,4,8,16 NEWTON_TOLS=3e-1,1e-1,3e-2,1e-2,3e-3,1e-3,1e-4 \
  LIN_FRACS=0.05,0.5 OUT_TAG=dense_mid_N256_M256 OUT_PREFIX="$ROOT/out/" \
  "$PY" sep_speed_r5.py || echo "SPEED FAILED"

cd "$ROOT/out" && sha256sum * > RESULTS.sha256
echo ALL-DONE
