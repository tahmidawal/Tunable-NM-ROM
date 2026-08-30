#!/bin/bash
#SBATCH --job-name=b2dtensor_n1024r512
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=04:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/b2dtensor/n1024r512/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/b2dtensor/n1024r512/logs/%j.err

# STRETCH (2026-08-30): the R=512 headline checkpoint (dn1024 dense_mid, K=32,
# R=512, N=1024, job 2837430) with the HEAD-PCA Tucker tensor: SVD of the head
# outputs over the 131072 training codes, K' at tail energy <= 1e-8, tensor
# (M=64, K', K') from the projected bank.  Arms full / ex (NNLS m=256 adv-only,
# the headline's ctrl set) / tensor.  Training truth NOT regenerated (4608
# trajectories; SKIP_TRAIN_TRUTH=1 -- positivity of the family is asserted on
# the canonical 576/104-trajectory draws in the N=64..1024 jobs and on the
# 8 test trajectories here).  H200 + 240G per the dn1024 sizing.
set -euo pipefail
ROOT=/cluster/tufts/paralab/tawal01/b2dtensor/n1024r512
VENV=/cluster/tufts/paralab/tawal01/ae-research/venv
PY="$VENV/bin/python"
export JAX_DEFAULT_MATMUL_PRECISION=highest

echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}', flush=True); sys.exit(0 if b=='gpu' else 42)"
( cd "$ROOT" && sha256sum -c MANIFEST.sha256 --quiet && echo "stage MANIFEST OK" )
export COMMIT=$(cat "$ROOT/COMMIT.txt")
echo "commit=$COMMIT"

cd "$ROOT/code"
env N=1024 K=32 R=512 EQ_M=64 EQ_MQ=256 EQ_CAND_CAP=65536 SEED0=0 TEST_SEED=1 N_TEST=8 \
  N_TRAIN=512 N_VAL=64 SKIP_TRAIN_TRUTH=1 HEAD_PCA=1 PCA_TAIL=1e-8 PCA_T0_TOL=1e-6 \
  TR_FACTOR=0.01 STEP_TOL=1e-9 STALL=1e-3 EXTRAP=1.0 IC_ENC_BUDGET=50 ENC_STEPS=12000 \
  TIME_REPS=5 BURN=2 FOM_REPS=5 PAIR_REPS=3 WARM=2 \
  NEWTON_TOLS=3e-1,1e-1,3e-2,1e-2,3e-3,1e-3,1e-4 LIN_FRACS=0.05,0.5 \
  ARMS=full,ex,tensor CKPT="$ROOT/in/sep_hfit_dense_mid_N1024_dense.pkl" \
  OUT="$ROOT/out/sep_b2d_tensor_n1024r512.json" CKPT_OUT="$ROOT/out/unused.pkl" \
  "$PY" sep_b2d_tensor.py

cd "$ROOT/out" && sha256sum *.json > RESULTS.sha256
echo ALL-DONE
