#!/bin/bash
#SBATCH --job-name=tensor_n65536
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100-80G
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/b1dtensor/tensor_n65536/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/b1dtensor/tensor_n65536/logs/%j.err

# JOB B/C (2026-08-29): N=65536.  Stage 1 = the standard sep_b1d_scale.py recipe
# (mirrors run_b1ds_n4096.sbatch: K=8 R=32 M=32, 40k steps, seed 0, same
# family; six ROM arms + tridiagonal FOM at FOM_TOLS=1e-3,1e-8).  The dense
# n x n gates J/T2 run at N=min(N,16384) with 1 trajectory (O(n^2) memory).
# Stage 2 = sep_b1d_tensor.py on the fresh checkpoint (oracle / NNLS-32 /
# learned-32 / tensor / tensor_nolean, ref + fast rollouts, 5 timed reps).
set -euo pipefail
ROOT=/cluster/tufts/paralab/tawal01/b1dtensor/tensor_n65536
VENV=/cluster/tufts/paralab/tawal01/ae-research/venv
PY="$VENV/bin/python"
export JAX_DEFAULT_MATMUL_PRECISION=highest

echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}', flush=True); sys.exit(0 if b=='gpu' else 42)"
( cd "$ROOT" && sha256sum -c MANIFEST.sha256 --quiet && echo "stage MANIFEST OK" )
export COMMIT=$(cat "$ROOT/COMMIT.txt")
echo "commit=$COMMIT"

cd "$ROOT/code"
env N=65536 K=8 R=32 M=32 N_TRAIN=512 N_TEST=8 TRAIN_STEPS=40000 \
  NODE_STEPS=2000 LR_NODES=3e-3 REFIT_EVERY=500 REFIT_JAC_STATES=16 \
  EVAL_EVERY=200 SAMP_REL=1 JAC_REL=1 SEED0=0 DATA_SEED=0 TEST_SEED=1 \
  TIME_REPS=5 BURN=2 FOM_TOLS=1e-3,1e-8 \
  DENSE_GATE_MAX_N=16384 T2_TRAJ=1 \
  OUT_TAG=n65536 OUT_PREFIX="$ROOT/out/" \
  "$PY" sep_b1d_scale.py

env N=65536 K=8 R=32 M=32 SEED0=0 TEST_SEED=1 N_TEST=8 TIME_REPS=5 BURN=2 \
  ARMS=oracle,base_tight,nodes_tight,tensor,tensor_nolean \
  DENSE_GATE_MAX_N=16384 T2_TRAJ=1 \
  CKPT_CACHE="$ROOT/out/sep_b1d_scale_n65536.pkl" \
  NODES_NPZ="$ROOT/out/sep_b1d_scale_n65536_nodes.npz" \
  BASE_JSON="$ROOT/out/sep_b1d_scale_n65536.json" \
  OUT="$ROOT/out/sep_b1d_tensor_n65536.json" \
  "$PY" sep_b1d_tensor.py

cd "$ROOT/out" && sha256sum *.json *.npz *.pkl > RESULTS.sha256
echo ALL-DONE
