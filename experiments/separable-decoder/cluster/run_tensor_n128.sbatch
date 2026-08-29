#!/bin/bash
#SBATCH --job-name=tensor_n128
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/b1dtensor/tensor_n128/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/b1dtensor/tensor_n128/logs/%j.err

# 1D Burgers sample-free TENSOR arm at N=128 (2026-08-29): oracle / NNLS-32 /
# learned-32 / tensor (lean + nolean) from the COMMITTED checkpoint + node
# sets, test trajectories regenerated in-job; gates J/T2/E/F/C/G/V +
# TB/TX/TA/T0/TS/TQ; reference and optimized rollouts interleaved, 5 timed
# reps, every rep persisted.
set -euo pipefail
ROOT=/cluster/tufts/paralab/tawal01/b1dtensor/tensor_n128
VENV=/cluster/tufts/paralab/tawal01/ae-research/venv
PY="$VENV/bin/python"
export JAX_DEFAULT_MATMUL_PRECISION=highest

echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}', flush=True); sys.exit(0 if b=='gpu' else 42)"
( cd "$ROOT" && sha256sum -c MANIFEST.sha256 --quiet && echo "stage MANIFEST OK" )
export COMMIT=$(cat "$ROOT/COMMIT.txt")
echo "commit=$COMMIT"

cd "$ROOT/code"
env N=128 K=8 R=32 M=32 SEED0=0 TEST_SEED=1 N_TEST=8 TIME_REPS=5 BURN=2 \
  ARMS=oracle,base_tight,nodes_tight,tensor,tensor_nolean \
  CKPT_CACHE="$ROOT/data/sep_b1d_scale_n128.pkl" \
  NODES_NPZ="$ROOT/data/sep_b1d_scale_n128_nodes.npz" \
  BASE_JSON="$ROOT/data/sep_b1d_scale_n128.json" \
  QREF="$ROOT/data/Q_n128.npy" \
  OUT="$ROOT/out/sep_b1d_tensor_n128.json" \
  "$PY" sep_b1d_tensor.py

cd "$ROOT/out" && sha256sum *.json > RESULTS.sha256
echo ALL-DONE
