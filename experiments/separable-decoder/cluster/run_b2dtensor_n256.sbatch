#!/bin/bash
#SBATCH --job-name=b2dtensor_n256
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/b2dtensor/n256/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/b2dtensor/n256/logs/%j.err

# 2D Burgers quadratic-tensor arm at N=256 (2026-08-30): arms full / ex (NNLS
# m=256 exlin) / tensor (+ ex_learned when a codesign node set exists), K=16
# R=64 M=64, 8 fresh-seed test trajectories regenerated in-job, training truth
# regenerated (streamed) for the positivity audit, gates asserted, arms
# interleaved (reps outermost, AB/BA), 2 burn + 5 timed reps, FOM tol-Newton
# ladder + matched-accuracy paired AB/BA on the same GPU.
set -euo pipefail
ROOT=/cluster/tufts/paralab/tawal01/b2dtensor/n256
VENV=/cluster/tufts/paralab/tawal01/ae-research/venv
PY="$VENV/bin/python"
export JAX_DEFAULT_MATMUL_PRECISION=highest

echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}', flush=True); sys.exit(0 if b=='gpu' else 42)"
( cd "$ROOT" && sha256sum -c MANIFEST.sha256 --quiet && echo "stage MANIFEST OK" )
export COMMIT=$(cat "$ROOT/COMMIT.txt")
echo "commit=$COMMIT"

cd "$ROOT/code"
env N=256 K=16 R=64 EQ_M=64 EQ_MQ=256 EQ_CAND_CAP=65536 SEED0=0 TEST_SEED=1 N_TEST=8 \
  TR_FACTOR=0.01 STEP_TOL=1e-9 STALL=1e-3 EXTRAP=1.0 IC_ENC_BUDGET=50 ENC_STEPS=12000 \
  TIME_REPS=5 BURN=2 FOM_REPS=5 PAIR_REPS=3 WARM=2 \
  NEWTON_TOLS=3e-1,1e-1,3e-2,1e-2,3e-3,1e-3,1e-4 LIN_FRACS=0.05,0.5 \
  ARMS=full,ex,tensor CKPT="$ROOT/in/sep_burgers_N256_K16_R64.pkl" NODES_NPZ="$ROOT/in/sep_codesign_n_N256_m64_nodes.npz" \
  OUT="$ROOT/out/sep_b2d_tensor_n256.json" CKPT_OUT="$ROOT/out/sep_b2d_tensor_n256_ckpt.pkl" \
  "$PY" sep_b2d_tensor.py

cd "$ROOT/out" && sha256sum *.json $(ls *.pkl 2>/dev/null) > RESULTS.sha256
echo ALL-DONE
