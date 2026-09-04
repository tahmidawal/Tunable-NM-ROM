#!/bin/bash
#SBATCH --job-name=b3d_n33
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100-80G
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/b3dtensor/n33/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/b3dtensor/n33/logs/%j.err
set -euo pipefail
ROOT=/cluster/tufts/paralab/tawal01/b3dtensor/n33
VENV=/cluster/tufts/paralab/tawal01/ae-research/venv
PY="$VENV/bin/python"
export JAX_DEFAULT_MATMUL_PRECISION=highest
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}', flush=True); sys.exit(0 if b=='gpu' else 42)"
( cd "$ROOT" && sha256sum -c MANIFEST.sha256 --quiet && echo "stage MANIFEST OK" )
export COMMIT=$(cat "$ROOT/COMMIT.txt")
echo "commit=$COMMIT"
cd "$ROOT/code"
env TABLE_DIR="$ROOT/code/runs/b3dtensor/tables" N=33 K=16 R=64 EQ_M=128 EQ_MQ=512 SEED0=0 TEST_SEED=1 N_TEST=8 TR_FACTOR=0.01 STEP_TOL=1e-9 STALL=1e-3 EXTRAP=1.0 IC_ENC_BUDGET=50 ENC_STEPS=12000 TIME_REPS=5 BURN=2 FOM_REPS=5 PAIR_REPS=5 WARM=2 STEPS=60000 LR=1e-3 P_SUB=16384 MAX_SNAPS=8192 OUT="$ROOT/out/sep_b3d_tensor_n33.json" CKPT_OUT="$ROOT/out/sep_b3d_tensor_n33_ckpt.pkl" KERNEL_OUT="$ROOT/out/sep_b3d_tensor_n33_kernel.npz" "$PY" sep_b3d_tensor.py
cd "$ROOT/out" && sha256sum $(ls *.json *.pkl *.npz 2>/dev/null) > RESULTS.sha256
echo ALL-DONE
