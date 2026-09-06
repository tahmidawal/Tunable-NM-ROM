#!/bin/bash
#SBATCH --job-name=ctol_b3dr_repro33
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100-80G
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/b3d_repair_20260906/repair_repro33/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/b3d_repair_20260906/repair_repro33/logs/%j.err
set -euo pipefail
TASK_ROOT=/cluster/tufts/paralab/tawal01/b3d_repair_20260906/repair_repro33
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_DEFAULT_MATMUL_PRECISION=highest
export JAX_ENABLE_X64=1
export OPENBLAS_NUM_THREADS=8
export OMP_NUM_THREADS=8
cd "$TASK_ROOT"
sha256sum -c MANIFEST.sha256 --quiet
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
export COMMIT=da479125b13aaa3e15b5cae33f72709705129838
cd "$TASK_ROOT/code"
env PILOT=1 TRAIN=0 N=33 K=32 R=128 EQ_M=256 EQ_MQ=1024 G_HIDDEN=256 GEN_CHUNK=64 CKPT="$TASK_ROOT/in/baseline.pkl" OUT="$TASK_ROOT/out/repro33.json" TABLE_DIR="$TASK_ROOT/code/runs/b3dtensor/tables" "$PY" -u sep_b3d_tensor.py
cd "$TASK_ROOT"
find out -type f -print0 | sort -z | xargs -0 sha256sum > RESULTS.sha256
