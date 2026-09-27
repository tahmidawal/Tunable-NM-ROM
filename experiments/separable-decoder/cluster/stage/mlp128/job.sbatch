#!/bin/bash
#SBATCH --job-name=ctol_b3da_mlp128
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100-80G
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/b3d_repair_20260906/mlp128/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/b3d_repair_20260906/mlp128/logs/%j.err
set -euo pipefail
TASK_ROOT=/cluster/tufts/paralab/tawal01/b3d_repair_20260906/mlp128
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_DEFAULT_MATMUL_PRECISION=highest
export JAX_ENABLE_X64=1
export OPENBLAS_NUM_THREADS=8
export OMP_NUM_THREADS=8
cd "$TASK_ROOT"
sha256sum -c MANIFEST.sha256 --quiet
finish() {
  task_exit=$?
  cd "$TASK_ROOT"
  printf '%s\n' "$task_exit" > EXIT_CODE
  find out -type f -print0 | sort -z | xargs -0 -r sha256sum > RESULTS.sha256
  exit "$task_exit"
}
trap finish EXIT
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
export COMMIT=8484a9e00b41d0bf79872e14f69ed3c17fc185c9
cd "$TASK_ROOT/code"
env CKPT="$TASK_ROOT/in/checkpoint.pkl" OUT="$TASK_ROOT/out/result.json" TABLE_DIR="$TASK_ROOT/code/runs/b3dtensor/tables" REFERENCE_TABLE="$TASK_ROOT/in/reference_table.npz" MODEL=b3d_arch_baseline HEAD_WIDTH=128 OPT_SEEDS=200,201 "$PY" -u b3d_arch_bench.py
