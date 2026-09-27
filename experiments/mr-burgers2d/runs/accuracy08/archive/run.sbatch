#!/bin/bash
#SBATCH --job-name=ctol_mr_burgers_accuracy08
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/mr_burgers2d_20260907/accuracy08/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/mr_burgers2d_20260907/accuracy08/logs/%j.err
set -euo pipefail
TASK_ROOT=/cluster/tufts/paralab/tawal01/mr_burgers2d_20260907/accuracy08
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export XDG_CACHE_HOME="$TASK_ROOT/cache" MPLCONFIGDIR="$TASK_ROOT/cache/matplotlib" TMPDIR="$TASK_ROOT/tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
cd "$TASK_ROOT"
sha256sum -c MANIFEST.sha256 --quiet
sha256sum -c INHERITED.sha256 --quiet
export COMMIT=$(cat COMMIT.txt)
echo "host=$(hostname) commit=$COMMIT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
"$PY" code/accuracy.py --config code/config-accuracy.json --checkpoint in/checkpoint.pkl --reuse-dir in/reuse --out out
printf 'ALL-DONE\n'
