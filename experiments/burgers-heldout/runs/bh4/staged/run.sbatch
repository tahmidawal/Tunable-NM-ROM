#!/bin/bash
#SBATCH --job-name=bh_bh4
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:h200:1

#SBATCH --exclude=pax007
#SBATCH --cpus-per-task=8
#SBATCH --mem=320G
#SBATCH --time=20:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/bheld_20260921/bh4/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/bheld_20260921/bh4/logs/%j.err
set -euo pipefail
TASK_ROOT=/cluster/tufts/paralab/tawal01/bheld_20260921/bh4
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export XDG_CACHE_HOME="$TASK_ROOT/cache" TMPDIR="$TASK_ROOT/tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TASK_ROOT/output"
cd "$TASK_ROOT"
sha256sum -c MANIFEST.sha256 --quiet
export SOURCE_COMMIT=$(cat COMMIT.txt)
echo "host=$(hostname) source_commit=$SOURCE_COMMIT"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
df -h /cluster/tufts/paralab | tail -1
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
EXPPATH="$TASK_ROOT/exp/experiments/mr-burgers2d:$TASK_ROOT/exp/experiments/separable-decoder:$TASK_ROOT/exp/experiments/head-ablation:$TASK_ROOT/exp/experiments/cheap-corrections:$TASK_ROOT/exp/experiments/b-ladder-top:$TASK_ROOT/exp/experiments/b-panel/speed:$TASK_ROOT/exp/experiments/hires-burgers:$TASK_ROOT/exp/experiments/burgers-heldout"

PYTHONPATH="$EXPPATH" "$PY" exp/experiments/burgers-heldout/bh_hires.py --config exp/experiments/burgers-heldout/config-4096-bh4.json --checkpoint in/bh_model.pkl \
  --inputs in --out output
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo ALL-DONE
