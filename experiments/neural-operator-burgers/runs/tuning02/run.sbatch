#!/bin/bash
#SBATCH --job-name=ctol_nob_tuning02
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100-80G
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:50:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/no_burgers_20260914/tuning02/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/no_burgers_20260914/tuning02/logs/%j.err
set -euo pipefail
TASK_ROOT=/cluster/tufts/paralab/tawal01/no_burgers_20260914/tuning02
CACHE=/cluster/tufts/paralab/tawal01/no_burgers_20260914/pilot-data01
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export XDG_CACHE_HOME="$TASK_ROOT/cache" MPLCONFIGDIR="$TASK_ROOT/cache/matplotlib" TMPDIR="$TASK_ROOT/tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
cd "$TASK_ROOT"
sha256sum -c MANIFEST.sha256 --quiet
export SOURCE_COMMIT=$(cat COMMIT.txt)
export PYTHONPATH="$TASK_ROOT/experiments/neural-operator-burgers:$TASK_ROOT/experiments/mr-burgers2d"
echo "host=$(hostname) source_commit=$SOURCE_COMMIT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
"$PY" experiments/neural-operator-burgers/tuning.py --phase validation \
  --reference "$CACHE/refinement/index.json" --dataset "$CACHE/validation/index.json" \
  --out "$TASK_ROOT/output" --seconds 6000 --shortlist "$TASK_ROOT/experiments/neural-operator-burgers/checks/tuning02-shortlist.json" 
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo ALL-DONE
