#!/bin/bash
#SBATCH --job-name=specfom_smoke3
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1

#SBATCH --exclude=pax007
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=06:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/specfom_20260923/smoke3/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/specfom_20260923/smoke3/logs/%j.err
set -euo pipefail
TASK_ROOT=/cluster/tufts/paralab/tawal01/specfom_20260923/smoke3
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export XDG_CACHE_HOME="$TASK_ROOT/cache" MPLCONFIGDIR="$TASK_ROOT/cache/matplotlib"
export TMPDIR="$TASK_ROOT/tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
cd "$TASK_ROOT"
sha256sum -c MANIFEST.sha256 --quiet
export SOURCE_COMMIT=$(cat COMMIT.txt)
echo "host=$(hostname) source_commit=$SOURCE_COMMIT"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
df -h /cluster/tufts/paralab | tail -1
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
cd "$TASK_ROOT/code"
export PYTHONPATH="$TASK_ROOT/code/experiments/mr-burgers2d:$TASK_ROOT/code/experiments/separable-decoder:$TASK_ROOT/code/experiments/head-ablation:$TASK_ROOT/code/experiments/cheap-corrections:$TASK_ROOT/code/experiments/b-ladder-top:$TASK_ROOT/code/experiments/b-panel/speed:$TASK_ROOT/code/experiments/hires-burgers:$TASK_ROOT/code/experiments/burgers-repanel:$TASK_ROOT/code/experiments/burgers-bank-knob"
"$PY" sp_burgers.py --config config-b2d-256.json --out ../output0
"$PY" sp_audit_np.py ../output0 --delete-fields || echo "AUDIT-FAILED ../output0"
cd "$TASK_ROOT"
rm -rf cache tmp
find output* -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo ALL-DONE
