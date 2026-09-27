#!/bin/bash
# sbatch ... panel.sbatch <config relative to code/> [jax memory fraction]
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
set -uo pipefail
CONFIG="$1"; JFRAC="${2:-0.6}"
TASK_ROOT="$PWD"
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
source /cluster/tufts/paralab/tawal01/ae-research/venv/bin/activate
export JAX_DEFAULT_MATMUL_PRECISION=highest JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=$JFRAC
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export XDG_CACHE_HOME="$TASK_ROOT/cache" MPLCONFIGDIR="$TASK_ROOT/cache/matplotlib" TMPDIR="$TASK_ROOT/tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
echo "host=$(hostname) job=$SLURM_JOB_ID config=$CONFIG jax_fraction=$JFRAC"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
df -h /cluster/tufts/paralab | tail -1
sha256sum -c --quiet SOURCE.sha256 || { echo "source manifest mismatch"; exit 3; }
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)" || exit 42
export SOURCE_COMMIT=none-uncommitted-see-SOURCE.sha256
cd code
"$PY" panel_l3d.py --config "$CONFIG" --out "$TASK_ROOT/output"
RC=$?
cd "$TASK_ROOT"
rm -rf cache tmp code/checkpoints
find output -type f -print0 | sort -z | xargs -0 -r sha256sum > OUTPUTS.sha256
echo "RUN_EXIT=$RC"
echo ALL-DONE
exit $RC
