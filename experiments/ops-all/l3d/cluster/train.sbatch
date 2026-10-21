#!/bin/bash
# sbatch ... train.sbatch <problem:mesh:arm:memfrac> ...   (all arms run CONCURRENTLY on the one GPU)
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --cpus-per-task=16
#SBATCH --time=01:45:00
set -uo pipefail
TASK_ROOT="$PWD"
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
source /cluster/tufts/paralab/tawal01/ae-research/venv/bin/activate
export JAX_DEFAULT_MATMUL_PRECISION=highest JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
export XDG_CACHE_HOME="$TASK_ROOT/cache" MPLCONFIGDIR="$TASK_ROOT/cache/matplotlib" TMPDIR="$TASK_ROOT/tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
echo "host=$(hostname) job=$SLURM_JOB_ID specs=$*"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
df -h /cluster/tufts/paralab | tail -1
sha256sum -c --quiet SOURCE.sha256 || { echo "source manifest mismatch"; exit 3; }
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)" || exit 42
cd code
PIDS=()
for spec in "$@"; do
  IFS=: read -r P N A F <<< "$spec"
  mkdir -p "$TASK_ROOT/output/$P$N"
  "$PY" train_l3d.py --problem "$P" --mesh "$N" --config "configs/ops/$A.json" --out "$TASK_ROOT/output/$P$N/$A" \
      --mem-frac "$F" > "$TASK_ROOT/logs/$P$N-$A.log" 2>&1 &
  PIDS+=("$!:$P$N-$A")
done
for pa in "${PIDS[@]}"; do wait "${pa%%:*}"; echo "EXIT ${pa#*:} $?"; done
cd "$TASK_ROOT"
rm -rf cache tmp
find output -type f -print0 | sort -z | xargs -0 -r sha256sum > OUTPUTS.sha256
echo ALL-DONE
