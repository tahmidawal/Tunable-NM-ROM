#!/bin/bash
# 256^3 second attempt (DESIGN A1): (1) whole-GPU memory probes of FNO-s and DeepONet-s at micro-batch 1
# (probe only, sequential); (2) the arms whose first attempt died in the harness's normalisation pass train concurrently.
# sbatch ... train256b.sbatch "<probe specs p:n:a:f ...>" <train specs p:n:a:f> ...
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --cpus-per-task=16
#SBATCH --time=01:55:00
set -uo pipefail
PROBES="$1"; shift
TASK_ROOT="$PWD"
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
source /cluster/tufts/paralab/tawal01/ae-research/venv/bin/activate
export JAX_DEFAULT_MATMUL_PRECISION=highest JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
export XDG_CACHE_HOME="$TASK_ROOT/cache" MPLCONFIGDIR="$TASK_ROOT/cache/matplotlib" TMPDIR="$TASK_ROOT/tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
echo "host=$(hostname) job=$SLURM_JOB_ID probes=$PROBES specs=$*"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
sha256sum -c --quiet SOURCE.sha256 || { echo "source manifest mismatch"; exit 3; }
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)" || exit 42
cd code
for spec in $PROBES; do
  IFS=: read -r P N A F <<< "$spec"
  "$PY" train_l3d.py --probe-only --problem "$P" --mesh "$N" --config "configs/ops/$A.json" --out "$TASK_ROOT/output/probe_$P$N/$A" \
      --mem-frac "$F" > "$TASK_ROOT/logs/probe-$P$N-$A.log" 2>&1; echo "PROBE-EXIT $P$N-$A $?"
done
PIDS=()
for spec in "$@"; do
  IFS=: read -r P N A F <<< "$spec"
  "$PY" train_l3d.py --problem "$P" --mesh "$N" --config "configs/ops/$A.json" --out "$TASK_ROOT/output/$P$N/$A" \
      --mem-frac "$F" > "$TASK_ROOT/logs/$P$N-$A.log" 2>&1 &
  PIDS+=("$!:$P$N-$A")
done
for pa in "${PIDS[@]}"; do wait "${pa%%:*}"; echo "EXIT ${pa#*:} $?"; done
cd "$TASK_ROOT"; rm -rf cache tmp
find output -type f -print0 | sort -z | xargs -0 -r sha256sum > OUTPUTS.sha256
echo ALL-DONE
