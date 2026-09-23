#!/bin/bash
# Usage (from submit.sh): sbatch --job-name=nshead_<job> --output=... job.sbatch <job> <config> [frozen-dir]
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
set -euo pipefail
JOB="$1"; CONFIG="$2"; FROZEN="${3:-}"
TASK_ROOT="$PWD"
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export XDG_CACHE_HOME="$TASK_ROOT/cache" MPLCONFIGDIR="$TASK_ROOT/cache/matplotlib"
export TMPDIR="$TASK_ROOT/tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TASK_ROOT/output"
echo "host=$(hostname) source_commit=$(cat COMMIT.txt) job=$SLURM_JOB_ID name=$JOB"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
df -h /cluster/tufts/paralab | tail -1
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
export SOURCE_COMMIT="$(cat COMMIT.txt)"
EXTRA=()
if [[ -n "$FROZEN" ]]; then EXTRA=(--frozen "$FROZEN"); fi
set +e
"$PY" experiments/ns3d-shift-head/head_ladder.py --config "$CONFIG" --out output "${EXTRA[@]}"
RC=$?
set -e
VRC=-1
set +e
if [[ "$RC" -eq 0 ]]; then
  "$PY" experiments/ns3d-shift-head/verify_head.py --out output; VRC=$?
fi
set -e
find output -type f -print0 | sort -z | xargs -0 -r sha256sum > OUTPUTS.sha256
# field files served the in-job independent audit; they are not transferred
rm -f output/fields_*.npy output/dev_truth.npy
echo "RUN_EXIT=$RC VERIFY_EXIT=$VRC"
exit "$RC"
