#!/bin/bash -l
# One GPU allocation of the measured Burgers stepper (amendment 3, Task A).  Per task line "dim L cases budget_s":
# reference solutions by the paper's solver (ref.py, JAX), then the cuDSS stepper (stepper.py, CuPy), then the
# reference arrays of that size are deleted.
# Usage: sbatch --job-name=ss_<tag> --time=HH:MM:SS --constraint=<gpu feature> cluster/run_step.sh <tasks file> [smoke]
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
set -eo pipefail
: "${SLURM_JOB_ID:?must be submitted as a batch job}"
NS=/cluster/tufts/paralab/tawal01/sparse-solvers-20261113
JOB_DIR=$(pwd -P)
case "$JOB_DIR" in $NS/jobs/*) ;; *) echo "bad job dir"; exit 64;; esac
VENV=/cluster/tufts/paralab/tawal01/ae-research/venv
PY=$VENV/bin/python
SP=$VENV/lib/python3.13/site-packages
export JAX_DEFAULT_MATMUL_PRECISION=highest JAX_ENABLE_X64=true PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TMPDIR="$JOB_DIR/tmp" XDG_CACHE_HOME="$JOB_DIR/cache" CUPY_CACHE_DIR="$JOB_DIR/cache/cupy"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$JOB_DIR/results" "$JOB_DIR/refs"
sha256sum -c --quiet SOURCE-SHA256SUMS && echo "sources verified"
sha256sum -c --quiet vendor/SHA256SUMS && echo "vendored paper solver verified"
cat SOURCE_COMMIT.txt; hostname; date
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
printf 'cpu_model='; lscpu | sed -n 's/^Model name:[[:space:]]*//p'
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}'); sys.exit(0 if b == 'gpu' else 42)"
export SS_PYDEPS=$NS/pydeps
NVL=$(ls -d $SP/nvidia/*/lib | tr '\n' ':')
BASE_LD="${LD_LIBRARY_PATH:-}"
grep -v '^#' "$1" | while read -r dim L cases budget; do
  [ -n "$dim" ] || continue
  out=results/b${dim}d_L${L}.json
  [ -e "$out" ] && continue
  echo "=== dim=$dim L=$L cases=$cases budget=$budget $(date)"
  if ! "$PY" ref.py --dim "$dim" --L "$L" --cases "$cases" --out refs; then
    echo "{\"task\": \"burgers_stepper\", \"dim\": $dim, \"L\": $L, \"status\": \"reference failed\"}" > "$out"
    echo "REFERENCE FAILED dim=$dim L=$L"; continue
  fi
  cp refs/b${dim}d_L${L}.json results/ref_b${dim}d_L${L}.json
  PYTHONPATH="$NS/pydeps:$JOB_DIR" LD_LIBRARY_PATH="$NS/pydeps/nvidia/cu12/lib:$NVL$BASE_LD" \
    "$PY" stepper.py --dim "$dim" --L "$L" --cases "$cases" --refs refs --out "$out" --budget "$budget" ${2:+--smoke} \
    || echo "STEPPER PROCESS FAILED dim=$dim L=$L rc=$?"
  rm -f refs/b${dim}d_L${L}_case*.npy
done
date
echo "task_completed"
