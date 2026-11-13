#!/usr/bin/env bash
# One AMG-lane allocation (CPU partition, 16 threads): pinned NGSolve wheels into a job prefix, pyamg from the lane-local
# target directory, then the task script.  Usage: sbatch --job-name=cylamg_<tag> --time=HH:MM:SS cluster/run.sh <task.sh>
#SBATCH --partition=batch
#SBATCH --constraint=sapphirerapids
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?must be submitted as a batch job}"
NS=/cluster/tufts/paralab/tawal01/cyl-amg-20261113
JOB_DIR=$(pwd -P)
case "$JOB_DIR" in $NS/*) ;; *) echo "bad job dir"; exit 64;; esac
export PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
export TMPDIR="$JOB_DIR/tmp" XDG_CACHE_HOME="$JOB_DIR/cache" MPLCONFIGDIR="$JOB_DIR/cache/mpl"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$JOB_DIR/results"
sha256sum -c --quiet SOURCE-SHA256SUMS && echo "sources verified"
cat SOURCE_COMMIT.txt
printf 'cpu_model='; lscpu | sed -n 's/^Model name:[[:space:]]*//p'; hostname; nproc
mkdir -p wheels && cp $NS/wheels/*.whl wheels/
"$PY" install_deps.py --wheel-dir "$JOB_DIR/wheels" --target "$JOB_DIR/deps"
DEPS_SITE=$("$PY" -c "import sysconfig; print(sysconfig.get_path('purelib', vars={'base':'$JOB_DIR/deps','platbase':'$JOB_DIR/deps'}))")
export PYTHONPATH="$DEPS_SITE:$NS/pydeps:$JOB_DIR:$JOB_DIR/vendor${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$JOB_DIR/deps/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$PY" -c "import ngsolve, pyamg, scipy; print('ngsolve', ngsolve.__version__, 'pyamg', pyamg.__version__, 'scipy', scipy.__version__)"
bash "$1"
echo "task_completed"
