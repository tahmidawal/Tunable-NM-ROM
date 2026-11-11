#!/usr/bin/env bash
# One G5 ROM-lane allocation: preflight, pinned NGSolve wheels into a job prefix, then the
# staged task script.  Usage: sbatch --job-name=ctol_cylr_<tag> --time=HH:MM:SS cluster/run.sh <task.sh>
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?must be submitted as a batch job}"
JOB_DIR=$(pwd -P)
case "$JOB_DIR" in /cluster/tufts/paralab/tawal01/cyl-unsteady-20261103/*) ;; *) echo "bad job dir"; exit 64;; esac
export PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_DEFAULT_MATMUL_PRECISION=highest JAX_ENABLE_X64=true PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR="$JOB_DIR/tmp" XDG_CACHE_HOME="$JOB_DIR/cache" MPLCONFIGDIR="$JOB_DIR/cache/mpl"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$JOB_DIR/results"
sha256sum -c SOURCE-SHA256SUMS
cat SOURCE_COMMIT.txt
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}'); print('jax_x64='+str(jax.config.x64_enabled)); sys.exit(0 if b=='gpu' and jax.config.x64_enabled else 42)"
printf 'cpu_model='; lscpu | sed -n 's/^Model name:[[:space:]]*//p'; nvidia-smi --query-gpu=name,uuid --format=csv; hostname
"$PY" install_deps.py --wheel-dir "$JOB_DIR/wheels" --target "$JOB_DIR/deps"
DEPS_SITE=$("$PY" -c "import sysconfig; print(sysconfig.get_path('purelib', vars={'base':'$JOB_DIR/deps','platbase':'$JOB_DIR/deps'}))")
export PYTHONPATH="$DEPS_SITE:$JOB_DIR:$JOB_DIR/vendor${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$JOB_DIR/deps/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$PY" -c "import ngsolve,jax; print('ngsolve', ngsolve.__version__, 'jax', jax.default_backend())"
bash "$1"
echo "task_completed"
