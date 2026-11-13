#!/bin/bash -l
# One GPU allocation of the sparse-solvers lane: preflight, versions, then the task list through bench.py driver.
# Usage: sbatch --job-name=ss_<tag> --time=HH:MM:SS --constraint=<gpu feature> cluster/run_gpu.sh <tasks file> [per-task timeout s]
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
export JAX_DEFAULT_MATMUL_PRECISION=highest PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TMPDIR="$JOB_DIR/tmp" XDG_CACHE_HOME="$JOB_DIR/cache" CUPY_CACHE_DIR="$JOB_DIR/cache/cupy"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$JOB_DIR/results"
sha256sum -c --quiet SOURCE-SHA256SUMS && echo "sources verified"
cat SOURCE_COMMIT.txt; hostname; date
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
printf 'cpu_model='; lscpu | sed -n 's/^Model name:[[:space:]]*//p'
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}'); sys.exit(0 if b == 'gpu' else 42)"
# AmgX was built against the cluster's cuda/12.9.0 + gcc/12.4.0 modules; only the AmgX worker gets these paths.
export SS_AMGX_LD=$(module load gcc/12.4.0 cuda/12.9.0 >/dev/null 2>&1; echo "$LD_LIBRARY_PATH")
export SS_AMGX_ROOT=$NS/amgx-build SS_AMGX_COMMIT=$(git -C $NS/amgx-build/src rev-parse HEAD)
export SS_PYDEPS=$NS/pydeps
export PYTHONPATH="$NS/pydeps:$JOB_DIR"
NVL=$(ls -d $SP/nvidia/*/lib | tr '\n' ':')
export LD_LIBRARY_PATH="$NS/pydeps/nvidia/cu12/lib:$NVL${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$PY" -c "import cupy, numpy, scipy, pyamg, jax; print('cupy', cupy.__version__, 'numpy', numpy.__version__, 'scipy', scipy.__version__, 'pyamg', pyamg.__version__, 'jax', jax.__version__)"
echo "amgx_commit=$SS_AMGX_COMMIT"
"$PY" bench.py driver --tasks "$1" --outdir "$JOB_DIR/results" --timeout "${2:-5400}"
date
echo "task_completed"
