#!/usr/bin/env bash
# One allocation of the cylinder-operators lane: GPU preflight, then the staged task script.
# Usage: sbatch --job-name=cylops_<tag> --time=HH:MM:SS cluster/run.sh <task.sh> [args]
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?must be submitted as a batch job}"
export JOB_DIR=$(pwd -P)
export NS=/cluster/tufts/paralab/tawal01/rebuttal-cyl-ops
case "$JOB_DIR" in $NS/*) ;; *) echo "bad job dir"; exit 64;; esac
export PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_DEFAULT_MATMUL_PRECISION=highest PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR="$JOB_DIR/tmp" XDG_CACHE_HOME="$JOB_DIR/cache" MPLCONFIGDIR="$JOB_DIR/cache/mpl"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$JOB_DIR/results"
sha256sum -c SOURCE-SHA256SUMS > /dev/null && echo "sources verified"
cat SOURCE_COMMIT.txt
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}'); sys.exit(0 if b == 'gpu' else 42)"
"$PY" -c "import torch,sys; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0)); sys.exit(0 if torch.cuda.is_available() else 42)"
printf 'cpu_model='; lscpu | sed -n 's/^Model name:[[:space:]]*//p'; nvidia-smi --query-gpu=name,uuid,memory.total --format=csv; hostname
export PYTHONPATH="$JOB_DIR${PYTHONPATH:+:$PYTHONPATH}"
T=$1; shift
bash "$T" "$@"
echo "task_completed"
