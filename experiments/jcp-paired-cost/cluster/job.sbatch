#!/usr/bin/env bash
#SBATCH --job-name=ctol_jpc_p01
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=job.log
#SBATCH --error=job.log
set -euo pipefail
TASK_ROOT="$SLURM_SUBMIT_DIR"
CONFIG=${1:-experiments/jcp-paired-cost/configs/pilot.json}
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export TMPDIR="$TASK_ROOT/tmp" XDG_CACHE_HOME="$TASK_ROOT/cache" CUDA_CACHE_PATH="$TASK_ROOT/cache/cuda"
export MPLCONFIGDIR="$TASK_ROOT/cache/matplotlib"
mkdir -p "$TMPDIR" "$CUDA_CACHE_PATH" "$MPLCONFIGDIR"
export SOURCE_COMMIT="$(cat "$TASK_ROOT/COMMIT.txt")"
cd "$TASK_ROOT/code"
echo "source_commit=$SOURCE_COMMIT job=$SLURM_JOB_ID host=$(hostname)"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
df -h /cluster/tufts/paralab
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
set +e
"$PY" -u experiments/jcp-paired-cost/run_paired.py --config "$CONFIG" --out "$TASK_ROOT/output"
RC=$?
set -e
echo "RUN_EXIT=$RC"
exit "$RC"
