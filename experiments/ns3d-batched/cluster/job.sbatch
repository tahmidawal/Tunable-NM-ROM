#!/bin/bash
# Usage (from submit.sh): sbatch ... job.sbatch [args to batched.py...]
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
set -euo pipefail
TASK_ROOT="$PWD"
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest XLA_PYTHON_CLIENT_MEM_FRACTION=0.94
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export XDG_CACHE_HOME="$TASK_ROOT/cache" MPLCONFIGDIR="$TASK_ROOT/cache/matplotlib"
export TMPDIR="$TASK_ROOT/tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TASK_ROOT/output"
echo "host=$(hostname) source_commit=$(cat COMMIT.txt) job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
set +e
"$PY" experiments/ns3d-batched/batched.py --out "$TASK_ROOT" "$@"
RC=$?
set -e
echo "RUN_EXIT=$RC"
exit "$RC"
