#!/bin/bash
# Usage (from submit_train.sh): sbatch ... train.sbatch <mesh> <arm>
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --time=02:30:00
set -euo pipefail
MESH="$1"; ARM="$2"
TASK_ROOT="$PWD"
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
source /cluster/tufts/paralab/tawal01/ae-research/venv/bin/activate
export JAX_DEFAULT_MATMUL_PRECISION=highest JAX_ENABLE_X64=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export XDG_CACHE_HOME="$TASK_ROOT/cache" MPLCONFIGDIR="$TASK_ROOT/cache/matplotlib" TMPDIR="$TASK_ROOT/tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TASK_ROOT/output"
echo "host=$(hostname) source_commit=$(cat COMMIT.txt) job=$SLURM_JOB_ID mesh=$MESH arm=$ARM"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
df -h /cluster/tufts/paralab | tail -1
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
set +e
"$PY" experiments/ns3d-operators/train_op.py --mesh "$MESH" --config "experiments/ns3d-operators/configs/ops/$ARM.json" --out output
RC=$?
set -e
find output -type f -print0 | sort -z | xargs -0 -r sha256sum > OUTPUTS.sha256
echo "RUN_EXIT=$RC"
exit "$RC"
