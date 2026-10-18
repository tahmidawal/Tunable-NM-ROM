#!/bin/bash
#SBATCH --job-name=hrb_t2d256a
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100-80G
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/jcp-hrb/t2d256a/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/jcp-hrb/t2d256a/logs/%j.err
set -euo pipefail
TASK_ROOT=/cluster/tufts/paralab/tawal01/jcp-hrb/t2d256a
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
source /cluster/tufts/paralab/tawal01/ae-research/venv/bin/activate
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
unset JCPREF_META
export JCPREF_STORE=/cluster/tufts/paralab/tawal01/jcp-ref/store
export XDG_CACHE_HOME="$TASK_ROOT/cache" TMPDIR="$TASK_ROOT/tmp" CUDA_CACHE_PATH="$TASK_ROOT/cache/cuda"
export JAX_COMPILATION_CACHE_DIR="$TASK_ROOT/cache/jax" PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$CUDA_CACHE_PATH" "$JAX_COMPILATION_CACHE_DIR"
cd "$TASK_ROOT"
sha256sum -c MANIFEST.sha256 --quiet
echo "host=$(hostname) source_commit=$(cat COMMIT.txt)"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
df -h /cluster/tufts/paralab | tail -1
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
cd experiments/jcp-hr-baselines
# hard driver deadline: 120 min wall minus 4 min reserved for checksumming the (per-phase checkpointed) evidence
set +e
timeout --signal=TERM --kill-after=30 $(( 120 * 60 - 240 )) "$PY" tune.py configs/tune_2d256.json "$TASK_ROOT/output"
echo "driver_exit=$?"
set -e
cd "$TASK_ROOT"
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo ALL-DONE   # job completion only; acceptance is smoke_accept.py's ALL (offline)
