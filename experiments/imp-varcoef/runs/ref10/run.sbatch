#!/bin/bash
#SBATCH --job-name=ivc_ref10
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --exclude=pax007
#SBATCH --no-requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/imp-varcoef/ref10/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/imp-varcoef/ref10/logs/%j.err
set -euo pipefail
TASK_ROOT=/cluster/tufts/paralab/tawal01/imp-varcoef/ref10
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
source /cluster/tufts/paralab/tawal01/ae-research/venv/bin/activate
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export XDG_CACHE_HOME="$TASK_ROOT/cache" TMPDIR="$TASK_ROOT/tmp"
export JCPREF_STORE=/cluster/tufts/paralab/tawal01/jcp-ref/store
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TASK_ROOT/output"
cd "$TASK_ROOT"
sha256sum -c MANIFEST.sha256 --quiet
export SOURCE_COMMIT=$(cat COMMIT.txt)
echo "host=$(hostname) source_commit=$SOURCE_COMMIT"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
df -h /cluster/tufts/paralab | tail -1
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
"$PY" -c "import jax; jax.config.update('jax_enable_x64', True); import jax.numpy as jnp, os; assert jnp.zeros(1).dtype == jnp.float64; assert os.environ['JAX_DEFAULT_MATMUL_PRECISION'] == 'highest'; print('x64+highest ok', flush=True)"
cd experiments/imp-varcoef
export JCPREF_META="$TASK_ROOT/experiments/jcp-fom2d/e0meta"
echo "STEP vref configs/ref10.json $(date -u +%FT%TZ)"
"$PY" vref.py --config configs/ref10.json --out "$TASK_ROOT/output/vref_ref10"
cd "$TASK_ROOT"
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo ALL-DONE
