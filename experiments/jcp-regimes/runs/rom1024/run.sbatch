#!/bin/bash
#SBATCH --job-name=jg_rom1024
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1

#SBATCH --exclude=pax007
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=01:30:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/jcp-regimes/rom1024/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/jcp-regimes/rom1024/logs/%j.err
set -euo pipefail
TASK_ROOT=/cluster/tufts/paralab/tawal01/jcp-regimes/rom1024
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
source /cluster/tufts/paralab/tawal01/ae-research/venv/bin/activate
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export XDG_CACHE_HOME="$TASK_ROOT/cache" TMPDIR="$TASK_ROOT/tmp"
export CUDA_CACHE_PATH="$TASK_ROOT/cache/cuda" PYTHONDONTWRITEBYTECODE=1    # contain CUDA / bytecode caches (job audit)
export JAX_COMPILATION_CACHE_DIR="$TASK_ROOT/cache/jax"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$CUDA_CACHE_PATH" "$JAX_COMPILATION_CACHE_DIR" "$TASK_ROOT/output"
cd "$TASK_ROOT"
sha256sum -c MANIFEST.sha256 --quiet
export SOURCE_COMMIT=$(cat COMMIT.txt)
echo "host=$(hostname) source_commit=$SOURCE_COMMIT"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
df -h /cluster/tufts/paralab | tail -1
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
cd experiments/jcp-regimes
JAX_PLATFORMS=cpu "$PY" test_regimes.py "$TASK_ROOT/output/unit_gates.json"
echo "STEP qacc configs/rom1024.json $(date -u +%FT%TZ)"
"$PY" qacc.py --config configs/rom1024.json --out "$TASK_ROOT/output/qacc_rom1024"
echo "STEP fverify configs/fverify-rom1024.json $(date -u +%FT%TZ)"
"$PY" fverify.py --config configs/fverify-rom1024.json --out "$TASK_ROOT/output/fverify_fverify-rom1024"
echo "STEP end - $(date -u +%FT%TZ)"
cd "$TASK_ROOT"
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo ALL-DONE
