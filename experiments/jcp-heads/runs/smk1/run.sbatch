#!/bin/bash
#SBATCH --job-name=jh_smk1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100-80G
#SBATCH --exclude=pax007
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/jcp-heads/smk1/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/jcp-heads/smk1/logs/%j.err
set -euo pipefail
TASK_ROOT=/cluster/tufts/paralab/tawal01/jcp-heads/smk1
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
source /cluster/tufts/paralab/tawal01/ae-research/venv/bin/activate
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export XDG_CACHE_HOME="$TASK_ROOT/cache" TMPDIR="$TASK_ROOT/tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TASK_ROOT/output"
cd "$TASK_ROOT"
sha256sum -c MANIFEST.sha256 --quiet
export SOURCE_COMMIT=$(cat COMMIT.txt)
echo "host=$(hostname) source_commit=$SOURCE_COMMIT"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
df -h /cluster/tufts/paralab | tail -1
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
cd experiments/jcp-heads
echo "STEP htrain configs/smk1train.json $(date -u +%FT%TZ)"
"$PY" htrain.py --config configs/smk1train.json --out "$TASK_ROOT/output/htrain_smk1train"
echo "STEP hacc configs/smk1acc.json $(date -u +%FT%TZ)"
"$PY" hacc.py --config configs/smk1acc.json --out "$TASK_ROOT/output/hacc_smk1acc"
echo "STEP hrule configs/smk1rule.json $(date -u +%FT%TZ)"
"$PY" hrule.py --config configs/smk1rule.json --out "$TASK_ROOT/output/hrule_smk1rule"
cd "$TASK_ROOT"
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo ALL-DONE
