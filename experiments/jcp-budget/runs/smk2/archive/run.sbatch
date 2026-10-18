#!/bin/bash
#SBATCH --job-name=jb_smk2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1

#SBATCH --exclude=pax007
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=02:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/jcp-budget/smk2/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/jcp-budget/smk2/logs/%j.err
set -euo pipefail
TASK_ROOT=/cluster/tufts/paralab/tawal01/jcp-budget/smk2
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
cd experiments/jcp-budget
echo "STEP ref/refrun configs/smk2_ref.json $(date -u +%FT%TZ)"
"$PY" ref/refrun.py --config configs/smk2_ref.json --out "$TASK_ROOT/output/ref_refrun_smk2_ref"
echo "STEP ref/refrun configs/smk2_x1.json $(date -u +%FT%TZ)"
"$PY" ref/refrun.py --config configs/smk2_x1.json --out "$TASK_ROOT/output/ref_refrun_smk2_x1"
echo "STEP b3run configs/smk2_b3.json $(date -u +%FT%TZ)"
"$PY" b3run.py --config configs/smk2_b3.json --out "$TASK_ROOT/output/b3run_smk2_b3"
cd "$TASK_ROOT"
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo ALL-DONE
