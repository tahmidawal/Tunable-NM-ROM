#!/bin/bash
#SBATCH --job-name=br3_val01
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=00:50:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/b3d-recon-20261027/val01/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/b3d-recon-20261027/val01/logs/%j.err
set -euo pipefail
TASK_ROOT=/cluster/tufts/paralab/tawal01/b3d-recon-20261027/val01
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export XDG_CACHE_HOME="$TASK_ROOT/cache" TMPDIR="$TASK_ROOT/tmp"
export JCPREF_STORE=/cluster/tufts/paralab/tawal01/jcp-ref/store
export JCPREF_META="$JCPREF_STORE"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TASK_ROOT/output"
cd "$TASK_ROOT"
sha256sum -c MANIFEST.sha256 --quiet
export SOURCE_COMMIT=$(cat COMMIT.txt)
echo "host=$(hostname) source_commit=$SOURCE_COMMIT"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
df -h /cluster/tufts/paralab
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
"$PY" -u experiments/burgers-recon3d/vendor/check_core.py && "$PY" -u experiments/burgers-recon3d/screen3d.py --phase val --prepared /cluster/tufts/paralab/tawal01/b3d-recon-20261027/prep01/output --out "$TASK_ROOT/output" && "$PY" -u experiments/burgers-recon3d/audit_artifacts.py "$TASK_ROOT/output" && "$PY" -u experiments/burgers-recon3d/export3d.py --input "$TASK_ROOT/output"
find output -type f ! -path 'output/regenerable/*' -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo ALL-DONE
