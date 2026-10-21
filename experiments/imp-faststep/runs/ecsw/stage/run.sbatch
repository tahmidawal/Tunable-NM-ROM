#!/bin/bash
#SBATCH --job-name=ifs_ecsw
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1

#SBATCH --exclude=pax007
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=01:15:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/imp-faststep/ecsw/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/imp-faststep/ecsw/logs/%j.err
set -euo pipefail
TASK_ROOT=/cluster/tufts/paralab/tawal01/imp-faststep/ecsw
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
source /cluster/tufts/paralab/tawal01/ae-research/venv/bin/activate
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export JCPREF_STORE=/cluster/tufts/paralab/tawal01/jcp-ref/store
export XDG_CACHE_HOME="$TASK_ROOT/cache" TMPDIR="$TASK_ROOT/tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TASK_ROOT/output"
cd "$TASK_ROOT"
sha256sum -c MANIFEST.sha256 --quiet
export SOURCE_COMMIT=$(cat COMMIT.txt)
echo "host=$(hostname) source_commit=$SOURCE_COMMIT"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
df -h /cluster/tufts/paralab | tail -1
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
cd experiments/imp-faststep
echo "STEP ecswjob configs/ecsw.json $(date -u +%FT%TZ)"
set +e
timeout --signal=TERM --kill-after=30 $(( 75 * 60 - 240 )) "$PY" ecswjob.py --config configs/ecsw.json --out "$TASK_ROOT/output/ecswjob_ecsw"
rc=$?
echo "driver_exit=$rc"
set -e
cd "$TASK_ROOT"
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
if [ $rc -eq 0 ] && [ ! -f "output/ecswjob_ecsw/COMPLETE" ]; then echo "no COMPLETE marker"; rc=5; fi
echo "ALL-DONE rc=$rc"
exit $rc
