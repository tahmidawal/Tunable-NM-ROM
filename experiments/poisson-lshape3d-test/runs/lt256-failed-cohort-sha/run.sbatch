#!/bin/bash
#SBATCH --job-name=pl3t_lt256
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100-80G
#SBATCH --exclude=pax007
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/pl3test_20260925/lt256/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/pl3test_20260925/lt256/logs/%j.err
set -euo pipefail
TASK_ROOT=/cluster/tufts/paralab/tawal01/pl3test_20260925/lt256
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export XDG_CACHE_HOME="$TASK_ROOT/cache" MPLCONFIGDIR="$TASK_ROOT/cache/matplotlib"
export TMPDIR="$TASK_ROOT/tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
cd "$TASK_ROOT"
sha256sum -c MANIFEST.sha256 --quiet
export SOURCE_COMMIT=$(cat COMMIT.txt)
echo "host=$(hostname) source_commit=$SOURCE_COMMIT"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
df -h /cluster/tufts/paralab | tail -1
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
# ---- field storage (poisson-lshape3d-test): saved fields go to node-local /tmp when it has room (tmpfs on the
# H200 nodes, charged to this job's memory, hence the larger --mem), else to the share if it keeps >= 100 GB free,
# else abort (exit 43) before any work. The audit deletes them; the EXIT trap removes the local copy in any case.
NEED_GB=1
mkdir -p "$TASK_ROOT/output"
LOCAL_FIELDS=/tmp/pl3t_lt256_${SLURM_JOB_ID}
AVAIL_LOCAL=$(df --output=avail -BG /tmp | tail -1 | tr -dc 0-9)
AVAIL_SHARE=$(df --output=avail -BG /cluster/tufts/paralab | tail -1 | tr -dc 0-9)
echo "fields_need_gb=$NEED_GB avail_local_gb=$AVAIL_LOCAL avail_share_gb=$AVAIL_SHARE"
if [ "$AVAIL_LOCAL" -ge $((NEED_GB * 11 / 10 + 10)) ]; then
  mkdir -p "$LOCAL_FIELDS"; trap 'rm -rf "$LOCAL_FIELDS"' EXIT
  ln -s "$LOCAL_FIELDS" "$TASK_ROOT/output/fields"; echo "fields_store=local $LOCAL_FIELDS"
elif [ "$AVAIL_SHARE" -ge $((NEED_GB + 100)) ]; then
  mkdir -p "$TASK_ROOT/output/fields"; echo "fields_store=share"
else
  echo "fields_store=NONE (need $NEED_GB GB)"; exit 43
fi
cd "$TASK_ROOT/code"
"$PY" plt_lshape.py --config config-lshape-256-test.json --out ../output
"$PY" pbk3_audit_np.py ../output --delete-fields
cd "$TASK_ROOT"
if [ -L output/fields ]; then rm output/fields; rm -rf "$LOCAL_FIELDS"; else rmdir output/fields 2>/dev/null || true; fi
rm -rf cache tmp
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo ALL-DONE
