#!/bin/bash
#SBATCH --job-name=hbank_h3d256
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=10:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/hbank_20260923/h3d256/job.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/hbank_20260923/h3d256/job.err
set -euo pipefail
ROOT=/cluster/tufts/paralab/tawal01/hbank_20260923/h3d256
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
source /cluster/tufts/paralab/tawal01/ae-research/venv/bin/activate
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 TMPDIR="$ROOT/tmp" XDG_CACHE_HOME="$ROOT/cache"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$ROOT/out"
cd "$ROOT"; sha256sum -c SOURCE.sha256 > /dev/null
export SOURCE_COMMIT=$(cat COMMIT.txt); echo "source_commit=$SOURCE_COMMIT job_id=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv
df -h /cluster/tufts/paralab | tail -1
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
cd code
STATUS=0
set +e
"$PY" hbk_run.py --config configs/h3d256.json --out "$ROOT/out/panel" || STATUS=1
"$PY" hbk_audit_np.py "$ROOT/out/panel" > "$ROOT/out/audit_np.json" 2> "$ROOT/out/audit_np.err" || STATUS=2
set -e
cd "$ROOT"
echo "run_exit=$STATUS" > out/EXIT_STATUS.txt; cp job.out out/job.out.copy 2>/dev/null || true
find out -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo "run_exit=$STATUS"; exit $STATUS
