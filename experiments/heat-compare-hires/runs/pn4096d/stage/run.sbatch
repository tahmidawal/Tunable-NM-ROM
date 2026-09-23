#!/bin/bash
#SBATCH --job-name=hcmp_pn4096d
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1

#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=07:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/hcmp_20260923/pn4096d/job.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/hcmp_20260923/pn4096d/job.err
set -euo pipefail
ROOT=/cluster/tufts/paralab/tawal01/hcmp_20260923/pn4096d
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
source /cluster/tufts/paralab/tawal01/ae-research/venv/bin/activate
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 TMPDIR="$ROOT/tmp" XDG_CACHE_HOME="$ROOT/cache"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$ROOT/out"
cd "$ROOT"; sha256sum -c SOURCE.sha256 > /dev/null
export SOURCE_COMMIT=$(cat COMMIT.txt); echo "source_commit=$SOURCE_COMMIT job_id=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv
df -h /cluster/tufts/paralab | tail -1
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
cd code/heat-compare-hires
set +e
PY=$PY OUT=$ROOT/out bash -c '"$PY" panel.py --config configs/pn4096.json --out "$OUT"'; STATUS=$?
set -e
cd "$ROOT"; find out -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo "run_exit=$STATUS"; exit $STATUS
