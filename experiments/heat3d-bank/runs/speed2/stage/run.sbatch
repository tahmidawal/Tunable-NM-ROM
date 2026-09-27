#!/bin/bash
#SBATCH --job-name=h3db_speed2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=06:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/h3dbank_20260921/speed2/job.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/h3dbank_20260921/speed2/job.err
set -euo pipefail
ROOT=/cluster/tufts/paralab/tawal01/h3dbank_20260921/speed2
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
cd code/heat3d-bank
STATUS=0
set +e
true || STATUS=1
if [ $STATUS -eq 0 ]; then
"$PY" run.py --config configs/speed2_vp_R256_K16.json --out "$ROOT/out/speed2_vp_R256_K16" || STATUS=1

fi
set -e
cd "$ROOT"; [ "-" != "-" ] && [ -d code/heat3d-bank/inputs/- ] && cp -r code/heat3d-bank/inputs/- out/trained_-
echo "run_exit=$STATUS" > out/EXIT_STATUS.txt; cp job.out out/job.out.copy 2>/dev/null || true
find out -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo "run_exit=$STATUS"; exit $STATUS
