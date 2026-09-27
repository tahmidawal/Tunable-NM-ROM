#!/bin/bash
#SBATCH --job-name=hires_h_h2d-wide02
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/hires_h_20260920/h2d-wide02/job.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/hires_h_20260920/h2d-wide02/job.err
set -euo pipefail
ROOT=/cluster/tufts/paralab/tawal01/hires_h_20260920/h2d-wide02
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 TMPDIR="$ROOT/tmp" XDG_CACHE_HOME="$ROOT/cache"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
cd "$ROOT"; sha256sum -c SOURCE.sha256 > /dev/null
export SOURCE_COMMIT=$(cat COMMIT.txt); echo "source_commit=$SOURCE_COMMIT job_id=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv
df -h /cluster/tufts/paralab | tail -1
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
cd code/hires-heat
set +e
"$PY" train.py --config configs/h2d-wide02.json --out inputs/wide2d
"$PY" run.py --config configs/h2d-wide02.json --out "$ROOT/out"; STATUS=$?
set -e
cd "$ROOT"; [ -n "wide2d" ] && cp -r code/hires-heat/inputs/wide2d out/trained_wide2d
find out -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo "run_exit=$STATUS"; exit $STATUS
