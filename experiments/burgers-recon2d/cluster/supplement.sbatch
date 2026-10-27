#!/usr/bin/env bash
#SBATCH --job-name=b2dr_inputplus
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
set -euo pipefail
: "${JOB_ROOT:?unique directory}"; : "${SOURCE_COMMIT:?frozen source}"; : "${EXPECTED_STAGE_SHA256:?reviewed manifest}"; : "${PREP_ROOT:?original preparation output}"; : "${EXPECTED_PREP_MANIFEST_SHA256:?reviewed source data manifest}"
cd "$JOB_ROOT"
test "$(sha256sum STAGED.sha256 | cut -d' ' -f1)" = "$EXPECTED_STAGE_SHA256"
sha256sum --check STAGED.sha256
export SEED=0 N_TRAIN=512 N_VAL=64 JAX_DEFAULT_MATMUL_PRECISION=highest JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}'); sys.exit(0 if b=='gpu' else 42)"
"$PY" supplement.py --data "$PREP_ROOT" --out output
