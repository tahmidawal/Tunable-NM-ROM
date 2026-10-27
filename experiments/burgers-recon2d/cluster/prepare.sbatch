#!/usr/bin/env bash
#SBATCH --job-name=b2dr_prepare
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
set -euo pipefail
: "${JOB_ROOT:?must set the unique absolute job directory}"
cd "$JOB_ROOT"
: "${SOURCE_COMMIT:?must identify the frozen lane commit}"
: "${EXPECTED_STAGE_SHA256:?must identify the reviewed staging manifest}"
test "$(sha256sum STAGED.sha256 | cut -d' ' -f1)" = "$EXPECTED_STAGE_SHA256"
sha256sum --check STAGED.sha256
export SEED=0 N_TRAIN=512 N_VAL=64
export JAX_DEFAULT_MATMUL_PRECISION=highest
export JAX_ENABLE_X64=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}'); sys.exit(0 if b=='gpu' else 42)"
"$PY" prepare.py --out output --phase all
