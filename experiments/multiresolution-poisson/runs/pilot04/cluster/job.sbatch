#!/bin/bash
#SBATCH --job-name=ctol_mr_poisson_pilot04
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/mr_poisson2d_20260907/pilot04/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/mr_poisson2d_20260907/pilot04/logs/%j.err
set -euo pipefail
TASK_ROOT=/cluster/tufts/paralab/tawal01/mr_poisson2d_20260907/pilot04
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_DEFAULT_MATMUL_PRECISION=highest
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=8
export OMP_NUM_THREADS=8
export COMMIT=0ed8384c30ff150d0f25adfe1dead41a8a739c26
cd "$TASK_ROOT"
sha256sum -c MANIFEST.sha256 --quiet
finish() {
  task_exit=$?
  cd "$TASK_ROOT"
  printf '%s\n' "$task_exit" > EXIT_CODE
  find out -type f -print0 | sort -z | xargs -0 -r sha256sum > RESULTS.sha256
  exit "$task_exit"
}
trap finish EXIT
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
"$PY" -u code/test_core.py
"$PY" -u code/test_followup.py
"$PY" -u code/test_kernel.py
"$PY" -u code/test_training.py
"$PY" -u code/pilot04.py --config code/config04.json --checkpoint in/model.pkl --out out/pilot
