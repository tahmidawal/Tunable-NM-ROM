#!/bin/bash
#SBATCH --job-name=ctol_modcp_wave_validation_reflective02
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/modcp_wave2d_20260910/validation_reflective02/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/modcp_wave2d_20260910/validation_reflective02/logs/%j.err
set -euo pipefail
TASK_ROOT=/cluster/tufts/paralab/tawal01/modcp_wave2d_20260910/validation_reflective02
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_DEFAULT_MATMUL_PRECISION=highest
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=8
export OMP_NUM_THREADS=8
export COMMIT=df737ba306f1edac21054939b9241f4f3f221320
cd "$TASK_ROOT"
sha256sum -c MANIFEST.sha256 --quiet
finish() {
  task_exit=$?
  cd "$TASK_ROOT"
  printf '%s\n' "$task_exit" > EXIT_CODE
  find out -type f ! -path '*/data/training_fields.npy' -print0 | sort -z | xargs -0 -r sha256sum > RESULTS.sha256
  exit "$task_exit"
}
trap finish EXIT
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
"$PY" -u code/modcp-eq/wave/test_physics.py
"$PY" -u code/modcp-eq/wave/test_weak.py
"$PY" -u code/modcp-eq/wave/compare.py --config code/modcp-eq/wave/config.json --boundary dirichlet --inputs inputs --out out/comparison --phase validation

