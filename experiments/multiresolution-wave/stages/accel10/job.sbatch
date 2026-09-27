#!/bin/bash
#SBATCH --job-name=ctol_mr_wave_accel10
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/mr_wave2d_20260907/accel10/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/mr_wave2d_20260907/accel10/logs/%j.err
set -euo pipefail
TASK_ROOT=/cluster/tufts/paralab/tawal01/mr_wave2d_20260907/accel10
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_DEFAULT_MATMUL_PRECISION=highest
export JAX_ENABLE_X64=1
export OPENBLAS_NUM_THREADS=8
export OMP_NUM_THREADS=8
export COMMIT=452cbabf21404a32300b93747e1254117fefa62f
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
"$PY" -u code/multiresolution-wave/test_pilot.py
"$PY" -u code/multiresolution-wave/test_dynamics.py

"$PY" -u code/multiresolution-wave/acceleration_replay.py --config code/multiresolution-wave/nested40-config.json --inputs in --out out/pilot
