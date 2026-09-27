#!/bin/bash
#SBATCH --job-name=ctol_mr_wave_iterative05
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/mr_wave2d_20260907/iterative05/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/mr_wave2d_20260907/iterative05/logs/%j.err
set -euo pipefail
TASK_ROOT=/cluster/tufts/paralab/tawal01/mr_wave2d_20260907/iterative05
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_DEFAULT_MATMUL_PRECISION=highest
export JAX_ENABLE_X64=1
export OPENBLAS_NUM_THREADS=8
export OMP_NUM_THREADS=8
export COMMIT=e1d377928913fc94c06c6052459a4ab4847c1eb1
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

"$PY" -u code/multiresolution-wave/iterative_replay.py --config code/multiresolution-wave/iterative-config.json --inputs in --out out/pilot
