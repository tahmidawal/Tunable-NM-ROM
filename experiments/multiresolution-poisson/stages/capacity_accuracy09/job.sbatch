#!/bin/bash
#SBATCH --job-name=ctol_mr_poisson_capacity_accuracy09
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/mr_poisson2d_20260907/capacity_accuracy09/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/mr_poisson2d_20260907/capacity_accuracy09/logs/%j.err
set -euo pipefail
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_DEFAULT_MATMUL_PRECISION=highest
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=8
export OMP_NUM_THREADS=8
export COMMIT=f3e3c21a440a31eb97b06fdb9ab9951662e17935
cd /cluster/tufts/paralab/tawal01/mr_poisson2d_20260907/capacity_accuracy09
sha256sum -c MANIFEST.sha256 --quiet
finish() {
 task_exit=$?
 printf '%s\n' "$task_exit" > EXIT_CODE
 find out -type f -print0 | sort -z | xargs -0 -r sha256sum > RESULTS.sha256
 exit "$task_exit"
}
trap finish EXIT
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
"$PY" -u code/test_staged_training.py
"$PY" -u code/test_capacity_training.py
"$PY" -u code/test_capacity_online.py
"$PY" -u code/capacity_accuracy.py --config code/config-capacity-accuracy.json --checkpoint in/model.pkl --out out/smoke --smoke
"$PY" -u code/capacity_accuracy.py --config code/config-capacity-accuracy.json --checkpoint in/model.pkl --out out/pilot
