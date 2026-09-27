#!/bin/bash
#SBATCH --job-name=ctol_nop_pilot01
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100-80G
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/no_poisson_20260914/pilot01/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/no_poisson_20260914/pilot01/logs/%j.err
set -euo pipefail
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_DEFAULT_MATMUL_PRECISION=highest
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=8
export OMP_NUM_THREADS=8
export CAMPAIGN_SOURCE_COMMIT=50098e583fdbb4384653d0bdc6a3885a7d9832a8
export MPLCONFIGDIR=/cluster/tufts/paralab/tawal01/no_poisson_20260914/pilot01/cache/matplotlib
export XDG_CACHE_HOME=/cluster/tufts/paralab/tawal01/no_poisson_20260914/pilot01/cache
cd /cluster/tufts/paralab/tawal01/no_poisson_20260914/pilot01
sha256sum -c MANIFEST.sha256 --quiet
finish() {
 status=$?
 echo "$status" > EXIT_CODE
 find out -type f -print0 | sort -z | xargs -0 -r sha256sum > RESULTS.sha256
 exit "$status"
}
trap finish EXIT
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
"$PY" -u source/experiments/neural-operator-poisson/dataset.py calibrate --output out/calibration
"$PY" -u source/experiments/neural-operator-poisson/dataset.py generate --output out/train --split train --count 128 --intervals 256 --calibration out/calibration/calibration.json
"$PY" -u source/experiments/neural-operator-poisson/dataset.py generate --output out/validation --split validation --count 32 --intervals 256 --calibration out/calibration/calibration.json --include-reference
"$PY" -u source/experiments/neural-operator-poisson/diagnose.py --index out/validation/index.json --output out/diagnosis
