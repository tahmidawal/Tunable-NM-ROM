#!/bin/bash
#SBATCH --job-name=ctol_nop_matched01
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100-80G
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/no_poisson_20260914/matched01/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/no_poisson_20260914/matched01/logs/%j.err
set -euo pipefail
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_DEFAULT_MATMUL_PRECISION=highest
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=8
export OMP_NUM_THREADS=8
export PYTHONDONTWRITEBYTECODE=1
export CAMPAIGN_SOURCE_COMMIT=6b906b122da97e0c0fb356595c05b01161d3dac2
export MPLCONFIGDIR=/cluster/tufts/paralab/tawal01/no_poisson_20260914/matched01/cache/matplotlib
export XDG_CACHE_HOME=/cluster/tufts/paralab/tawal01/no_poisson_20260914/matched01/cache
cd /cluster/tufts/paralab/tawal01/no_poisson_20260914/matched01
sha256sum -c MANIFEST.sha256 --quiet
sha256sum -c DATA.sha256 --quiet
finish() {
 status=$?
 echo "$status" > EXIT_CODE
 find out -type f -print0 | sort -z | xargs -0 -r sha256sum > RESULTS.sha256
 exit "$status"
}
trap finish EXIT
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
for rank in 128 256; do
    "$PY" -u source/experiments/neural-operator-poisson/train_matched.py --index out/data/train/index.json --output out/r$rank --rank "$rank"
    "$PY" -u source/experiments/neural-operator-poisson/diagnose.py --index out/data/validation/index.json --output out/diagnosis_r$rank --protocol out/r$rank/evaluation-protocol.json
done
