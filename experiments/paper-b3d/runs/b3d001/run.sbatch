#!/bin/bash
#SBATCH --job-name=ctol_paper_b3d001
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=pax007
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/paper_b3d_20260920/b3d001/slurm.%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/paper_b3d_20260920/b3d001/slurm.%j.err
set -uo pipefail
export JAX_DEFAULT_MATMUL_PRECISION=highest
export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.80
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export SOURCE_COMMIT=1d2010aacff49a77ff7f8fe2c3221b4c6c739dc6
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
cd /cluster/tufts/paralab/tawal01/paper_b3d_20260920/b3d001/code
sha256sum -c SOURCE.sha256 || exit 43
$PY -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)" || exit $?
mkdir -p ../out
date -u +%FT%TZ > ../STARTED_UTC
$PY -u run.py --config config-pilot.json --out ../out > ../driver.log 2>&1
code=$?
if [ "$code" -eq 0 ]; then
    $PY -u audit.py ../out > ../audit.log 2>&1
    code=$?
fi
printf '%s\n' "$code" > ../EXIT_CODE
date -u +%FT%TZ > ../FINISHED_UTC
cd ..
find code out -type f ! -path '*/__pycache__/*' -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
exit "$code"
