#!/usr/bin/env bash
#SBATCH --job-name=ctol_nsobs3d_g1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/ns-obstacle-3d-20261027/g1-smoke01/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/ns-obstacle-3d-20261027/g1-smoke01/%j.err
#SBATCH --chdir=/cluster/tufts/paralab/tawal01/ns-obstacle-3d-20261027/g1-smoke01
set -euo pipefail
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_ENABLE_X64=1 JAX_DEFAULT_MATMUL_PRECISION=highest
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR="$PWD/tmp" PIP_CACHE_DIR="$PWD/pip-cache"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"
sha256sum -c INPUTS.sha256
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
"$PY" install_deps.py --target "$PWD/deps" --metadata DEPENDENCIES.discovery.json
export PYTHONPATH="$PWD/deps"
timeout --signal=TERM 1000 "$PY" reference3d.py --config configs/g1-smoke.json --out output
