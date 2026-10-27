#!/usr/bin/env bash
#SBATCH --job-name=ctol_nsobs3d_g1b
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/ns-obstacle-3d-20261027/g1-smoke02/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/ns-obstacle-3d-20261027/g1-smoke02/%j.err
#SBATCH --chdir=/cluster/tufts/paralab/tawal01/ns-obstacle-3d-20261027/g1-smoke02
set -euo pipefail
run_started_seconds=$SECONDS
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR="$PWD/tmp" PIP_CACHE_DIR="$PWD/pip-cache"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"
sha256sum -c INPUTS.sha256
timeout --signal=TERM --kill-after=10 90 "$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); print('jax_x64='+str(jax.config.x64_enabled),flush=True); assert jax.config.x64_enabled; sys.exit(0 if b=='gpu' else 42)"
timeout --signal=TERM --kill-after=10 300 "$PY" install_deps.py --target "$PWD/deps" --metadata DEPENDENCIES.discovery.json
export PYTHONPATH="$PWD/deps${PYTHONPATH:+:$PYTHONPATH}"
remaining_seconds=$((1140 - (SECONDS - run_started_seconds)))
if (( remaining_seconds < 120 )); then
  echo 'Insufficient reserved time after dependency preflight; no FE case launched' >&2
  exit 43
fi
if (( remaining_seconds > 850 )); then
  remaining_seconds=850
fi
timeout --signal=TERM --kill-after=20 "$remaining_seconds" "$PY" reference3d.py --config configs/g1-smoke02.json --out output
