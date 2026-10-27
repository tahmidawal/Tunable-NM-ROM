#!/usr/bin/env bash
#SBATCH --job-name=nsob2d_smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?must be submitted as reviewed batch job}"
JOB_DIR=$(pwd -P)
case "$JOB_DIR" in /cluster/tufts/paralab/tawal01/ns-obstacle-2d-20261027/*) ;; *) exit 64;; esac
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_DEFAULT_MATMUL_PRECISION=highest JAX_ENABLE_X64=true
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR="$JOB_DIR/tmp" XDG_CACHE_HOME="$JOB_DIR/cache"
export MPLCONFIGDIR="$JOB_DIR/cache/matplotlib" PIP_CACHE_DIR="$JOB_DIR/cache/pip"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$PIP_CACHE_DIR" "$JOB_DIR/results"
sha256sum -c SOURCE-SHA256SUMS
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}'); print('jax_version='+jax.__version__); print('jax_devices='+str(jax.devices())); print('jax_x64='+str(jax.config.x64_enabled)); sys.exit(0 if b=='gpu' and jax.config.x64_enabled else 42)"
printf 'cpu_model='
lscpu | sed -n 's/^Model name:[[:space:]]*//p'
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv
"$PY" dependencies.py --wheel-dir wheels --install --target "$JOB_DIR/deps"
export PYTHONPATH="$JOB_DIR/deps${PYTHONPATH:+:$PYTHONPATH}"
"$PY" -c "import ngsolve,netgen,numpy,jax; print('ngsolve='+ngsolve.__version__); print('numpy='+numpy.__version__); print('jax_after_dependency_import='+jax.__version__); print('backend_after_dependency_import='+jax.default_backend())"
"$PY" fom.py --config configs/channel-smoke.json --out results/channel
"$PY" fom.py --config configs/manufactured-smoke.json --out results/manufactured
"$PY" fom.py --config configs/dfg-smoke.json --out results/dfg
"$PY" smoke_summary.py results
