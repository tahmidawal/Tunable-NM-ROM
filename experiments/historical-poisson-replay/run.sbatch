#!/bin/bash
#SBATCH --job-name=mr_pois_hist02
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/mr_poisson2d_20260907/historical_replay02/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/mr_poisson2d_20260907/historical_replay02/logs/%j.err
set -euo pipefail
ROOT=/cluster/tufts/paralab/tawal01/mr_poisson2d_20260907/historical_replay02
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_DEFAULT_MATMUL_PRECISION=highest
export JAX_ENABLE_X64=True
export OPENBLAS_NUM_THREADS=8
export OMP_NUM_THREADS=8
export XLA_PYTHON_CLIENT_PREALLOCATE=false
echo "host=$(hostname)"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}', flush=True); sys.exit(0 if b=='gpu' else 42)"
cd "$ROOT"
sha256sum -c MANIFEST.sha256 --quiet
export OUT="$ROOT/out"
export MESHES=1024,128,256,512
export N_SRC=16 REPS=12 WARM=3 GATE_NZ=32
set +e
"$PY" replay.py
code=$?
set -e
echo "$code" > EXIT_CODE
if [ "$code" -ne 0 ]; then exit "$code"; fi
find out -type f -exec sha256sum {} + > RESULTS.sha256
echo ALL-DONE
