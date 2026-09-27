#!/bin/bash
#SBATCH --job-name=sepdec512_j1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/sepdec_n512/j1/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/sepdec_n512/j1/logs/%j.err

set -euo pipefail
ROOT=/cluster/tufts/paralab/tawal01/sepdec_n512/j1
VENV=/cluster/tufts/paralab/tawal01/ae-research/venv
PY="$VENV/bin/python"
export JAX_DEFAULT_MATMUL_PRECISION=highest

echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
# mandatory GPU preflight: abort 42 on silent CPU fallback
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}', flush=True); sys.exit(0 if b=='gpu' else 42)"

cd "$ROOT" && sha256sum -c MANIFEST.sha256 --quiet && echo "manifest OK"
mkdir -p "$ROOT/out"
cd "$ROOT/code"

# Poisson N=512 broad cell sweep: f(K) over K in {8,16,24,32}, R=4K,
# n_ff=128, 100k train steps; balanced timing vs the CG ladder; fresh-seed
# cohort arm; data regenerated from seed in-job (once, shared by all cells).
N=512 CELLS="8:32,16:64,24:96,32:128" NFF=128 FF_SCALE=4.0 STEPS=100000 \
  N_TEST=16 TIME_REPS=7 TR_FACTOR=1.0 GN_ITERS=60 \
  OUT_DIR="$ROOT/out" "$PY" sep_poisson.py

cd "$ROOT/out" && sha256sum *.json *.pkl > RESULTS.sha256
echo ALL-DONE
