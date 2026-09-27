#!/bin/bash
#SBATCH --job-name=sepdec512_j4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/sepdec_n512/j4/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/sepdec_n512/j4/logs/%j.err

set -euo pipefail
ROOT=/cluster/tufts/paralab/tawal01/sepdec_n512/j4
VENV=/cluster/tufts/paralab/tawal01/ae-research/venv
PY="$VENV/bin/python"
export JAX_DEFAULT_MATMUL_PRECISION=highest

echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
# mandatory GPU preflight: abort 42 on silent CPU fallback
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}', flush=True); sys.exit(0 if b=='gpu' else 42)"

cd "$ROOT" && sha256sum -c MANIFEST.sha256 --quiet && echo "manifest OK"
mkdir -p "$ROOT/out"
cd "$ROOT/code"

# Poisson N=512 refinement (j1: err==oracle ~3e-2 plateau for K>=16, so the
# constraint is decoder capacity/training, not K): R=8K + n_ff=256 + 150k
# steps at K=16/24, plus a pure ff_scale=8 probe against the j1 K16 cell.
N=512 CELLS="16:128:256:4.0:150000,24:192:256:4.0:150000,16:64:128:8.0:100000" \
  NFF=128 FF_SCALE=4.0 STEPS=100000 \
  N_TEST=16 TIME_REPS=7 TR_FACTOR=1.0 GN_ITERS=60 \
  OUT_DIR="$ROOT/out" "$PY" sep_poisson.py

cd "$ROOT/out" && sha256sum *.json *.pkl > RESULTS.sha256
echo ALL-DONE
