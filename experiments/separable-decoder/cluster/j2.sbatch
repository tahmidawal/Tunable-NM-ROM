#!/bin/bash
#SBATCH --job-name=sepdec512_j2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/sepdec_n512/j2/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/sepdec_n512/j2/logs/%j.err

set -euo pipefail
ROOT=/cluster/tufts/paralab/tawal01/sepdec_n512/j2
VENV=/cluster/tufts/paralab/tawal01/ae-research/venv
PY="$VENV/bin/python"
export JAX_DEFAULT_MATMUL_PRECISION=highest

echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}', flush=True); sys.exit(0 if b=='gpu' else 42)"

cd "$ROOT" && sha256sum -c MANIFEST.sha256 --quiet && echo "manifest OK"
mkdir -p "$ROOT/out"
cd "$ROOT/code"

# Burgers N=512: f(K) over K in {8,16,24}, R=4K, n_ff=128, 60k steps;
# END-TO-END timing (IC fit + rollout + full decode) vs truth-generator and
# the tolerance-terminated Newton ladder (full ladder timed in cell 0);
# 6 fresh-seed test trajectories; data regenerated from seed in-job (once).
# MAX_SNAPS=6144 with all t=0 states kept (memory: N=512 states are 2 MB each).
N=512 GEN_CHUNK=8 CELLS="16:64,8:32,24:96" NFF=128 FF_SCALE=4.0 STEPS=60000 \
  MAX_SNAPS=6144 N_TEST=6 TIME_REPS=5 TIME_WARM=1 TR_FACTOR=0.01 \
  NEWTON_LADDER="1e-2:1e-3,1e-4:1e-5,1e-6:1e-7" BASELINE_CELLS=0 \
  OUT_DIR="$ROOT/out" "$PY" sep_burgers.py

cd "$ROOT/out" && sha256sum *.json *.pkl > RESULTS.sha256
echo ALL-DONE
