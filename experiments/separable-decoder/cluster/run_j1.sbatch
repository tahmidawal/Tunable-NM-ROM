#!/bin/bash
#SBATCH --job-name=sepdec_n128_j1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/sepdec_n128/j1/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/sepdec_n128/j1/logs/%j.err

# Job 1 of the N=128 separable-decoder scaling round: broad (K, R=6K) sweep,
# both PDEs, 60k train steps, HANDOFF-mandatory measurement rules implemented
# in sep_poisson.py / sep_burgers.py (end-to-end timing incl. IC fit, raw reps,
# balanced order, Newton-tolerance strong baseline, fresh-seed Poisson cohort).
set -euo pipefail
ROOT=/cluster/tufts/paralab/tawal01/sepdec_n128/j1
VENV=/cluster/tufts/paralab/tawal01/ae-research/venv
PY="$VENV/bin/python"
export JAX_DEFAULT_MATMUL_PRECISION=highest

echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
# mandatory GPU preflight: abort 42 on silent CPU fallback
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}', flush=True); sys.exit(0 if b=='gpu' else 42)"

cd "$ROOT/code"

pois () {  # pois <K>
  local k=$1 r=$((6*k))
  echo "=== poisson K=$k R=$r ==="
  N=128 K=$k R=$r M=$((4*k)) MQ=$((16*k)) STEPS=60000 N_FF=128 \
    N_TEST=16 REPS=7 TR_FACTOR=1.0 FRESH_SEED=1 \
    OUT="$ROOT/out/sep_poisson_K${k}_R${r}.json" \
    CKPT="$ROOT/out/sep_poisson_N128_K${k}_R${r}.pkl" \
    "$PY" sep_poisson.py
}
burg () {  # burg <K>
  local k=$1 r=$((6*k))
  echo "=== burgers K=$k R=$r ==="
  N=128 K=$k R=$r M=$((4*k)) MQ=$((16*k)) STEPS=60000 N_FF=128 \
    N_TEST=8 REPS=7 TR_FACTOR=0.01 NTOLS=3e-2,1e-2,1e-3,1e-5 \
    DATA_CACHE="$ROOT/data/burgers_n128.npz" \
    OUT="$ROOT/out/sep_burgers_K${k}_R${r}.json" \
    CKPT="$ROOT/out/sep_burgers_N128_K${k}_R${r}.pkl" \
    "$PY" sep_burgers.py
}

# primary K=16 first in each PDE so a late failure cannot cost the main cells
pois 16
pois 8
pois 24
pois 32
burg 16
burg 8
burg 24
burg 32

cd "$ROOT/out" && sha256sum *.json *.pkl > RESULTS.sha256
echo ALL-DONE
