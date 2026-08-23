#!/bin/bash
#SBATCH --job-name=sd1024_j1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/sepdec_n1024/j1/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/sepdec_n1024/j1/logs/%j.err

# j1: Poisson N=1024 feasibility + K/R sweep (K=8 R=32; K=16 R=64; K=16 R=128)
set -euo pipefail
ROOT=/cluster/tufts/paralab/tawal01/sepdec_n1024/j1
VENV=/cluster/tufts/paralab/tawal01/ae-research/venv
PY="$VENV/bin/python"
export JAX_DEFAULT_MATMUL_PRECISION=highest

echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}', flush=True); sys.exit(0 if b=='gpu' else 42)"

cd "$ROOT/code"
run () {  # run <K> <R>
  local k=$1 r=$2
  echo "=== poisson N=1024 K=$k R=$r ==="
  N=1024 K=$k R=$r M=$((4*k)) MQ=$((16*k)) STEPS=50000 \
    N_TEST=16 N_TEST_FRESH=16 TR_FACTOR=1.0 \
    REPS=7 PAIR_REPS=4 TAU_MAIN=1e-2 \
    OUT="$ROOT/out/sep_poisson_N1024_K${k}_R${r}.json" \
    CKPT="$ROOT/out/sep_poisson_N1024_K${k}_R${r}.pkl" \
    "$PY" sep_poisson.py
}

run 8  32
run 16 64
run 16 128

cd "$ROOT/out" && sha256sum *.json *.pkl > RESULTS.sha256
echo ALL-DONE
