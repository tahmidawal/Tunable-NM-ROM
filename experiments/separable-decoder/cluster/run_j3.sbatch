#!/bin/bash
#SBATCH --job-name=sd1024_j3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=04:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/sepdec_n1024/j3/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/sepdec_n1024/j3/logs/%j.err

# j3: Burgers N=1024 ACCURACY push. j2 (2826213) showed err 1.17e-1 (K16),
# coverage/IC-limited: 104 trajectories, 2048 states, R=4K. Here: 168
# trajectories, 3072 states, more capacity (R=8K for K16; K=24 f(K) point),
# longer training, and a wider IC multi-start.
set -euo pipefail
ROOT=/cluster/tufts/paralab/tawal01/sepdec_n1024/j3
VENV=/cluster/tufts/paralab/tawal01/ae-research/venv
PY="$VENV/bin/python"
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90

echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}', flush=True); sys.exit(0 if b=='gpu' else 42)"

cd "$ROOT/code"
run () {  # run <K> <R>
  local k=$1 r=$2
  echo "=== burgers N=1024 K=$k R=$r (accuracy push) ==="
  N=1024 N_TRAIN=160 N_VAL=8 K=$k R=$r M=$((4*k)) MQ=$((16*k)) STEPS=60000 \
    MAX_SNAPS=3072 N_TEST=4 TR_FACTOR=0.01 \
    REPS=5 FOM_REPS=3 PAIR_REPS=3 ICF_STARTS=32 ICF_ITERS=60 \
    NEWTON_TOLS=1e-2,1e-3,1e-4 \
    OUT="$ROOT/out/sep_burgers_N1024_K${k}_R${r}.json" \
    CKPT="$ROOT/out/sep_burgers_N1024_K${k}_R${r}.pkl" \
    "$PY" sep_burgers.py
}

run 16 128
run 24 96

cd "$ROOT/out" && sha256sum *.json *.pkl > RESULTS.sha256
echo ALL-DONE
