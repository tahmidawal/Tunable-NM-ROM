#!/bin/bash
#SBATCH --job-name=sd1024_j2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=04:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/sepdec_n1024/j2/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/sepdec_n1024/j2/logs/%j.err

# j2: Burgers N=1024 feasibility (K=16 R=64 first, then K=8 R=32 if time
# allows; each cell saves its JSON incrementally so a timeout loses nothing
# already measured).  96+8 training trajectories (memory bound: ~44 GB host
# per full state array), 2048 training states.
set -euo pipefail
ROOT=/cluster/tufts/paralab/tawal01/sepdec_n1024/j2
VENV=/cluster/tufts/paralab/tawal01/ae-research/venv
PY="$VENV/bin/python"
export JAX_DEFAULT_MATMUL_PRECISION=highest

echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}', flush=True); sys.exit(0 if b=='gpu' else 42)"

cd "$ROOT/code"
run () {  # run <K> <R>
  local k=$1 r=$2
  echo "=== burgers N=1024 K=$k R=$r ==="
  N=1024 N_TRAIN=96 N_VAL=8 K=$k R=$r M=$((4*k)) MQ=$((16*k)) STEPS=40000 \
    MAX_SNAPS=2048 N_TEST=4 TR_FACTOR=0.01 \
    REPS=5 FOM_REPS=3 PAIR_REPS=3 ICF_STARTS=16 ICF_ITERS=40 \
    NEWTON_TOLS=1e-2,1e-3,1e-4 \
    OUT="$ROOT/out/sep_burgers_N1024_K${k}_R${r}.json" \
    CKPT="$ROOT/out/sep_burgers_N1024_K${k}_R${r}.pkl" \
    "$PY" sep_burgers.py
}

run 16 64
run 8  32

cd "$ROOT/out" && sha256sum *.json *.pkl > RESULTS.sha256
echo ALL-DONE
