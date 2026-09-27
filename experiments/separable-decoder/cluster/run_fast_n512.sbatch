#!/bin/bash
#SBATCH --job-name=fast_n512
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/b1dqf/fast_n512/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/b1dqf/fast_n512/logs/%j.err

# A100 confirmation of the optimized ROM rollout at N=512: reference vs
# optimized (MODE=ab, interleaved, parity vs the committed baseline JSON)
# plus the labeled throughput arm (MODE=thru).  No retraining: decoder and
# node sets load from the committed scale-run artifacts in in/.
set -euo pipefail
ROOT=/cluster/tufts/paralab/tawal01/b1dqf/fast_n512
VENV=/cluster/tufts/paralab/tawal01/ae-research/venv
PY="$VENV/bin/python"
export JAX_DEFAULT_MATMUL_PRECISION=highest

echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}', flush=True); sys.exit(0 if b=='gpu' else 42)"
( cd "$ROOT" && sha256sum -c MANIFEST.sha256 --quiet && echo "stage MANIFEST OK" )

cd "$ROOT/code"
env N=512 MODE=ab TIME_REPS=7 BURN=2 \
  ARMS=oracle,base_tight,nodes_tight,base_half,nodes_half \
  CKPT_CACHE="$ROOT/in/sep_b1d_scale_n512.pkl" \
  NODES_NPZ="$ROOT/in/sep_b1d_scale_n512_nodes.npz" \
  BASE_JSON="$ROOT/in/sep_b1d_scale_n512.json" \
  OUT="$ROOT/out/fast_a100_n512.json" \
  "$PY" sep_b1d_fast.py
env N=512 MODE=thru TIME_REPS=7 BURN=2 ARMS=base_tight \
  CKPT_CACHE="$ROOT/in/sep_b1d_scale_n512.pkl" \
  NODES_NPZ="$ROOT/in/sep_b1d_scale_n512_nodes.npz" \
  BASE_JSON="$ROOT/in/sep_b1d_scale_n512.json" \
  OUT="$ROOT/out/thru_a100_n512.json" \
  "$PY" sep_b1d_fast.py

cd "$ROOT/out" && sha256sum *.json > RESULTS.sha256
echo ALL-DONE
