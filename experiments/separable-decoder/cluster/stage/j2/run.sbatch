#!/bin/bash
#SBATCH --job-name=sepdec_n128_j2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/sepdec_n128/j2/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/sepdec_n128/j2/logs/%j.err

# Job 2 of the N=128 round: accuracy push from j1's findings.
# j1 showed (a) Burgers error tracks the training floor and improves with K,
# (b) Poisson error is flat in K and rep/objective-limited, (c) EQ row p95
# grew with N -> one m=32K probe.  So: 200k steps, R=8K, MAX_SNAPS=16384.
set -euo pipefail
ROOT=/cluster/tufts/paralab/tawal01/sepdec_n128/j2
VENV=/cluster/tufts/paralab/tawal01/ae-research/venv
PY="$VENV/bin/python"
export JAX_DEFAULT_MATMUL_PRECISION=highest

echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}', flush=True); sys.exit(0 if b=='gpu' else 42)"

cd "$ROOT/code"

pois () {  # pois <K> <R> <MQ> <tag>
  local k=$1; local r=$2; local mq=$3; local tag=$4
  echo "=== poisson K=$k R=$r m=$mq $tag ==="
  N=128 K=$k R=$r M=$((4*k)) MQ=$mq STEPS=200000 N_FF=128 \
    N_TEST=16 REPS=7 TR_FACTOR=1.0 FRESH_SEED=1 \
    OUT="$ROOT/out/sep_poisson_${tag}.json" \
    CKPT="$ROOT/out/sep_poisson_N128_${tag}.pkl" \
    "$PY" sep_poisson.py
}
burg () {  # burg <K> <R> <tag>
  local k=$1; local r=$2; local tag=$3
  echo "=== burgers K=$k R=$r $tag ==="
  N=128 K=$k R=$r M=$((4*k)) MQ=$((16*k)) STEPS=200000 N_FF=128 \
    MAX_SNAPS=16384 N_TEST=8 REPS=7 TR_FACTOR=0.01 \
    NTOLS=3e-2,1e-2,1e-3,1e-5 DATA_CACHE="$ROOT/data/burgers_n128.npz" \
    OUT="$ROOT/out/sep_burgers_${tag}.json" \
    CKPT="$ROOT/out/sep_burgers_N128_${tag}.pkl" \
    "$PY" sep_burgers.py
}

FAILED=""
for c in "burg 16 128 K16_R128_S200k_sn16k" \
         "burg 32 256 K32_R256_S200k_sn16k" \
         "pois 16 128 256 K16_R128_S200k" \
         "pois 16 128 512 K16_R128_S200k_m32K" \
         "pois 32 256 512 K32_R256_S200k"; do
  if ! $c; then FAILED="$FAILED [$c]"; echo "CELL-FAILED: $c"; fi
done
if [ -n "$FAILED" ]; then echo "FAILED CELLS:$FAILED"; fi

cd "$ROOT/out" && sha256sum *.json *.pkl > RESULTS.sha256
echo ALL-DONE
