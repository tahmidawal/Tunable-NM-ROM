#!/bin/bash
#SBATCH --job-name=sepdec_r1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/sepdec/sepdec_r1/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/sepdec/sepdec_r1/logs/%j.err

set -euo pipefail
ROOT=/cluster/tufts/paralab/tawal01/sepdec/sepdec_r1
VENV=/cluster/tufts/paralab/tawal01/ae-research/venv
PY="$VENV/bin/python"
export JAX_DEFAULT_MATMUL_PRECISION=highest

echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
# mandatory GPU preflight: abort 42 on silent CPU fallback
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}', flush=True); sys.exit(0 if b=='gpu' else 42)"

cd "$ROOT/code"
run () {  # run <pde> <K> <R>
  local pde=$1 k=$2 r=$3
  echo "=== $pde K=$k R=$r ==="
  N=64 K=$k R=$r M=$((4*k)) MQ=$((16*k)) STEPS=40000 \
    OUT="$ROOT/out/sep_${pde}_K${k}_R${r}.json" \
    CKPT="$ROOT/out/sep_${pde}_N64_K${k}_R${r}.pkl" \
    "$PY" "sep_${pde}.py"
}

# Poisson: N_TEST=16, incumbent trust factor 1.0; Burgers: 4 trajectories, 0.01
N_TEST=16 TR_FACTOR=1.0  run poisson 8  32
N_TEST=16 TR_FACTOR=1.0  run poisson 16 64
N_TEST=4  TR_FACTOR=0.01 run burgers 8  32
N_TEST=4  TR_FACTOR=0.01 run burgers 16 64

cd "$ROOT/out" && sha256sum *.json *.pkl > RESULTS.sha256
echo ALL-DONE
