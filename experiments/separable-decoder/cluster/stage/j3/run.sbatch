#!/bin/bash
#SBATCH --job-name=sepdec_n256_j3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/sepdec_n256/j3/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/sepdec_n256/j3/logs/%j.err

# N=256 Burgers f(K) extension: K in {24,32} (j2 covered {8,16}).
# STEPS=40000 (K=16 training plateaued by 40k; keeps 2 cells in budget).
set -euo pipefail
ROOT=/cluster/tufts/paralab/tawal01/sepdec_n256/j3
VENV=/cluster/tufts/paralab/tawal01/ae-research/venv
PY="$VENV/bin/python"
export JAX_DEFAULT_MATMUL_PRECISION=highest

echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}', flush=True); sys.exit(0 if b=='gpu' else 42)"

cd "$ROOT/code"
env N=256 CELLS=24:96,32:128 STEPS=40000 N_TEST=8 MAX_SNAPS=8192 TR_FACTOR=0.01 \
  REPS=5 WARM=2 IC_TOP=12 NEWTON_TOLS=1e-2,1e-3,1e-5,1e-8 LIN_FACTOR=1e-2 MAX_NEWTON=20 \
  OUT_PREFIX="$ROOT/out/" \
  "$PY" sep_burgers.py

cd "$ROOT/out" && sha256sum *.json *.pkl > RESULTS.sha256
echo ALL-DONE
