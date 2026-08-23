#!/bin/bash
#SBATCH --job-name=sepdec_n256_j1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/sepdec_n256/j1/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/sepdec_n256/j1/logs/%j.err

# N=256 Poisson sweep: f(K) at K in {8,16,32} + one capacity-push K=16 cell.
set -euo pipefail
ROOT=/cluster/tufts/paralab/tawal01/sepdec_n256/j1
VENV=/cluster/tufts/paralab/tawal01/ae-research/venv
PY="$VENV/bin/python"
export JAX_DEFAULT_MATMUL_PRECISION=highest

echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
# mandatory GPU preflight: abort 42 on silent CPU fallback
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}', flush=True); sys.exit(0 if b=='gpu' else 42)"

cd "$ROOT/code"

env N=256 K=8  R=32  M=32  MQ=128 STEPS=100000 N_TEST=16 TR_FACTOR=1.0 REPS=7 FRESH_SEED=777 \
  OUT="$ROOT/out/sep_poisson_K8_R32.json"   CKPT="$ROOT/out/sep_poisson_N256_K8_R32.pkl"   "$PY" sep_poisson.py
env N=256 K=16 R=64  M=64  MQ=256 STEPS=100000 N_TEST=16 TR_FACTOR=1.0 REPS=7 FRESH_SEED=777 \
  OUT="$ROOT/out/sep_poisson_K16_R64.json"  CKPT="$ROOT/out/sep_poisson_N256_K16_R64.pkl"  "$PY" sep_poisson.py
env N=256 K=16 R=128 M=64  MQ=256 STEPS=100000 N_TEST=16 TR_FACTOR=1.0 REPS=7 FRESH_SEED=777 \
  N_FF=128 G_HIDDEN=256 H_HIDDEN=256 \
  OUT="$ROOT/out/sep_poisson_K16_R128big.json" CKPT="$ROOT/out/sep_poisson_N256_K16_R128big.pkl" "$PY" sep_poisson.py
env N=256 K=32 R=128 M=128 MQ=512 STEPS=100000 N_TEST=16 TR_FACTOR=1.0 REPS=7 FRESH_SEED=777 \
  OUT="$ROOT/out/sep_poisson_K32_R128.json" CKPT="$ROOT/out/sep_poisson_N256_K32_R128.pkl" "$PY" sep_poisson.py

cd "$ROOT/out" && sha256sum *.json *.pkl > RESULTS.sha256
echo ALL-DONE
