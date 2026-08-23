#!/bin/bash
#SBATCH --job-name=sepdec_n256_j4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/sepdec_n256/j4/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/sepdec_n256/j4/logs/%j.err

# N=256 Poisson refinement: best cell (K=16,R=64) at 200k steps with LOOSER
# CG rungs (5e-1,3e-1) so the ~3e-2-error regime has a strict iso-accuracy
# classical comparator; plus K=24 to densify f(K). REPS=9.
set -euo pipefail
ROOT=/cluster/tufts/paralab/tawal01/sepdec_n256/j4
VENV=/cluster/tufts/paralab/tawal01/ae-research/venv
PY="$VENV/bin/python"
export JAX_DEFAULT_MATMUL_PRECISION=highest

echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}', flush=True); sys.exit(0 if b=='gpu' else 42)"

cd "$ROOT/code"
env N=256 K=16 R=64 M=64 MQ=256 STEPS=200000 N_TEST=16 TR_FACTOR=1.0 REPS=9 FRESH_SEED=777 \
  FOM_LADDER=5e-1,3e-1,1e-1,3e-2,1e-2,1e-3,1e-6 \
  OUT="$ROOT/out/sep_poisson_K16_R64_s200k.json" CKPT="$ROOT/out/sep_poisson_N256_K16_R64_s200k.pkl" \
  "$PY" sep_poisson.py
env N=256 K=24 R=96 M=96 MQ=384 STEPS=100000 N_TEST=16 TR_FACTOR=1.0 REPS=9 FRESH_SEED=777 \
  FOM_LADDER=5e-1,3e-1,1e-1,3e-2,1e-2,1e-3,1e-6 \
  OUT="$ROOT/out/sep_poisson_K24_R96.json" CKPT="$ROOT/out/sep_poisson_N256_K24_R96.pkl" \
  "$PY" sep_poisson.py

cd "$ROOT/out" && sha256sum *.json *.pkl > RESULTS.sha256
echo ALL-DONE
