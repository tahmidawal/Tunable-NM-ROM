#!/bin/bash
#SBATCH --job-name=burgacc_ext256
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=01:30:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/burgacc/ext256/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/burgacc/ext256/logs/%j.err

# COEFFICIENT EXTRACTION at N=256 from the frozen round-3 reference checkpoint.
# Regenerates the canonical training draw and the fresh-seed test trajectories
# with the incumbent generator (<=1e-8 truth gate), projects every state onto
# the FROZEN learned bank, and emits (Gram, c*, ||u||^2, floor^2).  Trains
# nothing and changes no discretization.  Sized to finish well before the
# 2026-08-25T06:00 maintenance window.
set -euo pipefail
ROOT=/cluster/tufts/paralab/tawal01/burgacc/ext256
VENV=/cluster/tufts/paralab/tawal01/ae-research/venv
PY="$VENV/bin/python"
export JAX_DEFAULT_MATMUL_PRECISION=highest

echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}', flush=True); sys.exit(0 if b=='gpu' else 42)"
( cd "$ROOT" && sha256sum -c MANIFEST.sha256 --quiet && echo "stage MANIFEST OK" )

cd "$ROOT/code"
env N=256 K=16 R=512 MAX_SNAPS=16384 T_EARLY=5 N_TEST=8 SEED0=0 \
  GEN_CHUNK=64 PROJ_CHUNK=256 IDENT_ROWS=64 \
  CKPT="$ROOT/in/sep_burgers_r3_N256_K16_R512.pkl" OUT_PREFIX="$ROOT/out/" \
  "$PY" sep_coeff_extract.py

cd "$ROOT/out" && sha256sum *.json *.npz > RESULTS.sha256
echo ALL-DONE
