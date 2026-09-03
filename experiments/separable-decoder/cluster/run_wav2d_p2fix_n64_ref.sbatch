#!/bin/bash
#SBATCH --job-name=wav2d_p2fix_n64_ref
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/wav2d/p2fix_n64_ref/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/wav2d/p2fix_n64_ref/logs/%j.err
# Phase-2 gate re-run (corrected D1/G0a controls) on the cached heads of n64_ref; then phase 3 again on the same cache.
set -euo pipefail
ROOT=/cluster/tufts/paralab/tawal01/wav2d/p2fix_n64_ref
VENV=/cluster/tufts/paralab/tawal01/ae-research/venv
PY="$VENV/bin/python"
export JAX_DEFAULT_MATMUL_PRECISION=highest
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}', flush=True); sys.exit(0 if b=='gpu' else 42)"
( cd "$ROOT" && sha256sum -c MANIFEST.sha256 --quiet && echo "stage MANIFEST OK" )
echo "commit=$(cat $ROOT/COMMIT.txt)"
rm -rf "$ROOT/cache"; cp -r /cluster/tufts/paralab/tawal01/wav2d/n64_ref/cache "$ROOT/cache"
cd "$ROOT/code"
"$PY" wav2d_head_gates.py N=64 BC=ref R=64 STEPS=40000 OUT="$ROOT/out" CACHE="$ROOT/cache"
"$PY" wav2d_rom_gates.py  N=64 BC=ref R=64 STEPS=40000 M=64 RS=8,20,40 OUT="$ROOT/out" CACHE="$ROOT/cache"
cd "$ROOT/out" && sha256sum *.json > RESULTS.sha256
echo ALL-DONE
