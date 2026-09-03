#!/bin/bash
#SBATCH --job-name=wav2d_n128_ref
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/wav2d/n128_ref/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/wav2d/n128_ref/logs/%j.err

# Wave 2D mechanism cell (2026-09-03), N=128 BC=ref: phase 2 (data from seed, bank R=64, heads sup/auto/auto+vc
# at 40k steps, gates D0-D2 G0a/b) then phase 3 (ROM gates W0-W7, G0c, arms A and C, RS 8/20/40, 4T horizon).
set -euo pipefail
ROOT=/cluster/tufts/paralab/tawal01/wav2d/n128_ref
VENV=/cluster/tufts/paralab/tawal01/ae-research/venv
PY="$VENV/bin/python"
export JAX_DEFAULT_MATMUL_PRECISION=highest
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}', flush=True); sys.exit(0 if b=='gpu' else 42)"
( cd "$ROOT" && sha256sum -c MANIFEST.sha256 --quiet && echo "stage MANIFEST OK" )
echo "commit=$(cat $ROOT/COMMIT.txt)"
cd "$ROOT/code"
"$PY" wav2d_head_gates.py N=128 BC=ref R=64 STEPS=40000 OUT="$ROOT/out" CACHE="$ROOT/cache"
"$PY" wav2d_rom_gates.py  N=128 BC=ref R=64 STEPS=40000 M=64 RS=8,20,40 OUT="$ROOT/out" CACHE="$ROOT/cache"
cd "$ROOT/out" && sha256sum *.json > RESULTS.sha256
echo ALL-DONE
