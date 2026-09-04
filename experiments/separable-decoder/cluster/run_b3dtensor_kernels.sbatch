#!/bin/bash
#SBATCH --job-name=b3d_kernels
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100-80G
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/b3dtensor/kernels/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/b3dtensor/kernels/logs/%j.err
set -euo pipefail
ROOT=/cluster/tufts/paralab/tawal01/b3dtensor/kernels
VENV=/cluster/tufts/paralab/tawal01/ae-research/venv
PY="$VENV/bin/python"
export JAX_DEFAULT_MATMUL_PRECISION=highest
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}', flush=True); sys.exit(0 if b=='gpu' else 42)"
( cd "$ROOT" && sha256sum -c MANIFEST.sha256 --quiet && echo "stage MANIFEST OK" )
export COMMIT=$(cat "$ROOT/COMMIT.txt")
echo "commit=$COMMIT"
cd "$ROOT/code"
env TABLE_DIR="$ROOT/code/runs/b3dtensor/tables" KERNELS="$ROOT/in/sep_b3d_tensor_n33_kernel.npz,$ROOT/in/sep_b3d_tensor_n65_kernel.npz,$ROOT/in/sep_b3d_tensor_n129_kernel.npz" TIME_REPS=5 BURN=2 FLAT_MAX_RATIO=1.25 OUT="$ROOT/out/sep_b3d_kernels.json" "$PY" sep_b3d_kernels.py
cd "$ROOT/out" && sha256sum $(ls *.json *.pkl *.npz 2>/dev/null) > RESULTS.sha256
echo ALL-DONE
