#!/bin/bash
#SBATCH --job-name=bqt_spectral01
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100-80G
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=30
#SBATCH --output=/cluster/tufts/paralab/tawal01/burgers-quadtree-20261117/spectral01/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/burgers-quadtree-20261117/spectral01/logs/%j.err
set -euo pipefail
source /cluster/tufts/paralab/tawal01/ae-research/venv/bin/activate
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export SOURCE_COMMIT=a18a62b6b43fe40eec793ab9fec6a1bb8b7073bd
export XDG_CACHE_HOME=/cluster/tufts/paralab/tawal01/burgers-quadtree-20261117/spectral01/cache TMPDIR=/cluster/tufts/paralab/tawal01/burgers-quadtree-20261117/spectral01/tmp
mkdir -p "$XDG_CACHE_HOME" "$TMPDIR"
cd /cluster/tufts/paralab/tawal01/burgers-quadtree-20261117/spectral01
sha256sum -c MANIFEST.sha256 --quiet
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
$PY -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
cd experiments/burgers-quadtree
$PY spectral_cost.py --out /cluster/tufts/paralab/tawal01/burgers-quadtree-20261117/spectral01/output
cd /cluster/tufts/paralab/tawal01/burgers-quadtree-20261117/spectral01
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
