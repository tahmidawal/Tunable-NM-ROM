#!/bin/bash
#SBATCH --job-name=bqt_c1tiles

#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100-80G
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=60
#SBATCH --output=/cluster/tufts/paralab/tawal01/burgers-quadtree-20261117/c1tiles/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/burgers-quadtree-20261117/c1tiles/logs/%j.err
set -euo pipefail
source /cluster/tufts/paralab/tawal01/ae-research/venv/bin/activate
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export SOURCE_COMMIT=df43da21a35f081e207def228bed7613a33ff5c7
export XDG_CACHE_HOME=/cluster/tufts/paralab/tawal01/burgers-quadtree-20261117/c1tiles/cache TMPDIR=/cluster/tufts/paralab/tawal01/burgers-quadtree-20261117/c1tiles/tmp
mkdir -p "$XDG_CACHE_HOME" "$TMPDIR"
cd /cluster/tufts/paralab/tawal01/burgers-quadtree-20261117/c1tiles
sha256sum -c MANIFEST.sha256 --quiet
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
$PY -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
cd experiments/burgers-quadtree
$PY c1_tiles.py --out /cluster/tufts/paralab/tawal01/burgers-quadtree-20261117/c1tiles/output
cd /cluster/tufts/paralab/tawal01/burgers-quadtree-20261117/c1tiles
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
