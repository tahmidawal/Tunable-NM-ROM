#!/bin/bash
#SBATCH --job-name=bqt_c3befficiency

#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100-80G
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=120
#SBATCH --output=/cluster/tufts/paralab/tawal01/burgers-quadtree-20261117/c3befficiency/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/burgers-quadtree-20261117/c3befficiency/logs/%j.err
set -euo pipefail
source /cluster/tufts/paralab/tawal01/ae-research/venv/bin/activate
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export SOURCE_COMMIT=a0f5d002efb0f1cd990addbce178f3778e8a9550
export XDG_CACHE_HOME=/cluster/tufts/paralab/tawal01/burgers-quadtree-20261117/c3befficiency/cache TMPDIR=/cluster/tufts/paralab/tawal01/burgers-quadtree-20261117/c3befficiency/tmp
mkdir -p "$XDG_CACHE_HOME" "$TMPDIR"
cd /cluster/tufts/paralab/tawal01/burgers-quadtree-20261117/c3befficiency
sha256sum -c MANIFEST.sha256 --quiet
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
$PY -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
cd experiments/burgers-quadtree
$PY c3_b_efficiency.py --out /cluster/tufts/paralab/tawal01/burgers-quadtree-20261117/c3befficiency/output
cd /cluster/tufts/paralab/tawal01/burgers-quadtree-20261117/c3befficiency
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
