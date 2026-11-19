#!/bin/bash
#SBATCH --job-name=bte_b7parentpolish
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100-80G
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=30
#SBATCH --chdir=/cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3/b7parentpolish
#SBATCH --output=/cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3/b7parentpolish/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3/b7parentpolish/logs/%j.err
set -euo pipefail
source /cluster/tufts/paralab/tawal01/ae-research/venv/bin/activate
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export SOURCE_COMMIT=7ee867310c09c5572664d9297cf5f38bc952fada STUDY_NAMESPACE=/cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3
export XDG_CACHE_HOME=/cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3/b7parentpolish/cache TMPDIR=/cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3/b7parentpolish/tmp
mkdir -p "$XDG_CACHE_HOME" "$TMPDIR"
cd /cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3/b7parentpolish
sha256sum -c SOURCE.sha256 --quiet
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
$PY -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
cd experiments/burgers-tree-embeddings
$PY polish_stable_parent.py --out /cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3/b7parentpolish/output
cd /cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3/b7parentpolish
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
