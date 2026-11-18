#!/bin/bash
#SBATCH --job-name=bte_b1globalfull
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100-80G
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=30
#SBATCH --chdir=/cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3/b1globalfull
#SBATCH --output=/cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3/b1globalfull/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3/b1globalfull/logs/%j.err
set -euo pipefail
source /cluster/tufts/paralab/tawal01/ae-research/venv/bin/activate
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export SOURCE_COMMIT=1b499d41d07a90ce30fcaeff4f503d3c09fe4396 STUDY_NAMESPACE=/cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3
export XDG_CACHE_HOME=/cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3/b1globalfull/cache TMPDIR=/cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3/b1globalfull/tmp
mkdir -p "$XDG_CACHE_HOME" "$TMPDIR"
cd /cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3/b1globalfull
sha256sum -c SOURCE.sha256 --quiet
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
$PY -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
cd experiments/burgers-tree-embeddings
$PY extend_global_controls.py --out /cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3/b1globalfull/output --train /cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3/archive/refstrain01/output,/cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3/b1refextend/output/train --dev /cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3/archive/refsdev01/output,/cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3/b1refextend/output/dev
cd /cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3/b1globalfull
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
