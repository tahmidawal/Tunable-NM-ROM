#!/bin/bash
#SBATCH --job-name=bte_b1stableglobal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100-80G
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=20
#SBATCH --chdir=/cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3/b1stableglobal
#SBATCH --output=/cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3/b1stableglobal/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3/b1stableglobal/logs/%j.err
set -euo pipefail
source /cluster/tufts/paralab/tawal01/ae-research/venv/bin/activate
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export SOURCE_COMMIT=0bf2e9a6dfa6f0f336ef638a533db33ee7e9b277 STUDY_NAMESPACE=/cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3
export XDG_CACHE_HOME=/cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3/b1stableglobal/cache TMPDIR=/cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3/b1stableglobal/tmp
mkdir -p "$XDG_CACHE_HOME" "$TMPDIR"
cd /cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3/b1stableglobal
sha256sum -c SOURCE.sha256 --quiet
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
$PY -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
cd experiments/burgers-tree-embeddings
$PY stable_global_controls.py --out /cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3/b1stableglobal/output --span /cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3/b1expanded/output --train /cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3/archive/refstrain01/output,/cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3/b1refextend/output/train --dev /cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3/archive/refsdev01/output,/cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3/b1refextend/output/dev
cd /cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3/b1stableglobal
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
