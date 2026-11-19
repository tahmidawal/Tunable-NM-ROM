#!/bin/bash
#SBATCH --job-name=bte_b2rspace
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100-80G
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=25
#SBATCH --chdir=/cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3/b2rspace
#SBATCH --output=/cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3/b2rspace/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3/b2rspace/logs/%j.err
set -euo pipefail
source /cluster/tufts/paralab/tawal01/ae-research/venv/bin/activate
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export SOURCE_COMMIT=e4ca16b0eb9a808444853f4d5ba27a41f02d3473 STUDY_NAMESPACE=/cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3
export XDG_CACHE_HOME=/cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3/b2rspace/cache TMPDIR=/cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3/b2rspace/tmp
mkdir -p "$XDG_CACHE_HOME" "$TMPDIR"
cd /cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3/b2rspace
sha256sum -c SOURCE.sha256 --quiet
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
$PY -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
cd experiments/burgers-tree-embeddings
$PY fixed_tree_pilot.py --out /cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3/b2rspace/output --L 32768 --dt .00125 --integrator cn --test-factor 16 --quadrature 512 --stable-global /cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3/archive/b1stableglobal/output --cache /cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3/archive/b1cache/output --span /cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3/archive/b1expanded/output --admission /cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3/archive/b1directaudit/output --dev /cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3/archive/refsdev01/output,/cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3/archive/b1refextend/output/dev --compare-previous /cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3/b2space/output
cd /cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3/b2rspace
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
