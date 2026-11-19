#!/bin/bash
#SBATCH --job-name=bte_b1cnM16Q1024
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100-80G
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=30
#SBATCH --chdir=/cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3/b1cnM16Q1024
#SBATCH --output=/cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3/b1cnM16Q1024/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3/b1cnM16Q1024/logs/%j.err
set -euo pipefail
source /cluster/tufts/paralab/tawal01/ae-research/venv/bin/activate
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export SOURCE_COMMIT=9c71519b1a501c49c2c1d1afded177f2fea7d24f STUDY_NAMESPACE=/cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3
export XDG_CACHE_HOME=/cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3/b1cnM16Q1024/cache TMPDIR=/cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3/b1cnM16Q1024/tmp
mkdir -p "$XDG_CACHE_HOME" "$TMPDIR"
cd /cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3/b1cnM16Q1024
sha256sum -c SOURCE.sha256 --quiet
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
$PY -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
cd experiments/burgers-tree-embeddings
$PY fixed_tree_pilot.py --out /cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3/b1cnM16Q1024/output --L 2048 --dt .0025 --integrator cn --test-factor 16 --quadrature 1024 --compare-previous /cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3/b1cnM16Q512/output --cache /cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3/b1cache/output --span /cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3/b1expanded/output --admission /cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3/b1directaudit/output --dev /cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3/archive/refsdev01/output,/cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3/b1refextend/output/dev
cd /cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3/b1cnM16Q1024
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
