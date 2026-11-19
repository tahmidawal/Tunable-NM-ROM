#!/bin/bash
#SBATCH --job-name=bte_b4time
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100-80G
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=90
#SBATCH --chdir=/cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3/b4time
#SBATCH --output=/cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3/b4time/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3/b4time/logs/%j.err
set -euo pipefail
source /cluster/tufts/paralab/tawal01/ae-research/venv/bin/activate
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export SOURCE_COMMIT=b8aa8ac8171a26dbf6dae02d998c020052a3b81d STUDY_NAMESPACE=/cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3
export XDG_CACHE_HOME=/cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3/b4time/cache TMPDIR=/cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3/b4time/tmp
mkdir -p "$XDG_CACHE_HOME" "$TMPDIR"
cd /cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3/b4time
sha256sum -c SOURCE.sha256 --quiet
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
$PY -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
cd experiments/burgers-tree-embeddings
$PY fixed_tree_pilot.py --out /cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3/b4time/output --L 16384 --dt .000625 --integrator cn --test-factor 16 --quadrature 512 --stable-global /cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3/archive/b1stableglobal/output --cache /cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3/archive/b1cache/output --span /cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3/archive/b1expanded/output --admission /cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3/archive/b1directaudit/output --dev /cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3/archive/refsdev01/output,/cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3/archive/b1refextend/output/dev --packed-fields --labels dev000,dev001,dev002,dev003,dev004,dev005,dev006,dev007,dev008,dev009,dev010,dev011,dev012,dev013,dev014,dev015,dev016,dev017,dev018,dev019,dev020,dev021,dev022,dev023,dev024,dev025,dev026,dev027,dev028,dev029,dev030,dev031,align000,align001,align002,align003,align004,align005,align006,align007,align008,align009,align010,align011 --save-prediction-only --compare-previous /cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3/b4base/output
cd /cluster/tufts/paralab/tawal01/burgers-tree-embeddings-20261118-1044bfe3/b4time
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
