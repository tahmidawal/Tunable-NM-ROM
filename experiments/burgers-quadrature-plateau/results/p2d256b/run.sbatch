#!/bin/bash
#SBATCH --job-name=bqp_p2d256b
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100-80G
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=30
#SBATCH --output=/cluster/tufts/paralab/tawal01/bqplateau-20261105/p2d256b/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/bqplateau-20261105/p2d256b/logs/%j.err
set -euo pipefail
source /cluster/tufts/paralab/tawal01/ae-research/venv/bin/activate
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export XDG_CACHE_HOME="/cluster/tufts/paralab/tawal01/bqplateau-20261105/p2d256b/cache" TMPDIR="/cluster/tufts/paralab/tawal01/bqplateau-20261105/p2d256b/tmp"
export JCPREF_STORE=/cluster/tufts/paralab/tawal01/jcp-ref/store
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
cd "/cluster/tufts/paralab/tawal01/bqplateau-20261105/p2d256b"
sha256sum -c MANIFEST.sha256 --quiet
export SOURCE_COMMIT=$(cat COMMIT.txt) SLURM_JOB_ID
echo "host=$(hostname) source_commit=$SOURCE_COMMIT"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
df -h /cluster/tufts/paralab | tail -1
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
cd "/cluster/tufts/paralab/tawal01/bqplateau-20261105/p2d256b/experiments/burgers-quadrature-plateau"
"$PY" plateau2d.py --config configs/p2d_256.json --out "/cluster/tufts/paralab/tawal01/bqplateau-20261105/p2d256b/output"
cd "/cluster/tufts/paralab/tawal01/bqplateau-20261105/p2d256b"
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo ALL-DONE
