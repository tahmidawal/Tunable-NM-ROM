#!/bin/bash
#SBATCH --job-name=q3_ho65
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1

#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=08:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/quad3d_20261001/ho65/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/quad3d_20261001/ho65/logs/%j.err
set -euo pipefail
TASK_ROOT=/cluster/tufts/paralab/tawal01/quad3d_20261001/ho65
source /cluster/tufts/paralab/tawal01/ae-research/venv/bin/activate
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export XDG_CACHE_HOME="$TASK_ROOT/cache" TMPDIR="$TASK_ROOT/tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
cd "$TASK_ROOT"
sha256sum -c MANIFEST.sha256 --quiet
export SOURCE_COMMIT=$(cat COMMIT.txt) SLURM_JOB_ID
echo "host=$(hostname) source_commit=$SOURCE_COMMIT"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
df -h /cluster/tufts/paralab | tail -1
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
cd "$TASK_ROOT/code"
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python; "$PY" experiments/quadrature-burgers3d/qpanel.py --config experiments/quadrature-burgers3d/configs/ho_n65.json --out output
cd "$TASK_ROOT"
find code/output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo ALL-DONE
