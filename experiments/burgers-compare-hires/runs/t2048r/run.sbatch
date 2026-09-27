#!/bin/bash
#SBATCH --job-name=bcmp_t2048r
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:h200:1

#SBATCH --exclude=pax007
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=04:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/bcmp_20260923/t2048r/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/bcmp_20260923/t2048r/logs/%j.err
set -euo pipefail
TASK_ROOT=/cluster/tufts/paralab/tawal01/bcmp_20260923/t2048r
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export XDG_CACHE_HOME="$TASK_ROOT/cache" TMPDIR="$TASK_ROOT/tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
cd "$TASK_ROOT"
sha256sum -c MANIFEST.sha256 --quiet
export SOURCE_COMMIT=$(cat COMMIT.txt)
echo "host=$(hostname) source_commit=$SOURCE_COMMIT"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
df -h /cluster/tufts/paralab | tail -1
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
export PYTHONPATH="$TASK_ROOT/experiments/mr-burgers2d:$TASK_ROOT/experiments/separable-decoder:$TASK_ROOT/experiments/head-ablation:$TASK_ROOT/experiments/cheap-corrections:$TASK_ROOT/experiments/b-ladder-top:$TASK_ROOT/experiments/b-panel/speed:$TASK_ROOT/experiments/quadratic-manifold:$TASK_ROOT/experiments/burgers-compare-hires/lib:$TASK_ROOT/experiments/burgers-compare-hires"
cd experiments/burgers-compare-hires
"$PY" opdata.py --mesh 2048 --out data
"$PY" -c "import torch,sys; ok=torch.cuda.is_available(); print('torch_cuda', ok, torch.cuda.get_device_name() if ok else None, flush=True); sys.exit(0 if ok else 42)"
"$PY" ops/worker.py specs/train-retry.json
find out data -name '*.json' -o -name 'best.pt' | sort | xargs sha256sum > OUTPUTS.sha256
echo ALL-DONE
