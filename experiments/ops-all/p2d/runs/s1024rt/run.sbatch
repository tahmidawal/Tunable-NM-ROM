#!/bin/bash
#SBATCH --job-name=p2d_s1024rt
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100-80G|h100-80G|h200-141G"
#SBATCH --exclude=pax007
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:45:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/opsall_20260924/p2d/s1024rt/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/opsall_20260924/p2d/s1024rt/logs/%j.err
set -euo pipefail
TASK_ROOT=/cluster/tufts/paralab/tawal01/opsall_20260924/p2d/s1024rt
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XDG_CACHE_HOME="$TASK_ROOT/cache" MPLCONFIGDIR="$TASK_ROOT/cache/matplotlib" TMPDIR="$TASK_ROOT/tmp"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
cd "$TASK_ROOT"
sha256sum -c MANIFEST.sha256 --quiet
echo "host=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
df -h /cluster/tufts/paralab | tail -1
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
"$PY" -c "import torch,sys; ok=torch.cuda.is_available(); print(f'torch_cuda={ok}',flush=True); sys.exit(0 if ok else 42)"
cd "$TASK_ROOT/code/ops"
PIDS=""
"$PY" train_ops.py --problem square --mesh 1024 --config ../configs/ops/transolver.json --out "$TASK_ROOT/output/transolver" --wall-seconds 3000 --train-count 0 --val-count 16 --cg-batch 32 --mem-share 1.0 --co-scheduled transolver > "$TASK_ROOT/logs/transolver.log" 2>&1 &
PIDS="$PIDS $!"
RC=0; for p in $PIDS; do wait $p || RC=1; done
tail -n 3 "$TASK_ROOT"/logs/*.log
cd "$TASK_ROOT"; rm -rf cache tmp
echo "members rc=$RC"
echo ALL-DONE
