#!/bin/bash
#SBATCH --job-name=bom-c1
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100-80G|h200-141G"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --signal=B:USR1@240
#SBATCH --output=/cluster/tufts/paralab/tawal01/rebuttal-burgers-ops/c1/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/rebuttal-burgers-ops/c1/logs/%j.err
set -euo pipefail
JOB=/cluster/tufts/paralab/tawal01/rebuttal-burgers-ops/c1
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
source /cluster/tufts/paralab/tawal01/ae-research/venv/bin/activate
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 WANDB_MODE=offline WANDB_DISABLED=true PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="$JOB/cache" MPLCONFIGDIR="$JOB/cache/matplotlib" TORCH_HOME="$JOB/cache/torch" TMPDIR="$JOB/tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
cd "$JOB"
echo "host=$(hostname) source_commit=f1aa9e734de53254671c9a629d189d38e60308cd job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b == 'gpu' else 42)"
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
export PYTHONPATH="$JOB/code"
PIDS=()
trap 'kill -USR1 "${PIDS[@]}" 2>/dev/null || true' USR1
mkdir -p $JOB/runs; "$PY" code/train.py --pool $JOB/../data/pool --budget-hours 3.5 --rung 4608 --arm F2 --seed 0 --resume-from /cluster/tufts/paralab/tawal01/rebuttal-burgers-ops/tr5/runs/F2-n4608-s0 --out $JOB/runs/F2-n4608-s0 & PIDS+=($!); "$PY" code/train.py --pool $JOB/../data/pool --budget-hours 3.5 --rung 4608 --arm U0 --seed 0 --resume-from /cluster/tufts/paralab/tawal01/rebuttal-burgers-ops/tr1/runs/U0-n4608-s0 --out $JOB/runs/U0-n4608-s0 & PIDS+=($!); "$PY" code/train.py --pool $JOB/../data/pool --budget-hours 3.5 --rung 4608 --arm U0 --seed 2 --resume-from /cluster/tufts/paralab/tawal01/rebuttal-burgers-ops/tr3/runs/U0-n4608-s2 --out $JOB/runs/U0-n4608-s2 & PIDS+=($!); "$PY" code/train.py --pool $JOB/../data/pool --budget-hours 3.5 --rung 4608 --arm U2 --seed 0 --resume-from /cluster/tufts/paralab/tawal01/rebuttal-burgers-ops/tr5/runs/U2-n4608-s0 --out $JOB/runs/U2-n4608-s0 & PIDS+=($!); wait || true; wait || true; wait || true; wait || true; wait || true
echo ALL-DONE
