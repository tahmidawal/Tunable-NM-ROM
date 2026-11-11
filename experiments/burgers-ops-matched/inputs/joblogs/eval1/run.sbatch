#!/bin/bash
#SBATCH --job-name=bom-eval1
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100-80G"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --signal=B:USR1@240
#SBATCH --output=/cluster/tufts/paralab/tawal01/rebuttal-burgers-ops/eval1/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/rebuttal-burgers-ops/eval1/logs/%j.err
set -euo pipefail
JOB=/cluster/tufts/paralab/tawal01/rebuttal-burgers-ops/eval1
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
source /cluster/tufts/paralab/tawal01/ae-research/venv/bin/activate
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 WANDB_MODE=offline WANDB_DISABLED=true PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="$JOB/cache" MPLCONFIGDIR="$JOB/cache/matplotlib" TORCH_HOME="$JOB/cache/torch" TMPDIR="$JOB/tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
cd "$JOB"
echo "host=$(hostname) source_commit=6effd83e9ca39d293e30456be6ff0863bf7d1ed7 job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b == 'gpu' else 42)"
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
export PYTHONPATH="$JOB/code"
PIDS=()
trap 'kill -USR1 "${PIDS[@]}" 2>/dev/null || true' USR1
"$PY" code/evaluate.py --manifest /cluster/tufts/paralab/tawal01/rebuttal-burgers-ops/selection_manifest.json --cohort-dir /cluster/tufts/paralab/tawal01/rebuttal-burgers-ops/data/test --cohort test64 --out-dir $JOB/out
echo ALL-DONE
