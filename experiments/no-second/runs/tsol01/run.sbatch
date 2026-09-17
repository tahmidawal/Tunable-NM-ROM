#!/bin/bash
#SBATCH --job-name=ctol_nos_tsol01
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=pax007
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=04:40:00
#SBATCH --signal=B:USR1@180
#SBATCH --output=/cluster/tufts/paralab/tawal01/no_second_20260917/tsol01/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/no_second_20260917/tsol01/logs/%j.err
set -euo pipefail
TASK_ROOT=/cluster/tufts/paralab/tawal01/no_second_20260917/tsol01
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export WANDB_MODE=offline WANDB_DISABLED=true
export XDG_CACHE_HOME="$TASK_ROOT/cache" MPLCONFIGDIR="$TASK_ROOT/cache/matplotlib"
export TORCH_HOME="$TASK_ROOT/cache/torch" TMPDIR="$TASK_ROOT/tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
cd "$TASK_ROOT"
sha256sum -c MANIFEST.sha256 --quiet
export SOURCE_COMMIT=$(cat COMMIT.txt)
echo "host=$(hostname) source_commit=$SOURCE_COMMIT job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv
df -h /cluster/tufts/paralab | tail -1
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
# Data: copy the verified shared cache into this attempt (never modify the cache), then re-verify.
test ! -e data
mkdir data
cp -r /cluster/tufts/paralab/tawal01/no_burgers_20260914/pilot-data01/train /cluster/tufts/paralab/tawal01/no_burgers_20260914/pilot-data01/validation /cluster/tufts/paralab/tawal01/no_burgers_20260914/pilot-data01/refinement /cluster/tufts/paralab/tawal01/no_burgers_20260914/pilot-data01/DATA.sha256 data/
( cd data && sha256sum -c DATA.sha256 --quiet && echo "data_verified=$(wc -l < DATA.sha256)" )
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
"$PY" code/worker_second.py code/specs/tsol01.json
find out data/diagnosis-cohort -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo ALL-DONE
