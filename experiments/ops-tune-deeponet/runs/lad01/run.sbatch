#!/bin/bash
#SBATCH --job-name=opstune_don_lad01
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100-80G
#SBATCH --exclude=pax007
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=08:30:00
#SBATCH --signal=B:USR1@180
#SBATCH --output=/cluster/tufts/paralab/tawal01/opstune_don_20260922/lad01/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/opstune_don_20260922/lad01/logs/%j.err
set -euo pipefail
TASK_ROOT=/cluster/tufts/paralab/tawal01/opstune_don_20260922/lad01
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
# Data: read-only symlinks to verified caches (never modified), verified through the link.
test ! -e data
mkdir data
mkdir -p data/pinned
ln -s /cluster/tufts/paralab/tawal01/no_burgers_20260914/pilot-data01/train data/pinned/train
ln -s /cluster/tufts/paralab/tawal01/no_burgers_20260914/pilot-data01/validation data/pinned/validation
ln -s /cluster/tufts/paralab/tawal01/no_burgers_20260914/pilot-data01/refinement data/pinned/refinement
ln -s /cluster/tufts/paralab/tawal01/no_burgers_20260914/pilot-data01/DATA.sha256 data/pinned/DATA.sha256
( cd data/pinned && sha256sum -c DATA.sha256 --quiet && echo "data_verified_pinned=$(wc -l < DATA.sha256)" )
mkdir -p data/ext
ln -s /cluster/tufts/paralab/tawal01/opstune_don_20260922/pool02/train data/ext/train
ln -s /cluster/tufts/paralab/tawal01/opstune_don_20260922/pool02/pool data/ext/pool
ln -s /cluster/tufts/paralab/tawal01/opstune_don_20260922/pool02/DATA.sha256 data/ext/DATA.sha256
( cd data/ext && sha256sum -c DATA.sha256 --quiet && echo "data_verified_ext=$(wc -l < DATA.sha256)" )
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
# Forward Slurm's USR1 (sent to this batch shell 180 s before the limit) to the worker,
# which asks train.py for a stop at the next evaluation; then wait for it.
"$PY" code/worker_tune.py code/specs/lad01.json &
WORKER=$!
trap 'kill -USR1 "$WORKER" 2>/dev/null || true' USR1
wait "$WORKER"
find out $(test -d data/diagnosis-cohort && echo data/diagnosis-cohort) -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
du -sh out | tail -1
echo ALL-DONE
