#!/usr/bin/env bash
# Stage this lane's code into a fresh job directory on the cluster and submit one job.
#   submit.sh <jobname> <hours> <constraint> <mem> '<command run from the job dir; $PY is the venv python>'
# Each job gets its own directory NS/<jobname>/ with a frozen copy of code/ (never resubmits an existing name).
set -euo pipefail
NAME=$1; HOURS=$2; CONSTRAINT=$3; MEM=$4; CMD=$5
NS=/cluster/tufts/paralab/tawal01/rebuttal-op-scaling
HERE=$(cd "$(dirname "$0")/.." && pwd)
COMMIT=$(git -C "$HERE" rev-parse HEAD)
ssh -o BatchMode=yes tufts-login "test ! -e $NS/$NAME && mkdir -p $NS/$NAME/logs $NS/$NAME/code"
rsync -a --delete "$HERE/code/" "tufts-login:$NS/$NAME/code/"
cat > /tmp/claude-1002/-home-tahmid-Dev-Tunable-NM-ROM/e8c5e8de-cca5-59d3-a21a-7fbede9c98ab/scratchpad/osc-$NAME.sbatch <<SB
#!/bin/bash
#SBATCH --job-name=osc-$NAME
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint="$CONSTRAINT"
#SBATCH --cpus-per-task=8
#SBATCH --mem=$MEM
#SBATCH --time=$HOURS:00:00
#SBATCH --signal=B:USR1@240
#SBATCH --output=$NS/$NAME/logs/%j.out
#SBATCH --error=$NS/$NAME/logs/%j.err
set -euo pipefail
JOB=$NS/$NAME
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
source /cluster/tufts/paralab/tawal01/ae-research/venv/bin/activate
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 WANDB_MODE=offline WANDB_DISABLED=true PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="\$JOB/cache" MPLCONFIGDIR="\$JOB/cache/matplotlib" TORCH_HOME="\$JOB/cache/torch" TMPDIR="\$JOB/tmp"
mkdir -p "\$TMPDIR" "\$XDG_CACHE_HOME" "\$MPLCONFIGDIR"
cd "\$JOB"
echo "host=\$(hostname) source_commit=$COMMIT job=\$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv
"\$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b == 'gpu' else 42)"
"\$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
export PYTHONPATH="\$JOB/code"
PIDS=()
trap 'kill -USR1 "\${PIDS[@]}" 2>/dev/null || true' USR1
$CMD
echo ALL-DONE
SB
scp -q /tmp/claude-1002/-home-tahmid-Dev-Tunable-NM-ROM/e8c5e8de-cca5-59d3-a21a-7fbede9c98ab/scratchpad/osc-$NAME.sbatch "tufts-login:$NS/$NAME/run.sbatch"
ssh -o BatchMode=yes tufts-login "cd $NS/$NAME && echo $COMMIT > COMMIT.txt && sbatch run.sbatch"
