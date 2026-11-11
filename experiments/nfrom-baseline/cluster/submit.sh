#!/usr/bin/env bash
# Stage this lane's code into a fresh job directory on the cluster and submit one GPU job.
#   submit.sh <jobname> <HH:MM:SS> <constraint> <mem> '<command run from the job dir; $PY is the venv python>'
# Each job gets its own directory NS/<jobname>/ with a frozen copy of code/ (an existing name is refused).
set -euo pipefail
NAME=$1; TIME=$2; CONSTRAINT=$3; MEM=$4; CMD=$5
NS=/cluster/tufts/paralab/tawal01/rebuttal-nfrom
HERE=$(cd "$(dirname "$0")/.." && pwd)
COMMIT=$(git -c gc.auto=0 -C "$HERE" rev-parse HEAD)
TMP=${TMPDIR:-/tmp}/nfrom-$NAME.sbatch
ssh -o BatchMode=yes tufts-login "test ! -e $NS/$NAME && mkdir -p $NS/$NAME/logs $NS/$NAME/code"
rsync -a --delete --exclude __pycache__ "$HERE/code/" "tufts-login:$NS/$NAME/code/"
cat > "$TMP" <<SB
#!/bin/bash
#SBATCH --job-name=nf-$NAME
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint="$CONSTRAINT"
#SBATCH --cpus-per-task=8
#SBATCH --mem=$MEM
#SBATCH --time=$TIME
#SBATCH --output=$NS/$NAME/logs/%j.out
#SBATCH --error=$NS/$NAME/logs/%j.err
set -euo pipefail
JOB=$NS/$NAME
NS=$NS
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
source /cluster/tufts/paralab/tawal01/ae-research/venv/bin/activate
export JAX_DEFAULT_MATMUL_PRECISION=highest XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export XDG_CACHE_HOME="\$JOB/cache" TMPDIR="\$JOB/tmp"
mkdir -p "\$TMPDIR" "\$XDG_CACHE_HOME"
cd "\$JOB"
echo "host=\$(hostname) source_commit=$COMMIT job=\$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv
"\$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b == 'gpu' else 42)"
export PYTHONPATH="\$JOB/code"
$CMD
echo ALL-DONE
SB
scp -q "$TMP" "tufts-login:$NS/$NAME/run.sbatch"
ssh -o BatchMode=yes tufts-login "cd $NS/$NAME && echo $COMMIT > COMMIT.txt && sbatch run.sbatch"
