#!/usr/bin/env bash
# Stage this lane's code into a fresh job directory on the cluster and submit one GPU job (template: nfrom-baseline lane).
#   submit.sh <jobname> <HH:MM:SS> <constraint> <mem> '<command run from the job dir>'
# Env pythons: $CROM_PY (torch 1.11 / PL 1.6.5) and $COLORA_PY (jax 0.4.19 / flax 0.7.5); official repos at $EXT (pinned commits).
set -euo pipefail
NAME=$1; TIME=$2; CONSTRAINT=$3; MEM=$4; CMD=$5
NS=/cluster/tufts/paralab/tawal01/rebuttal-official
HERE=$(cd "$(dirname "$0")/.." && pwd)
COMMIT=$(git -c gc.auto=0 -C "$HERE" rev-parse HEAD)
TMP=${TMPDIR:-/tmp}/official-$NAME.sbatch
ssh -o BatchMode=yes tufts-login "test ! -e $NS/jobs/$NAME && mkdir -p $NS/jobs/$NAME/logs $NS/jobs/$NAME/code"
rsync -a --delete --exclude __pycache__ "$HERE/code/" "tufts-login:$NS/jobs/$NAME/code/"
cat > "$TMP" <<SB
#!/bin/bash
#SBATCH --job-name=of-$NAME
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint="$CONSTRAINT"
#SBATCH --cpus-per-task=8
#SBATCH --mem=$MEM
#SBATCH --time=$TIME
#SBATCH --output=$NS/jobs/$NAME/logs/%j.out
#SBATCH --error=$NS/jobs/$NAME/logs/%j.err
set -euo pipefail
NS=$NS
JOB=$NS/jobs/$NAME
EXT=$NS/ext
DATA=/cluster/tufts/paralab/tawal01/rebuttal-nfrom/data
TEST=/cluster/tufts/paralab/tawal01/rebuttal-burgers-ops/data/test/test64-L256
CROM_PY=$NS/env/crom/bin/python
COLORA_PY=$NS/env/colora/bin/python
AE_PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MPLBACKEND=Agg
export XDG_CACHE_HOME="\$JOB/cache" TMPDIR="\$JOB/tmp" XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p "\$TMPDIR" "\$XDG_CACHE_HOME"
cd "\$JOB"
echo "host=\$(hostname) source_commit=$COMMIT job=\$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv
export NF_GPU_NAME="\$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
"\$CROM_PY" -c "import torch,sys; ok=torch.cuda.is_available(); print('torch_cuda=',ok, torch.cuda.get_device_name(0) if ok else ''); sys.exit(0 if ok else 42)"
"\$COLORA_PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b == 'gpu' else 42)"
$CMD
echo ALL-DONE
SB
scp -q "$TMP" "tufts-login:$NS/jobs/$NAME/run.sbatch"
ssh -o BatchMode=yes tufts-login "cd $NS/jobs/$NAME && echo $COMMIT > COMMIT.txt && sbatch run.sbatch"
