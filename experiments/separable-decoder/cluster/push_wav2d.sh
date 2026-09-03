#!/bin/bash
# rsync one staged job dir to the wav2d cluster namespace and submit it.  One experiment = one namespace
# (shared account); one job = one directory.  squeue BEFORE and AFTER (duplicate-submit race).
# Usage: ./push_wav2d.sh <jobname> <sbatch-file>
set -euo pipefail
JOB=${1:?jobname}; SB=${2:?sbatch file}
HERE=$(cd "$(dirname "$0")" && pwd)
REM=/cluster/tufts/paralab/tawal01/wav2d/$JOB
echo "--- squeue BEFORE ---"; ssh -o BatchMode=yes tufts-login "squeue -u tawal01 -h -o '%i %j %T'" || true
ssh -o BatchMode=yes tufts-login "mkdir -p $REM/logs"
rsync -a --delete --exclude=logs/ "$HERE/stage/$JOB/" tufts-login:$REM/
rsync -a "$HERE/$SB" tufts-login:$REM/job.sbatch
JID=$(ssh -o BatchMode=yes tufts-login "cd $REM && sbatch --parsable job.sbatch")
echo "submitted $JOB -> job $JID"
echo "--- squeue AFTER ---"; ssh -o BatchMode=yes tufts-login "squeue -u tawal01 -h -o '%i %j %T'"
