#!/bin/bash
# rsync one staged job dir to the b3dtensor namespace and submit it.
# One experiment = one namespace; one job = one directory.  squeue BEFORE and AFTER.
# The login node drops roughly every other connection (2026-09-04), so every remote
# call is retried; the sbatch itself is guarded by a BEFORE check that no job of
# this name is queued (duplicate-submit race).
# Usage: ./push_b3dtensor.sh <jobname> <sbatch-file>
set -euo pipefail
JOB=${1:?jobname}; SB=${2:?sbatch file}
HERE=$(cd "$(dirname "$0")" && pwd)
REM=/cluster/tufts/paralab/tawal01/b3dtensor/$JOB
retry() { for i in 1 2 3 4 5 6; do if "$@"; then return 0; fi; echo "  (retry $i: $1 ...)" >&2; sleep 12; done; return 1; }
rssh() { timeout 60 ssh -o ConnectTimeout=20 -o BatchMode=yes tufts-login "$@"; }
echo "--- squeue BEFORE ---"; retry rssh "squeue -u tawal01 -h -o '%i %j %T'" | tee /tmp/sq_before_$JOB.txt
if grep -q "b3d_$JOB " /tmp/sq_before_$JOB.txt; then echo "ABORT: a job named b3d_$JOB is already queued"; exit 3; fi
retry rssh "mkdir -p $REM/logs $REM/out"
# never delete an earlier out/ or logs/; a re-push of a finished job needs a new job dir
retry rsync -a --delete --exclude=logs/ --exclude=out/ -e "ssh -o ConnectTimeout=20 -o BatchMode=yes" "$HERE/stage/$JOB/" tufts-login:$REM/
retry rsync -a -e "ssh -o ConnectTimeout=20 -o BatchMode=yes" "$HERE/$SB" tufts-login:$REM/job.sbatch
# IDEMPOTENT submission: the job id is written to a remote JID file by the same shell that runs
# sbatch, so a dropped connection after a successful sbatch cannot submit twice
JID=$(retry rssh "cd $REM && if [ -s JID ]; then cat JID; else sbatch --parsable job.sbatch > JID.tmp && mv JID.tmp JID && cat JID; fi")
echo "submitted $JOB -> job $JID"
echo "--- squeue AFTER ---"; retry rssh "squeue -u tawal01 -h -o '%i %j %T'"
