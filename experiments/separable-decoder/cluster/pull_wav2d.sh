#!/bin/bash
# Pull one finished wav2d job's out/ + logs/ into runs/wav2d/<jobname>/ and verify RESULTS.sha256.
# Usage: ./pull_wav2d.sh <jobname>
set -euo pipefail
JOB=${1:?jobname}
HERE=$(cd "$(dirname "$0")" && pwd)
SEP=$(dirname "$HERE")
REM=/cluster/tufts/paralab/tawal01/wav2d/$JOB
DST=$SEP/runs/wav2d/$JOB
mkdir -p "$DST"
rsync -a tufts-login:$REM/out/ "$DST/out/"
rsync -a tufts-login:$REM/logs/ "$DST/logs/"
rsync -a tufts-login:$REM/COMMIT.txt "$DST/COMMIT.txt"
( cd "$DST/out" && sha256sum -c RESULTS.sha256 && echo "RESULTS.sha256 OK ($JOB)" )
