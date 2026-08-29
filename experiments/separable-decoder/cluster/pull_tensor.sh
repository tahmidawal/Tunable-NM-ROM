#!/bin/bash
# Pull one finished b1dtensor job's out/ + logs/ into runs/b1dtensor/<jobname>/
# and verify RESULTS.sha256 locally.  Usage: ./pull_tensor.sh <jobname>
set -euo pipefail
JOB=${1:?jobname}
HERE=$(cd "$(dirname "$0")" && pwd)
SEP=$(dirname "$HERE")
REM=/cluster/tufts/paralab/tawal01/b1dtensor/$JOB
DST=$SEP/runs/b1dtensor/$JOB
mkdir -p "$DST"
rsync -a tufts-login:$REM/out/ "$DST/out/"
rsync -a tufts-login:$REM/logs/ "$DST/logs/"
( cd "$DST/out" && sha256sum -c RESULTS.sha256 && echo "RESULTS.sha256 OK ($JOB)" )
