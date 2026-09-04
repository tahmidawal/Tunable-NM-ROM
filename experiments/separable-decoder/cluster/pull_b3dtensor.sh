#!/bin/bash
# Pull out/ logs/ COMMIT.txt of one job into runs/b3dtensor/<jobname>/ and verify RESULTS.sha256.
# Usage: ./pull_b3dtensor.sh <jobname>
set -euo pipefail
JOB=${1:?jobname}
HERE=$(cd "$(dirname "$0")" && pwd)
DST="$HERE/../runs/b3dtensor/$JOB"
REM=/cluster/tufts/paralab/tawal01/b3dtensor/$JOB
mkdir -p "$DST"
rsync -a tufts-login:$REM/out/ "$DST/out/"
rsync -a tufts-login:$REM/logs/ "$DST/logs/"
rsync -a tufts-login:$REM/COMMIT.txt "$DST/COMMIT.txt"
( cd "$DST/out" && sha256sum -c RESULTS.sha256 --quiet && echo "RESULTS.sha256 OK ($JOB)" )
