#!/bin/bash
# Pull a finished sepdec_n512 job's out/ + logs/ into runs/, verify checksums.
# Usage: ./pull_job.sh <jobname>            (e.g. j1)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SEP="$(dirname "$HERE")"
JOB=${1:?jobname}
DEST="$SEP/runs/sepdec_n512_$JOB"
mkdir -p "$DEST"
rsync -a "tufts-login:/cluster/tufts/paralab/tawal01/sepdec_n512/$JOB/out/" "$DEST/out/"
rsync -a "tufts-login:/cluster/tufts/paralab/tawal01/sepdec_n512/$JOB/logs/" "$DEST/logs/"
rsync -a "tufts-login:/cluster/tufts/paralab/tawal01/sepdec_n512/$JOB/MANIFEST.sha256" "$DEST/"
( cd "$DEST/out" && sha256sum -c RESULTS.sha256 --quiet && echo "RESULTS checksums OK" )
echo "pulled -> $DEST"
