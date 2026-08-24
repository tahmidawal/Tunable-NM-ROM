#!/bin/bash
# Pull one finished round-3 n256_push job: results + logs, checksum both sides,
# verify the integrity markers, then (only with --delete) remove the remote dir.
# Usage: ./runs/pull_r3.sh r3a 2835788 [--delete]
set -euo pipefail
JOB=${1:?job dir name, e.g. r3a}; JID=${2:?slurm job id}; DEL=${3:-}
HERE=$(cd "$(dirname "$0")" && pwd)
DST=$HERE/push_$JOB
REM=/cluster/tufts/paralab/tawal01/n256_push/$JOB
mkdir -p "$DST/out" "$DST/logs"
rsync -a tufts-login:$REM/out/ "$DST/out/"
rsync -a tufts-login:$REM/logs/ "$DST/logs/"
rsync -a tufts-login:$REM/MANIFEST.sha256 "$DST/"
( cd "$DST/out" && sha256sum -c RESULTS.sha256 --quiet && echo "local RESULTS.sha256 OK" )
grep -q "jax_backend=gpu" "$DST/logs/$JID.out" && echo "gpu preflight OK"
grep -q "stage MANIFEST OK" "$DST/logs/$JID.out" && echo "stage manifest OK"
grep -q "ALL-DONE" "$DST/logs/$JID.out" && echo "ALL-DONE OK"
grep -i "large amount of constants" "$DST/logs/$JID."* && echo "!! CAPTURED-CONSTANT WARNING" || echo "no captured-constant warning"
if [ "$DEL" = "--delete" ]; then
  ssh tufts-login "rm -rf $REM" && echo "remote $REM deleted"
fi
