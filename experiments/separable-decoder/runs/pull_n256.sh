#!/bin/bash
# Pull one finished sepdec_n256 job: results + logs, checksum both sides,
# then (only after verification) delete the remote job dir.
# Usage: ./runs/pull_n256.sh j1 2825729 [--delete]
set -euo pipefail
JOB=${1:?j1|j2|j3|j4}; JID=${2:?slurm job id}; DEL=${3:-}
HERE=$(cd "$(dirname "$0")" && pwd)
DST=$HERE/n256_$JOB
REM=/cluster/tufts/paralab/tawal01/sepdec_n256/$JOB
mkdir -p "$DST/out" "$DST/logs"
rsync -a tufts-login:$REM/out/ "$DST/out/"
rsync -a tufts-login:$REM/logs/ "$DST/logs/"
rsync -a tufts-login:$REM/MANIFEST.sha256 "$DST/"
( cd "$DST/out" && sha256sum -c RESULTS.sha256 --quiet && echo "local RESULTS.sha256 OK" )
grep -q "jax_backend=gpu" "$DST/logs/$JID.out" && echo "gpu preflight OK"
grep -q "ALL-DONE" "$DST/logs/$JID.out" && echo "ALL-DONE OK"
if [ "$DEL" = "--delete" ]; then
  ssh tufts-login "rm -rf $REM" && echo "remote $REM deleted"
fi
