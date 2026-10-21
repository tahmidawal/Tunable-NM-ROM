#!/bin/bash
# g2d: pull ONLY small records of one job (json, logs, job.out/err, checksums) into runs/<job>/pull/. Never fields,
# checkpoints or datasets. Records the remote sha256 of every pulled file and checks it locally.
set -euo pipefail
JOB=$1; G2D=$(cd "$(dirname "$0")/.." && pwd); NS=/cluster/tufts/paralab/tawal01/opsall_20260924/g2d
mkdir -p "$G2D/runs/$JOB/pull"
rsync -a --prune-empty-dirs --exclude='src/***' --exclude='cache/***' --exclude='tmp/***' --include='*/' \
  --include='*.json' --include='*.log' --include='job.out' --include='job.err' --include='*.sha256' --exclude='*' \
  "tufts-login:$NS/$JOB/" "$G2D/runs/$JOB/pull/"
ssh -n tufts-login "cd $NS/$JOB && find . -path ./src -prune -o -path ./cache -prune -o -path ./tmp -prune -o -type f \( -name '*.json' -o -name '*.log' -o -name job.out -o -name job.err \) -print0 | xargs -0 sha256sum" > "$G2D/runs/$JOB/pull/REMOTE.sha256"
(cd "$G2D/runs/$JOB/pull" && sha256sum -c REMOTE.sha256 --quiet) && echo "pulled $JOB: checksums OK"
