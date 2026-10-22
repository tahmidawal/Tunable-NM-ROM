#!/bin/bash
# usage: collect.sh <jobname> <name>   (name = cell e.g. L_wide_S1, or a probe job name e.g. meas6)
# 1) if the job has out/cell: copy it to the paralab archive (restore source for test jobs) and verify a sha256 listing;
# 2) pull every file < 100 MB of the job directory (except code/) into runs/cells/<name>/ and VERIFY the local copies
#    against remote sha256 sums; 3) delete the cluster job directory only after both verifications succeed.
set -euo pipefail
JOB=$1; NAME=$2
LANE=$(cd "$(dirname "$0")/.." && pwd)
NS=/cluster/tufts/paralab/tawal01/imp-geometry
ARCH=/cluster/tufts/paralab/tawal01/burgers-archive/imp-geometry/$NAME
DEST="$LANE/runs/cells/$NAME"
mkdir -p "$DEST"
STATE=$(timeout 60 ssh -o ConnectTimeout=20 tufts-login "if [ -d $NS/$JOB/out/cell ]; then echo present; else echo absent; fi") \
  || { echo "transport failure while checking $JOB: nothing collected or deleted" >&2; exit 2; }
case "$NAME" in meas*|smoke*) ;; *) [ "$STATE" = present ] || { echo "cell job without out/cell: refusing" >&2; exit 2; };; esac
if [ "$STATE" = present ]; then
  timeout 3600 ssh -o ConnectTimeout=20 tufts-login "set -euo pipefail; mkdir -p $ARCH; cp -a $NS/$JOB/out/cell/. $ARCH/; cd $ARCH && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS; cd $NS/$JOB/out/cell && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum | diff - $ARCH/SHA256SUMS"
  timeout 120 scp -q -o ConnectTimeout=20 tufts-login:$ARCH/SHA256SUMS "$DEST/ARCHIVE-SHA256SUMS"
  echo "archive verified: $ARCH"
fi
timeout 1800 ssh -o ConnectTimeout=20 tufts-login "set -euo pipefail; cd $NS/$JOB && find . -path ./code -prune -o -type f -size -100M -print | sort > .pull-list && xargs -a .pull-list sha256sum > .pull-sums && tar -czf - -T .pull-list .pull-sums" | tar -C "$DEST" -xzf -
( cd "$DEST" && sha256sum -c --quiet .pull-sums )
echo "local copies verified"
timeout 300 ssh -o ConnectTimeout=20 tufts-login "rm -rf $NS/$JOB"
echo "collected $JOB -> runs/cells/$NAME"
