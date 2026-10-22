#!/bin/bash
# usage: collect.sh <jobname> <name> [critical]   (name = cell e.g. W40_A2, or a smoke name smk-A2)
#        mode 'critical' stops after phase 1 (accounting + every critical file, any size) and deletes nothing
# Phases (job audits 1-3): 0) terminal state required; 1) critical evidence (JSON/JSONL/logs/markers/memory samples, each
# < 20 MB) pulled and verified; 2) ledger entry (accounting) recorded -- REQUIRED before anything is deleted; 3) the
# remaining files < 100 MB pulled and verified; 4) out/cell archived to burgers-archive/imp-box/<name> with a verified
# sha256 listing (training arrays regenerable from the seed are not archived); 5) the cluster job directory deleted only
# after phases 1-4 all succeeded.
set -euo pipefail
JOB=$1; NAME=$2; MODE=${3:-full}
[[ "$JOB" =~ ^[A-Za-z0-9_-]+$ && "$NAME" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "bad names" >&2; exit 2; }
LANE=$(cd "$(dirname "$0")/.." && pwd)
NS=/cluster/tufts/paralab/tawal01/imp-box
ARCH=/cluster/tufts/paralab/tawal01/burgers-archive/imp-box/$NAME
DEST="$LANE/runs/cells/$NAME"
mkdir -p "$DEST"
JID=$(timeout 60 ssh -o ConnectTimeout=20 tufts-login "cat $NS/$JOB/submitted") || { echo "no submitted id" >&2; exit 2; }
JST=$(timeout 60 ssh -o ConnectTimeout=20 tufts-login "sacct -j $JID -X -n -P -o State | head -1 | cut -d' ' -f1")
case "$JST" in COMPLETED|FAILED|CANCELLED*|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL) ;; *) echo "job $JID not terminal ($JST): refusing" >&2; exit 2;; esac
STATE=$(timeout 60 ssh -o ConnectTimeout=20 tufts-login "if [ -d $NS/$JOB/out/cell ]; then echo present; else echo absent; fi") \
  || { echo "transport failure while checking $JOB: nothing collected or deleted" >&2; exit 2; }
case "$NAME" in meas*|smoke*|smk-*|smk2-*) ;; *) [ "$STATE" = present ] || { echo "cell job without out/cell: refusing" >&2; exit 2; };; esac
/home/tahmid/Dev/.venv/bin/python "$LANE/ledger.py" add "$JID" "$JOB" || echo "ledger attempt failed (retried below)" >&2
pull() {   # $1 = find predicate (remote), $2 = list tag, $3 = timeout seconds
  timeout ${3:-900} ssh -o ConnectTimeout=20 tufts-login "set -euo pipefail; cd $NS/$JOB && find . -path ./code -prune -o -type f $1 ! -name '.pull-*' -print | sort > .pull-$2 && xargs -a .pull-$2 -r sha256sum > .pull-$2.sums && tar -czf - -T .pull-$2 .pull-$2.sums" | tar -C "$DEST" -xzf -
  if [ -s "$DEST/.pull-$2.sums" ]; then ( cd "$DEST" && sha256sum -c --quiet ".pull-$2.sums" ); fi   # empty set = nothing to verify
}
CRIT="\( -name '*.json' -o -name '*.jsonl' -o -name '*.log' -o -name '*.out' -o -name '.done_*' -o -name '*.txt' -o -name submitted -o -name '*.sbatch' \)"
# phase 1: inventory of every critical file (any size); small ones (< 20 MiB, byte-exact) then large ones; completeness
timeout 120 ssh -o ConnectTimeout=20 tufts-login "set -o pipefail; cd $NS/$JOB && find . -path ./code -prune -o -type f $CRIT ! -name '.pull-*' -print | sort" \
  > "$DEST/.critical-inventory" || { echo "critical inventory failed: INCOMPLETE" >&2; exit 4; }
pull "-size -20971520c $CRIT" crit 300
pull "-size +20971519c $CRIT" critbig 1200
missing=0
while IFS= read -r f; do [ -f "$DEST/$f" ] || { echo "missing critical file $f" >&2; missing=1; }; done < "$DEST/.critical-inventory"
[ $missing -eq 0 ] || { echo "critical evidence INCOMPLETE" >&2; exit 4; }
date -u +%FT%TZ > "$DEST/CRITICAL-COMPLETE"
echo "critical evidence complete and verified ($(wc -l < "$DEST/.critical-inventory") files)"
[ -f "$DEST/allocated_gpu.txt" ] && /home/tahmid/Dev/.venv/bin/python "$LANE/ledger.py" add "$JID" "$JOB" "$DEST/allocated_gpu.txt" || true
[ "$MODE" = critical ] && exit 0
/home/tahmid/Dev/.venv/bin/python "$LANE/ledger.py" add "$JID" "$JOB" || { echo "ledger failed: nothing deleted" >&2; exit 3; }
pull "-size -100M" rest
echo "local copies verified"
if [ "$STATE" = present ]; then
  EXCL="! -name train_coef.npy ! -name trainc_coef.npy ! -name bank_U.npy"
  timeout 3600 ssh -o ConnectTimeout=20 tufts-login "set -euo pipefail; mkdir -p $ARCH; cd $NS/$JOB/out/cell && find . -type f $EXCL ! -name SHA256SUMS -print0 | cpio -0 -pdm --quiet $ARCH; cd $ARCH && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS; cd $NS/$JOB/out/cell && find . -type f $EXCL ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum | diff - $ARCH/SHA256SUMS"
  timeout 120 scp -q -o ConnectTimeout=20 tufts-login:$ARCH/SHA256SUMS "$DEST/ARCHIVE-SHA256SUMS"
  echo "archive verified: $ARCH"
fi
# every job-level file outside out/cell and code/ (logs of any size, memory samples) is preserved before deletion
timeout 1800 ssh -o ConnectTimeout=20 tufts-login "set -euo pipefail; mkdir -p $ARCH/_job; cd $NS/$JOB && find . -path ./code -prune -o -path ./out/cell -prune -o -type f -print0 | cpio -0 -pdm --quiet $ARCH/_job; cd $ARCH/_job && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 -r sha256sum > SHA256SUMS; cd $NS/$JOB && find . -path ./code -prune -o -path ./out/cell -prune -o -type f -print0 | sort -z | xargs -0 -r sha256sum | diff - $ARCH/_job/SHA256SUMS"
echo "job-level files preserved: $ARCH/_job"
[ -f "$DEST/CRITICAL-COMPLETE" ] || { echo "no critical completeness record: nothing deleted" >&2; exit 4; }
timeout 300 ssh -o ConnectTimeout=20 tufts-login "rm -rf $NS/$JOB"
echo "collected $JOB -> runs/cells/$NAME"
