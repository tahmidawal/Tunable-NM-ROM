#!/bin/bash
# Collect one finished attempt (COMMON-BRIEF 6): require a terminal sacct state; pull logs and every output file
# < 100 MB, verifying each against the job's OUTPUTS.sha256 (explicit list; a listed small file that is missing is an
# error); move the whole remote output/ to burgers-archive/imp-multimesh/<attempt>/output (later jobs read sources from
# there), record a sha256 listing of the archive, then delete the remote job directory.
# Usage: collect.sh <attempt> <jobid>
set -euo pipefail
A="$1"; JID="$2"; LANE=$(cd "$(dirname "$0")/.." && pwd); PY=/home/tahmid/Dev/.venv/bin/python
NS=/cluster/tufts/paralab/tawal01/imp-multimesh; ARC=/cluster/tufts/paralab/tawal01/burgers-archive/imp-multimesh
ST=$(timeout 60 ssh -o ConnectTimeout=20 tufts-login "sacct -j $JID -X -n -o State" | awk '{print $1}' | head -1)
case "$ST" in COMPLETED|FAILED|CANCELLED*|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL) ;; *) echo "job $JID not terminal ($ST)"; exit 2;; esac
D="$LANE/runs/$A/pulled"; mkdir -p "$D"
rsync -a "tufts-login:$NS/$A/logs" "tufts-login:$NS/$A/OUTPUTS.sha256" "$D/"
timeout 120 ssh -o ConnectTimeout=20 tufts-login "cd $NS/$A && find output -type f -size -100M -print0 | xargs -0 -r ls" > "$D/small.list"
rsync -a --files-from="$D/small.list" "tufts-login:$NS/$A/" "$D/"
"$PY" - "$D" <<'PYEOF'
import hashlib, sys, pathlib
d = pathlib.Path(sys.argv[1]); small = set(l.strip() for l in open(d / 'small.list') if l.strip())
n = 0
for ln in open(d / 'OUTPUTS.sha256'):
    h, f = ln.split(maxsplit=1); f = f.strip()
    if f in small:
        assert (d / f).exists(), f'missing pulled file {f}'
        assert hashlib.sha256((d / f).read_bytes()).hexdigest() == h, f'hash mismatch {f}'
        n += 1
assert n == len(small), (n, len(small))
print('pulled and verified', n, 'files')
PYEOF
timeout 1200 ssh -o ConnectTimeout=20 tufts-login "mkdir -p $ARC/$A && test ! -e $ARC/$A/output && mv $NS/$A/output $ARC/$A/output && cp $NS/$A/OUTPUTS.sha256 $ARC/$A/ && cd $ARC/$A && sha256sum -c OUTPUTS.sha256 --quiet && echo archive-verified"
timeout 600 ssh -o ConnectTimeout=20 tufts-login "cd $ARC/$A && find output -type f -size +100M -print0 | sort -z | xargs -0 -r sha256sum" > "$LANE/runs/$A/LARGE_FILES.sha256"
echo "archived to $ARC/$A ($(wc -l < "$LANE/runs/$A/LARGE_FILES.sha256") large files)"
timeout 300 ssh -o ConnectTimeout=20 tufts-login "rm -rf $NS/$A"
echo "remote job dir removed"
