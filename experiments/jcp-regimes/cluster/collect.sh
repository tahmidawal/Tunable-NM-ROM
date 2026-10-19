#!/bin/bash
# jcp-regimes: collect one finished job (COMMON-BRIEF §16). Pull JSONs, logs, small arrays (slim references, R files,
# controls, probes); archive every large raw array (full reference npz, F_ fields) to burgers-archive with a verified
# sha256 listing; record sacct (UTC); assemble store/ from slim outputs; then delete the namespace job directory.
set -euo pipefail
A="$1"; J="$2"; [[ "$A" =~ ^[a-zA-Z0-9]+$ ]]
LANE=$(cd "$(dirname "$0")/.." && pwd)
NS=/cluster/tufts/paralab/tawal01/jcp-regimes/$A
AR=/cluster/tufts/paralab/tawal01/burgers-archive/jcp-regimes/$A
LOCAL=$LANE/runs/$A
mkdir -p "$LOCAL"
# raw arrays to the archive: full reference case npz (c*.npz under refjob_*/2d) and F_ files
timeout 1800 ssh -o ConnectTimeout=20 tufts-login "set -euo pipefail; mkdir -p $AR; cd $NS; \
  find output -type f \( -path 'output/refjob_*/2d/*/ST/c*.npz' -o -name 'F_*.npz' -o -path 'output/refjob_*/2d/*/PS/*.npz' \) -print0 | LC_ALL=C sort -z > /tmp/jg_$A.list; \
  xargs -0 -a /tmp/jg_$A.list -I{} cp --parents {} $AR/; \
  (xargs -0 -a /tmp/jg_$A.list sha256sum) | LC_ALL=C sort -k2 > $AR/ARCHIVE.sha256; \
  (cd $AR && sha256sum -c ARCHIVE.sha256 --quiet) && echo ARCHIVE-VERIFIED; rm -f /tmp/jg_$A.list" > "$LOCAL/ARCHIVE.log"
grep -q ARCHIVE-VERIFIED "$LOCAL/ARCHIVE.log"
timeout 120 ssh -o ConnectTimeout=20 tufts-login "cat $AR/ARCHIVE.sha256" > "$LOCAL/ARCHIVE.sha256"
# pull everything except the archived raw arrays (PS npz are small and needed: re-include them)
rsync -a --exclude 'F_*.npz' --exclude 'output/refjob_*/2d/*/ST/c*.npz' --exclude cache --exclude tmp --exclude experiments \
  "tufts-login:$NS/" "$LOCAL/"
S=$(timeout 60 ssh -o ConnectTimeout=20 tufts-login "sacct -j $J -X -n -P -o State,Elapsed,Start,End" | head -1)
/home/tahmid/Dev/.venv/bin/python - "$S" "$LOCAL/SACCT.json" "$J" <<'PY'
import sys, json, datetime as dt, time
st, el, s, e = sys.argv[1].split('|')
off = -time.altzone if time.localtime().tm_isdst else -time.timezone      # cluster and this host share US/Eastern
tz = dt.timezone(dt.timedelta(seconds=off))
iso = lambda x: dt.datetime.fromisoformat(x).replace(tzinfo=tz).isoformat()
json.dump(dict(State=st, Elapsed=el, Start=iso(s), End=iso(e), job=sys.argv[3]), open(sys.argv[2], 'w'), indent=1)
PY
# store/ from slim outputs
for d in "$LOCAL"/output/refslim_*/2d/*/ST; do
  [ -d "$d" ] || continue
  g=$(basename "$(dirname "$d")"); mkdir -p "$LANE/store/2d/$g/ST"; cp -p "$d"/c*.npz "$d"/c*.json "$LANE/store/2d/$g/ST/"
done
grep -q "ALL-DONE" "$LOCAL"/logs/*.out
timeout 120 ssh -o ConnectTimeout=20 tufts-login "case $NS in /cluster/tufts/paralab/tawal01/jcp-regimes/[a-zA-Z0-9]*) rm -rf $NS;; esac"
echo "COLLECTED $A"
