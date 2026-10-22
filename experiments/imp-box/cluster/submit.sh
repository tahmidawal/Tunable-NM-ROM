#!/bin/bash
# usage: submit.sh <jobname> <gpu> <mem> <time> <bodyfile>
# Stages the COMMITTED lane tree (git archive of HEAD) into /cluster/.../imp-box/<jobname>/code and submits.
# Two acquisitions of the campaign lock: the reservation (directory must not exist; lane cap 3; account cap 8) and the
# sbatch itself (caps re-checked). Prints the job id. After submission, launch cluster/wait_collect.sh and record its PID.
set -euo pipefail
NAME=$1; GPU=$2; MEM=$3; TIME=$4; BODY=$5
[[ "$NAME" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "bad job name" >&2; exit 2; }
[ "$GPU" = h200 ] || { echo "imp-box times on H200 only (DESIGN §5)" >&2; exit 2; }
LANE=$(cd "$(dirname "$0")/.." && pwd)
REPO=$(git -C "$LANE" rev-parse --show-toplevel)
REL=${LANE#$REPO/}
NS=/cluster/tufts/paralab/tawal01/imp-box
DIR=$NS/$NAME
LOCK=/cluster/tufts/paralab/tawal01/.campaign-sbatch.lock
FULL=$(git -C "$LANE" rev-parse HEAD)
COMMIT=${FULL:0:9}
git -C "$LANE" diff --quiet HEAD -- . ":(exclude)HEARTBEAT.json" ":(exclude)runs" || { echo "uncommitted changes in the lane: commit first" >&2; exit 4; }
# budget gate: the job must be listed in the committed release file (written from resources.py after the job audit)
# the approval binds the exact job specification: body content hash, GPU, memory, time (code audit 2)
BSHA=$(git -C "$REPO" show "$FULL:$REL/${BODY#$LANE/}" | sha256sum | cut -d' ' -f1)
git -C "$REPO" show "$FULL:$REL/cluster/RELEASE.json" | /home/tahmid/Dev/.venv/bin/python -c "
import json, sys
r = json.load(sys.stdin)['approved'].get(sys.argv[1])
ok = r is not None and r['body_sha256'] == sys.argv[2] and r['gpu'] == sys.argv[3] and r['mem'] == sys.argv[4] \
    and r['time'] == sys.argv[5]
sys.exit(0 if ok else 6)" "$NAME" "$BSHA" "$GPU" "$MEM" "$TIME" \
  || { echo "job $NAME (body $BSHA, $GPU $MEM $TIME) is not approved in the committed cluster/RELEASE.json" >&2; exit 6; }
if git -C "$REPO" show "$FULL:$REL/${BODY#$LANE/}" | grep -q -- '--stages time\|in .*time'; then
  git -C "$REPO" show "$FULL:$REL/${BODY#$LANE/}" | grep -q controls || { echo "body runs time without controls" >&2; exit 2; }
fi
CAPS='set -o pipefail; q=$(squeue -u tawal01 -h -o %j) || exit 76; n=$(printf "%s" "$q" | grep -c . || true); l=$(printf "%s" "$q" | grep -c "^ibox-" || true); [ $n -lt 8 ] && [ $l -lt 3 ]'
timeout 700 ssh -o ConnectTimeout=20 tufts-login "flock -w 600 $LOCK bash -c '$CAPS || exit 75; test ! -e $DIR || { echo job dir exists; exit 3; }; mkdir -p $DIR/code && touch $DIR/reserved'"
git -C "$REPO" archive --format=tar "$FULL" "$REL" | tar -xOf - >/dev/null   # verify archive of the recorded commit
git -C "$REPO" archive --format=tar "$FULL" "$REL" | timeout 300 ssh -o ConnectTimeout=20 tufts-login "tar -C $DIR/code -xf - --strip-components=$(echo "$REL" | awk -F/ '{print NF}')"
git -C "$REPO" ls-tree -r "$FULL" "$REL" | timeout 60 ssh -o ConnectTimeout=20 tufts-login "cat > $DIR/code/.git_tree.txt"   # test-stage authentication
TMP=$(mktemp)
CONS=none
# template and body are read from the recorded commit (executed inputs are committed content)
git -C "$REPO" show "$FULL:$REL/cluster/job.sbatch.tmpl" | sed -e "s#__NAME__#$NAME#g; s#__GPU__#$GPU#g; s#__MEM__#$MEM#g; s#__TIME__#$TIME#g; s#__DIR__#$DIR#g; s#__COMMIT__#$COMMIT#g; s#__CONSTRAINT__#$CONS#g" > "$TMP"
BODYC=$(mktemp); git -C "$REPO" show "$FULL:$REL/${BODY#$LANE/}" > "$BODYC" || { echo "body must be a committed lane file (path relative to the lane)"; exit 5; }
BODY=$BODYC
/home/tahmid/Dev/.venv/bin/python - "$BODY" "$TMP" <<'PY'
import sys
body = open(sys.argv[1]).read(); f = sys.argv[2]; s = open(f).read().replace('__BODY__', body); open(f, 'w').write(s)
PY
timeout 60 ssh -o ConnectTimeout=20 tufts-login "cat > $DIR/run.sbatch" < "$TMP"; rm -f "$TMP"
timeout 60 ssh -o ConnectTimeout=20 tufts-login 'squeue -u tawal01 -h | wc -l' | sed 's/^/before: queued+running=/'
timeout 700 ssh -o ConnectTimeout=20 tufts-login "flock -w 600 $LOCK bash -c '$CAPS || exit 75; test ! -e $DIR/submitted || exit 3; cd $DIR && sbatch --parsable run.sbatch > submitted && cat submitted'"
timeout 60 ssh -o ConnectTimeout=20 tufts-login "squeue -u tawal01 -h -o '%i %j %T'"
