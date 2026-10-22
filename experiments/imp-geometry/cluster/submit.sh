#!/bin/bash
# usage: submit.sh <jobname> <gpu> <mem> <time> <bodyfile>
# Stages the COMMITTED lane tree (git archive of HEAD) into /cluster/.../imp-geometry/<jobname>/code and submits.
# One flock on the campaign lock covers the reservation (directory must not exist; lane cap 3; account cap 8) and the
# sbatch itself. Prints the job id.
set -euo pipefail
NAME=$1; GPU=$2; MEM=$3; TIME=$4; BODY=$5
LANE=$(cd "$(dirname "$0")/.." && pwd)
REPO=$(git -C "$LANE" rev-parse --show-toplevel)
REL=${LANE#$REPO/}
NS=/cluster/tufts/paralab/tawal01/imp-geometry
DIR=$NS/$NAME
LOCK=/cluster/tufts/paralab/tawal01/.campaign-sbatch.lock
FULL=$(git -C "$LANE" rev-parse HEAD)
COMMIT=${FULL:0:9}
git -C "$LANE" diff --quiet HEAD -- . ":(exclude)HEARTBEAT.json" || { echo "uncommitted changes in the lane: commit first" >&2; exit 4; }
CAPS='set -o pipefail; q=$(squeue -u tawal01 -h -o %j) || exit 76; n=$(printf "%s" "$q" | grep -c . || true); l=$(printf "%s" "$q" | grep -c "^igeo-" || true); [ $n -lt 8 ] && [ $l -lt 3 ]'
timeout 700 ssh -o ConnectTimeout=20 tufts-login "flock -w 600 $LOCK bash -c '$CAPS || exit 75; test ! -e $DIR || { echo job dir exists; exit 3; }; mkdir -p $DIR/code && touch $DIR/reserved'"
git -C "$REPO" archive --format=tar "$FULL" "$REL" | tar -xOf - >/dev/null   # verify archive of the recorded commit
git -C "$REPO" archive --format=tar "$FULL" "$REL" | timeout 300 ssh -o ConnectTimeout=20 tufts-login "tar -C $DIR/code -xf - --strip-components=$(echo "$REL" | awk -F/ '{print NF}')"
TMP=$(mktemp)
CONS=$([ "$GPU" = a100 ] && echo a100-80G || echo "$GPU")
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
