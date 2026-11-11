#!/usr/bin/env bash
# Stage the committed lane sources into a FRESH cluster attempt directory and submit once.
# Usage: cluster/submit.sh <attempt-name> <HH:MM:SS> <task.sh> [task args...]
# Optional env GRES (default gpu:1), e.g. GRES=gpu:a100:1.
set -euo pipefail
cd "$(dirname "$0")/.."
ATTEMPT=$1; WALL=$2; shift 2
NS=/cluster/tufts/paralab/tawal01/rebuttal-cyl-ops
DIR=$NS/$ATTEMPT
git -c gc.auto=0 diff --quiet HEAD -- . || { echo "uncommitted changes in lane; commit first"; exit 2; }
[ -z "$(git -c gc.auto=0 ls-files --others --exclude-standard -- '*.py' '*.sh')" ] || { echo "untracked sources; commit first"; exit 2; }
COMMIT=$(git -c gc.auto=0 rev-parse HEAD)
mapfile -t FILES < <(git -c gc.auto=0 ls-files -- . ':!ledger' ':!artifacts' ':!*.md')
echo "squeue before:"; ssh tufts-login "squeue -u \$USER -h -o '%i %j %T %Z'" | grep rebuttal-cyl-ops || echo "(no lane jobs)"
ssh tufts-login "mkdir -p $NS && test ! -e $DIR && mkdir $DIR" || { echo "attempt directory exists or cannot be created: $DIR"; exit 4; }
tar -cf - "${FILES[@]}" | ssh tufts-login "cd $DIR && tar -xf - && echo $COMMIT > SOURCE_COMMIT.txt && sha256sum ${FILES[*]} > SOURCE-SHA256SUMS"
[ "$(sha256sum "${FILES[@]}")" = "$(ssh tufts-login "cd $DIR && cat SOURCE-SHA256SUMS")" ] || { echo "staged checksum mismatch"; exit 3; }
JOB=$(ssh tufts-login "cd $DIR && sbatch --parsable --gres=${GRES:-gpu:1} --job-name=cylops_$ATTEMPT --time=$WALL cluster/run.sh $*")
AFTER=$(ssh tufts-login "squeue -u \$USER -h -o '%i %T %Z'")
echo "squeue after:"; echo "$AFTER" | grep rebuttal-cyl-ops || true
n=$(echo "$AFTER" | awk -v d="$DIR" '$3==d' | wc -l)
[ "$n" -eq 1 ] || { echo "expected exactly one job in $DIR, found $n (job $JOB)"; exit 5; }
echo "submitted job=$JOB dir=$DIR commit=$COMMIT"
mkdir -p ledger
printf '{"attempt":"%s","job":"%s","dir":"%s","commit":"%s","wall":"%s","task":"%s","submitted_utc":"%s"}\n' \
  "$ATTEMPT" "$JOB" "$DIR" "$COMMIT" "$WALL" "$*" "$(date -u +%FT%TZ)" >> ledger/JOBS.jsonl
