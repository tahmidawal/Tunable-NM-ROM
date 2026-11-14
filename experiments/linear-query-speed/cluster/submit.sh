#!/usr/bin/env bash
# Stage the committed lane files into a FRESH job directory of this lane's namespace and submit once.
# Usage: experiments/linear-query-speed/cluster/submit.sh <attempt> <HH:MM:SS> <gpu feature> "<selection meshes>" "<test meshes>" ["<paper-driver meshes>"]
set -euo pipefail
cd "$(dirname "$0")/.."
ATTEMPT=$1; WALL=$2; FEAT=$3; SEL=$4; MESHES=$5; PAPER=${6:-}
# if <selection meshes> is the word "testfrozen": submit.sh <attempt> <wall> <feature> testfrozen <frozen file> "<meshes>" ["<extra args>"]
if [ "$SEL" = testfrozen ]; then RUNARGS="testfrozen '$5' '${6:-}' '${7:-}'"; MESHES="${6:-}"; PAPER="frozen=$5 extra=${7:-}"; elif [ "$SEL" = heat2d ]; then RUNARGS="heat2d '$5'"; PAPER=""; else RUNARGS="auto '$SEL' '$MESHES' '$PAPER'"; fi
NS=/cluster/tufts/paralab/tawal01/linear-query-speed-20261114
DIR=$NS/jobs/$ATTEMPT
[ -z "$(git status --porcelain -- . ":!ledger" ":!LEDGER.md")" ] || { echo "uncommitted changes; commit first"; exit 2; }
COMMIT=$(git rev-parse HEAD)
mapfile -t FILES < <(git ls-files -- '*.py' '*.json' paper_code paper_code_heat cluster/run_gpu.sh | sort -u)
mine=$(ssh tufts-login "squeue -u \$USER -h -p gpu -o '%j'" | grep -c '^lqs_' || true)
[ "$mine" -lt 2 ] || { echo "already $mine GPU jobs of this lane queued or running"; exit 6; }
ssh tufts-login "test ! -e $DIR && mkdir -p $DIR" || { echo "attempt directory exists: $DIR"; exit 4; }
tar -cf - "${FILES[@]}" | ssh tufts-login "cd $DIR && tar -xf - && echo $COMMIT > SOURCE_COMMIT.txt"
sha256sum "${FILES[@]}" | ssh tufts-login "cd $DIR && cat > SOURCE-SHA256SUMS && sha256sum -c --quiet SOURCE-SHA256SUMS" || { echo "staged checksum mismatch"; exit 3; }
JOB=$(ssh tufts-login "cd $DIR && sbatch --parsable --job-name=lqs_$ATTEMPT --time=$WALL --constraint=$FEAT ${LQS_MEM:+--mem=$LQS_MEM} cluster/run_gpu.sh $RUNARGS")
n=$(ssh tufts-login "squeue -u \$USER -h -o '%i %Z'" | awk -v d="$DIR" '$2==d' | wc -l)
[ "$n" -eq 1 ] || { echo "expected exactly one job in $DIR, found $n (job $JOB)"; exit 5; }
echo "submitted job=$JOB dir=$DIR commit=$COMMIT"
mkdir -p ledger
printf '{"attempt":"%s","job":"%s","dir":"%s","commit":"%s","wall":"%s","gpu_feature":"%s","selection_meshes":"%s","test_meshes":"%s","paper_driver_meshes":"%s","submitted_utc":"%s"}\n' \
  "$ATTEMPT" "$JOB" "$DIR" "$COMMIT" "$WALL" "$FEAT" "$SEL" "$MESHES" "$PAPER" "$(date -u +%FT%TZ)" >> ledger/JOBS.jsonl
