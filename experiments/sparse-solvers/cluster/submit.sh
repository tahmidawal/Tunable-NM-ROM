#!/usr/bin/env bash
# Stage the committed lane files into a FRESH job directory of this lane's namespace and submit once.
# Usage: experiments/sparse-solvers/cluster/submit.sh <attempt> <HH:MM:SS> <gpu feature> <tasks file> [per-task timeout s]
set -euo pipefail
cd "$(dirname "$0")/.."
ATTEMPT=$1; WALL=$2; FEAT=$3; TASKS=$4; TMO=${5:-5400}
NS=/cluster/tufts/paralab/tawal01/sparse-solvers-20261113
DIR=$NS/jobs/$ATTEMPT
[ -z "$(git status --porcelain -- .)" ] || { echo "uncommitted changes; commit first"; exit 2; }
COMMIT=$(git rev-parse HEAD)
FILES=(bench.py matrices.py cluster/run_gpu.sh "$TASKS")
mine=$(ssh tufts-login "squeue -u \$USER -h -p gpu -o '%j'" | grep -c '^ss_' || true)
[ "$mine" -lt 3 ] || { echo "already $mine GPU jobs of this lane queued or running"; exit 6; }
ssh tufts-login "test ! -e $DIR && mkdir -p $DIR" || { echo "attempt directory exists: $DIR"; exit 4; }
tar -cf - "${FILES[@]}" | ssh tufts-login "cd $DIR && tar -xf - && echo $COMMIT > SOURCE_COMMIT.txt && sha256sum ${FILES[*]} > SOURCE-SHA256SUMS"
[ "$(sha256sum "${FILES[@]}")" = "$(ssh tufts-login "cd $DIR && cat SOURCE-SHA256SUMS")" ] || { echo "staged checksum mismatch"; exit 3; }
JOB=$(ssh tufts-login "cd $DIR && sbatch --parsable --job-name=ss_$ATTEMPT --time=$WALL --constraint=$FEAT cluster/run_gpu.sh $TASKS $TMO")
n=$(ssh tufts-login "squeue -u \$USER -h -o '%i %Z'" | awk -v d="$DIR" '$2==d' | wc -l)
[ "$n" -eq 1 ] || { echo "expected exactly one job in $DIR, found $n (job $JOB)"; exit 5; }
echo "submitted job=$JOB dir=$DIR commit=$COMMIT"
mkdir -p ledger
printf '{"attempt":"%s","job":"%s","dir":"%s","commit":"%s","wall":"%s","gpu_feature":"%s","tasks":"%s","submitted_utc":"%s"}\n' \
  "$ATTEMPT" "$JOB" "$DIR" "$COMMIT" "$WALL" "$FEAT" "$TASKS" "$(date -u +%FT%TZ)" >> ledger/JOBS.jsonl
