#!/usr/bin/env bash
# Stage the committed sources (source-lane files + this lane's files, flattened into one job directory as the source
# lane does) into a FRESH attempt directory of this lane's namespace and submit once.
# Usage: experiments/cylinder-amg/cluster/submit.sh <attempt> <HH:MM:SS> <task.sh>
set -euo pipefail
cd "$(dirname "$0")/.."
LANE=$(pwd); SRC=$LANE/../cylinder-unsteady-rom
ATTEMPT=$1; WALL=$2; TASK=$3
NS=/cluster/tufts/paralab/tawal01/cyl-amg-20261113
DIR=$NS/$ATTEMPT
git diff --quiet HEAD -- . ../cylinder-unsteady-rom || { echo "uncommitted changes; commit first"; exit 2; }
COMMIT=$(git rev-parse HEAD)
SFILES=(vendor/ref/transient.py vendor/ref/cycles.py vendor/ref/vendor/scalars.py vendor/ref/vendor/oriented_normals.py
        vendor/dependencies.py vendor/DEPENDENCIES.json rom_r4.py rom_core.py cycles.py fe_ladder2.py inputs/admitted_mesh.vol.bin
        tasks/manifest_production.json)
LFILES=(amg_stepper.py fe_amg.py install_deps.py cluster/run.sh "$TASK" tasks/MESHES.json tasks/entries_dev.json
        tasks/entries_smoke.json tasks/entries_heldout.json)
ssh tufts-login "test ! -e $DIR && mkdir -p $DIR" || { echo "attempt directory exists: $DIR"; exit 4; }
( cd "$SRC" && tar -cf - "${SFILES[@]}" ) | ssh tufts-login "cd $DIR && tar -xf -"
tar -cf - "${LFILES[@]}" | ssh tufts-login "cd $DIR && tar -xf - && echo $COMMIT > SOURCE_COMMIT.txt && sha256sum ${SFILES[*]} ${LFILES[*]} > SOURCE-SHA256SUMS"
local_sums=$( (cd "$SRC" && sha256sum "${SFILES[@]}"); sha256sum "${LFILES[@]}")
remote_sums=$(ssh tufts-login "cd $DIR && cat SOURCE-SHA256SUMS")
[ "$local_sums" = "$remote_sums" ] || { echo "staged checksum mismatch"; exit 3; }
JOB=$(ssh tufts-login "cd $DIR && sbatch --parsable --job-name=cylamg_$ATTEMPT --time=$WALL cluster/run.sh $TASK")
n=$(ssh tufts-login "squeue -u \$USER -h -o '%i %Z'" | awk -v d="$DIR" '$2==d' | wc -l)
[ "$n" -eq 1 ] || { echo "expected exactly one job in $DIR, found $n (job $JOB)"; exit 5; }
echo "submitted job=$JOB dir=$DIR commit=$COMMIT"
mkdir -p ledger
printf '{"attempt":"%s","job":"%s","dir":"%s","commit":"%s","wall":"%s","task":"%s","submitted_utc":"%s"}\n' \
  "$ATTEMPT" "$JOB" "$DIR" "$COMMIT" "$WALL" "$TASK" "$(date -u +%FT%TZ)" >> ledger/JOBS.jsonl
