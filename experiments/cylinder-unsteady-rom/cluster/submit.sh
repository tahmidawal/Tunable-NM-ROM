#!/usr/bin/env bash
# Stage the committed lane sources into a FRESH cluster attempt directory and submit once.
# Usage: cluster/submit.sh <attempt-name> <HH:MM:SS> <config.json> [...]
# Requires a clean committed tree; checks squeue before and after; one job per directory.
set -euo pipefail
cd "$(dirname "$0")/.."
LANE=$(pwd)
ATTEMPT=$1; WALL=$2; shift 2
NS=/cluster/tufts/paralab/tawal01/cyl-unsteady-20261103
DIR=$NS/$ATTEMPT
git diff --quiet HEAD -- . || { echo "uncommitted changes in lane; commit first"; exit 2; }
COMMIT=$(git rev-parse HEAD)
FILES=(inputs/EXPORT-SHA256SUMS rom_export.py rom_project.py rom_qcheck.py rom_r5.py vendor/ref/transient.py vendor/ref/cycles.py vendor/ref/vendor/scalars.py vendor/ref/vendor/oriented_normals.py rom_core.py rom_r2.py rom_r4.py bank_train.py cycles.py install_deps.py vendor/geometry.py vendor/accuracy_v2/__init__.py vendor/accuracy_v2/spatial_bank.py vendor/accuracy_v2/fast_spatial_bank.py vendor/accuracy_v2/residual_head_bank.py
       vendor/dependencies.py vendor/DEPENDENCIES.json inputs/run09_bank.npz inputs/admitted_mesh.vol.bin cluster/run.sh "$@")
echo "squeue before:"; ssh tufts-login "squeue -u \$USER -h -o '%i %j %Z'" | grep cyl-unsteady-20261103 || echo "(no campaign jobs)"
ssh tufts-login "test ! -e $DIR && mkdir -p $DIR" || { echo "attempt directory exists or cannot be created: $DIR"; exit 4; }
tar -cf - "${FILES[@]}" | ssh tufts-login "cd $DIR && tar -xf - && echo $COMMIT > SOURCE_COMMIT.txt && sha256sum ${FILES[*]} > SOURCE-SHA256SUMS"
# Verify staged bytes against the local committed bytes.
local_sums=$(sha256sum "${FILES[@]}")
remote_sums=$(ssh tufts-login "cd $DIR && cat SOURCE-SHA256SUMS")
[ "$local_sums" = "$remote_sums" ] || { echo "staged checksum mismatch"; exit 3; }
CFGS=("$@")
JOB=$(ssh tufts-login "cd $DIR && sbatch --parsable --job-name=ctol_cylr_$ATTEMPT --time=$WALL cluster/run.sh ${CFGS[*]}")
echo "squeue after:"; AFTER=$(ssh tufts-login "squeue -u \$USER -h -o '%i %Z'")
echo "$AFTER" | grep cyl-unsteady-20261103 || true
n=$(echo "$AFTER" | awk -v d="$DIR" '$2==d' | wc -l)
[ "$n" -eq 1 ] || { echo "expected exactly one job in $DIR, found $n (job $JOB)"; exit 5; }
echo "submitted job=$JOB dir=$DIR commit=$COMMIT"
mkdir -p ledger
printf '{"attempt":"%s","job":"%s","dir":"%s","commit":"%s","wall":"%s","configs":"%s","submitted_utc":"%s"}\n' \
  "$ATTEMPT" "$JOB" "$DIR" "$COMMIT" "$WALL" "${CFGS[*]}" "$(date -u +%FT%TZ)" >> ledger/JOBS.jsonl
