#!/usr/bin/env bash
# Collect only the compact evidence of a finished attempt (local disk is nearly full): result JSON
# files under results/ (no arrays, no orientation dumps), the job logs, the source commit and
# checksums, and a SHA-256 listing of every remote result file.  Nothing remote is deleted.
# Usage: cluster/collect_json.sh <attempt-name>
set -euo pipefail
cd "$(dirname "$0")/.."
ATTEMPT=$1
DIR=/cluster/tufts/paralab/tawal01/cyl-unsteady-20261103/$ATTEMPT
DEST=artifacts/$ATTEMPT
mkdir -p "$DEST"
ssh tufts-login "cd $DIR && find ./results ./slurm-*.out ./slurm-*.err ./SOURCE_COMMIT.txt ./SOURCE-SHA256SUMS -type f -print0 | sort -z | xargs -0 sha256sum" > "$DEST/REMOTE-SHA256SUMS"
ssh tufts-login "cd $DIR && { find ./results -type f -name '*.json' ! -name normal_orientation.json -size -1000k -print0; find . -maxdepth 1 -type f \( -name 'slurm-*' -o -name SOURCE_COMMIT.txt -o -name SOURCE-SHA256SUMS \) -print0; find ./tasks -type f -print0; } | tar --null -T - -cf -" | tar -xf - -C "$DEST"
( cd "$DEST" && find . -type f ! -name REMOTE-SHA256SUMS | sed 's#^#  #' | grep -F -f - REMOTE-SHA256SUMS | sha256sum -c --quiet - ) && echo "local copies verified: $ATTEMPT"
du -sh "$DEST"
