#!/usr/bin/env bash
set -euo pipefail
ATTEMPT=${1:?usage: collect.sh attempt numeric_job_id}
JOB=${2:?numeric job id}
[[ "$ATTEMPT" =~ ^[a-zA-Z0-9]+$ && "$JOB" =~ ^[0-9]+$ ]] || exit 2
HERE=$(cd "$(dirname "$0")" && pwd)
REMOTE=/cluster/tufts/paralab/tawal01/mr_burgers2d_20260907/$ATTEMPT
DST=$HERE/../runs/$ATTEMPT
ACTIVE=$(ssh tufts-login "squeue -u tawal01 -h -o '%i'")
[[ ! "$ACTIVE" =~ (^|[[:space:]])$JOB($|[[:space:]]) ]] || { echo "Job still queued; refusing collection/cleanup"; exit 3; }
# Create checksums after the scheduler closes logs, including failed attempts.
ssh tufts-login "cd '$REMOTE' && find out logs -type f -print0 | sort -z | xargs -0 sha256sum > COLLECT.sha256"
mkdir -p "$DST"
rsync -a "tufts-login:$REMOTE/out/" "$DST/out/"
rsync -a "tufts-login:$REMOTE/logs/" "$DST/logs/"
for file in COMMIT.txt PROVENANCE.json MANIFEST.sha256 COLLECT.sha256 run.sbatch; do
  scp "tufts-login:$REMOTE/$file" "$DST/$file"
done
(cd "$DST" && sha256sum -c COLLECT.sha256 --quiet)
ssh tufts-login "rm -rf -- '$REMOTE'; test ! -e '$REMOTE'; squeue -u tawal01 -o '%.12i %.32j %.8T %.10M %.18R'"
printf '%s\n' "$JOB" > "$DST/JOB_ID.txt"
echo "Verified and removed exact directory: $REMOTE"
