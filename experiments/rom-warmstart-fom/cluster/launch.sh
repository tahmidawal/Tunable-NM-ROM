#!/usr/bin/env bash
# launch.sh <cell> [gpu-constraint]
# scp the staged cell DIRECTLY into the paralab namespace wsfom/<cell> (never via
# a login-node /tmp: login nodes are load-balanced with node-local /tmp and a
# stale staged script silently corrupted a round once), verify checksums, check
# squeue BEFORE and AFTER, then sbatch.  ONE JOB PER DIRECTORY.
set -euo pipefail
cell="$1"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE="$HERE/stage/$cell"
REMOTE="/cluster/tufts/paralab/tawal01/wsfom/$cell"
[[ -d "$STAGE" ]] || { echo "no stage dir $STAGE" >&2; exit 1; }

before=$(ssh tufts-login "squeue -u tawal01 -h -o '%j' | grep -c '^$cell\$' || true")
if [[ "$before" != "0" ]]; then
  echo "a job named $cell is already queued ($before) -- abort (one job per dir)" >&2; exit 2
fi
ssh tufts-login "mkdir -p $REMOTE"
scp -q -r "$STAGE"/. "tufts-login:$REMOTE/"
remote_sum=$(ssh tufts-login "cd $REMOTE && find . -type f -not -name MANIFEST.sha256 -exec sha256sum {} \; | sort | sha256sum | cut -d' ' -f1")
local_sum=$(cd "$STAGE" && find . -type f -not -name MANIFEST.sha256 -exec sha256sum {} \; | sort | sha256sum | cut -d' ' -f1)
if [[ "$remote_sum" != "$local_sum" ]]; then
  echo "CHECKSUM MISMATCH after scp for $cell ($local_sum vs $remote_sum)" >&2; exit 3
fi
jid=$(ssh tufts-login "cd $REMOTE && sbatch --parsable run.sbatch")
after=$(ssh tufts-login "squeue -u tawal01 -h -o '%j' | grep -c '^$cell\$' || true")
echo "$cell job=$jid queued_with_name=$after remote=$REMOTE"
[[ "$after" == "1" ]] || { echo "WARNING: $after jobs named $cell in the queue" >&2; }
