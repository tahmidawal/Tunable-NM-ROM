#!/usr/bin/env bash
# launch.sh <cell> : stage (already built by make_cell.sh) -> scp into paralab
# blat/<cell> -> verify checksums -> squeue before/after -> sbatch.
# One job per cell dir.  Prints the job id.
set -euo pipefail
cell="$1"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE="$HERE/stage/$cell"
REMOTE="/cluster/tufts/paralab/tawal01/blat2/$cell"
[[ -d "$STAGE" ]] || { echo "no stage dir $STAGE" >&2; exit 1; }
before=$(ssh tufts-login "squeue -u tawal01 -h -o '%i %j' | grep -c ' $cell\$' || true")
if [[ "$before" != "0" ]]; then echo "job named $cell already queued ($before) -- abort" >&2; exit 2; fi
[[ "$cell" =~ ^[A-Za-z0-9_]+$ ]] || { echo "refusing to touch remote dir for cell name '$cell'" >&2; exit 4; }
# recreate the cell dir from scratch: leftovers from an earlier run would break the
# post-scp checksum comparison (and could be silently mistaken for this run's output)
ssh tufts-login "rm -rf $REMOTE && mkdir -p $REMOTE"
scp -q -r "$STAGE"/. "tufts-login:$REMOTE/"
remote_sum=$(ssh tufts-login "cd $REMOTE && find . -type f -not -name MANIFEST.sha256 -exec sha256sum {} \; | sort | sha256sum | cut -d' ' -f1")
local_sum=$(cd "$STAGE" && find . -type f -not -name MANIFEST.sha256 -exec sha256sum {} \; | sort | sha256sum | cut -d' ' -f1)
if [[ "$remote_sum" != "$local_sum" ]]; then echo "CHECKSUM MISMATCH after scp for $cell" >&2; exit 3; fi
jid=$(ssh tufts-login "cd $REMOTE && sbatch --parsable run.sbatch")
after=$(ssh tufts-login "squeue -u tawal01 -h -o '%i %j' | grep -c ' $cell\$' || true")
echo "$cell job=$jid queued_with_name=$after"
[[ "$after" == "1" ]] || { echo "WARNING: $after jobs named $cell in queue" >&2; }
