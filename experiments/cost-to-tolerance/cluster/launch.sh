#!/usr/bin/env bash
# launch.sh <cell> [--gpu a100|h100|h200|l40s]
#   scp the staged cell DIRECTLY into /cluster/tufts/paralab/tawal01/ctol/<cell>
#   (never through login /tmp -- login nodes are load balanced with node-local
#   /tmp and a stale script silently corrupted a round once), verify sha256,
#   check squeue BEFORE and AFTER, sbatch.  One job per directory.
set -euo pipefail
cell="$1"; shift || true
gpu=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) gpu="$2"; shift 2;;
    *) echo "unknown flag $1" >&2; exit 1;;
  esac
done
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE="$HERE/stage/$cell"
REMOTE="/cluster/tufts/paralab/tawal01/ctol/$cell"
[[ -d "$STAGE" ]] || { echo "no stage dir $STAGE (run make_cells.sh first)" >&2; exit 1; }
[[ "$cell" =~ ^[A-Za-z0-9_]+$ ]] || { echo "refusing cell name '$cell'" >&2; exit 4; }

before=$(ssh tufts-login "squeue -u tawal01 -h -o '%j' | grep -cx '$cell' || true")
[[ "$before" == "0" ]] || { echo "a job named $cell is already queued ($before) -- abort" >&2; exit 2; }

ssh tufts-login "rm -rf $REMOTE && mkdir -p $REMOTE"
scp -q -r "$STAGE"/. "tufts-login:$REMOTE/"
remote_sum=$(ssh tufts-login "cd $REMOTE && find . -type f -not -name MANIFEST.sha256 -exec sha256sum {} \; | sort | sha256sum | cut -d' ' -f1")
local_sum=$(cd "$STAGE" && find . -type f -not -name MANIFEST.sha256 -exec sha256sum {} \; | sort | sha256sum | cut -d' ' -f1)
[[ "$remote_sum" == "$local_sum" ]] || { echo "CHECKSUM MISMATCH after scp for $cell" >&2; exit 3; }

extra=""
[[ -n "$gpu" ]] && extra="--constraint=$gpu"
jid=$(ssh tufts-login "cd $REMOTE && sbatch --parsable $extra run.sbatch")
after=$(ssh tufts-login "squeue -u tawal01 -h -o '%j' | grep -cx '$cell' || true")
echo "$cell job=$jid queued_with_name=$after"
[[ "$after" == "1" ]] || { echo "WARNING: $after jobs named $cell in the queue" >&2; exit 5; }
