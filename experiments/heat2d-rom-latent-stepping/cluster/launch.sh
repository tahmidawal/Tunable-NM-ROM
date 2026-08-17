#!/usr/bin/env bash
# launch.sh <cell> : stage (already built by make_cell.sh) -> scp into paralab
# blat/<cell> -> verify checksums -> squeue before/after -> sbatch.
# One job per cell dir.  Prints the job id.
set -euo pipefail
cell="$1"
[[ "$cell" =~ ^[a-z0-9_]+$ ]] || { echo "bad cell name '$cell'" >&2; exit 1; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE="$HERE/stage/$cell"
REMOTE="/cluster/tufts/paralab/tawal01/hlat/$cell"
[[ -d "$STAGE" ]] || { echo "no stage dir $STAGE" >&2; exit 1; }
before=$(ssh tufts-login "squeue -u tawal01 -h -o '%i %j' | grep -c ' $cell\$' || true")
if [[ "$before" != "0" ]]; then echo "job named $cell already queued ($before) -- abort" >&2; exit 2; fi
# refuse to reuse a directory that already holds results (one job per dir, ever)
existing=$(ssh tufts-login "ls $REMOTE/out 2>/dev/null | wc -l" || echo 0)
if [[ "${existing:-0}" != "0" ]]; then echo "$REMOTE/out is non-empty -- refusing to reuse" >&2; exit 4; fi
ssh tufts-login "mkdir -p $REMOTE"
scp -q -r "$STAGE"/. "tufts-login:$REMOTE/"
remote_sum=$(ssh tufts-login "cd $REMOTE && find . -type f -not -name MANIFEST.sha256 -exec sha256sum {} \; | sort | sha256sum | cut -d' ' -f1")
local_sum=$(cd "$STAGE" && find . -type f -not -name MANIFEST.sha256 -exec sha256sum {} \; | sort | sha256sum | cut -d' ' -f1)
if [[ "$remote_sum" != "$local_sum" ]]; then echo "CHECKSUM MISMATCH after scp for $cell" >&2; exit 3; fi
jid=$(ssh tufts-login "cd $REMOTE && sbatch --parsable run.sbatch")
after=$(ssh tufts-login "squeue -u tawal01 -h -o '%i %j' | grep -c ' $cell\$' || true")
echo "$cell job=$jid queued_with_name=$after local_manifest=$local_sum"
[[ "$after" == "1" ]] || { echo "FATAL: $after jobs named $cell in queue -- cancel one" >&2; exit 5; }
