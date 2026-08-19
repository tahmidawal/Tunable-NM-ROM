#!/usr/bin/env bash
# Directly stage one approved cell to paralab, verify content, then submit.
set -euo pipefail

cell="$1"
[[ "$cell" =~ ^nda_[A-Za-z0-9_]+$ ]] || { echo "invalid cell $cell" >&2; exit 2; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE="$HERE/stage/$cell"
REMOTE_ROOT=/cluster/tufts/paralab/tawal01/nmrom_nonlinear_decoder
REMOTE="$REMOTE_ROOT/$cell"
[[ -d "$STAGE" ]] || { echo "missing stage $STAGE" >&2; exit 3; }

echo "queue before submit"
ssh tufts-login "squeue -u tawal01 -o '%i %j %T %M %R'"
queued="$(ssh tufts-login "squeue -u tawal01 -h -o '%j' | grep -cx '$cell' || true")"
[[ "$queued" == 0 ]] || { echo "$cell already queued" >&2; exit 4; }
exists="$(ssh tufts-login "test -e '$REMOTE' && echo yes || echo no")"
[[ "$exists" == no ]] || { echo "remote cell already exists: $REMOTE" >&2; exit 5; }
ssh tufts-login "df -h /cluster/tufts/paralab/tawal01 && mkdir -p '$REMOTE'"
scp -q -r "$STAGE"/. "tufts-login:$REMOTE/"
local_sum="$(cd "$STAGE" && find . -type f -exec sha256sum {} \; | sort | sha256sum | cut -d' ' -f1)"
remote_sum="$(ssh tufts-login "cd '$REMOTE' && find . -type f -exec sha256sum {} \\; | sort | sha256sum | cut -d' ' -f1")"
[[ "$local_sum" == "$remote_sum" ]] || { echo "post-scp checksum mismatch" >&2; exit 6; }
jid="$(ssh tufts-login "cd '$REMOTE' && sbatch --parsable run.sbatch")"
echo "queue after submit"
ssh tufts-login "squeue -u tawal01 -o '%i %j %T %M %R'"
echo "$cell job=$jid tree_sha256=$local_sum"

