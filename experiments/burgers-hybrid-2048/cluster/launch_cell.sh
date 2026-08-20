#!/usr/bin/env bash
# launch_cell.sh <cell>: stage directly to the isolated paralab namespace.
set -euo pipefail
cell="$1"
if [[ "$cell" != ctol_hybb2048_* ]]; then
  echo "cell must match ctol_hybb2048_*" >&2
  exit 2
fi
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE="$HERE/stage/$cell"
REMOTE="/cluster/tufts/paralab/tawal01/hybb2048/$cell"

[[ -f "$STAGE/run.sbatch" ]] || { echo "missing stage for $cell" >&2; exit 2; }
ssh tufts-login "test ! -e '$REMOTE' || { echo remote-exists >&2; exit 3; }; mkdir -p '$REMOTE'"
scp -q -r "$STAGE"/. "tufts-login:$REMOTE/"
ssh tufts-login "cd '$REMOTE' && sha256sum -c MANIFEST.sha256 && sbatch run.sbatch"
