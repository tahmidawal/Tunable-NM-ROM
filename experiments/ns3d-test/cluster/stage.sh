#!/bin/bash
# stage.sh <DEST>: copy this worktree's code (python/json/sh/frozen npz only) into one fresh
# cluster job directory, plus COMMIT.txt. Refuses a dirty tree and refuses to reuse a directory.
set -euo pipefail
DEST="$1"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
if [[ -n "$(git -C "$HERE" status --porcelain -- experiments ':!experiments/ns3d-test/runs')" ]]; then
  echo "refusing: uncommitted changes under experiments/"; exit 1
fi
ssh tufts-login "mkdir -p $(dirname $DEST) && mkdir $DEST" || { echo "refusing: $DEST exists"; exit 1; }
ssh tufts-login "mkdir -p $DEST/logs $DEST/output $DEST/experiments"
for pkg in ns3d ns3d-grok ns3d-shift ns3d-shift-head ns3d-operators ns3d-test; do
  rsync -az --delete \
    --exclude='artifacts/' --exclude='runs/' --exclude='data/' --exclude='checks/' --exclude='__pycache__/' \
    --exclude='reports/' --include='*/' --include='*.py' --include='*.json' --include='*.sbatch' --include='*.sh' \
    --include='frozen/**.npz' --exclude='*' \
    "$HERE/experiments/$pkg/" "tufts-login:$DEST/experiments/$pkg/"
done
git -C "$HERE" rev-parse HEAD | ssh tufts-login "cat > $DEST/COMMIT.txt"
