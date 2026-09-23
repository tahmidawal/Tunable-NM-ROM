#!/bin/bash
# stage.sh <DEST>: copy this worktree's code (python/json/sh/frozen npz only) into one fresh
# cluster job directory, plus COMMIT.txt. Refuses to reuse a directory.
set -euo pipefail
DEST="$1"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
if ssh tufts-login "test -e $DEST"; then echo "refusing: $DEST exists (dirs are never reused)"; exit 1; fi
ssh tufts-login "mkdir -p $DEST/logs $DEST/output $DEST/experiments"
for pkg in ns3d ns3d-grok ns3d-shift ns3d-shift-head ns3d-operators; do
  rsync -az --delete \
    --exclude='artifacts/' --exclude='runs/' --exclude='data/' --exclude='checks/' --exclude='__pycache__/' \
    --include='*/' --include='*.py' --include='*.json' --include='*.sbatch' --include='*.sh' \
    --include='frozen/**.npz' --include='NEURALOPERATOR-LICENSE' --exclude='*' \
    "$HERE/experiments/$pkg/" "tufts-login:$DEST/experiments/$pkg/"
done
if [[ -n "$(git -C "$HERE" status --porcelain -- experiments/ns3d-operators)" ]]; then
  echo "refusing: uncommitted changes in experiments/ns3d-operators"; exit 1
fi
git -C "$HERE" rev-parse HEAD | ssh tufts-login "cat > $DEST/COMMIT.txt"
