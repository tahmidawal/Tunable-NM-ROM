#!/bin/bash
# stage.sh <DEST> <mesh>: copy this worktree's code (python/json/sh/frozen npz only) and the mesh's
# 8 operator checkpoints (from the ns3d-operators worktree, read-only) into one fresh cluster job
# directory, plus COMMIT.txt. Refuses to reuse a directory or to stage uncommitted code.
set -euo pipefail
DEST="$1"; MESH="$2"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OPS_RUNS="$HERE/../2026-09-23-ns3d-operators/experiments/ns3d-operators/runs"
if [[ -n "$(git -C "$HERE" status --porcelain -- experiments ':!experiments/t2-ns3d-test/runs' ':!experiments/t2-ns3d-test/reports' ':!experiments/t2-ns3d-test/checks')" ]]; then
  echo "refusing: uncommitted changes under experiments/"; exit 1
fi
ssh tufts-login "mkdir -p $(dirname $DEST) && mkdir $DEST" || { echo "refusing: $DEST exists"; exit 1; }
ssh tufts-login "mkdir -p $DEST/logs $DEST/output $DEST/experiments $DEST/checkpoints"
for pkg in ns3d ns3d-grok ns3d-shift ns3d-shift-head ns3d-operators t2-ns3d-test; do
  rsync -az --delete \
    --exclude='artifacts/' --exclude='runs/' --exclude='data/' --exclude='checks/' --exclude='__pycache__/' \
    --include='*/' --include='*.py' --include='*.json' --include='*.sbatch' --include='*.sh' \
    --include='frozen/**.npz' --include='NEURALOPERATOR-LICENSE' --exclude='*' \
    "$HERE/experiments/$pkg/" "tufts-login:$DEST/experiments/$pkg/"
done
TMPD=$(mktemp -d)
for arm in fno-s fno-l unet-s unet-l tsol-s tsol-l don-s don-l; do
  ln -s "$OPS_RUNS/tr${MESH}_${arm}/output/best.pt" "$TMPD/tr${MESH}_${arm}.pt"
done
( cd "$TMPD" && sha256sum *.pt ) > "$TMPD/CHECKPOINTS.sha256"
rsync -azL "$TMPD/" "tufts-login:$DEST/checkpoints/"
rm -rf "$TMPD"
ssh tufts-login "cd $DEST/checkpoints && sha256sum -c --quiet CHECKPOINTS.sha256" && echo "checkpoints verified on cluster"
git -C "$HERE" rev-parse HEAD | ssh tufts-login "cat > $DEST/COMMIT.txt"
