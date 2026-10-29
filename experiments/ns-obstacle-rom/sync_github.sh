#!/usr/bin/env bash
# Mirror this lane's committed work to GitHub without pushing the lane branch itself.
#
# The lane branch descends from history carrying multi-GB archive blobs that cannot
# reach GitHub, so the lane is mirrored onto origin/codeonly/<branch>, whose history is
# the filtered code-only mirror. The mirror tree is the parent mirror's base tree plus
# every path changed on the lane since LOCAL_BASE (taken from HEAD), minus files over
# MAX_BYTES, which are listed on stdout and must be tracked by SHA256 in a manifest.
# Idempotent: commits and pushes only when the mirrored tree changes.
#
# Usage (after every `git commit` on the lane): bash experiments/ns-obstacle-rom/sync_github.sh
set -euo pipefail
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=gc.auto GIT_CONFIG_VALUE_0=0
cd "$(git rev-parse --show-toplevel)"

LOCAL_BASE=93f399faad67f94ea2b56fe25f13003143e64694
REMOTE_BASE=5a64899fdcfbeb6cfe5b3371af1a8d8a3d2b568f
MIRROR=codeonly/$(git rev-parse --abbrev-ref HEAD)
MAX_BYTES=$((50 * 1024 * 1024))
SOURCE_HEAD=$(git rev-parse HEAD)

git fetch -q origin "+refs/heads/$MIRROR:refs/remotes/origin/$MIRROR" 2>/dev/null || true
PARENT=$(git rev-parse -q --verify "refs/remotes/origin/$MIRROR" || echo "$REMOTE_BASE")

export GIT_INDEX_FILE
GIT_INDEX_FILE=$(mktemp)
trap 'rm -f "$GIT_INDEX_FILE"' EXIT
git read-tree "$REMOTE_BASE^{tree}"

/home/tahmid/Dev/.venv/bin/python experiments/ns-obstacle-rom/retention/build_mirror_index.py \
  --source "$SOURCE_HEAD" --base "$LOCAL_BASE" --max-bytes "$MAX_BYTES"

TREE=$(git write-tree)
if [ "$TREE" = "$(git rev-parse "$PARENT^{tree}")" ]; then
  echo "mirror up to date: origin/$MIRROR"
  exit 0
fi
MSG="mirror of $(git rev-parse --short "$SOURCE_HEAD"): $(git log -1 --format=%s "$SOURCE_HEAD")"
NEW=$(git commit-tree "$TREE" -p "$PARENT" -m "$MSG")
git push -q origin "$NEW:refs/heads/$MIRROR"
echo "pushed origin/$MIRROR -> ${NEW:0:9} ($MSG)"
