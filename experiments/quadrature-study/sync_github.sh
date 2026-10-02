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
# Usage (after every `git commit` on the lane): bash experiments/quadrature-study/sync_github.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

LOCAL_BASE=21c175a1b5cef353845d370cbed9599b545e2248    # exp/2026-10-01-ns3d-coordnet-bank (fork point)
REMOTE_BASE=1716d80deadb7d383c836d668840d5acc0730fc2   # its codeonly/ mirror
MIRROR=codeonly/$(git rev-parse --abbrev-ref HEAD)
MAX_BYTES=$((50 * 1024 * 1024))

git fetch -q origin "+refs/heads/$MIRROR:refs/remotes/origin/$MIRROR" 2>/dev/null || true
PARENT=$(git rev-parse -q --verify "refs/remotes/origin/$MIRROR" || echo "$REMOTE_BASE")

export GIT_INDEX_FILE
GIT_INDEX_FILE=$(mktemp)
trap 'rm -f "$GIT_INDEX_FILE"' EXIT
git read-tree "$REMOTE_BASE^{tree}"

git diff --no-renames --name-status -z "$LOCAL_BASE" HEAD |
while IFS= read -r -d '' status && IFS= read -r -d '' path; do
  if [ "$status" = D ]; then
    git update-index --force-remove -- "$path"
    continue
  fi
  read -r mode type sha size _ < <(git ls-tree -l HEAD -- "$path")
  if [ "$type" = blob ] && [ "$size" -gt "$MAX_BYTES" ]; then
    echo "skipped (>${MAX_BYTES} bytes, track by SHA256): $path"
    git update-index --force-remove -- "$path"
    continue
  fi
  git update-index --add --cacheinfo "$mode,$sha,$path"
done

TREE=$(git write-tree)
if [ "$TREE" = "$(git rev-parse "$PARENT^{tree}")" ]; then
  echo "mirror up to date: origin/$MIRROR"
  exit 0
fi
MSG="mirror of $(git rev-parse --short HEAD): $(git log -1 --format=%s HEAD)"
NEW=$(git commit-tree "$TREE" -p "$PARENT" -m "$MSG")
git push -q origin "$NEW:refs/heads/$MIRROR"
echo "pushed origin/$MIRROR -> ${NEW:0:9} ($MSG)"
