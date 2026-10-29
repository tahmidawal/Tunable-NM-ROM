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
# Usage (after every `git commit` on the lane): bash experiments/ns-obstacle-2d/sync_github.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
# Freeze the committed source while a prior snapshot uploads asynchronously.
SOURCE_HEAD=$(git rev-parse HEAD)

LOCAL_BASE=93f399faad67f94ea2b56fe25f13003143e64694    # exp/2026-09-25-t2-ns3d-test (lane fork point)
REMOTE_BASE=5a64899fdcfbeb6cfe5b3371af1a8d8a3d2b568f   # origin/codeonly/exp/2026-09-25-t2-ns3d-test (its mirror)
MIRROR=codeonly/$(git rev-parse --abbrev-ref HEAD)
MAX_BYTES=$((50 * 1024 * 1024))

git fetch -q origin "+refs/heads/$MIRROR:refs/remotes/origin/$MIRROR" 2>/dev/null || true
PARENT=$(git rev-parse -q --verify "refs/remotes/origin/$MIRROR" || echo "$REMOTE_BASE")

export GIT_INDEX_FILE
GIT_INDEX_FILE=$(mktemp)
trap 'rm -f "$GIT_INDEX_FILE"' EXIT
git read-tree "$REMOTE_BASE^{tree}"

git diff --no-renames --name-status -z "$LOCAL_BASE" "$SOURCE_HEAD" |
while IFS= read -r -d '' status && IFS= read -r -d '' path; do
  if [ "$status" = D ]; then
    git update-index --force-remove -- "$path"
    continue
  fi
  read -r mode type sha size _ < <(git ls-tree -l "$SOURCE_HEAD" -- "$path")
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
MSG="mirror of $(git rev-parse --short "$SOURCE_HEAD"): $(git log -1 --format=%s "$SOURCE_HEAD")"
NEW=$(git commit-tree "$TREE" -p "$PARENT" -m "$MSG")
/home/tahmid/Dev/.venv/bin/python experiments/ns-obstacle-2d/g2/seed_mirror_objects.py "$NEW" "$PARENT"
SEED_REF="refs/heads/codeonly/seed/nsobs2d-${TREE:0:16}"
SEED_PARENT=$(git ls-remote origin "$SEED_REF" | awk '{print $1}')
if [ -n "$SEED_PARENT" ]; then
  git merge-base --is-ancestor "$PARENT" "$SEED_PARENT"
  NEW=$(git commit-tree "$TREE" -p "$SEED_PARENT" -m "$MSG")
  /home/tahmid/Dev/.venv/bin/python - "$NEW" "$SEED_PARENT" <<'AUDIT'
import subprocess,sys
ids=[x.split()[0] for x in subprocess.check_output(['git','rev-list','--objects',sys.argv[1],'^'+sys.argv[2]],text=True).splitlines()]
metadata=subprocess.check_output(['git','cat-file','--batch-check=%(objectsize)'],input=('\n'.join(ids)+'\n').encode()).decode()
size=sum(int(x) for x in metadata.splitlines())
print('final mirror proposed new bytes='+str(size),flush=True)
if size>900*1024**2:raise SystemExit('final payload exceeds bounded push; inspect before upload')
AUDIT
fi
git push -q origin "$NEW:refs/heads/$MIRROR"
echo "pushed origin/$MIRROR -> ${NEW:0:9} ($MSG)"
