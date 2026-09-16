#!/usr/bin/env bash
#
# mirror-code-only.sh — build a code-only mirror of this repository, for pushing to GitHub.
#
# WHY THIS EXISTS
# ---------------
# The working repository is ~212 GiB of Git objects.  99.6 % of that is raw run
# archives: chunked tarballs (`*.part0000`, `archive-parts/part-0000`), `.npz`
# output fields and `.pkl` checkpoints.  Four single commits on the
# 2026-09-13-consolidated lineage each add 3.9–8.6 GB of heat archive chunks,
# which is far above GitHub's ~2 GB per-push pack limit, and a single commit
# cannot be split across pushes.  No branch descending from those commits can
# reach GitHub as-is.
#
# This script rewrites history with every raw archive path removed, leaving the
# code, the reports, the audited result JSONs and the SHA256 manifests that
# describe the removed archives.  It never touches the source repository's refs
# or objects: all rewriting happens inside a throwaway clone.
#
# SAFETY CONTRACT
# ---------------
#   * The source repository is opened read-only.  No ref, note, reflog or object
#     in it is created, moved or deleted.
#   * The clone is made with `--shared`, so no object is copied and no repack of
#     the source repository ever runs.  `git repack` on the source reaches ~48 GB
#     resident on the shared GB10 and must not be run.
#   * Nothing is pushed.  The script prints the push command and stops.
#
# USAGE
#   tools/mirror-code-only.sh <output-dir> [ref ...]
#
#   <output-dir>  a path that does not yet exist; the throwaway mirror is built there
#   [ref ...]     refs to mirror; default: --all
#
# REQUIREMENTS
#   git only.  `git filter-repo` is NOT installed on this box and nothing is
#   installed by this script; the rewrite uses `git filter-branch --index-filter`,
#   which operates on the index alone and therefore never reads or writes the
#   multi-gigabyte blob contents.  If `git filter-repo` is later made available
#   (e.g. `/home/tahmid/Dev/.venv/bin/pip install --no-deps git-filter-repo`), it
#   is the better tool: same result, one pass, no per-commit subprocess.
#
set -euo pipefail

SRC="${MIRROR_SRC:-/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude}"
OUT="${1:?usage: mirror-code-only.sh <output-dir> [ref ...]}"
shift || true
REFS=("$@")
if [ ${#REFS[@]} -eq 0 ]; then REFS=(--all); fi

# Every path matching this ERE is removed from every commit.  Anchored on the
# basename so that a manifest such as `MANIFEST.sha256` or `ARCHIVE.json` that
# sits beside a chunk directory is kept.
DROP_RE='(\.part-?[0-9]+$)|(/archive-parts/part-?[0-9]+$)|(\.(npz|npy|pkl|pt|pth|h5|hdf5)$)|(\.(tar|tgz|zip)$)|(\.tar\.gz$)|(\.gz$)'

if [ -e "$OUT" ]; then echo "refusing to write into existing path: $OUT" >&2; exit 1; fi

mkdir -p "$(dirname "$OUT")"
INVENTORY="${OUT%.git}.objects.txt"

echo "== 1/5 measuring the source object set (read-only)"
git -C "$SRC" rev-list --objects "${REFS[@]}" \
  | git -C "$SRC" cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(objectsize:disk) %(rest)' \
  > "$INVENTORY"
awk -v re="$DROP_RE" '
  $1=="blob" {
    path=""; for (i=5;i<=NF;i++) path=path (i>5?" ":"") $i
    if (path ~ re) { dn++; ds+=$3 } else { kn++; ks+=$3 }
  }
  END { printf "   dropped blobs: %d, %.3f GiB\n   kept blobs:    %d, %.3f GiB\n", dn, ds/1073741824, kn, ks/1073741824 }
' "$INVENTORY"

echo "== 2/5 throwaway --shared clone (no object is copied)"
git clone --bare --shared "$SRC" "$OUT" >/dev/null 2>&1
git -C "$OUT" config gc.auto 0

echo "== 3/5 rewriting history with every raw-archive path removed"
export FILTER_BRANCH_SQUELCH_WARNING=1
git -C "$OUT" filter-branch --force --prune-empty \
  --index-filter "git ls-files -z | grep -zE '$DROP_RE' | git update-index -z --force-remove --stdin || :" \
  --tag-name-filter cat -- --all >/dev/null

echo "== 4/5 dropping the filter-branch backup refs and packing"
git -C "$OUT" for-each-ref --format='%(refname)' refs/original/ \
  | xargs -r -n1 git -C "$OUT" update-ref -d
git -C "$OUT" reflog expire --expire=now --all
# --local would exclude the kept blobs, which live in the source object store via
# the alternate; they must be copied in so the mirror is self-contained and
# pushable.  This packs ~0.9 GiB of JSON/markdown/python, not the archives.
git -C "$OUT" repack -a -d -f --window=50 --depth=50 >/dev/null
rm -f "$OUT/objects/info/alternates"
git -C "$OUT" fsck --no-dangling --connectivity-only >/dev/null

echo "== 5/5 result"
du -sh "$OUT"
git -C "$OUT" count-objects -vH
echo
echo "largest blobs surviving in the mirror:"
git -C "$OUT" rev-list --objects --all \
  | git -C "$OUT" cat-file --batch-check='%(objecttype) %(objectsize) %(rest)' \
  | awk '$1=="blob"{printf "%10.2f MiB  %s\n", $2/1048576, substr($0, index($0,$3))}' \
  | sort -rn | head -10
echo
echo "NOTHING HAS BEEN PUSHED.  To publish the mirror (from a session that may push):"
echo "  git -C $OUT push --mirror https://github.com/<owner>/<code-only-mirror-repo>.git"
echo "The source repository at $SRC is unchanged."
