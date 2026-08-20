#!/usr/bin/env bash
# make_phase7_train_cell.sh CELL EXPECTED_COMMIT P4_CELL P5_CELL P6_CELL
set -euo pipefail
cell="$1"; expected_commit="$2"; p4_cell="$3"; p5_cell="$4"; p6_cell="$5"
[[ "$cell" == p7_g1_s11_r1 ]] || { echo "invalid Phase7 cell" >&2; exit 2; }
[[ "$expected_commit" =~ ^[0-9a-f]{40}$ ]] || exit 2
[[ "$p4_cell" == p4_d_r3 && "$p5_cell" == p5_d_r1 && "$p6_cell" == p6_d_r1 ]] || exit 2
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; EXP="$(dirname "$HERE")"
WORKTREE="$(cd "$EXP/../.." && pwd)"; ROOT="$(cd "$WORKTREE/../.." && pwd)"
STAGE="$HERE/stage/$cell"; P4="$EXP/runs/$p4_cell"; P5="$EXP/runs/$p5_cell"; P6="$EXP/runs/$p6_cell"
[[ "$(git -C "$WORKTREE" rev-parse HEAD)" == "$expected_commit" ]] || exit 3
sources=(b10_common.py b10_spline.py b10_spline_train.py b10_s0_spline.py b10_phase3.py b10_phase4.py b10_phase5.py b10_phase7.py b10_phase7_train.py b10_audit_phase7_train.py)
for name in "${sources[@]}"; do
  path="experiments/burgers-1e3-10x/$name"
  git -C "$WORKTREE" diff --quiet HEAD -- "$path" || exit 3
  git -C "$WORKTREE" cat-file -e "$expected_commit:$path"
done
for path in experiments/burgers-1e3-10x/PHASE-7-PRE-REGISTRATION.md experiments/burgers-1e3-10x/cluster/phase7_train.sbatch; do
  git -C "$WORKTREE" diff --quiet HEAD -- "$path" || exit 3
  git -C "$WORKTREE" cat-file -e "$expected_commit:$path"
done
git -C "$WORKTREE" diff --quiet HEAD -- experiments/burgers-hybrid-1024/bh_common.py || exit 3
git -C "$WORKTREE" cat-file -e "$expected_commit:experiments/burgers-hybrid-1024/bh_common.py"
for prior in "$P4" "$P5" "$P6"; do
  [[ -f "$prior/LOCAL.sha256" ]] || exit 4
  (cd "$prior" && sha256sum -c LOCAL.sha256)
  [[ "$(jq -r .status "$prior/out/AUDIT.json")" == pass ]] || exit 4
done
[[ "$(sha256sum "$P4/out/phase4_d.json" | cut -d' ' -f1)" == ff425dfa1f73ac2d8559df2d780ef09dc1ade179ed5636458e53e0f389f5d617 ]] || exit 4
[[ "$(sha256sum "$P5/out/phase5_d.json" | cut -d' ' -f1)" == 97f8bc6bb9e1d67d0baf4652bd57e6fb69dab484fc8f99ce12018e9f6c1d0c96 ]] || exit 4
[[ "$(sha256sum "$P6/out/phase6_d.json" | cut -d' ' -f1)" == 9fe2d49bbb0324fd08ef5da906c3afab0338a1f3bbfb6dfc1ef73f89603c139a ]] || exit 4
rm -rf "$STAGE"
mkdir -p "$STAGE/logs" "$STAGE/out" "$STAGE/code/deps/burgers2d-coord-rom" \
  "$STAGE/code/deps/p4" "$STAGE/code/deps/p5/targets" "$STAGE/code/deps/p6"
for name in "${sources[@]}"; do cp "$EXP/$name" "$STAGE/code/"; done
cp "$EXP/PHASE-7-PRE-REGISTRATION.md" "$STAGE/code/"
cp "$WORKTREE/experiments/burgers-hybrid-1024/bh_common.py" "$STAGE/code/"
cp "$ROOT/worktrees/2026-08-14-burgers2d-coord-rom/experiments/burgers2d-coord-rom/burgers2d_film.py" "$STAGE/code/deps/burgers2d-coord-rom/"
cp "$P4/out/phase4_d.json" "$P4/out/phase4_d.npz" "$P4/out/AUDIT.json" "$P4/MANIFEST.sha256" "$STAGE/code/deps/p4/"
cp "$P5/out/phase5_d.json" "$P5/out/phase5_d.npz" "$P5/out/AUDIT.json" "$P5/MANIFEST.sha256" "$STAGE/code/deps/p5/"
cp "$P5/out/targets/"*.npz "$STAGE/code/deps/p5/targets/"
cp "$P6/out/phase6_d.json" "$P6/out/phase6_d.npz" "$P6/out/AUDIT.json" "$P6/MANIFEST.sha256" "$STAGE/code/deps/p6/"
film_sha="$(awk '$2=="./code/deps/burgers2d-coord-rom/burgers2d_film.py" {print $1}' "$P4/MANIFEST.sha256")"
bh_sha="$(awk '$2=="./code/bh_common.py" {print $1}' "$P4/MANIFEST.sha256")"
[[ "$(sha256sum "$STAGE/code/deps/burgers2d-coord-rom/burgers2d_film.py" | cut -d' ' -f1)" == "$film_sha" ]] || exit 4
[[ "$(sha256sum "$STAGE/code/bh_common.py" | cut -d' ' -f1)" == "$bh_sha" ]] || exit 4
cp "$HERE/phase7_train.sbatch" "$STAGE/run.sbatch"
(cd "$STAGE" && find . -type f ! -path './MANIFEST.sha256' -exec sha256sum {} \; | sort > MANIFEST.sha256)
echo "stage=$STAGE"
echo "commit=$expected_commit"
echo "manifest_file_sha256=$(sha256sum "$STAGE/MANIFEST.sha256" | cut -d' ' -f1)"
