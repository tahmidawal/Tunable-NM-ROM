#!/usr/bin/env bash
# make_phase4_d_cell.sh CELL EXPECTED_COMMIT S0_CELL P3_CELL
set -euo pipefail
cell="$1"; expected_commit="$2"; s0_cell="$3"; p3_cell="$4"
[[ "$cell" =~ ^p4_d_r[0-9]+$ ]] || { echo "invalid P4-D cell" >&2; exit 2; }
[[ "$expected_commit" =~ ^[0-9a-f]{40}$ ]] || { echo "invalid commit" >&2; exit 2; }
[[ "$s0_cell" =~ ^s0_spline_r[0-9]+$ ]] || exit 2
[[ "$p3_cell" =~ ^p3_d_r[0-9]+$ ]] || exit 2
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP="$(dirname "$HERE")"
WORKTREE="$(cd "$EXP/../.." && pwd)"
ROOT="$(cd "$WORKTREE/../.." && pwd)"
STAGE="$HERE/stage/$cell"
S0="$EXP/runs/$s0_cell"; P3="$EXP/runs/$p3_cell"
[[ "$(git -C "$WORKTREE" rev-parse HEAD)" == "$expected_commit" ]] || exit 3
sources=(b10_common.py b10_spline.py b10_s0_spline.py b10_phase3.py b10_phase4.py b10_phase4_d.py b10_audit_phase4_d.py)
for name in "${sources[@]}"; do
  git -C "$WORKTREE" diff --quiet HEAD -- "experiments/burgers-1e3-10x/$name" || exit 3
  git -C "$WORKTREE" cat-file -e "$expected_commit:experiments/burgers-1e3-10x/$name"
done
git -C "$WORKTREE" diff --quiet HEAD -- experiments/burgers-hybrid-1024/bh_common.py || {
  echo "dirty bh_common dependency" >&2; exit 3;
}
git -C "$WORKTREE" cat-file -e "$expected_commit:experiments/burgers-hybrid-1024/bh_common.py"
for run in "$S0" "$P3"; do
  [[ -f "$run/LOCAL.sha256" ]] || exit 4
  (cd "$run" && sha256sum -c LOCAL.sha256)
  [[ "$(jq -r .status "$run/out/AUDIT.json")" == pass ]] || exit 4
done
rm -rf "$STAGE"
mkdir -p "$STAGE/logs" "$STAGE/out" "$STAGE/code/deps/s0" \
  "$STAGE/code/deps/p3" "$STAGE/code/deps/burgers2d-coord-rom"
for name in "${sources[@]}"; do cp "$EXP/$name" "$STAGE/code/"; done
cp "$WORKTREE/experiments/burgers-hybrid-1024/bh_common.py" "$STAGE/code/"
cp "$ROOT/worktrees/2026-08-14-burgers2d-coord-rom/experiments/burgers2d-coord-rom/burgers2d_film.py" "$STAGE/code/deps/burgers2d-coord-rom/"
cp "$S0/out/s0.json" "$S0/out/s0.npz" "$S0/out/AUDIT.json" "$STAGE/code/deps/s0/"
cp "$P3/out/phase3_d.json" "$P3/out/phase3_d.npz" "$P3/out/AUDIT.json" "$STAGE/code/deps/p3/"
cp "$P3/MANIFEST.sha256" "$STAGE/code/deps/p3/MANIFEST.sha256"
film_sha="$(awk '$2=="./code/deps/burgers2d-coord-rom/burgers2d_film.py" {print $1}' "$P3/MANIFEST.sha256")"
bh_sha="$(awk '$2=="./code/bh_common.py" {print $1}' "$P3/MANIFEST.sha256")"
[[ "$film_sha" =~ ^[0-9a-f]{64}$ && "$bh_sha" =~ ^[0-9a-f]{64}$ ]] || exit 4
[[ "$(sha256sum "$STAGE/code/deps/burgers2d-coord-rom/burgers2d_film.py" | cut -d' ' -f1)" == "$film_sha" ]] || exit 4
[[ "$(sha256sum "$STAGE/code/bh_common.py" | cut -d' ' -f1)" == "$bh_sha" ]] || exit 4
cp "$HERE/phase4_d.sbatch" "$STAGE/run.sbatch"
(cd "$STAGE" && find . -type f -not -name MANIFEST.sha256 -exec sha256sum {} \; | sort > MANIFEST.sha256)
echo "stage=$STAGE"
echo "commit=$expected_commit"
echo "manifest_file_sha256=$(sha256sum "$STAGE/MANIFEST.sha256" | cut -d' ' -f1)"
