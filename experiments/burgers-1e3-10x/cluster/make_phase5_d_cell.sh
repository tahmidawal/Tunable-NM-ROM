#!/usr/bin/env bash
# make_phase5_d_cell.sh CELL EXPECTED_COMMIT P4_CELL
set -euo pipefail
cell="$1"; expected_commit="$2"; p4_cell="$3"
[[ "$cell" =~ ^p5_d_r[0-9]+$ ]] || { echo "invalid P5-D cell" >&2; exit 2; }
[[ "$expected_commit" =~ ^[0-9a-f]{40}$ ]] || { echo "invalid commit" >&2; exit 2; }
[[ "$p4_cell" =~ ^p4_d_r[0-9]+$ ]] || exit 2
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP="$(dirname "$HERE")"
WORKTREE="$(cd "$EXP/../.." && pwd)"
ROOT="$(cd "$WORKTREE/../.." && pwd)"
STAGE="$HERE/stage/$cell"
P4="$EXP/runs/$p4_cell"
[[ "$(git -C "$WORKTREE" rev-parse HEAD)" == "$expected_commit" ]] || exit 3
sources=(b10_common.py b10_spline.py b10_s0_spline.py b10_phase3.py b10_phase4.py b10_phase4_d.py b10_phase5.py b10_phase5_d.py b10_audit_phase5_d.py b10_phase5_rank_diagnostic.py)
for name in "${sources[@]}"; do
  git -C "$WORKTREE" diff --quiet HEAD -- "experiments/burgers-1e3-10x/$name" || exit 3
  git -C "$WORKTREE" cat-file -e "$expected_commit:experiments/burgers-1e3-10x/$name"
done
for name in PHASE-5-PRE-REGISTRATION.md PHASE-5-RANK-CHECKPOINT.md phase5_rank_diagnostic.json; do
  git -C "$WORKTREE" diff --quiet HEAD -- "experiments/burgers-1e3-10x/$name" || exit 3
  git -C "$WORKTREE" cat-file -e "$expected_commit:experiments/burgers-1e3-10x/$name"
done
git -C "$WORKTREE" diff --quiet HEAD -- experiments/burgers-1e3-10x/cluster/phase5_d.sbatch || exit 3
git -C "$WORKTREE" cat-file -e "$expected_commit:experiments/burgers-1e3-10x/cluster/phase5_d.sbatch"
git -C "$WORKTREE" diff --quiet HEAD -- experiments/burgers-hybrid-1024/bh_common.py || {
  echo "dirty bh_common dependency" >&2; exit 3;
}
git -C "$WORKTREE" cat-file -e "$expected_commit:experiments/burgers-hybrid-1024/bh_common.py"
[[ -f "$P4/LOCAL.sha256" ]] || exit 4
(cd "$P4" && sha256sum -c LOCAL.sha256)
[[ "$(jq -r .status "$P4/out/AUDIT.json")" == pass ]] || exit 4
[[ "$(jq -r .decision.selected_spatial_arm "$P4/out/phase4_d.json")" == H1 ]] || exit 4
rm -rf "$STAGE"
mkdir -p "$STAGE/logs" "$STAGE/out/targets" "$STAGE/code/deps/p4" \
  "$STAGE/code/deps/phase5" "$STAGE/code/deps/burgers2d-coord-rom"
for name in "${sources[@]}"; do cp "$EXP/$name" "$STAGE/code/"; done
cp "$WORKTREE/experiments/burgers-hybrid-1024/bh_common.py" "$STAGE/code/"
cp "$ROOT/worktrees/2026-08-14-burgers2d-coord-rom/experiments/burgers2d-coord-rom/burgers2d_film.py" "$STAGE/code/deps/burgers2d-coord-rom/"
cp "$P4/out/phase4_d.json" "$P4/out/phase4_d.npz" "$P4/out/AUDIT.json" "$STAGE/code/deps/p4/"
cp "$P4/MANIFEST.sha256" "$STAGE/code/deps/p4/MANIFEST.sha256"
cp "$EXP/PHASE-5-PRE-REGISTRATION.md" "$EXP/PHASE-5-RANK-CHECKPOINT.md" \
  "$EXP/phase5_rank_diagnostic.json" "$STAGE/code/deps/phase5/"
film_sha="$(awk '$2=="./code/deps/burgers2d-coord-rom/burgers2d_film.py" {print $1}' "$P4/MANIFEST.sha256")"
bh_sha="$(awk '$2=="./code/bh_common.py" {print $1}' "$P4/MANIFEST.sha256")"
[[ "$film_sha" =~ ^[0-9a-f]{64}$ && "$bh_sha" =~ ^[0-9a-f]{64}$ ]] || exit 4
[[ "$(sha256sum "$STAGE/code/deps/burgers2d-coord-rom/burgers2d_film.py" | cut -d' ' -f1)" == "$film_sha" ]] || exit 4
[[ "$(sha256sum "$STAGE/code/bh_common.py" | cut -d' ' -f1)" == "$bh_sha" ]] || exit 4
cp "$HERE/phase5_d.sbatch" "$STAGE/run.sbatch"
(cd "$STAGE" && find . -type f ! -path './MANIFEST.sha256' -exec sha256sum {} \; | sort > MANIFEST.sha256)
grep -q ' ./code/deps/p4/MANIFEST.sha256$' "$STAGE/MANIFEST.sha256"
echo "stage=$STAGE"
echo "commit=$expected_commit"
echo "manifest_file_sha256=$(sha256sum "$STAGE/MANIFEST.sha256" | cut -d' ' -f1)"
