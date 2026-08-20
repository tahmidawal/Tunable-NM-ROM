#!/usr/bin/env bash
# make_phase6_d_cell.sh CELL EXPECTED_COMMIT P5_CELL
set -euo pipefail
cell="$1"; expected_commit="$2"; p5_cell="$3"
[[ "$cell" =~ ^p6_d_r[0-9]+$ ]] || { echo "invalid P6-D cell" >&2; exit 2; }
[[ "$expected_commit" =~ ^[0-9a-f]{40}$ ]] || { echo "invalid commit" >&2; exit 2; }
[[ "$p5_cell" =~ ^p5_d_r[0-9]+$ ]] || exit 2
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP="$(dirname "$HERE")"; WORKTREE="$(cd "$EXP/../.." && pwd)"; ROOT="$(cd "$WORKTREE/../.." && pwd)"
STAGE="$HERE/stage/$cell"; P5="$EXP/runs/$p5_cell"
[[ "$(git -C "$WORKTREE" rev-parse HEAD)" == "$expected_commit" ]] || exit 3
sources=(b10_common.py b10_spline.py b10_s0_spline.py b10_phase3.py b10_phase4.py b10_phase4_d.py b10_phase5.py b10_phase6.py b10_phase6_d.py b10_audit_phase6_d.py)
for name in "${sources[@]}"; do
  git -C "$WORKTREE" diff --quiet HEAD -- "experiments/burgers-1e3-10x/$name" || exit 3
  git -C "$WORKTREE" cat-file -e "$expected_commit:experiments/burgers-1e3-10x/$name"
done
for name in PHASE-6-PRE-REGISTRATION.md PHASE-6-RESOURCE-ESTIMATE.md; do
  git -C "$WORKTREE" diff --quiet HEAD -- "experiments/burgers-1e3-10x/$name" || exit 3
  git -C "$WORKTREE" cat-file -e "$expected_commit:experiments/burgers-1e3-10x/$name"
done
git -C "$WORKTREE" diff --quiet HEAD -- experiments/burgers-1e3-10x/cluster/phase6_d.sbatch || exit 3
git -C "$WORKTREE" cat-file -e "$expected_commit:experiments/burgers-1e3-10x/cluster/phase6_d.sbatch"
git -C "$WORKTREE" diff --quiet HEAD -- experiments/burgers-hybrid-1024/bh_common.py || exit 3
git -C "$WORKTREE" cat-file -e "$expected_commit:experiments/burgers-hybrid-1024/bh_common.py"
[[ -f "$P5/LOCAL.sha256" ]] || exit 4
(cd "$P5" && sha256sum -c LOCAL.sha256)
[[ "$(jq -r .status "$P5/out/AUDIT.json")" == pass ]] || exit 4
[[ "$(jq -r .decision.phase5_hard_stop "$P5/out/phase5_d.json")" == true ]] || exit 4
[[ "$(sha256sum "$P5/out/phase5_d.json" | cut -d' ' -f1)" == 97f8bc6bb9e1d67d0baf4652bd57e6fb69dab484fc8f99ce12018e9f6c1d0c96 ]] || exit 4
[[ "$(sha256sum "$P5/out/phase5_d.npz" | cut -d' ' -f1)" == 5235b81b19c4ed459e7fda4291fe67eb3f4b87ba07413eb36e861a0b147dfe54 ]] || exit 4
[[ "$(sha256sum "$P5/out/AUDIT.json" | cut -d' ' -f1)" == c84ee29e1b9fe84f5e90949e18be26c07a6c54c00320a2f7a82bd1cb8dee0eff ]] || exit 4
[[ "$(sha256sum "$P5/MANIFEST.sha256" | cut -d' ' -f1)" == 6135791d3a5cca08b0ff1c424d93451579f3dd1048bef2a5cf314e2b8bf317d6 ]] || exit 4
rm -rf "$STAGE"
mkdir -p "$STAGE/logs" "$STAGE/out" "$STAGE/code/deps/p5" "$STAGE/code/deps/phase6" "$STAGE/code/deps/burgers2d-coord-rom"
for name in "${sources[@]}"; do cp "$EXP/$name" "$STAGE/code/"; done
cp "$WORKTREE/experiments/burgers-hybrid-1024/bh_common.py" "$STAGE/code/"
cp "$ROOT/worktrees/2026-08-14-burgers2d-coord-rom/experiments/burgers2d-coord-rom/burgers2d_film.py" "$STAGE/code/deps/burgers2d-coord-rom/"
cp "$P5/out/phase5_d.json" "$P5/out/phase5_d.npz" "$P5/out/AUDIT.json" "$STAGE/code/deps/p5/"
cp "$P5/MANIFEST.sha256" "$STAGE/code/deps/p5/MANIFEST.sha256"
cp "$EXP/PHASE-6-PRE-REGISTRATION.md" "$STAGE/code/deps/phase6/"
film_sha="$(awk '$2=="./code/deps/burgers2d-coord-rom/burgers2d_film.py" {print $1}' "$P5/MANIFEST.sha256")"
bh_sha="$(awk '$2=="./code/bh_common.py" {print $1}' "$P5/MANIFEST.sha256")"
[[ "$film_sha" =~ ^[0-9a-f]{64}$ && "$bh_sha" =~ ^[0-9a-f]{64}$ ]] || exit 4
[[ "$(sha256sum "$STAGE/code/deps/burgers2d-coord-rom/burgers2d_film.py" | cut -d' ' -f1)" == "$film_sha" ]] || exit 4
[[ "$(sha256sum "$STAGE/code/bh_common.py" | cut -d' ' -f1)" == "$bh_sha" ]] || exit 4
cp "$HERE/phase6_d.sbatch" "$STAGE/run.sbatch"
(cd "$STAGE" && find . -type f ! -path './MANIFEST.sha256' -exec sha256sum {} \; | sort > MANIFEST.sha256)
grep -q ' ./code/deps/p5/MANIFEST.sha256$' "$STAGE/MANIFEST.sha256"
echo "stage=$STAGE"
echo "commit=$expected_commit"
echo "manifest_file_sha256=$(sha256sum "$STAGE/MANIFEST.sha256" | cut -d' ' -f1)"
