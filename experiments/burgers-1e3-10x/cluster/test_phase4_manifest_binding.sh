#!/usr/bin/env bash
# Non-scientific static fixture for nested staging-manifest provenance.
set -euo pipefail
p3_manifest="$1"; expected_commit="$2"
[[ -f "$p3_manifest" && "$expected_commit" =~ ^[0-9a-f]{40}$ ]] || exit 2
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP="$(dirname "$HERE")"
WORKTREE="$(cd "$EXP/../.." && pwd)"
ROOT="$(cd "$WORKTREE/../.." && pwd)"
PY=/home/tahmid/Dev/.venv/bin/python
fixture="$(mktemp -d)"
trap 'rm -rf "$fixture"' EXIT
mkdir -p "$fixture/code/deps/p3" "$fixture/code/deps/burgers2d-coord-rom"
cp "$p3_manifest" "$fixture/code/deps/p3/MANIFEST.sha256"
cp "$WORKTREE/experiments/burgers-hybrid-1024/bh_common.py" "$fixture/code/"
cp "$ROOT/worktrees/2026-08-14-burgers2d-coord-rom/experiments/burgers2d-coord-rom/burgers2d_film.py" "$fixture/code/deps/burgers2d-coord-rom/"
(cd "$fixture" && find . -type f ! -path './MANIFEST.sha256' -exec sha256sum {} \; | sort > MANIFEST.sha256)
nested_sha="$(sha256sum "$p3_manifest" | cut -d' ' -f1)"
grep -q "^$nested_sha  ./code/deps/p3/MANIFEST.sha256$" "$fixture/MANIFEST.sha256"
! grep -q '  ./MANIFEST.sha256$' "$fixture/MANIFEST.sha256"
"$PY" "$EXP/b10_test_phase4_chain_schema.py" \
  --s0-json "$EXP/runs/s0_spline_r1/out/s0.json" \
  --s0-npz "$EXP/runs/s0_spline_r1/out/s0.npz" \
  --s0-audit "$EXP/runs/s0_spline_r1/out/AUDIT.json" \
  --p3-json "$EXP/runs/p3_d_r1/out/phase3_d.json" \
  --p3-npz "$EXP/runs/p3_d_r1/out/phase3_d.npz" \
  --p3-audit "$EXP/runs/p3_d_r1/out/AUDIT.json" \
  --p3-manifest "$p3_manifest" --stage-manifest "$fixture/MANIFEST.sha256" \
  --expected-commit "$expected_commit" --worktree "$WORKTREE"
echo phase4_stage_manifest_fixture=PASS
