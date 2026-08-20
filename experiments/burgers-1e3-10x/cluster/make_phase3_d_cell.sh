#!/usr/bin/env bash
# make_phase3_d_cell.sh CELL EXPECTED_COMMIT S0_CELL
set -euo pipefail
cell="$1"; expected_commit="$2"; s0_cell="$3"
[[ "$cell" =~ ^p3_d_r[0-9]+$ ]] || { echo "invalid P3-D cell" >&2; exit 2; }
[[ "$expected_commit" =~ ^[0-9a-f]{40}$ ]] || { echo "invalid commit" >&2; exit 2; }
[[ "$s0_cell" =~ ^s0_spline_r[0-9]+$ ]] || { echo "invalid S0 cell" >&2; exit 2; }
command -v jq >/dev/null || { echo "jq is required" >&2; exit 2; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP="$(dirname "$HERE")"
WORKTREE="$(cd "$EXP/../.." && pwd)"
ROOT="$(cd "$WORKTREE/../.." && pwd)"
STAGE="$HERE/stage/$cell"
S0_RUN="$EXP/runs/$s0_cell"
[[ "$(git -C "$WORKTREE" rev-parse HEAD)" == "$expected_commit" ]] || {
  echo "HEAD is not expected commit" >&2; exit 3;
}
sources=(b10_common.py b10_spline.py b10_s0_spline.py b10_phase3.py b10_phase3_d.py)
for name in "${sources[@]}"; do
  git -C "$WORKTREE" diff --quiet HEAD -- "experiments/burgers-1e3-10x/$name" || {
    echo "dirty scientific source: $name" >&2; exit 3;
  }
  git -C "$WORKTREE" cat-file -e "$expected_commit:experiments/burgers-1e3-10x/$name"
done
[[ -f "$S0_RUN/LOCAL.sha256" ]] || { echo "missing pulled S0" >&2; exit 4; }
(cd "$S0_RUN" && sha256sum -c LOCAL.sha256)
s0_json_sha="$(sha256sum "$S0_RUN/out/s0.json" | cut -d' ' -f1)"
s0_npz_sha="$(sha256sum "$S0_RUN/out/s0.npz" | cut -d' ' -f1)"
[[ "$(jq -r .status "$S0_RUN/out/AUDIT.json")" == pass ]] || exit 4
[[ "$(jq -r .source_json_sha256 "$S0_RUN/out/AUDIT.json")" == "$s0_json_sha" ]] || exit 4
[[ "$(jq -r .source_npz_sha256 "$S0_RUN/out/AUDIT.json")" == "$s0_npz_sha" ]] || exit 4

# The staging directory is recoverable and is never a pulled scientific run.
rm -rf "$STAGE"
mkdir -p "$STAGE/logs" "$STAGE/out" "$STAGE/code/deps/s0" \
  "$STAGE/code/deps/burgers2d-coord-rom"
for name in "${sources[@]}"; do cp "$EXP/$name" "$STAGE/code/"; done
cp "$WORKTREE/experiments/burgers-hybrid-1024/bh_common.py" "$STAGE/code/"
cp "$ROOT/worktrees/2026-08-14-burgers2d-coord-rom/experiments/burgers2d-coord-rom/burgers2d_film.py" \
  "$STAGE/code/deps/burgers2d-coord-rom/"
cp "$S0_RUN/out/s0.json" "$S0_RUN/out/s0.npz" "$S0_RUN/out/AUDIT.json" \
  "$STAGE/code/deps/s0/"
cp "$HERE/phase3_d.sbatch" "$STAGE/run.sbatch"
(cd "$STAGE" && find . -type f -not -name MANIFEST.sha256 -exec sha256sum {} \; | sort > MANIFEST.sha256)
echo "stage=$STAGE"
echo "commit=$expected_commit"
echo "manifest_file_sha256=$(sha256sum "$STAGE/MANIFEST.sha256" | cut -d' ' -f1)"
