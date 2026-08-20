#!/usr/bin/env bash
# make_phase8_d_cell.sh p8_d_r1 EXPECTED_COMMIT p4_d_r3 p5_d_r1 p6_d_r1 p7_g1_s11_r3
set -euo pipefail
cell="$1"; expected_commit="$2"; p4_cell="$3"; p5_cell="$4"; p6_cell="$5"; p7_cell="$6"
[[ "$cell" == p8_d_r1 && "$expected_commit" =~ ^[0-9a-f]{40}$ ]] || exit 2
[[ "$p4_cell" == p4_d_r3 && "$p5_cell" == p5_d_r1 && "$p6_cell" == p6_d_r1 \
   && "$p7_cell" == p7_g1_s11_r3 ]] || exit 2
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; EXP="$(dirname "$HERE")"
WORKTREE="$(cd "$EXP/../.." && pwd)"; ROOT="$(cd "$WORKTREE/../.." && pwd)"
STAGE="$HERE/stage/$cell"; P4="$EXP/runs/$p4_cell"; P5="$EXP/runs/$p5_cell"
P6="$EXP/runs/$p6_cell"; P7="$EXP/runs/$p7_cell"
[[ "$(git -C "$WORKTREE" rev-parse HEAD)" == "$expected_commit" ]] || exit 3
sources=(b10_common.py b10_spline.py b10_spline_train.py b10_s0_spline.py b10_phase3.py b10_phase4.py b10_phase5.py b10_phase7.py b10_phase7_train.py b10_phase8_d.py b10_audit_phase8_d.py)
tracked=(experiments/burgers-1e3-10x/PHASE-8-PRE-REGISTRATION.md experiments/burgers-1e3-10x/PHASE-8-RESOURCE-ESTIMATE.md experiments/burgers-1e3-10x/cluster/phase8_d.sbatch experiments/burgers-1e3-10x/cluster/verify_manifest_file_set.sh)
for name in "${sources[@]}"; do tracked+=("experiments/burgers-1e3-10x/$name"); done
for path in "${tracked[@]}" experiments/burgers-hybrid-1024/bh_common.py; do
  git -C "$WORKTREE" diff --quiet HEAD -- "$path" || exit 3
  git -C "$WORKTREE" cat-file -e "$expected_commit:$path"
done
for prior in "$P4" "$P5" "$P6" "$P7"; do
  [[ -f "$prior/LOCAL.sha256" ]] || exit 4
  (cd "$prior" && sha256sum -c LOCAL.sha256)
  [[ "$(jq -r .status "$prior/out/AUDIT.json")" == pass ]] || exit 4
done
[[ "$(sha256sum "$P4/out/phase4_d.json" | cut -d' ' -f1)" == ff425dfa1f73ac2d8559df2d780ef09dc1ade179ed5636458e53e0f389f5d617 ]] || exit 4
[[ "$(sha256sum "$P5/out/phase5_d.json" | cut -d' ' -f1)" == 97f8bc6bb9e1d67d0baf4652bd57e6fb69dab484fc8f99ce12018e9f6c1d0c96 ]] || exit 4
[[ "$(sha256sum "$P6/out/phase6_d.json" | cut -d' ' -f1)" == 9fe2d49bbb0324fd08ef5da906c3afab0338a1f3bbfb6dfc1ef73f89603c139a ]] || exit 4
[[ "$(sha256sum "$P7/out/phase7_train.json" | cut -d' ' -f1)" == a59e92640aae787003d5753d4614de1b6fe90c68fdba7f2daa81a125a3c0956c ]] || exit 4
[[ "$(sha256sum "$P7/out/phase7_train.npz" | cut -d' ' -f1)" == fd40d339c0746b48c07408d5d017dff8819595350c365f7af5fc0d40673946ae ]] || exit 4
[[ "$(sha256sum "$P7/out/checkpoint.pkl" | cut -d' ' -f1)" == 113101637ef2b4fb75fe2ca0ba0dabe5a90c35d03a5c563b765438385c62db8c ]] || exit 4
[[ "$(sha256sum "$P7/out/AUDIT.json" | cut -d' ' -f1)" == 35c95ee38dea7622f40b6c199f4164b6c27ec3f37ad5561a51de6cd3b0a79322 ]] || exit 4
[[ "$(sha256sum "$P7/MANIFEST.sha256" | cut -d' ' -f1)" == a95fc4621a90cef13071df1ad1db363187deac0f7fc7c8c774c0fedd7c9c3a19 ]] || exit 4
rm -rf "$STAGE"
mkdir -p "$STAGE/logs" "$STAGE/out" "$STAGE/code/deps/burgers2d-coord-rom" \
  "$STAGE/code/deps/p4" "$STAGE/code/deps/p5/targets" "$STAGE/code/deps/p6" "$STAGE/code/deps/p7"
for name in "${sources[@]}"; do cp "$EXP/$name" "$STAGE/code/"; done
cp "$EXP/PHASE-8-PRE-REGISTRATION.md" "$STAGE/code/"
cp "$WORKTREE/experiments/burgers-hybrid-1024/bh_common.py" "$STAGE/code/"
cp "$ROOT/worktrees/2026-08-14-burgers2d-coord-rom/experiments/burgers2d-coord-rom/burgers2d_film.py" "$STAGE/code/deps/burgers2d-coord-rom/"
cp "$P4/out/phase4_d.json" "$P4/out/phase4_d.npz" "$P4/out/AUDIT.json" "$P4/MANIFEST.sha256" "$STAGE/code/deps/p4/"
cp "$P5/out/phase5_d.json" "$P5/out/phase5_d.npz" "$P5/out/AUDIT.json" "$P5/MANIFEST.sha256" "$STAGE/code/deps/p5/"
cp "$P5/out/targets/"*.npz "$STAGE/code/deps/p5/targets/"
cp "$P6/out/phase6_d.json" "$P6/out/phase6_d.npz" "$P6/out/AUDIT.json" "$P6/MANIFEST.sha256" "$STAGE/code/deps/p6/"
cp "$P7/out/phase7_train.json" "$P7/out/phase7_train.npz" "$P7/out/checkpoint.pkl" "$P7/out/AUDIT.json" "$P7/MANIFEST.sha256" "$STAGE/code/deps/p7/"
film_sha="$(awk '$2=="./code/deps/burgers2d-coord-rom/burgers2d_film.py" {print $1}' "$P4/MANIFEST.sha256")"
bh_sha="$(awk '$2=="./code/bh_common.py" {print $1}' "$P4/MANIFEST.sha256")"
[[ "$(sha256sum "$STAGE/code/deps/burgers2d-coord-rom/burgers2d_film.py" | cut -d' ' -f1)" == "$film_sha" ]] || exit 4
[[ "$(sha256sum "$STAGE/code/bh_common.py" | cut -d' ' -f1)" == "$bh_sha" ]] || exit 4
cp "$HERE/phase8_d.sbatch" "$STAGE/run.sbatch"
cp "$HERE/verify_manifest_file_set.sh" "$STAGE/verify_manifest_file_set.sh"
(cd "$STAGE" && find . -type f ! -path './MANIFEST.sha256' -exec sha256sum {} \; | sort > MANIFEST.sha256)
"$STAGE/verify_manifest_file_set.sh" "$STAGE" "$STAGE/MANIFEST.sha256"
(cd "$STAGE" && sha256sum -c MANIFEST.sha256)
echo "stage=$STAGE"; echo "commit=$expected_commit"
echo "manifest_file_sha256=$(sha256sum "$STAGE/MANIFEST.sha256" | cut -d' ' -f1)"
