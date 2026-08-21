#!/usr/bin/env bash
# make_phase9_train_cell.sh p9_t{1,2}_s11_r1 COMMIT p4_d_r3 p5_d_r1 p6_d_r1 p7_g1_s11_r3 p8_d_r2 [p9_t1_s11_r1]
set -euo pipefail
cell="$1"; commit="$2"; p4c="$3"; p5c="$4"; p6c="$5"; p7c="$6"; p8c="$7"; t1c="${8:-}"
[[ "$cell" =~ ^p9_t[12]_s11_r1$ && "$commit" =~ ^[0-9a-f]{40}$ ]] || exit 2
if [[ "$cell" == p9_t1_s11_r1 ]]; then [[ -z "$t1c" ]] || exit 2; else [[ "$t1c" == p9_t1_s11_r1 ]] || exit 2; fi
[[ "$p4c $p5c $p6c $p7c $p8c" == "p4_d_r3 p5_d_r1 p6_d_r1 p7_g1_s11_r3 p8_d_r2" ]] || exit 2
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; EXP="$(dirname "$HERE")"; WORKTREE="$(cd "$EXP/../.." && pwd)"; ROOT="$(cd "$WORKTREE/../.." && pwd)"
STAGE="$HERE/stage/$cell"; P4="$EXP/runs/$p4c"; P5="$EXP/runs/$p5c"; P6="$EXP/runs/$p6c"; P7="$EXP/runs/$p7c"; P8="$EXP/runs/$p8c"
T1="${t1c:+$EXP/runs/$t1c}"
[[ "$(git -C "$WORKTREE" rev-parse HEAD)" == "$commit" ]] || exit 3
sources=(b10_common.py b10_spline.py b10_spline_train.py b10_s0_spline.py b10_phase3.py b10_phase4.py b10_phase5.py b10_phase7.py b10_phase7_train.py b10_phase8_d.py b10_phase9_train.py b10_audit_phase9_train.py)
tracked=(experiments/burgers-1e3-10x/PHASE-9-PRE-REGISTRATION.md experiments/burgers-1e3-10x/PHASE-9-RESOURCE-ESTIMATE.md experiments/burgers-1e3-10x/cluster/phase9_train.sbatch experiments/burgers-1e3-10x/cluster/verify_manifest_file_set.sh)
for name in "${sources[@]}"; do tracked+=("experiments/burgers-1e3-10x/$name"); done
for path in "${tracked[@]}" experiments/burgers-hybrid-1024/bh_common.py; do
  git -C "$WORKTREE" diff --quiet HEAD -- "$path" || exit 3; git -C "$WORKTREE" cat-file -e "$commit:$path"
done
for prior in "$P4" "$P5" "$P6" "$P7" "$P8"; do
  [[ -f "$prior/LOCAL.sha256" ]]; (cd "$prior" && sha256sum -c LOCAL.sha256); [[ "$(jq -r .status "$prior/out/AUDIT.json")" == pass ]]
done
if [[ -n "$T1" ]]; then
  [[ -f "$T1/LOCAL.sha256" ]]; (cd "$T1" && sha256sum -c LOCAL.sha256)
  [[ "$(jq -r .status "$T1/out/AUDIT.json")" == pass && "$(jq -r .decision.g2_licensed "$T1/out/AUDIT.json")" == true ]]
fi
[[ "$(sha256sum "$P4/out/phase4_d.json"|cut -d' ' -f1)" == ff425dfa1f73ac2d8559df2d780ef09dc1ade179ed5636458e53e0f389f5d617 ]]
[[ "$(sha256sum "$P5/out/phase5_d.json"|cut -d' ' -f1)" == 97f8bc6bb9e1d67d0baf4652bd57e6fb69dab484fc8f99ce12018e9f6c1d0c96 ]]
[[ "$(sha256sum "$P6/out/phase6_d.json"|cut -d' ' -f1)" == 9fe2d49bbb0324fd08ef5da906c3afab0338a1f3bbfb6dfc1ef73f89603c139a ]]
[[ "$(sha256sum "$P7/out/phase7_train.json"|cut -d' ' -f1)" == a59e92640aae787003d5753d4614de1b6fe90c68fdba7f2daa81a125a3c0956c ]]
[[ "$(sha256sum "$P8/out/phase8_d.json"|cut -d' ' -f1)" == be0ca15e5c36f45a1f9a8fdfe86b79a592b530c069ae6d25c1f573c59c974f4f ]]
rm -rf "$STAGE"; mkdir -p "$STAGE/logs" "$STAGE/out" "$STAGE/code/deps/"{p4,p5/targets,p6,p7,p8,burgers2d-coord-rom}
[[ -z "$T1" ]] || mkdir -p "$STAGE/code/deps/t1"
for name in "${sources[@]}"; do cp "$EXP/$name" "$STAGE/code/"; done
cp "$EXP/PHASE-9-PRE-REGISTRATION.md" "$STAGE/code/"; cp "$WORKTREE/experiments/burgers-hybrid-1024/bh_common.py" "$STAGE/code/"
cp "$ROOT/worktrees/2026-08-14-burgers2d-coord-rom/experiments/burgers2d-coord-rom/burgers2d_film.py" "$STAGE/code/deps/burgers2d-coord-rom/"
cp "$P4/out/phase4_d.json" "$P4/out/phase4_d.npz" "$P4/out/AUDIT.json" "$P4/MANIFEST.sha256" "$STAGE/code/deps/p4/"
cp "$P5/out/phase5_d.json" "$P5/out/phase5_d.npz" "$P5/out/AUDIT.json" "$P5/MANIFEST.sha256" "$STAGE/code/deps/p5/"; cp "$P5/out/targets/"*.npz "$STAGE/code/deps/p5/targets/"
cp "$P6/out/phase6_d.json" "$P6/out/phase6_d.npz" "$P6/out/AUDIT.json" "$P6/MANIFEST.sha256" "$STAGE/code/deps/p6/"
cp "$P7/out/phase7_train.json" "$P7/out/phase7_train.npz" "$P7/out/checkpoint.pkl" "$P7/out/AUDIT.json" "$P7/MANIFEST.sha256" "$STAGE/code/deps/p7/"
cp "$P8/out/phase8_d.json" "$P8/out/phase8_d.npz" "$P8/out/AUDIT.json" "$P8/MANIFEST.sha256" "$STAGE/code/deps/p8/"
[[ -z "$T1" ]] || cp "$T1/out/phase9_train.json" "$T1/out/phase9_train.npz" "$T1/out/checkpoint.pkl" "$T1/out/AUDIT.json" "$T1/MANIFEST.sha256" "$STAGE/code/deps/t1/"
cp "$HERE/phase9_train.sbatch" "$STAGE/run.sbatch"; cp "$HERE/verify_manifest_file_set.sh" "$STAGE/"
(cd "$STAGE" && find . -type f ! -path './MANIFEST.sha256' -exec sha256sum {} \; | sort > MANIFEST.sha256)
"$STAGE/verify_manifest_file_set.sh" "$STAGE" "$STAGE/MANIFEST.sha256"; (cd "$STAGE" && sha256sum -c MANIFEST.sha256)
echo "stage=$STAGE"; echo "commit=$commit"; echo "manifest_file_sha256=$(sha256sum "$STAGE/MANIFEST.sha256"|cut -d' ' -f1)"
