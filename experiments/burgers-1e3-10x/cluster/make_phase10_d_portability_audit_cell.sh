#!/usr/bin/env bash
# make_phase10_d_portability_audit_cell.sh p10_d_r2_audit_r1 COMMIT p4_d_r3 p5_d_r1 p6_d_r1 p7_g1_s11_r3 p8_d_r2 p9_t1_s11_r1_failed p9_terminal_recovery_s11_r1_failed p9_terminal_recovery_audit_r1 p10_d_r1_failed p10_d_r2_failed
set -euo pipefail
cell="$1"; commit="$2"; p4c="$3"; p5c="$4"; p6c="$5"; p7c="$6"; p8c="$7"; failedc="$8"; recoveryc="$9"; acceptedc="${10}"; r1c="${11}"; r2c="${12}"
[[ "$cell" == p10_d_r2_audit_r1 && "$commit" =~ ^[0-9a-f]{40}$ ]] || exit 2
[[ "$p4c $p5c $p6c $p7c $p8c $failedc $recoveryc $acceptedc $r1c $r2c" == "p4_d_r3 p5_d_r1 p6_d_r1 p7_g1_s11_r3 p8_d_r2 p9_t1_s11_r1_failed p9_terminal_recovery_s11_r1_failed p9_terminal_recovery_audit_r1 p10_d_r1_failed p10_d_r2_failed" ]] || exit 2
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; EXP="$(dirname "$HERE")"; WORKTREE="$(cd "$EXP/../.." && pwd)"; ROOT="$(cd "$WORKTREE/../.." && pwd)"
STAGE="$HERE/stage/$cell"; P4="$EXP/runs/$p4c"; P5="$EXP/runs/$p5c"; P6="$EXP/runs/$p6c"; P7="$EXP/runs/$p7c"; P8="$EXP/runs/$p8c"; FAILED="$EXP/runs/$failedc"; RECOVERY="$EXP/runs/$recoveryc"; ACCEPTED="$EXP/runs/$acceptedc"; R1="$EXP/runs/$r1c"; R2="$EXP/runs/$r2c"
[[ "$(git -C "$WORKTREE" rev-parse HEAD)" == "$commit" ]] || exit 3
sources=(b10_common.py b10_spline.py b10_spline_train.py b10_s0_spline.py b10_phase3.py b10_phase4.py b10_phase5.py b10_phase7.py b10_phase7_train.py b10_phase8_d.py b10_phase9_train.py b10_audit_phase9_train.py b10_phase9_terminal_recovery.py b10_phase10_d.py b10_audit_phase10_d.py b10_audit_phase10_d_portability.py)
tracked=(experiments/burgers-1e3-10x/PHASE-10-PRE-REGISTRATION.md experiments/burgers-1e3-10x/PHASE-10-D-AUDIT-RESOURCE-ESTIMATE.md experiments/burgers-1e3-10x/cluster/phase10_d_portability_audit.sbatch experiments/burgers-1e3-10x/cluster/verify_manifest_file_set.sh)
for name in "${sources[@]}"; do tracked+=("experiments/burgers-1e3-10x/$name"); done
for path in "${tracked[@]}" experiments/burgers-hybrid-1024/bh_common.py; do git -C "$WORKTREE" diff --quiet HEAD -- "$path" || exit 3; git -C "$WORKTREE" cat-file -e "$commit:$path"; done
for prior in "$P4" "$P5" "$P6" "$P7" "$P8"; do (cd "$prior" && sha256sum -c LOCAL.sha256); [[ "$(jq -r .status "$prior/out/AUDIT.json")" == pass ]]; done
for prior in "$FAILED" "$RECOVERY" "$ACCEPTED" "$R1" "$R2"; do (cd "$prior" && sha256sum -c LOCAL.sha256); done
[[ "$(sha256sum "$R2/out/phase10_d.json"|cut -d' ' -f1)" == f0fab7583d1f1590efb58cd97d0128f34cc80cf4ab7ce8e3271066158f239523 ]]
[[ "$(sha256sum "$R2/out/phase10_d.npz"|cut -d' ' -f1)" == 35d139fedcfff539aef25a335c556d47f2ec5e8581066a9e1f9b6ad8fcef4664 ]]
[[ "$(jq -r .status "$R2/out/AUDIT.json")" == fail ]]
rm -rf "$STAGE"; mkdir -p "$STAGE/logs" "$STAGE/out" "$STAGE/code/deps/"{p4,p5/targets,p6,p7,p8,failed,recovery,accepted,p10_r1_failed,r2_failed,burgers2d-coord-rom}
for name in "${sources[@]}"; do cp "$EXP/$name" "$STAGE/code/"; done
cp "$EXP/PHASE-10-PRE-REGISTRATION.md" "$STAGE/code/"; cp "$WORKTREE/experiments/burgers-hybrid-1024/bh_common.py" "$STAGE/code/"
cp "$ROOT/worktrees/2026-08-14-burgers2d-coord-rom/experiments/burgers2d-coord-rom/burgers2d_film.py" "$STAGE/code/deps/burgers2d-coord-rom/"
cp "$P4/out/phase4_d.json" "$P4/out/phase4_d.npz" "$P4/out/AUDIT.json" "$P4/MANIFEST.sha256" "$STAGE/code/deps/p4/"
cp "$P5/out/phase5_d.json" "$P5/out/phase5_d.npz" "$P5/out/AUDIT.json" "$P5/MANIFEST.sha256" "$STAGE/code/deps/p5/"; cp "$P5/out/targets/"*.npz "$STAGE/code/deps/p5/targets/"
cp "$P6/out/phase6_d.json" "$P6/out/phase6_d.npz" "$P6/out/AUDIT.json" "$P6/MANIFEST.sha256" "$STAGE/code/deps/p6/"
cp "$P7/out/phase7_train.json" "$P7/out/phase7_train.npz" "$P7/out/checkpoint.pkl" "$P7/out/AUDIT.json" "$P7/MANIFEST.sha256" "$STAGE/code/deps/p7/"
cp "$P8/out/phase8_d.json" "$P8/out/phase8_d.npz" "$P8/out/AUDIT.json" "$P8/MANIFEST.sha256" "$STAGE/code/deps/p8/"
cp -a "$FAILED/." "$STAGE/code/deps/failed/"; cp -a "$RECOVERY/." "$STAGE/code/deps/recovery/"; cp -a "$ACCEPTED/." "$STAGE/code/deps/accepted/"; cp -a "$R1/." "$STAGE/code/deps/p10_r1_failed/"; cp -a "$R2/." "$STAGE/code/deps/r2_failed/"
cp "$HERE/phase10_d_portability_audit.sbatch" "$STAGE/run.sbatch"; cp "$HERE/verify_manifest_file_set.sh" "$STAGE/"
(cd "$STAGE" && find . -type f ! -path './MANIFEST.sha256' -exec sha256sum {} \; | LC_ALL=C sort > MANIFEST.sha256)
"$STAGE/verify_manifest_file_set.sh" "$STAGE" "$STAGE/MANIFEST.sha256"; (cd "$STAGE" && sha256sum -c MANIFEST.sha256)
echo "stage=$STAGE"; echo "commit=$commit"; echo "manifest_file_sha256=$(sha256sum "$STAGE/MANIFEST.sha256"|cut -d' ' -f1)"; echo "file_count=$(wc -l < "$STAGE/MANIFEST.sha256")"
