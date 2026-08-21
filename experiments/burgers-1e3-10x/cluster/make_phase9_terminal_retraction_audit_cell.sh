#!/usr/bin/env bash
# make_phase9_terminal_retraction_audit_cell.sh p9_terminal_recovery_audit_r1 COMMIT p4_d_r3 p5_d_r1 p6_d_r1 p7_g1_s11_r3 p8_d_r2 p9_t1_s11_r1_failed p9_terminal_recovery_s11_r1_failed
set -euo pipefail
cell="$1"; commit="$2"; p4c="$3"; p5c="$4"; p6c="$5"; p7c="$6"; p8c="$7"; failedc="$8"; recoveryc="$9"
[[ "$cell" == p9_terminal_recovery_audit_r1 && "$commit" =~ ^[0-9a-f]{40}$ ]] || exit 2
[[ "$p4c $p5c $p6c $p7c $p8c $failedc $recoveryc" == "p4_d_r3 p5_d_r1 p6_d_r1 p7_g1_s11_r3 p8_d_r2 p9_t1_s11_r1_failed p9_terminal_recovery_s11_r1_failed" ]] || exit 2
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; EXP="$(dirname "$HERE")"; WORKTREE="$(cd "$EXP/../.." && pwd)"; ROOT="$(cd "$WORKTREE/../.." && pwd)"
STAGE="$HERE/stage/$cell"; P4="$EXP/runs/$p4c"; P5="$EXP/runs/$p5c"; P6="$EXP/runs/$p6c"; P7="$EXP/runs/$p7c"; P8="$EXP/runs/$p8c"; FAILED="$EXP/runs/$failedc"; RECOVERY="$EXP/runs/$recoveryc"
[[ "$(git -C "$WORKTREE" rev-parse HEAD)" == "$commit" ]] || exit 3
sources=(b10_common.py b10_spline.py b10_spline_train.py b10_s0_spline.py b10_phase3.py b10_phase4.py b10_phase5.py b10_phase7.py b10_phase7_train.py b10_phase8_d.py b10_phase9_train.py b10_phase9_terminal_recovery.py b10_audit_phase9_terminal_recovery.py b10_audit_phase9_terminal_retraction.py)
tracked=(experiments/burgers-1e3-10x/PHASE-9-TERMINAL-RECOVERY-PRE-REGISTRATION.md experiments/burgers-1e3-10x/PHASE-9-TERMINAL-RETRACTION-AUDIT-RESOURCE-ESTIMATE.md experiments/burgers-1e3-10x/cluster/phase9_terminal_retraction_audit.sbatch experiments/burgers-1e3-10x/cluster/verify_manifest_file_set.sh)
for name in "${sources[@]}"; do tracked+=("experiments/burgers-1e3-10x/$name"); done
for path in "${tracked[@]}" experiments/burgers-hybrid-1024/bh_common.py; do
  git -C "$WORKTREE" diff --quiet HEAD -- "$path" || exit 3; git -C "$WORKTREE" cat-file -e "$commit:$path"
done
for prior in "$P4" "$P5" "$P6" "$P7" "$P8"; do
  [[ -f "$prior/LOCAL.sha256" ]]; (cd "$prior" && sha256sum -c LOCAL.sha256); [[ "$(jq -r .status "$prior/out/AUDIT.json")" == pass ]]
done
(cd "$FAILED" && sha256sum -c LOCAL.sha256)
(cd "$RECOVERY" && sha256sum -c LOCAL.sha256)
[[ "$(sha256sum "$RECOVERY/out/phase9_terminal_recovery.json"|cut -d' ' -f1)" == 8d86ab90c1d293e4810c3d3a9391db48f635f8c77a7a8c30e71809a112bc7f8d ]]
[[ "$(sha256sum "$RECOVERY/out/phase9_terminal_recovery.npz"|cut -d' ' -f1)" == d6e5001d8dcb5ba7fb269241b989492533379807bcbc4cb65a384f7d429c773e ]]
[[ "$(sha256sum "$RECOVERY/out/terminal_checkpoint.pkl"|cut -d' ' -f1)" == 90e9df6388bf3c05905d52c5ff073f331728ffd493a965e4376e4df285bb07d9 ]]
[[ "$(sha256sum "$RECOVERY/out/AUDIT.json"|cut -d' ' -f1)" == a1c6c0ee23e16da4aab1e4320aedcab4bd7039ec650429908feb15a471f99cec ]]
[[ "$(jq -r .remote_preserved "$RECOVERY/FAILURE.json")" == true && "$(jq -r .remote_cleaned "$RECOVERY/FAILURE.json")" == false ]]
rm -rf "$STAGE"; mkdir -p "$STAGE/logs" "$STAGE/out" "$STAGE/code/deps/"{p4,p5/targets,p6,p7,p8,failed,recovery,burgers2d-coord-rom}
for name in "${sources[@]}"; do cp "$EXP/$name" "$STAGE/code/"; done
cp "$EXP/PHASE-9-TERMINAL-RECOVERY-PRE-REGISTRATION.md" "$STAGE/code/"
cp "$WORKTREE/experiments/burgers-hybrid-1024/bh_common.py" "$STAGE/code/"
cp "$ROOT/worktrees/2026-08-14-burgers2d-coord-rom/experiments/burgers2d-coord-rom/burgers2d_film.py" "$STAGE/code/deps/burgers2d-coord-rom/"
cp "$P4/out/phase4_d.json" "$P4/out/phase4_d.npz" "$P4/out/AUDIT.json" "$P4/MANIFEST.sha256" "$STAGE/code/deps/p4/"
cp "$P5/out/phase5_d.json" "$P5/out/phase5_d.npz" "$P5/out/AUDIT.json" "$P5/MANIFEST.sha256" "$STAGE/code/deps/p5/"; cp "$P5/out/targets/"*.npz "$STAGE/code/deps/p5/targets/"
cp "$P6/out/phase6_d.json" "$P6/out/phase6_d.npz" "$P6/out/AUDIT.json" "$P6/MANIFEST.sha256" "$STAGE/code/deps/p6/"
cp "$P7/out/phase7_train.json" "$P7/out/phase7_train.npz" "$P7/out/checkpoint.pkl" "$P7/out/AUDIT.json" "$P7/MANIFEST.sha256" "$STAGE/code/deps/p7/"
cp "$P8/out/phase8_d.json" "$P8/out/phase8_d.npz" "$P8/out/AUDIT.json" "$P8/MANIFEST.sha256" "$STAGE/code/deps/p8/"
cp -a "$FAILED/." "$STAGE/code/deps/failed/"; cp -a "$RECOVERY/." "$STAGE/code/deps/recovery/"
cp "$HERE/phase9_terminal_retraction_audit.sbatch" "$STAGE/run.sbatch"; cp "$HERE/verify_manifest_file_set.sh" "$STAGE/"
(cd "$STAGE" && find . -type f ! -path './MANIFEST.sha256' -exec sha256sum {} \; | sort > MANIFEST.sha256)
"$STAGE/verify_manifest_file_set.sh" "$STAGE" "$STAGE/MANIFEST.sha256"; (cd "$STAGE" && sha256sum -c MANIFEST.sha256)
echo "stage=$STAGE"; echo "commit=$commit"; echo "manifest_file_sha256=$(sha256sum "$STAGE/MANIFEST.sha256"|cut -d' ' -f1)"
