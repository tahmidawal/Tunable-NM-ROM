#!/usr/bin/env bash
# make_phase11_g2_cell.sh p11_g2_s11_r1 COMMIT
set -euo pipefail
[[ $# -eq 2 ]] || exit 2
cell="$1"; commit="$2"
[[ "$cell" == p11_g2_s11_r1 && "$commit" =~ ^[0-9a-f]{40}$ ]] || exit 2
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; EXP="$(dirname "$HERE")"; WORKTREE="$(cd "$EXP/../.." && pwd)"; ROOT="$(cd "$WORKTREE/../.." && pwd)"
STAGE="$HERE/stage/$cell"
P4="$EXP/runs/p4_d_r3"; P5="$EXP/runs/p5_d_r1"; P6="$EXP/runs/p6_d_r1"; P7="$EXP/runs/p7_g1_s11_r3"; P8="$EXP/runs/p8_d_r2"
P9R="$EXP/runs/p9_terminal_recovery_s11_r1_failed"; P9A="$EXP/runs/p9_terminal_recovery_audit_r1"; P9T="$EXP/runs/p9_t1_s11_r1_failed"
P10="$EXP/runs/p10_d_r2_failed"; P10A="$EXP/runs/p10_d_r2_audit_r2"
[[ "$(git -C "$WORKTREE" rev-parse HEAD)" == "$commit" ]] || exit 3
sources=(b10_common.py b10_spline.py b10_spline_train.py b10_s0_spline.py b10_phase3.py b10_phase4.py b10_phase5.py b10_phase7.py b10_phase7_train.py b10_phase8_d.py b10_phase9_train.py b10_audit_phase9_train.py b10_phase11_g2.py b10_audit_phase11_g2.py)
tracked=(experiments/burgers-1e3-10x/PHASE-11-PRE-REGISTRATION.md experiments/burgers-1e3-10x/PHASE-11-RESOURCE-ESTIMATE.md experiments/burgers-1e3-10x/cluster/phase11_g2.sbatch experiments/burgers-1e3-10x/cluster/verify_manifest_file_set.sh)
for name in "${sources[@]}"; do tracked+=("experiments/burgers-1e3-10x/$name"); done
for path in "${tracked[@]}" experiments/burgers-hybrid-1024/bh_common.py; do
  git -C "$WORKTREE" diff --quiet HEAD -- "$path" || exit 3
  git -C "$WORKTREE" cat-file -e "$commit:$path"
done
for prior in "$P4" "$P5" "$P6" "$P7" "$P8" "$P9R" "$P9A" "$P9T" "$P10" "$P10A"; do
  [[ -f "$prior/LOCAL.sha256" ]]; (cd "$prior" && sha256sum -c LOCAL.sha256)
done
[[ "$(jq -r .status "$P9A/out/AUDIT.json")" == pass && "$(jq -r .status "$P10A/out/AUDIT.json")" == pass ]]
rm -rf "$STAGE"; mkdir -p "$STAGE/logs" "$STAGE/out" "$STAGE/code/deps/"{p4,p5/targets,p6,p7,p8,p9,p10,burgers2d-coord-rom}
for name in "${sources[@]}"; do cp "$EXP/$name" "$STAGE/code/"; done
cp "$EXP/PHASE-11-PRE-REGISTRATION.md" "$STAGE/code/"; cp "$WORKTREE/experiments/burgers-hybrid-1024/bh_common.py" "$STAGE/code/"
cp "$ROOT/worktrees/2026-08-14-burgers2d-coord-rom/experiments/burgers2d-coord-rom/burgers2d_film.py" "$STAGE/code/deps/burgers2d-coord-rom/"
cp "$P4/out/phase4_d.json" "$P4/out/phase4_d.npz" "$P4/out/AUDIT.json" "$P4/MANIFEST.sha256" "$STAGE/code/deps/p4/"
cp "$P5/out/phase5_d.json" "$P5/out/phase5_d.npz" "$P5/out/AUDIT.json" "$P5/MANIFEST.sha256" "$STAGE/code/deps/p5/"; cp "$P5/out/targets/"*.npz "$STAGE/code/deps/p5/targets/"
cp "$P6/out/phase6_d.json" "$P6/out/phase6_d.npz" "$P6/out/AUDIT.json" "$P6/MANIFEST.sha256" "$STAGE/code/deps/p6/"
cp "$P7/out/phase7_train.json" "$P7/out/phase7_train.npz" "$P7/out/checkpoint.pkl" "$P7/out/AUDIT.json" "$P7/MANIFEST.sha256" "$STAGE/code/deps/p7/"
cp "$P8/out/phase8_d.json" "$P8/out/phase8_d.npz" "$P8/out/AUDIT.json" "$P8/MANIFEST.sha256" "$STAGE/code/deps/p8/"
cp "$P9R/out/phase9_terminal_recovery.json" "$P9R/out/phase9_terminal_recovery.npz" "$STAGE/code/deps/p9/"
cp "$P9R/out/terminal_checkpoint.pkl" "$STAGE/code/deps/p9/checkpoint.pkl"; cp "$P9A/out/AUDIT.json" "$P9A/out/AUDIT-WORK.npz" "$P9A/MANIFEST.sha256" "$STAGE/code/deps/p9/"; cp "$P9T/out/work_checkpoint.pkl" "$STAGE/code/deps/p9/"
cp "$P10/out/phase10_d.json" "$P10/out/phase10_d.npz" "$P10/out/work_checkpoint.pkl" "$P10/MANIFEST.sha256" "$STAGE/code/deps/p10/"
cp "$P10A/out/AUDIT.json" "$STAGE/code/deps/p10/"; cp "$P10A/MANIFEST.sha256" "$STAGE/code/deps/p10/AUDIT-MANIFEST.sha256"
cp "$HERE/phase11_g2.sbatch" "$STAGE/run.sbatch"; cp "$HERE/verify_manifest_file_set.sh" "$STAGE/"
(cd "$STAGE" && find . -type f ! -path './MANIFEST.sha256' -exec sha256sum {} \; | sort > MANIFEST.sha256)
"$STAGE/verify_manifest_file_set.sh" "$STAGE" "$STAGE/MANIFEST.sha256"; (cd "$STAGE" && sha256sum -c MANIFEST.sha256)
echo "stage=$STAGE"; echo "commit=$commit"; echo "file_count=$(wc -l < "$STAGE/MANIFEST.sha256")"; echo "manifest_file_sha256=$(sha256sum "$STAGE/MANIFEST.sha256"|cut -d' ' -f1)"
