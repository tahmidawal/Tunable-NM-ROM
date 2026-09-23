# heat-bank-knob — PROGRESS (kept current; read after any interruption)

Branch `exp/2026-09-23-heat-bank-knob` (LOCAL commits only — never push). Lane dir `experiments/heat-bank-knob/`.
Cluster namespace `/cluster/tufts/paralab/tawal01/hbank_20260923/<job>`. Design + pre-registered rule: `DESIGN.md`.
Submit: `cluster/submit.sh <job> h200 <HH:MM:SS> 240G <config>` (clean committed tree). Collect: `cluster/collect.sh <job> [--remove]`.

## Milestones
- 2026-09-23 ~03:00 EDT: code (hbk_core/hbk_run/hbk_audit_np/hbk_summarize), pinned rotations `prep_2d.npz` / `prep_3d.npz`
  (training data only), configs. Local 2D smoke (64², 128², dev cohort): parity 3.6e-14, audit passed with both controls
  detected; neighbour gate fails on the shared GB10 (expected noise; only the H200 value counts).

## Jobs
| job | slurm id | config | state |
|---|---|---|---|
| h2d | 4197350 | configs/h2d.json (2D 1024²/2048²/4096², val 791001 ×16 + sealed 791099 ×16) | submitted 2026-09-23 ~03:25 EDT, source 962ced91, H200 |
| h3d | 4197416 | configs/h3d.json (3D 32³/64³/128³, val 921777 ×16 + sealed 921099 ×64) | submitted ~03:50 EDT, source bae9e1e6, H200 |
| h3d256 | - | configs/h3d256.json (3D 256³) | to submit when one of the above finishes (≤ 2 jobs) |

- Local 3D smoke (16³, dev cohort): parity ≤ 7.5e-14 on all 12 paired arms; audit passed, both controls detected.
