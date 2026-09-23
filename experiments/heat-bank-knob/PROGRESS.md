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
