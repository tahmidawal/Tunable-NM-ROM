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
| h2d | 4197350 | configs/h2d.json (2D 1024²/2048²/4096², val 791001 ×16 + sealed 791099 ×16) | COMPLETED 06:44 EDT (H200 pax010, 3.6 h); collected, checksums ok, audit v1+v2 passed, remote removed |
| h3d | 4197416 | configs/h3d.json (3D 32³/64³/128³, val 921777 ×16 + sealed 921099 ×64) | submitted ~03:50 EDT, source bae9e1e6, H200 |
| h3d256 | 4206722 | configs/h3d256.json (3D 256³) | submitted 06:50 EDT, source 00147a52 |

- Local 3D smoke (16³, dev cohort): parity ≤ 7.5e-14 on all 12 paired arms; audit passed, both controls detected.

## h2d findings (generated: REPORT.md / runs/h2d/summary.json)
- Parity 1e-13 at every mesh; determinism ok; audit v2 passed. Sealed q0/q32 errors reproduce the paper (1.3616 / 0.4876 %).
- **Neighbour (order-effect) gate FAILS at 4096²**: nmrom_R64_q0_bf 1.104 > 1.10 (bf NM-ROM arms run ~8 % slower right after a CG solve); passes at 1024² (1.024) and 2048² (1.071). Linear-rung arms pass (≤ 1.02).
- Linear rung dominates: accurate = lin_R128 (0.133 %) in both families; fast = lin_R48 (0.276 %).
- Nested column-block storage costs overhead at R' = R: lin_R128_cn 8.42 ms vs unrotated parent_lin_cn 5.66 ms at 4096².
- q = 0 error is not monotone in R' (1.362 → 1.319 % at R'=48, then 1.571 %); linear rung and q = R'−K are monotone.
