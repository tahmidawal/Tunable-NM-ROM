# poisson-bank-knob-3d — progress (kept current)

Branch `exp/2026-09-23-poisson-bank-knob-3d` (local commits only, never pushed). Namespace
`/cluster/tufts/paralab/tawal01/pbank3_20260923/<attempt>/`. Design: `DESIGN.md`. Numbers: `reports/` (generated).

## Milestones

- 2026-09-23 ~03:00 EDT — code, rotations (training data only), configs committed (`50aeed0f`).
  Rotation diagnostics: cube `R_G` cond 1.7e4, training-projection worst floor 0.20 % at R'=64, 0.12 % at 96, 0.117 % at 128;
  L-shape `R_G` cond 2.3e5, `||LT-I||` 2.3e-11, training worst floor 12 % at R'=64, 3.9 % at 128, 1.94 % at 256, 1.90 % at 512.
- Cluster smokes (A100): smokec1 4197332, smokel1 4197333, smokec2 4197411, smokel2 4197414 — all gates + NumPy audit PASS
  (NumPy linear-rung recompute 1e-15, wrong-R' control detected). Smoke numbers are not results. Remote dirs deleted.
- c32 4197502 (A100): all numbers fine but the pre-registered neighbour gate FAILED (+1.1 ms on every arm). Diagnostics
  c32diag 4197813 and c32diag2 4198000 traced it to the missing device-guard call in the re-time, not to the CG
  predecessor; back-to-back steady-state times match the main phase. Amendment A1 (DESIGN.md): paired, interleaved
  order gate. c32 superseded. c64 4197504 cancelled at start.
- Collected, audit PASS, all gates (A1 order gate): l256 4197817 (A1 recomputed offline from 6x1 pairs), c32b 4198670,
  c64b 4198674, l512 4199770, c32final 4200246, c64final 4202245, l1024 4201792.
- 32^3/64^3 settings FROZEN on development (commits f6c10ffe, d2c775a9) before the final jobs were staged:
  accurate R128_linear, fast R64_linear at both meshes.
- Pending (H200): c256 4203255, l2048 4203305. Next: c128 (H200).

## Queue plan (max 2 of this lane at once)

c32, c64 (dev, A100) -> freeze 32/64 settings -> l256, l512 (A100) -> l1024 (A100) -> l2048, c128, c256 (H200) ->
c32final, c64final (A100, reserved final cohort, frozen settings, once).
