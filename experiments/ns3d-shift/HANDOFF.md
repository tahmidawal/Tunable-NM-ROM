# Handoff — ns3d-shift

Updated 2026-09-22. Branch `exp/2026-09-22-ns3d-shift-decoder`, forked from
`exp/2026-09-21-ns3d-grok` @ 8852b7cd. Namespace
`/cluster/tufts/paralab/tawal01/ns3dshift_20260922/`.

## Idea

The translation is an **online unknown of the reduced least-squares problem**.
Implemented in the co-moving ("freezing") form: `u(x,t) = v(x - c(t), t)` with
`v = G a` in a fixed **centered** bank turns the weak residual into the ordinary
one plus a single term linear in the frame increment `delta = c^{n+1} - c^n`.
`A = Phi^T G`, the advection tensor and `D_d = Phi^T d_d G` are all offline, so
**no basis is ever shifted at run time**. See `DESIGN.md`.

The mechanism is free for *any* fixed bank. A coordinate network is **not** what
makes the translation cheap; the co-moving formulation is. The learned-bank arm
(F) was deferred for that reason and never run.

## What was run

| job | config | what it settled |
|---|---|---|
| 4176514 `pilot01` | `configs/pilot01.json` | floors, harness checks, identifiability, arms B0/B1/B2 + ladder, tracker, CNAB2. `results/pilot01.md` |
| 4176596 `cost02` | `configs/cost02.json` | M and dt sweep at gauge 0 with the per-piece cost split. `results/cost02.md` |

Both A100, `jax_backend=gpu`, f64, matmul precision highest, one job directory
each, remote directories deleted after a checksum-verified pull. Budget used: 2
of 6. Field `.npy` files are not committed (75 MB each); they live in the session
scratch and are regenerable from the recorded seeds.

## Headline

Accuracy **passes** and speed **fails**, so stop rule 3 applies: no sealed draw.
The sealed seed named in `DESIGN.md` (202609221) has **not** been drawn. Seeds
202609203 and 202609211 remain closed.

## Next

See the closing section of `../../reports/2026-09-22-ns3d-shift-decoder.md`.
