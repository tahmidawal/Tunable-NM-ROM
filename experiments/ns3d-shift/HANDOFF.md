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
| 4176669 `mesh03` | `configs/mesh03.json` | exploratory N=64 probe (DESIGN amendment). `results/mesh03.md` |
| 4178112 `fast04` | `configs/fast04.json` | the driver fix measured before/after in one job, plus the 32^3 frontier |
| 4178149 `ladder64` | `configs/ladder64.json` | 64^3 frontier, floors, tracker, CNAB2 stability |
| 4178207 `ladder96` | `configs/ladder96.json` | 96^3 frontier |

All A100 (`mesh03` on an 80GB card, the others on 40GB), `jax_backend=gpu`, f64, matmul precision highest, one job directory
each, remote directories deleted after a checksum-verified pull. Budget used: 3
of 6. Field `.npy` files are not committed (75 MB each); they live in the session
scratch and are regenerable from the recorded seeds.

## The driver fix

`ns2d_rom.make_lm` -> `shift_rom.make_frozen_run`: fixed damped Gauss-Newton
sweeps, analytic Jacobian, statically unrolled scan, extrapolated warm start,
constants hoisted, Cholesky instead of LU, one contraction against `T + T^T`
instead of two. Measured 2.52x at matched rank/step with field parity 1.46e-08.
`build_operators_fast` also removes the dense test matrix (14.7 GB at 128^3).
`test_fast_solver.py` is the local gate; run it before touching either.

## Headline

Accuracy passes comfortably: **0.449 %** evolved worst, 0/16 over the 5 % target,
against 2.538 % for the centroid tracker on the same cohort and a 0.128 % oracle
floor. The frame is recovered with **no oracle and no gauge**; the gauge was
tried at three weights and hurt at all of them.

Speed loses at $N=32$ (**0.232x**, best setting 0.422x) and **crosses at $N=64$**
(**1.789x** at 2.073 % evolved worst, 0/16 over target). The $N=32$ cost is ~90 %
generic damped-LM driver overhead, not physics, and that overhead carries no $N$.

Stop rule 3 applied after `pilot01`: **no sealed draw**. `mesh03` was an
explicitly exploratory DESIGN amendment and carries no licence to open one
either. Seed **202609221 has not been drawn**; 202609203 and 202609211 remain
closed.

## Next

See the closing section of `../../reports/2026-09-22-ns3d-shift-decoder.md`.
