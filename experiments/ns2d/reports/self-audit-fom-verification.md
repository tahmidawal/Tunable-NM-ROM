# ns2d — self-audit of the FOM verification (Codex substitute)

The lane protocol requires an independent Codex audit of the design before the first job and
of the FOM verification before Phase 2. Codex was unavailable on 2026-09-17 (usage limit
exhausted until 2026-09-19 11:33; the launched run also failed its bubblewrap sandbox before
reading any file — `codex-design-audit` produced no output). A Claude subagent audit launched
as a stopgap was killed by the API session limit before reporting. This document is the
**self-audit** the coordinator asked for in their place; it is written by the lane owner and
must be read as such. Codex will be re-run on 2026-09-19 if the quota returns, against this
document and the run JSONs.

## What the verification checks, gate by gate (DESIGN.md → `ns2d_phase1.py`)

| DESIGN gate | code | what would make it fail | negative control in code |
|---|---|---|---|
| F-LAP | `gate_lap` | a wrong DFT symbol or a non-zero mean mode | — (identity; a wrong symbol reads O(1)) |
| F-JAC | `gate_jac` | any transcription error in `arakawa` (the three sub-Jacobians must cancel in the sums) | plain centred Jacobian `jac_centred` must fail the $\sum\omega J$ identity (reads 2e-2 / 4e-3 in the smoke) |
| F-JAC-ORDER | main | a first-order stencil | — (order test) |
| F-TG-exact (§A1) | `gate_tg` | wrong $\lambda_h$, wrong Poisson solve, a non-antisymmetric $J_A$, wrong time scheme | wrong-$\lambda$ (4/3) and backward Euler, both against the same closed form |
| F-TG-semi, F-TG-cont | `gate_tg`, main | time error above $10^{-6}$; spatial order $\ne 2$; measured continuum error not matching the closed-form prediction to 2 % | — |
| F-MMS | `gate_mms`, main | wrong advection sign, wrong source, wrong midpoint evaluation time | **flipped `jac_sign`** must fail at $\ge 10^{-2}$ (reads 4.9e-2 at 16²) |
| F-BUDGET | `gate_budget` | any term of the residual not evaluated at the midpoint; a non-symmetric Poisson inverse | backward Euler at $\nu=0$ must drift $\ge 10^{-4}$ (reads 3e-4) |
| F-MESH | `gate_mesh` | resolution loss, order $\ne 2$ | — |
| F-INDEP | `gate_indep`, `ns2d_indep.py` | any difference between the JAX stencils and independently written NumPy stencils, Poisson solve, Jacobian and Newton | — (an independent implementation is its own control) |
| F-CFL, F-NEWTON, F-DATA | `generate` | unconverged Newton steps; CFL > 2 | — |

## Bugs the verification caught before the first cluster job

1. **`ns2d_indep.arakawa_np` had the opposite sign** (Arakawa's eq. 46 is written as $J(\zeta,\psi)$).
   Caught by the analytic-$J$ check: the relative error converged to exactly 2.0 while every
   conservation identity held at 1e-17 — i.e. the identities alone would have passed a wrong
   sign. Fixed by negating; the error then converges at second order (8.8e-2, 2.3e-2, 5.7e-3,
   1.4e-3 at N = 32…256). Lesson recorded: conservation identities are sign-blind; only an
   analytic-$J$ or MMS check fixes the advection sign, which is why F-MMS carries the flipped
   sign control.
2. **The mesh-order estimator was biased.** Using the finest level as the reference over
   three levels turns a pure $Ch^2$ error into an observed order of $\log_2 5 = 2.32$ (the
   smoke read 2.28 and 2.21). Replaced by successive-level differences (1.95 / 1.88 on the
   16/32/64 smoke, pre-asymptotic at 16²). The `errors_vs_finest` are still reported as a
   diagnostic. This is a retraction of the gate arithmetic, not of the scheme.
3. **Roundoff-level audit comparisons.** The NumPy audit initially compared the budget
   identity residuals (∼1e-13) to the JAX values at 1e-12 relative and "mismatched"; these are
   value gates against the threshold across implementations (lane-protocol landmine 14),
   now handled as such.

## Smoke evidence (local GB10, 16² and 32², T = 0.2, `runs/`… scratch; not a certified run)

Recorded here because the same code runs on the cluster; the certified numbers are those of
job 3780151 (attempt `ns101`) and appear in the generated report.

- F-TG-exact 1.0e-14 / 5.1e-15; semi 3.2e-8; continuum 2.02e-3 / 5.07e-4 against predictions
  matching to 1.6e-5 / 6.4e-5 relative; wrong-$\lambda$ 5.3e-2, backward Euler 1.2e-4.
- F-MMS spatial 6.7e-3 → 1.7e-3 (order 1.97 at 16→32), temporal order 2.000006 from the three
  $\Delta t$ levels, flipped-sign control 4.9e-2.
- F-BUDGET identities 1.0e-13 … 2.2e-13; $\nu=0$ drifts 1.6e-16 / 4.4e-16; BE control 3e-4.
- F-INDEP worst 3.5e-16 over 10 steps at 32² (JAX 1.5 s vs NumPy dense-Newton 55 s).
- Newton: at most 2 iterations per step, worst relative residual 7e-17 (ntol 1e-11 requested).

## What this self-audit cannot do

It cannot replace an audit by a different model family. Specifically unverified by anyone
but the author: (i) that the Arakawa form in `ns2d_fom.arakawa` is the standard one rather
than merely a conserving, antisymmetric, second-order bilinear form (the gates test the
latter properties, which are what the ROM tensor needs; the former is a naming question);
(ii) the pre-registered 2× criterion's calibration (it was chosen before any NS number
existed, by analogy with Burgers' 3.6×). Both are flagged for the Codex pass on 2026-09-19.

## To be completed from job 3780151

- [ ] every Phase-1 gate at 64/128/256 (+512 in F-MESH) with its number
- [ ] F-JAC-ORDER at 64→128→256 (expected 2.0 ± 0.1; the 16→32 smoke value 1.83 is pre-asymptotic)
- [ ] F-MESH orders from successive differences at Re 100 and Re 1000
- [ ] F-INDEP at 64² over 100 steps
- [ ] `audit_phase1.py` ALL_MATCH on the collected output
