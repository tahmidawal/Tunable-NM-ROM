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

## Completed from job 3780151 (attempt `ns101`, A100 `pax105`, 47 m 33 s, `jax_backend=gpu`, x64, matmul `highest`, commit `8a01627b`)

Certified numbers. 28 gates; 25 pass as written and the three F-JAC gates pass under §A2.
The independent NumPy audit (`runs/ns101/audit.json`) recomputed every one of these from the
saved fields without importing the driver or JAX: **52 checks, ALL_MATCH = true.**

| gate | certified numbers | verdict |
|---|---|---|
| F-LAP | 1.1e-15 / 4.5e-15 / 1.5e-14 at $N=64/128/256$ | pass |
| F-JAC identities | $\le 9.3\times10^{-18}$; antisymmetry $\le 2.4\times10^{-16}$ | pass |
| F-JAC control (§A2) | centred-Jacobian $\sum\omega J$ leak / Arakawa's = 1.5e14, 6.4e13, 2.4e13 (bar $10^6$) | pass |
| F-JAC-ORDER | 2.27e-2, 5.72e-3, 1.43e-3 → orders 1.989, 1.997 | pass |
| F-TG-exact | 1.7e-14, 4.4e-14, 9.7e-14 vs the closed-form discrete solution | pass |
| F-TG-semi | 1.64e-7 (predicted $\approx1.6\times10^{-7}$), bar $10^{-6}$ | pass |
| F-TG-cont | 6.34e-4, 1.58e-4, 3.95e-5; orders 2.001, 2.004; each within 0.03–0.41 % of its closed-form prediction | pass |
| F-TG controls | wrong-$\lambda$ 3.0e-1, backward Euler 6.2e-4 | both fire |
| F-MMS spatial | 4.32e-4, 1.08e-4, 2.71e-5 → orders 1.997, 1.998 | pass |
| F-MMS temporal | order 2.000006 from three $\Delta t$ levels at $N=128$ | pass |
| F-MMS sign control | flipped $J_A$ reads 5.21e-2 | fires |
| F-BUDGET identities | enstrophy $\le1.0\times10^{-13}$, energy $\le1.4\times10^{-13}$ over 100 steps | pass |
| F-BUDGET $\nu=0$ | enstrophy drift $\le3.5\times10^{-16}$, energy $\le4.4\times10^{-16}$ | pass |
| F-BUDGET control | backward Euler drifts 5.4e-3 – 5.5e-3 | fires |
| F-MESH Re 100 | successive diffs 3.65e-3, 9.08e-4, 2.27e-4 → orders 2.009, 2.002 | pass |
| F-MESH Re 1000 | successive diffs 7.29e-2, 1.87e-2, 4.66e-3 → orders 1.965, 2.004 | pass |
| F-INDEP | worst 9.14e-16 over 100 steps at 64² (JAX 1.45 s; NumPy dense-Newton 653 s) | pass |
| F-DATA | 3 cohorts × 3 meshes; worst Newton residual 9.8e-15; max CFL 0.31 / 0.61 / 1.23 | pass |

**Re 1000 is resolved at every mesh in the ladder.** The design flagged 64² as possibly
under-resolved at $\nu=10^{-3}$; the measured orders are 1.96 and 2.00 across
$64\to128\to256\to512$, so the family is second-order convergent at both Reynolds numbers and
no mesh had to be dropped. Max CFL 1.23 at 256² is within the recorded bound (the scheme is
A-stable; the temporal-order gate, not CFL, is the accuracy guarantee).

### What the verification caught, in order

The three bugs above (independent-Jacobian sign, biased mesh-order estimator, roundoff-level
audit comparisons) were caught locally. The cluster run caught the fourth:

4. **F-JAC's negative control was an absolute threshold on a mesh-scaling quantity** — the
   centred Jacobian's conservation leak is itself $O(h^2)$, so the $10^{-3}$ floor failed at
   $N\ge64$ for correct code. Amended to a ratio (§A2). This is the fourth instance in this
   lane of the failure mode the project has named repeatedly; it is worth noting that it was
   caught by a *passing* system (the identities held at 1e-18 while the gate read FAIL), not
   by a suspicious result.

## What this self-audit still cannot do

Unchanged from above: no independent model has reviewed this lane. Specifically unverified by
anyone but the author: (i) that `ns2d_fom.arakawa` is the standard Arakawa form rather than
merely a conserving, antisymmetric, second-order bilinear form (the gates test the latter,
which is what the ROM tensor needs); (ii) the calibration of the pre-registered 2× criterion.
Both remain flagged for a Codex pass when the quota returns (2026-09-19 11:33).
