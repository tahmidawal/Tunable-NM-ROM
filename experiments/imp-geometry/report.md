# Geometry without a fast transform (E2, S1 wide family): on the holed box the bank is slower than the strongest tested competitor (L-shape: descriptive only)

**Status (2026-10-22): final for the two executed S1 wide-family cells (H qualified; L descriptive only, its validation references fail the eligibility gate); S2 and the box controls were not run.** Lane `exp/2026-10-21-imp-geometry`. Pre-registered A10 outcome: **not supported at S1; S2 untested** (geometry L: S1-only, wide family; WIN not decidable without S2 and box controls; geometry H: WITHDRAWN (measured loss)). Every measured number below is read by `make_report.py` from the run JSONs; protocol constants (cap, thresholds, cohort sizes) are those registered in DESIGN.md; references are conditional Richardson estimates. Every S is an upper bound on our speed-up relative to more thoroughly tuned competitors (finite ladders), with the selected bank fixed.

## 1. In brief

- **L | wide | S1 (descriptive; reference-ineligible):** S (bank vs best competitor, validation) ranges 0.0147–0.214 over the three targets and two statistics; the best competitor is a POD reduced model (ecsw, podg). Decision labels per target (worst error): τ=0.01: no evidence (reference); τ=0.03: no evidence (reference); τ=0.1: no evidence (reference).
- **H | wide | S1:** S (bank vs best competitor, validation) ranges 0.0292–0.216 over the three targets and two statistics; the best competitor is a POD reduced model (ecsw, podg). Decision labels per target (worst error): τ=0.01: loss; τ=0.03: loss; τ=0.1: loss.
- Against the high-order spectral-element competitor alone the bank is also slower at every target (per-family S in §3); it is faster than FD + multigrid only at the loosest targets.
- GPU-hours spent: 30.68 of the 62 cap.

## 2. What ran

| job | name | GPU | state | GPU-h |
|---|---|---|---|---|
| 6158037 | smoke1 | a100 | COMPLETED | 0.0267 |
| 6158118 | smoke2 | a100 | COMPLETED | 0.366 |
| 6158303 | smoke3 | a100 | COMPLETED | 0.05 |
| 6158754 | smoke4 | a100 | COMPLETED | 0.131 |
| 6158976 | smoke5 | a100 | COMPLETED | 0.5 |
| 6175734 | meas6 | a100-80G | COMPLETED | 0.0294 |
| 6175736 | L_wide_S1 | a100-80G | TIMEOUT | 11.6 |
| 6192623 | L_wide_S1_r | a100-80G | COMPLETED | 6 |
| 6192666 | H_wide_S1 | a100-80G | COMPLETED | 11.5 |
| 6202632 | test_L_wide_S1 | a100-80G | COMPLETED | 0.189 |
| 6209321 | test_H_wide_S1 | a100-80G | COMPLETED | 0.255 |

Table: Slurm ledger (`runs/GPU-LEDGER.json`, elapsed × GPUs from `sacct`).

| cell | eligible refs (E0 / corner / ineligible) | U | median D_bnd (val / test) |
|---|---|---|---|
| L | 7 / 21 / 4 | 0.00092 | 0.0925 / 0.0813 |
| H | 32 / 0 / 0 | 0.000273 | 0.0995 / 0.0676 |

Table: references (val32) and the boundary-interaction diagnostic (A3.2; gate 0.05).

## 3. Speed ratios

| cell | τ | statistic | S val | S test | best competitor | spectral elements (C-SEM) | FD + multigrid (C-FDMG) | FD + RK4 (C-RK4) | POD + ECSW (C-ECSW) | POD-Galerkin (R-PODG) |
|---|---|---|---|---|---|---|---|---|---|---|
| L (descriptive) | 0.01 | worst | 0.0336 | – | POD + ECSW (C-ECSW) | 0.0856 | 0.324 | 0.0719 | 0.0336 | 0.0422 |
| L (descriptive) | 0.01 | median | 0.0147 | 0.0128 | POD-Galerkin (R-PODG) | 0.0391 | 0.172 | 0.0377 | 0.0149 | 0.0147 |
| L (descriptive) | 0.03 | worst | 0.0399 | 0.0242 | POD-Galerkin (R-PODG) | 0.107 | 0.471 | 0.106 | 0.0415 | 0.0399 |
| L (descriptive) | 0.03 | median | 0.0601 | 0.0625 | POD-Galerkin (R-PODG) | 0.0925 | 0.727 | 0.162 | 0.0606 | 0.0601 |
| L (descriptive) | 0.1 | worst | 0.0916 | 0.0903 | POD-Galerkin (R-PODG) | 0.108 | 1.36 | 0.299 | 0.105 | 0.0916 |
| L (descriptive) | 0.1 | median | 0.214 | – | POD-Galerkin (R-PODG) | 0.298 | 3.59 | 0.837 | 0.244 | 0.214 |
| H | 0.01 | worst | 0.0332 | 0.0357 | POD + ECSW (C-ECSW) | 0.0865 | 0.334 | 2.19 | 0.0332 | 0.0435 |
| H | 0.01 | median | 0.0292 | 0.0306 | POD-Galerkin (R-PODG) | 0.078 | 0.341 | 2.26 | 0.0301 | 0.0292 |
| H | 0.03 | worst | 0.0677 | 0.0667 | POD + ECSW (C-ECSW) | 0.152 | 0.653 | 4.29 | 0.0677 | 0.0891 |
| H | 0.03 | median | 0.0949 | 0.0955 | POD-Galerkin (R-PODG) | 0.146 | 1.12 | 7.34 | 0.0951 | 0.0949 |
| H | 0.1 | worst | 0.119 | 0.12 | POD-Galerkin (R-PODG) | 0.234 | 1.77 | 11.8 | 0.137 | 0.119 |
| H | 0.1 | median | 0.216 | – | POD-Galerkin (R-PODG) | 0.304 | 3.66 | 24 | 0.246 | 0.216 |

Table: S = median paired ratio t_competitor / t_bank (S < 1: the bank is slower); "S val" is the max–min over finalists on val[0:8], "S test" the same on test[0:8] with test-gated finalists; per-family columns are validation S against each family.

![per-family S](plots/S_per_family.png)

Figure: validation $S$ of each competitor family against the bank (worst-error matching; log scale; dashed line $S=2$ is the win bar, $S<1$ means the bank is slower).

## 4. Decision (pre-registered rules)

- A10 S1 readout: **not supported at S1; S2 untested**.
- Geometry L: **S1-only, wide family; WIN not decidable without S2 and box controls** (A4.2: a WIN needs both wide levels to win; a level with a measured loss withdraws the geometry; the box controls only decide attribution of a WIN).
- Geometry H: **WITHDRAWN (measured loss)** (A4.2: a WIN needs both wide levels to win; a level with a measured loss withdraws the geometry; the box controls only decide attribution of a WIN).
- Test-once finalists that missed τ or failed Q1/Q2 on test32 are excluded from the test S (A4.2); several bank test finalists failed Q2 or the tight-solve bar, which is why some test S are empty.
- Limitation: the test stage stores only family-level T-dbl pass flags, not the individual ratios.
- L labels are "no evidence (reference)" because 4 of 32 validation references fail the Richardson order gate (A3.3 requires all 32); the measured S values are nevertheless below 2 at every target.

## 5. Deviations, retractions, limitations

- The first geometry (side-2 domains) gave no interaction between bumps and features; it was redesigned by the interaction diagnostic (DESIGN A2). **Retraction (A3.1):** A2 claimed no solver or ROM result existed at selection time; smoke ladder rows had in fact been produced and partly inspected, so the redesign was diagnostic-guided problem development, not outcome-blind.
- The first L cell job (6175736) timed out in the ladder (11.6 GPU-h); the registered reductions were applied (A9: reduced-arm rank ladder r in {16,32,64,128,256,384}; implementation variants screened only for the 3 fastest candidates; spectral-element grading off/0.5) and the cell was resumed; the narrow family, the S2 cells and the box controls were deferred (A8, A9).
- The pre-release smoke allowance of 1 GPU-h was exceeded by 0.074 GPU-h (recorded).
- Results are scoped to this family, the field output contract, these implementations and one A100-80GB.
- Competitors are admitted on validity and accuracy; their Q1 is reported, not gated (A4.2); this can only lower S.
- Static condensation and multi-domain Chebyshev were not built; C-SEM uses Jacobi-PCG or an exact eigendecomposition solve.

## 6. Glossary

- **S:** speed ratio: competitor time over bank time at matched error; S ≥ 2 is the win bar; S < 1 means the bank is slower
- **τ, worst / median:** error target; the worst or median relative L2 error over the 32 validation cases
- **best competitor:** the competitor family whose matched configuration gives the smallest S
- **bank:** the coordinate-network ordered span with a distance envelope (our method), CN-R LSPG stepper
- **POD-Galerkin / POD + ECSW:** linear reduced models built from the same training data on the FD grid, the second with empirical cubature
- **U, eligible:** reference allowance (2 × max Richardson estimate + time check); a target is usable only if all 32 references are eligible and U ≤ τ/5
- **D_bnd:** difference between the domain solution and a box evolution of the same initial condition: how much the feature matters
- **test-once:** a held-out cohort scored once with the frozen validation-selected configurations
- **A4.2, A10:** the registered decision table and the coordinator S1 conditional (DESIGN.md)
- **L, H:** the L-shaped domain (unit square minus a corner quadrant) and the unit square with a circular hole
- **S1, S2:** the two viscosity (steepness) levels; only S1 was run
- **wide family:** initial bumps with centres spread over a 0.4-wide square (minus a small exclusion zone near the features)
- **E0 / corner-limited:** reference cases whose observed Richardson order is in [1.8, 2.2] / [1.3, 2.5]; others are ineligible
- **Richardson estimate:** reference-error estimate from three grid levels; a conditional estimate, not a bound
- **Q1 / Q2:** numerical qualification of a reduced configuration: solver certificates (Q1) and quadrature-rule convergence (Q2)
- **T-dbl, T-id, sentinel:** timing controls: serial two-case call vs the sum of single calls; timed outputs equal the official outputs; a reference timing repeated before and after
- **FD, RK4, SEM:** finite differences (with multigrid or explicit Runge-Kutta 4) and nodal spectral elements
- **CN-R LSPG:** Crank-Nicolson with two backward-Euler start steps, least-squares Petrov-Galerkin residual minimisation
- **finalists, max-min:** up to two fastest qualified configurations per family; S = max over bank finalists of min over competitor finalists
- **S test:** the same statistic on the held-out test cases, using only finalists that meet the target and qualify on test
