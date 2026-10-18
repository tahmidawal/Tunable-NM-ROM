**Overall: RESULTS-CONFIRMED — part 1 only.** No incorrect numerical label found. CPU/NumPy recomputation only; no files changed, no scratch written, no GPU or cluster activity.

| Item | Verdict | Evidence |
|---|---|---|
| Q1/Q2 predicates and labels | **CORRECT** | Checked GAL, failing CNR and BE configurations at every mesh, all four fallbacks, ext22 and inherited P1 samples. All 96 lane blocks have zero non-certified production steps. Failures correctly arise from tight-distance or Q2 rollout bars—not certificate counts. |
| Every-case Q1 distance | **CORRECT** | At 1024², R128/R256 CNR, factor 2: worst tight distances **3.59942e-4 / 4.03904e-4**, both failures despite passing Q2. Tight/tighter checks pass. |
| Pooled ext22 qualification | **CORRECT** | Each of four extensions produces 38 distinct cases after pooling; **0/1900** non-certified steps. Independently checked extension distances and inherited P1 anchors. |
| Combined reproduction | **CORRECT** | Recomputed **21 gates × 38 cases** directly against `W_R*.npz`; worst relative discrepancy **1.78893e-15**. |
| E0 scoring and identity | **CORRECT** | Selected raw-field scores reproduce exactly; eligibility, ε_ref and reference hashes match. Case/time/Δt/reference-kind swaps reject. All 112 reported configurations contain 38 ST-eligible cases. |
| Fallbacks, §2.3 | **CORRECT** | Exactly four required: R128 CNR factor 2.5 at every mesh; R384 CNR factor 2.5 at 256². Originals remain failed. All fallbacks pass Q2; **only R384 qualifies**—R128 still fails Q1. |
| H-acc-N / H-acc-r | **CORRECT** | R512 BE factor 1: H-acc-N passes, populations **38/16**, both 100% within margin; TOST p **3.638e-12 / 1.526e-5**. At 4096², BE transitions have improve/worsen/tie counts **38/0/0**, **24/1/13**, **7/16/15**: monotone through 384, not through 512. |
| B4 and TH06 | **CORRECT** | Raw coefficients reproduce R384 CNR **p=1.957277**, TH06 **p=0.998432**; six interpretable cases each. Anchors `/64`, `/128`, solver, quadrature and finer-rule checks pass. Order-2 wording explicitly limits the claim to the fitted finite range. |
| 3D Q2 and offline agreement | **CORRECT** | Independently recomputed four configurations, including both winners and control. Winner successor distances **8.14028e-4 / 9.50795e-4** fail; finer distances **2.86627e-5 / 3.07470e-5** pass. Independent distances/rho agree to round-off. All **17** reproduction gates pass; worst **1.84944e-15**. |
| Open candidates, A2.1 | **CORRECT** | All 17 tested candidates fail Q2; **436−17=419** open candidates, exactly matching the recorded list. |
| Controls | **CORRECT** | Acceptance records **85/85 required checks passing**. Recomputed: budget-zero rejects **300/300** steps for each form; zero-update motion **0.510–0.515**; 2D under-resolution distance **0.0336638**, rho **1.58164**; 3D control distance **0.00430295**. |

**Ranked reporting corrections required when carrying earlier claims forward:**

1. **Combined qualification scope:** its `report.md:267,351,355` “qualified” R128/R256 fast arms **fail this lane’s Q1**. State the changed qualification result explicitly; retain historical measurements and identify the different criteria.
2. **Speed3d qualification scope:** both H2 winners fail Q2 here. Historical speed ratios are not retracted; describing them as Q1+Q2-qualified would be wrong. Speed3d already calls reduced rules “surrogates, not quadrature-equivalent” (`report.md:155`).
3. **P1 population/scope:** no numerical retraction found. Preserve its 16-case 4096² population distinction; neither its mesh-equivalence verdict nor B4’s finite-range order supports a universal claim.

**Computational must-fixes: none found.** Timing panels and fronts remain unaudited.