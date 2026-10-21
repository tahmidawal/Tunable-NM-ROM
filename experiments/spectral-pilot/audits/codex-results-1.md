**The main numerical results reproduce, but the report needs substantive corrections to its interpretation and reporting.** I used independent, read-only calculations from the raw JSONs and reference arrays; no files were changed.

All measured values below are **PILOT**. Target-ordered lists use **PILOT τ = (0.03, 0.01, 0.003, 0.001)**.

1. **CORRECT — Minimum feasible sizes and speed ratios.** Independently applying `worst + U ≤ τ` reproduces:

   | PILOT level | PILOT FOM minimum size | PILOT ROM minimum rank | PILOT S |
   |---|---|---|---|
   | A1 | N: 12, 12, 16, 16 | 96, 128, 192, 192 | 0.218065, 0.128618, 0.108046, 0.110044 |
   | A2 | N: 16, 24, 32, 32 | 192, 256, 384, 384 | 0.101955, 0.079876, 0.081057, 0.080914 |
   | A3 | N: 48, 48, 64, 96 | 384, censored, censored, censored | 0.152214, censored, censored, censored |
   | AB | N: 16, 24, 24, 32 | 16, 32, 48, 64 | 2.947386, 1.768718, 1.668892, 0.624517 |
   | B | L: 64, 64, 128, 192, for both FD families | 16, 32, 48, 64 | 29.322450, 17.012905, 64.181669, 114.849556 |

   The reported sine-ROM minimum ranks and POD floor ranks also reproduce. No numerical correction needed.

2. **CORRECT — Matched configurations and target qualification.** The finalists match the fastest fully timed screen candidates. Recomputing medians from the stored repetition arrays reproduces their times and ratios.

   | PILOT level | PILOT matched FOM configurations | PILOT matched ROM configurations |
   |---|---|---|
   | A1 | (N, steps): (12,5), (12,5), (16,10), (16,10) | (form,r,steps): (Q,96,5), (Q,128,10), (Q,192,10), (Q,192,10) |
   | A2 | (16,5), (24,10), (32,20), (32,20) | (Q,192,10), (Q,256,10), (Q,384,20), (Q,384,20) |
   | A3 | (48,20), then censored pairs | (Q,384,20), then censored |
   | AB | (16,5), (24,10), (24,10), (32,20) | (T,24,5), (T,32,10), (T,48,10), (T,64,20) |
   | B | (family,L,steps): (RK4,64,160), (RK4,64,160), (IMEX,128,80), (IMEX,192,160) | (r,steps,L_init): (24,5,32), (32,10,64), (48,10,64), (64,20,64) |

   Every finalist satisfies the allowance-adjusted target, including the ROM’s rescored initializer. **Fix:** call these “screen-selected fastest among evaluated candidates,” disclose pruning, and include Part-A `Mq` and `Ninit`; the current table omits parameters needed to identify the timed configurations.

3. **CORRECT — Hypothesis arithmetic; WRONG — scope of the crossover diagnostic.** At **PILOT τ = 0.03**, independent results are:

   - PILOT H-A1 exponent: **0.630930**, consistent.
   - PILOT H-A2 ratios: **0.793388, 0.853333, 0.173834**, consistent under the registered endpoint rule—not a monotone decrease.
   - PILOT tensor-only H-A2′ ratios: **1.295287, 2.471919, 3.075073**, inconsistent.
   - PILOT log-speed slope: **−0.163619**.
   - PILOT tensor-only crossover ranks, target-ordered: A1 **74.115, 74.115, 75.868, 75.868**; A2 **77.672, 94.540, 103.147, 103.147**; A3 **124.875, 124.875, 146.430, 189.900**.

   These reproduce the report, but `analyze.py` uses only tensor costs. The design’s cheaper-form definition is not explicitly withdrawn, and every matched Part-A ROM uses Q.

   Taking the cheaper measured evaluation form gives **PILOT crossover ranks 105.209, 110.647, 366.902** and **PILOT ratios 0.912469, 1.735249, 1.046601** at the common target. The endpoint verdict remains inconsistent, but the claimed steadily worsening trend disappears. These unrestricted Q costs do not establish target qualification.

   **Fix:** explicitly label the existing diagnostic tensor-only, resolve the design discrepancy, and avoid using it to explain all ROM whole-query costs.

4. **CORRECT — Censoring.** The tighter A3 targets are genuinely unreached through **PILOT r = 512**. Their ratios are absent, and the corresponding trend verdicts correctly say “too few levels.” **Fix:** distinguish “ROM censored” from “not timed” in individual cells instead of sharing one dash symbol.

5. **CORRECT — Required controls and allowance arithmetic; WRONG — completeness of the disclosure.** Stored SW1a/SW1b fixtures pass, required G4 and available SW4 controls pass, G7 passes, and independently reconstructed cohorts satisfy disjointness. I found no D2 violations in **PILOT 1,255 checked rows**; spectral floors also agree with the reference arrays.

   Recomputed allowances are **PILOT A1 2.742634e−9, A2 6.506401e−9, A3 1.534763e−8, AB 3.193566e−9, B 3.243812e−5**. Part-B observed orders are **PILOT 2.004580 and 2.001152**.

   SW3 was superseded by diagnostic H1; it should not be treated as a missing mandatory failure test. G9 is likewise diagnostic.

   **Fix:** report the allowance coverage explicitly: Part-A padding checks cover only the steepest cases specified by the design; Part-B time evidence extrapolates from a partial fine-grid check. Include G6/D2 evidence and distinguish required controls, diagnostics, and unimplemented checks. “All pass” does not mean every original diagnostic ran.

6. **CORRECT — Sentinel timing validity; WRONG — implying complete protocol compliance.** Screen/final timestamp links match, and every sentinel drift passes. Maximum screen drift is **PILOT 6.097%**; maximum final drift is **PILOT 0.862%**. Records establish CPU, f64, single-core affinity and the thread checks.

   However, timing metadata records `precision: null`, and the launcher does not set the required `JAX_DEFAULT_MATMUL_PRECISION=highest`. Native timings are batched rather than interleaved, as already acknowledged in the design. The report omits timing outlier counts: examples include **PILOT two ROM outliers** at each of B’s loosest targets and **PILOT one** for AB at the next target.

   **Fix:** distinguish sentinel-valid from fully protocol-compliant; disclose the unset precision setting and native-timing limitation, and publish spreads/outliers. This does not by itself demonstrate erroneous CPU results.

7. **WRONG — “which floors every ROM error.”** The Part-B FD-operator/reference discrepancy of **PILOT 9.274321e−5** is reproduced, but it is not a proven lower bound on ROM error. Projection, temporal and spatial errors can cancel.

   **Fix:** replace with “a measured discretization discrepancy that may limit accuracy.” Reserve “floor” for the actual sampled-span projection floor.

8. **CORRECT — Break-even arithmetic; OVERSTATED if read as deployment accounting.** The selected offline components total **PILOT 1,018.082 seconds**. The tight-target saving is **PILOT 0.442031 seconds/query**, giving **PILOT 2,303.194 queries**.

   **Fix:** retain the indicative qualification and call this a component-based estimate. It mixes parallel/shared-machine offline wall time with single-core queries and does not establish an all-inclusive deployment break-even.

9. **OVERSTATED — “family width drives the outcome.”** AB and B have identical test descriptors, confirmed independently. But A versus AB changes centre location and spread, bump-width distribution, viscosity distribution and sampled cases. Parameter-range width is also not synonymous with Kolmogorov n-width.

   **Fix:** “The narrow-family AB result motivates a controlled family-width experiment; this pilot does not establish that width dominates viscosity or front sharpness.” Hold the other distributions fixed in that experiment.

10. **CORRECT — B/AB arithmetic; OVERSTATED — causal decomposition.** The ratios reproduce as **PILOT 9.948630, 9.618778, 38.457663, 183.901461**. They are ratios of speed ratios, not an isolated “remaining factor” caused by replacing spectral with FD. Geometry, reference/scoring norms, bases, initialization and output costs also change.

    **Fix:** call them descriptive cross-panel ratios. Delete “the advantage has two parts” as a quantitative attribution. Also replace the prediction that an untested high-order solver “would sit between these” with “its performance is unknown.”

11. **OVERSTATED — “no crossover,” impossibility for linear spans, exponential convergence and data sufficiency.** “No observed crossover on the evaluated Part-A configurations” is supported. “Not with a linear span” reinstates the transfer/impossibility claim withdrawn by the design. The speed sequence is nonmonotone, and only one target supports the registered trend fit.

    Likewise, the pilot does not demonstrate asymptotic exponential convergence, prove that nonlinear representations are necessary, or establish that more training data improves rollout speed. The training comparison supports sensitivity of projection floors only.

    **Fix:** scope conclusions to this POD implementation, evaluated ladders, in-sample cohort and CPU contract. Treat other representations and additional training as hypotheses.

12. **WRONG — A4 omission rationale and proposal consistency.** The report invokes an earlier generation-time rule that amendment A2.6 explicitly replaced. The measured generation duration is correct, but the stated governing rule is not.

    The proposal also mixes different steepness-grid sizes, budgets fewer training sizes than it lists, omits family-width multiplicity from retraining costs, and calls the sum of its component budgets a total “with contingency” without allocating contingency. Its “Part-A spread” description uses a bump-width range different from the actual amended pilot family.

    **Fix:** document the actual omission decision; reconcile a single experiment matrix and budget; label costs unvalidated planning estimates. Define arm-specific cost diagnostics instead of applying tensor crossover ranks to nonlinear or transport-aware arms.

13. **WRONG — Strict per-number PILOT labeling.** The opening blanket statement and plot titles identify the pilot, but several numerical tables and interpretation paragraphs lack local PILOT labels.

    **Fix:** put “PILOT” in every numerical table caption and numerical finding paragraph; keep proposed budgets explicitly marked “planning estimates, not measured results.” Make corrections in `make_report.py` and `report_text.py`, then regenerate.

OVERALL: NEEDS-FIX