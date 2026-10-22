**Overall: NEEDS-RESTATEMENT.** Numerical checks support both reported stop-rule failures. The report overstates the explanation for the small speed gains.

1. **CORRECT — F2/F3 stop rules.**
   - F2’s eight candidates, in registered order: **fail, fail, fail, fail, pass, fail, pass, pass** for Q2.
   - All four candidates below Gauss96 have demonstrated Q2 failures; Gauss96 passes Q1/Q2. Thus **m* = 9,216 is identified**, despite unevaluated Q1 on larger Gauss112.
   - Gauss96 fails both `no_worse` and `faster`: **F2 fired (tested failure)**.
   - All six F3a/F3b × κ=4/8/16 pass Q1 but fail Q2. Their maximum rollout distances are respectively **0.0193054, 0.00626067, 0.000541709; 0.0400071, 0.0186106, 0.00249500**, all above 0.00025. **F3 fired (tested failure)**.
   - R qualifies; production populations match. Both projected and binding readings give these outcomes under the amended precedence.

2. **CORRECT — Q2 arrays versus result.json.**
   - Recomputed every saved Q2 successor/check rollout distance; no disagreements at the registered tolerance.
   - Independently recomputed 36 tested-vector ρ values: six states × two cases × R/F2_gauss96/F3b_k8. Maximum absolute discrepancy: **4.71×10⁻¹⁴**.
   - Example, F3b_k8 on val32|5: distance **0.01861057869**, final-state ρ **0.1359244591**; both reproduce.

3. **CORRECT — conditional E0 errors and `no_worse`.**
   - Rescored all 38 cases for each requested arm; scores reproduce exactly.

   | Arm | Worst error | Median error |
   |---|---:|---:|
   | R | 1.850974% | 0.049478% |
   | F2_gauss96 | 2.131174% | 0.284504% |
   | F3b_k8 | 1.850661% | 0.049353% |

   - In fractional-error units, **2ε* = 0.00001203033537** for both comparisons.
   - F2 difference **+0.002802001814** → false; F3b_k8 difference **−0.000003127607** → true. The worst-case reference uncertainties are selected correctly.

4. **CORRECT — timing and controls.**
   - Recomputed paired ratios and 2,000-resample, seed-0 case-bootstrap CIs from repetitions; exact agreement.

   | Arm | End-to-end median [95% CI] | Evolution median [95% CI] |
   |---|---|---|
   | F2_gauss96 | 1.022153 [0.992452, 1.068039] | 1.021648 [0.992153, 1.068354] |
   | F3b_k8 | 1.178596 [1.178175, 1.179007] | 1.180123 [1.179573, 1.180462] |
   | F2F3_k16 | 1.069998 [1.034947, 1.117952] | 1.070825 [1.035500, 1.119353] |

   - Zero registered timing outliers in these blocks; recorded output-tie checks pass.
   - C6: **0.999799908**, inside [0.9, 1.1]. C5′: **2.138603559 ≥ 1.5**, reproduced from raw Jacobian timing arrays.

5. **NEEDS-RESTATEMENT — report and EXPORT.**
   - Both reproduce byte-for-byte through in-memory generation; checked numerical entries agree with their source records and rounding.
   - Internal-ablation scope, conditional E0 estimates, and exploratory, selection-unadjusted CIs are appropriately labelled.
   - [Report line 13](experiments/imp-hrcost/report.md:13): “close to the GPU latency floor” is **not established** by these measurements. State that point-count reductions yield modest query gains; latency is a possible explanation requiring profiling.
   - The glossary’s “every step converged” misdefines Q1: the registered criterion permits **≤1% non-converged steps**, although these production arms actually have zero.

Read-only throughout: no repository/log changes, jobs, SSH, or scratch files.