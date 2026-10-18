**Overall: READY for the A1.11 selection freeze.** No numerical blocker found. The report needs the nonblocking wording corrections below; replication remains pending.

Audited `cf120dfca`. Recomputed all 55 timing panels, checked 6,688 solver rows, decoded 359 case-trajectories against E0, and independently reproduced all 21 tight/tighter ADT maxima. Sampled score discrepancies were ≤ \(2.8\times10^{-17}\). The analysis and generated report/EXPORT reproduce exactly.

| Item | Verdict | Evidence |
|---|---|---|
| 1. Job integrity | **CORRECT** | Jobs 5352455/5352457: `b4b2cfe4b`, H200, GPU backend, f64/highest, cohort 923801 cases 0–63, complete. Output checksums and source provenance match; stderr empty; no log anomalies. |
| 2. H2 confirmed gains | **CORRECT** | Recomputed \(S\): **2.00985, 5.36717, 17.20196** against C_old/C_W768/C_W1024. All pass. |
| 2. B*/scheme decomposition | **NEEDS-RESTATEMENT** | Direct-panel factors reproduce: **1.7463 / 1.1582**, **2.1345 / 2.4950**, **1.9103 / 8.9636**. Header implies an exact quotient that separate-panel medians do not satisfy. |
| 2. H1.4 | **CORRECT** | **2 pass, 20 fail, 3 rescue, 5 unavailable.** Pass ratios: W1024 BDF2 0.025: **0.597540**; W768 BDF2 0.0125: **0.538518**. |
| 2. “BDF2/CN beats W1024 BE” | **WRONG as a combined claim** | Selected BDF2 passes both accuracy comparisons; selected CN fails W1024’s worst-error comparison. The report’s comparator-specific assignments are correct. |
| 3. H3.1/G3 | **CORRECT** | Step-loop shares **95.37–98.81%**; component-sum/fused **0.9820–1.0028**, all within 15%. Indicative profiling caveat present. |
| 3. H3.2/K5 | **CORRECT** | All four qualify; certified nonstationarity ≤ **0.469%**, ADT distances ≤ **2.26944e-4**. Full-rank gl12 controls fail for all four. |
| 3. H3.3 | **CORRECT** | All G4 distances ≤ **1.24e-14**; K6 fires and K6+ passes. B16/B1 throughput ratios **0.9711, 3.9353, 1.2878**: only W_C_W768 reaches 2×. |
| 3. A1.13/reduced ADT | **CORRECT** | \(\delta_{\rm quad}\) maxima **1.756e-4–9.554e-4**; reduced-rule ADT maxima **9.90e-9–5.37e-8**, below 2.5e-5. Surrogate qualification is correctly distinguished from quadrature equivalence. |
| 4. G5/K7; mesh eligibility | **CORRECT** | Archived G5 errors **1.06–1.75e-15**; K7 differences ≈ **0.02353**. All 16 mesh-ROM cells eligible; errors reproduce; all mesh ADT checks pass. |
| 4. H4.1/H4.2 | **CORRECT** | Flatness ratios **1.0069, 1.0683, 1.0178, 1.0906**, all pass. Same-mesh ratios reproduce; all eight FOM diagnostics calibrated/converged/scored, explicitly eight-case and unequal-accuracy. |
| 4. H4.3/H-comp/K2 | **CORRECT** | Primary CC ratios **0.09479–1.97067**, PS **0.007465–0.134545**. At L384, W_C_W768/CC = **2.02663**, but every union ratio is below 2; **K2 fires**. |
| 5. Spectral observation | **CORRECT** | Paired panels confirm spectral superiority against the selected L64 ROMs. Comparator is **PS N16**, cross-resolution—not a same-resolution spectral 64³ comparison. |
| 6. Bounds, scope and publication | **NEEDS-RESTATEMENT** | Upper-bound direction, mesh-free/scoring-lattice scope, mesh-specific matching and screening/confirmation distinctions are correct. Preparation-cost and timing-population caveats are incomplete. |

Details for the non-CORRECT items:

- **W1024 comparison:** BDF2’s median/worst errors are **0.006418/0.044079**, versus W1024 BE’s **0.015299/0.045866**. CN’s worst is **0.046460**; even after uncertainty adjustment, **0.046376 > 0.045950**. Say “BDF2 beats W1024’s deployed BE; CN wins against C_W768.”

- **Decomposition:** Rename the final column “direct paired B*/W* speed-up.” State that each factor has its own confirmation panel, their product need not equal total \(S\), and neither factor isolates one methodological change. This is required by DESIGN A2.7(b).

- **Prepared-input costs:** The report omits the explicit prepared-input latency qualification and accompanying preparation costs required by A1.9/A2.7(e). This matters: m1 records median PS-N16 preparation of **12.0 ms**, outside its roughly **4 ms** timed solve. Report preparation separately; these measurements do not establish end-to-end speed ratios by simply adding medians.

- **Other reporting caveats:** The glossary’s universal “confirmed = 8 cases × 3 repetitions” needs route 4’s **4 × 2** exception. Also disclose that central-CN shares E0’s discretization family, so its error partly measures the reference’s own level difference.

**K2 interpretation is sound:** the governing criterion concerns the *best validated FOM*, hence the union. The central-CN-only 2.02663× cell remains a valid family-specific pass, while its spectral/union ratio is **0.15824×**. Scope this conclusion to this lane’s selected 3D configurations.

Audit limits: G5/K7 and batching-control certificates were checked from archived measurements; their underlying table/control arrays and raw timing hashes are not retained. Individual drift excursions numbered **12/1,024** in p1 and **2/672** in m1; every subject-median drift passed the registered rule.

[Raw audit evidence](/tmp/speed3d-audit-scratch/results_audit.log) · [E0 decode checks](/tmp/speed3d-audit-scratch/decode_audit.log) · [ADT checks](/tmp/speed3d-audit-scratch/adt_audit.log)

No repository, main or LAB-LOG changes; no jobs submitted. CPU only, arrays read in place, scratch approximately **48 KB**.