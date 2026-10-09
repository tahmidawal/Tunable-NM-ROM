**REPORT-OK is NO.** The sampled numerical results check out, but several claims and plot labels need correction before presentation. No files were modified.

References below use line numbers in [report.md](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-wide-bank/experiments/jcp-wide-bank/report.md) and [make_report.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-wide-bank/experiments/jcp-wide-bank/make_report.py).

1. **Number provenance — FAIL overall; numerical spot-checks PASS.**

   I checked more than 40 numerical values across the summary, 2D tables, training, A6, and 3D sections. All sampled measured values match their JSON sources at printed precision. Examples:

   | JSON source and fields | Verified report values |
   |---|---|
   | `runs/j1/archive/output/result.json`: deployed-arm `rows[].ref_ST_evolved`, worst/median for ranks 128/256/384/512 | 4.64/1.04, 3.06/0.91, 2.75/0.91, 2.84/0.91 % |
   | Same, `ref_S_evolved` | 3.45/0.41, 1.49/0.09, 1.05/0.05, 1.60/0.06 % |
   | Same, `final_timing.subjects[].median_ms` | 28.4, 71.2, 200.3, 688.8 ms |
   | Same, paired ratios recomputed from final-panel invocations | Rank-384 trim ratio 0.75; unrounded 0.752900 |
   | Same, `gates.converged_R512_M2048.gref_check_worst_distance` | 3.4e-09 |
   | `runs/j3b/archive/output/training.json`: duration, checkpoint, validation error | 1.85 h, step 11000, 1.31 % |
   | Same, `floors_full.129` at ranks 512/768/1024 | 3.23, 1.93, 1.29 % |
   | Same, ordering inverse check and bank condition | 5.8e-12; 1.01e+04 |
   | `runs/s2c/archive/output/result.json`: three G1 diagnostics | 1.9e-06, 1.3e-07, 2.5e-13 |
   | `runs/j2/archive/output/result_A7.json`: rank-512, κ=4 | Converged 3.17/1.12 %; 104.3 ms |
   | `runs/j4/archive/output/result.json`: 129-node W1024 converged worst, ranks 256/512/768/1024 | 4.70, 3.66, 4.20, 3.44 % |
   | Same, rank-512 query and rank-1024 tensor storage | 49.9 ms; 34.37 GB |

   `checks/j1-audit.json`, `j2-audit.json`, and `j4-audit.json` all contain `accepted: true`, matching the report. Reading those flags is distinct from rerunning their underlying field audits.

   **Failure:** “Every number … from the run JSONs” at line 3 is literally false. The generator hard-codes failed-job IDs and exit code at lines 743–745, configuration/threshold numbers in prose, and the plots’ 141 GB hardware line at lines 666–667. The 2048-column memory figure is a labelled calculation, not a run measurement.

   **Fix:** distinguish measured results, registered constants, calculated extrapolations, and operational metadata. Source operational numbers from a machine-readable ledger. I found no hand-typed measured error or timing among the sampled results.

2. **Document structure, provisional labels, mathematics and glossary — FAIL.**

   **PASS:** level-1 title, opening status, mathematical LaTeX, and closing glossary exist.

   **Fixes required:**

   - Put **PROVISIONAL** in each accuracy table caption/header and in the tensor-comparison paragraphs at lines 156 and 190. Section-wide labels can disappear when tables are copied into slides.
   - Correct line 3: training span floors use native-grid validation fields, as line 111 explains; not every accuracy number is against the refined reference.
   - Define missing columns/terms: `cond A` and \(A\), query-time contents and aggregation, Jacobian timing, floor check, flux check, M2/W1024, POD, Smolyak, and decimal GB.
   - Explain **nodes versus cells**: the 3D tables say “mesh” but print 65/129, while surrounding text says 64³/128³.
   - Correct the glossary’s exact \(\kappa=M/R'\) definition for 3D: κ is nominal; completing eigenvalue shells changes actual \(M\).

3. **Honesty, design compliance and timing — FAIL overall, with substantial disclosures passing.**

   **PASS:** A6, A7’s post-hoc chronology, as-run versus amended J2 selection, missing confirmation timings, A8’s solver change, failed/cancelled attempts, RSS defect, and the contradicted 1d mechanism are plainly disclosed. GPU types are identified; I found no cross-job timing ratio. Conditioning and time-error explanations are labelled hypotheses.

   **Failures and concrete fixes:**

   - **Overstatement at lines 10 and 200:** “widening … does not lower the rollout error” contradicts the printed diagnostic decrease, including **3.66 → 3.44 %** at 128³. Replace with: **“H4 is unavailable. Diagnostic errors vary non-monotonically with width; gains beyond 512 columns are modest.”**
   - **Line 203:** “no better than M2” hides slightly better medians. Say **“worse worst-case error, slightly lower median error.”**
   - **Missing mandatory limitation:** DESIGN A3-13 explicitly requires reporting that **G2/G4 use only 32 columns and do not validate rank-dependent behaviour above 512**. Add it beside the J4 audit statement.
   - **Missing scope:** state prominently that this is the **linear-span coefficient solve, with no nonlinear-manifold head**, as required by A0-1.
   - **Incomplete registered reporting:** 3D \(m_d/m_\rho\) diagnostics required by A0-3 are absent. The registered peak-device-memory and LM-iteration/exit summaries are also absent. Add generated summaries or explicitly identify omissions.
   - Scope “storage is no longer the binding constraint” to the **advection blocks at tested settings**. The displayed storage is not total device memory.
   - Line 103 attributes cost growth to increasing quadrature size without isolating that cause. State that \(m\), \(R'\), and \(M\) increase together.

4. **All 12 PNGs — FAIL overall.**

   All were visually inspected; sampled plotted values agree with the tables.

   | PNG(s) | Verdict and fix |
   |---|---|
   | `2d_error_vs_Rp.png` | **FAIL:** logarithmic y-axis is unlabelled and has only one numbered tick, “1”. Label the scale and add useful ticks. |
   | `2d_cost_vs_Rp.png`, `2d_memory_vs_Rp.png` | **PASS:** readable, labelled, consistent with sampled data. |
   | `2d_mstar_vs_M.png` | **PASS with improvement:** label logarithmic axes and add ticks at actual rule sizes. |
   | All four `3d_j2_*.png` | **FAIL:** omit **A7 POST-HOC SELECTION**. The error plot’s “registered selection unavailable” marker explanation is particularly misleading when its filled rank-512 marker comes from amended selection. |
   | `3d_j4_error_vs_Rp.png` | **FAIL for presentation:** legend overlaps the projection-floor curve. Move it outside; identify hollow points explicitly as gate-failed diagnostics. |
   | `3d_j4_cost_vs_Rp.png` | **FAIL:** both diagnostic mesh series use identical hollow grey circles; distinguish their markers or colours. |
   | `3d_j4_mstar_vs_M.png` | **FAIL:** pools M2 and W1024 without identifying banks; overlapping points obscure distinct outcomes. Separate banks and annotate unavailable wide-rank settings. |
   | `3d_j4_memory_vs_Rp.png` | **FAIL for interpretation:** identify bank/mesh for the different off-mesh points and label the hardware line as total capacity, not available advection storage. |

   For J2’s memory plot, also identify the H200 capacity line as a hypothetical comparison: J2 ran on an H100.

5. **Coordinator readiness — FAIL.**

   The current report could lead the presenter to claim that:

   - widening produces no error reduction;
   - these results test a nonlinear-manifold head;
   - J2 deployments were selected prospectively;
   - 1.61 GB is total solver memory;
   - all wide-rank implementation checks ran at full rank.

   Correct those points above. The defensible headline is: **the tested 2D dial worsens beyond 384 columns; no 3D trim passes the accuracy bar; wider-bank projection floors improve, but the registered wide-rank rollout verdict remains unavailable because K-conv fails. Accuracy conclusions remain provisional.**

REPORT-OK: NO