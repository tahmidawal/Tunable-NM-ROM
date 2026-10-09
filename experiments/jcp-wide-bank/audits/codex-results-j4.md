**The numerical tables and H4 verdict are CORRECT. The headline “widening … does not lower rollout error” is WRONG as written: all 64 cases improve from \(R'=512\) to 1024 at both meshes.** The typical improvement is small, but it is systematic.

No files were modified. I independently recomputed statistics, selection and timing from `result.json`; inspected the four PNGs and their plotting code; and reviewed the saved NumPy audit. I did not rerun its full field-to-reference reconstruction.

1. **Tables and registered H4 outcome — CORRECT.**

   Every J4 table entry matches the raw records at the printed precision: deployed/converged errors, floors, both \(m^\star\) thresholds, query/Jacobian medians and storage. Errors below are percentages; each pair is worst/median.

   | Mesh | Bank | \(R'\) | Deployed error | Gauss-56 diagnostic error | Floor |
   |---|---|---:|---:|---:|---:|
   | 64³ | W1024 | 256 | 4.71144 / 1.24968 | 4.70673 / 1.24965 | 4.01086 |
   | 64³ | W1024 | 512 | 3.69865 / 1.10903 | 3.70982 / 1.10903 | 2.26315 |
   | 64³ | W1024 | 768 | unavailable | 4.36845 / 1.10224 | 1.62590 |
   | 64³ | W1024 | 1024 | unavailable | 3.57561 / 1.10153 | 1.19240 |
   | 64³ | M2 | 256 | 4.70252 / 1.25118 | 4.70263 / 1.25130 | 4.00958 |
   | 64³ | M2 | 512 | 3.16502 / 1.11749 | 3.16560 / 1.11749 | 2.25209 |
   | 128³ | W1024 | 256 | 4.70016 / 1.24943 | 4.69560 / 1.24944 | 4.01086 |
   | 128³ | W1024 | 512 | 3.64640 / 1.11148 | 3.65703 / 1.11148 | 2.26315 |
   | 128³ | W1024 | 768 | unavailable | 4.20114 / 1.10037 | 1.62590 |
   | 128³ | W1024 | 1024 | unavailable | 3.44420 / 1.09963 | 1.19240 |
   | 128³ | M2 | 256 | 4.69153 / 1.25025 | 4.69162 / 1.25017 | 4.00958 |
   | 128³ | M2 | 512 | 3.12140 / 1.12423 | 3.12147 / 1.12421 | 2.25209 |

   Independently reselecting all **48 family/threshold entries** reproduces the recorded selections. At both meshes:

   | Bank, \(R'\) | Primary \(m^\star\), lattice/Gauss | Secondary, lattice/Gauss |
   |---|---:|---:|
   | W1024, 256 | 8192 / 13824 | 4096 / 8000 |
   | W1024, 512 | 16384 / 32768 | 16384 / 13824 |
   | W1024, 768 or 1024 | unavailable / unavailable | unavailable / unavailable |
   | M2, 256 | 16384 / 13824 | 4096 / 8000 |
   | M2, 512 | 32768 / 32768 | 16384 / 13824 |

   Tensor worst errors also match: at 64³, W1024 256/512 gives **10.43707/10.21322%**, M2 gives **10.43811/10.21857%**; at 128³, **6.68680/6.10933%** and **6.68881/6.10512%**, respectively. No tensor rollout was measured above 512.

   Query and Jacobian medians recomputed from repetition records:

   | Mesh | Bank, \(R'\) | Rule | Query ms | Jacobian ms | Off-mesh GB |
   |---|---|---|---:|---:|---:|
   | 64³ | W1024, 256 | lat8192 | 14.4097 | 0.236982 | 0.100860 |
   | 64³ | W1024, 512 | lat16384 | 43.1652 | 0.763052 | 0.403177 |
   | 64³ | M2, 256 | gl24 | 18.1107 | 0.334124 | 0.170201 |
   | 64³ | M2, 512 | lat32768 | 70.7344 | 1.430302 | 0.806355 |
   | 128³ | W1024, 256 | lat8192 | 14.6956 | 0.189074 | 0.100663 |
   | 128³ | W1024, 512 | lat16384 | 49.9108 | 0.835387 | 0.402784 |
   | 128³ | M2, 256 | gl24 | 17.5267 | 0.257547 | 0.169869 |
   | 128³ | M2, 512 | gl32 | 79.2755 | 1.579535 | 0.805569 |

   These are **per-setting panel** times, as the cost plot correctly states.

   K-conv’s failures are real, against the unchanged \(2.5\times10^{-5}\) bar:

   | Mesh | \(R'=768\) | \(R'=1024\) |
   |---|---:|---:|
   | 64³ | \(3.05615379\times10^{-5}\) | \(1.71648694\times10^{-4}\) |
   | 128³ | \(2.85582314\times10^{-5}\) | \(1.56606720\times10^{-4}\) |

   **A2-7 therefore requires H4 = unavailable at both meshes**, before testing improvement ratios. A8 permits diagnostic Gauss-56 errors; it does not waive K-conv or authorize a substitute verdict. Even ignoring the gate, those diagnostic worst-error ratios are **0.96382 and 0.94180**, short of H4’s 0.8 bar—but that is not the registered result.

2. **Diagnostic claims — mixed.**

   - **CORRECT, with “nearly”: median nearly flat from 512–1024.** At 128³ it decreases **1.11148 → 1.10037 → 1.09963%**. “Flat” should describe the small magnitude, not imply zero improvement.
   - **CORRECT: worst error non-monotone.** At 128³: **3.65703 → 4.20114 → 3.44420%**.
   - **NEEDS-RESTATEMENT: “floor halves.”** It falls **2.26315 → 1.19240%**, a **47.31% reduction**. “Nearly halves” is accurate.
   - **NEEDS-RESTATEMENT: “mesh dependence grows with width.”** Recorded worst distances are **0.00065455, 0.00184295, 0.00352158, 0.00335942**. Growth is substantial overall, but **not monotone**: 1024 is slightly below 768. These comparisons also change \(M\), and mesh-specific test counts differ at the narrower settings.
   - **NEEDS-RESTATEMENT: “512-column prefix is no better than M2.”** Its **worst error is worse**, but its **median is slightly better**, and its selected rule uses fewer points and less time. Specify the metric; there is no overall dominance.
   - **CORRECT: gate-ignored 1024 rule.** `lat32768` requires exactly **1,610,874,880 bytes**; query/Jacobian medians are **261.075/5.81115 ms** at 64³ and **268.871/5.78534 ms** at 128³. Tensor storage is computed as **34,368,126,976 bytes**.
   - **NEEDS-RESTATEMENT: “storage is no longer the binding constraint.”** The **advection blocks** are small. This is not total execution memory: the largest sampled device footprint is **75.76 GB**, with possible missed spikes between 0.2-second samples. Scope the conclusion to advection storage and this H200 execution.

3. **Challenge to the accuracy narrative — WRONG headline; hypothesis correctly labelled.**

   The claim appears in both [“Answers in brief”](experiments/jcp-wide-bank/report.md:10) and the [detailed conclusion](experiments/jcp-wide-bank/report.md:200).

   Paired Gauss-56 case maxima, with change defined as **1024 minus 512**:

   | Statistic | 64³ | 128³ |
   |---|---:|---:|
   | Cases improved | **64/64** | **64/64** |
   | Cases worsened | 0 | 0 |
   | Median paired change | −0.004366 pp | −0.004347 pp |
   | Mean paired change | −0.035832 pp | −0.036343 pp |
   | Cases improving by >0.05 pp | 12/64 | 12/64 |
   | Cohort worst-error change | −0.134205 pp | −0.212828 pp |
   | Relative worst-error reduction | 3.62% | 5.82% |

   **The improvement’s direction is not driven by a few cases; its magnitude is concentrated in a minority.** Removing case 5, the worst case at both widths, still gives worst errors **2.62913 → 2.32361%** at 64³ and **2.62433 → 2.32761%** at 128³. Removing the five cases with largest errors also preserves improvement in every remaining case.

   Thus “does not lower rollout error” is contradicted by the recorded diagnostics. “Produces small typical improvements, much smaller than the floor reduction” is supported. Because K-conv fails, these remain diagnostics rather than certified deployment gains.

   **CORRECT:** the backward-Euler explanation is explicitly labelled *“Hypothesis, not isolated here.”* Neither this panel nor the 2D comparison demonstrates the cause in 3D. Also describe H4 as a joint **\((R',M)\)** comparison, not an isolated rank experiment.

4. **Gates, controls and K-time — CORRECT execution; NEEDS-RESTATEMENT of scope.**

   - GPU/f64/highest precision appear in the log; all settings record `adaptive_first=6`.
   - The wide-rank K-conv failures are **quadrature-distance failures**, not A7 eligibility failures. At 1024, conv/check each have **2/1600 non-stationary validation steps**, zero reason-3 exits, and zero non-stationary certification steps.
   - All target checks pass: **\(2.72\times10^{-8}\) to \(7.88\times10^{-7}\)** against \(10^{-5}\).
   - Both controls fail on numerical criteria, independently of eligibility: across all settings, their minimum worst distance is **0.3383** and minimum \(\rho\) is **2.3297**, far above either threshold.
   - All **14 timing panels / 172 subject-panel combinations** pass recomputed drift checks: **0.96603–1.04327**; timed output differences are **zero**. Each subject has 32 records per A phase.
   - The saved NumPy audit reports **12,032 error-record comparisons**, worst relative discrepancy **\(5.65\times10^{-14}\)**, plus **5,632 distance comparisons** and successful fault injections. Its distance reconstruction covers **65 nodes only**, as disclosed.
   - **Missing required disclosure:** A3-13 says the report must explain that **G2/G4 use only 32 columns** and do not certify rank-dependent behavior above 512. That limitation is absent.
   - A8’s J2 comparability warning is present. Explicitly extend it to the historical three-adaptive-step lane if making that comparison.

5. **Plots — CORRECT values; some presentation needs restatement.**

   - **Error:** matches the **converged-rule diagnostic** columns, not deployed errors; title and hollow markers disclose this correctly.
   - **Cost:** matches per-setting timing medians; wide-rank points are correctly labelled gate-ignored diagnostics.
   - **Memory:** correctly plots computed nominal \(M=4R'\) tensor storage and measured-setting off-mesh storage. Small differences from actual-\(M\) table values are expected and labelled.
   - **\(m^\star\):** plotted points are correct and unavailable wide settings are omitted. **NEEDS-RESTATEMENT:** distinguish W1024 from M2 and annotate the missing 768/1024 settings; otherwise multiple values at the same \(M\) have no bank identity.
   - Minor readability issue: the error plot’s legend overlaps the floor curve.

**Statements safe to report:**

> H4 is unavailable at both meshes because the registered quadrature-agreement gate fails at \(R'=1024\); it also fails at 768. Under A8’s diagnostic fallback, widening from 512 to 1024 improves every validation case, but the median paired improvement is only about 0.00435 percentage points. Worst error decreases by 3.62% and 5.82%, while the projection floor decreases by 47.31%. Error and mesh dependence are not monotone across the full width ladder. The 1024-column diagnostic lattice rule stores 1.61 GB of advection data versus a computed 34.37 GB tensor, at 261–269 ms per query. These are first-order-reference-relative diagnostics; time-error dominance remains a hypothesis.