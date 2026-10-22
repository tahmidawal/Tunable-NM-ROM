**NEEDS-RESTATEMENT overall.** The primary **FIRES** outcome survives the checks performed; both **NOT FIRED** sensitivities reproduce. There are evidence and reporting gaps, plus an order-dependent construction-cost assignment.

1. **E0 errors — NEEDS-RESTATEMENT.** No `Phi_lat_*.npy` files are present locally, so the requested independent `W @ Phi_lat.T` reconstruction could not be completed. Using the saved `F` arrays instead, I recomputed **324 dev6 errors** against E0: maximum discrepancy **3.55×10⁻¹⁴**. All 38 reference cases are eligible. This verifies scoring, but not coefficient-to-field reconstruction or val32 fields.

2. **Qualification and J5 replication — CORRECT at the retained-evidence level.** All **18 QUALIFIED** ours records satisfy the recorded Q1/Q2 predicates. Across competitors, **49 MEASURED-Q1 and 5 MEASURED-FAIL** labels agree with validity, pooled stationarity, tight-distance and anchor evidence. J5’s **596 shared-case errors** match p1 within **3.72×10⁻¹⁶**. Raw certification recomputation remains limited by the pulled artefacts.

3. **Costs, aliases and witnesses — NEEDS-RESTATEMENT.** Independent enumeration reproduces all primary witnesses. Shared `needs` are charged once, and the specified aliases agree. Example totals:

   | Workload / ours | CP cost | Our cost | CP worst / our worst |
   |---|---:|---:|---:|
   | 256, q=10 / R128-BE | 385.338 s | 3,887.616 s | 3.261% / 4.847% |
   | All meshes, q=1000 / R384-CNR | 523.255 s | 4,164.017 s | 1.516% / 1.813% |

   However, [analysis.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-multimesh/experiments/imp-multimesh/analysis.py) overwrites shared resampling costs with the later arm’s measurement. For example, `1024→4096` becomes **1.305 s**, replacing **7.675 s**. Thus the chosen construction cost depends on arm order. Using the largest measured resampling cost for each shared artefact leaves all three headline outcomes unchanged.

4. **Bank-fit probe — CORRECT.** J2 yields **3,146.9923 s** projected bank fit, **660.681 s** head dependency, and **3,863.6733 s** total offline cost. Independent A100-log parsing gives **10,698.9735 s** versus **10,684.65 s** recorded: **0.1341%** discrepancy, within 5%.

5. **Timing panels — CORRECT.** Accepted panels have the required observation counts and passing recorded identity checks. Sentinel ratios span **0.99478–1.01085**; A2/A1 ratios span **0.99259–1.01993**. J6’s invalid 4096² panel is excluded; J6b supplies that mesh. All twelve fresh compile probes exist without suspected cache hits.

6. **Controls and gates — CORRECT in retained evidence.** S1c records the required failures and passing unmutated fixture. Budget-zero produces **300/300 nonstationary steps**. Basis/rank certificates pass; failed competitor configurations are excluded. No overlooked numerical gate was identified.

7. **Stop-rule readings — CORRECT; reporting NEEDS-RESTATEMENT.** Independent exhaustive searches reproduce **126 primary MATCH**, **84 casewise MATCH / 42 NO-MATCH**, and **126 rollout NO-MATCH**. All fifteen generated verdict tables, including adverse compilation assignments, reproduce. The sensitivity failures do **not** establish a cost advantage.

   Reporting omissions remain: registered **A_sel**, fresh-versus-in-process compile comparisons, and timing medians/outlier counts. Break-even counts are also grid approximations: the reported **473,151** for 4096² R128-BE/CNR corresponds to calculated crossings near **465,634 / 467,668**, respectively.

**Required fixes:**

- Supply checksum-verified lattice bases to complete coefficient-based E0 checks.
- Make shared construction-cost selection explicit and independent of arm ordering.
- Complete the omitted registered reporting, or explicitly disclose deviations.
- Label break-even counts as grid estimates or calculate exact crossings.

No files were changed, no jobs submitted, and no scratch files created.