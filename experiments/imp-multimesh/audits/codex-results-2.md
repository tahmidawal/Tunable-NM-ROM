**NEEDS-RESTATEMENT overall.** The primary **FIRES** outcome survives independent checks. Both **NOT FIRED** sensitivities reproduce. Remaining defects concern reporting.

Previous-round fixes:

| Issue | Round-2 verdict |
|---|---|
| Coefficient-based E0 check | **FIXED** |
| Order-dependent shared resampling costs | **FIXED** |
| Reporting omissions | **NOT FIXED** — partial additions, with errors below |
| Break-even grid labelling | **FIXED** |

1. **E0 errors — CORRECT.** Retrieved two archived lattice bases into permitted scratch and verified their SHA256 hashes against J1/J4 manifests. Independently reconstructed `to_lattice(W @ Phi_lat.T)` for **456 errors**, covering every dev6/val32 case and six configurations of PM-256 at 256² and CPU-256 at 4096². Maximum absolute discrepancies were **6.94×10⁻¹⁷** and **8.67×10⁻¹⁷**, respectively.

2. **Qualification and J5 replication — CORRECT at the retained-evidence level.** All **18 QUALIFIED** records satisfy their Q1/Q2 evidence; J5’s raw result rows also satisfy the checked validity, stationarity-certification, tight-distance and successor predicates. All **49 MEASURED-Q1 / 5 MEASURED-FAIL** competitor labels agree with their recorded predicates. J5’s **596 shared-case errors** match p1 within **3.72×10⁻¹⁶**.

3. **Costs, aliases and witnesses — CORRECT.** Independent enumeration reproduces the cheapest admissible costs. Shared `needs` are charged once; source-mesh aliases and CPU-1024-at-256 aliasing agree. Shared resampling now uses the maximum measurement independently of arm order.

   | Workload / ours | CP cost | Our cost | CP worst / our worst |
   |---|---:|---:|---:|
   | 256, q=10 / R128-BE | 385.338 s | 3,887.616 s | 3.261% / 4.847% |
   | 4096, q=100 / R256-CNR | 428.590 s | 3,886.398 s | 1.337% / 2.52% |
   | All meshes, q=1000 / R384-CNR | 530.229 s | 4,164.017 s | 1.516% / 1.813% |

   For example, the first competitor cost is independently reconstructed as **383.298086 shared construction + 0.167943 arm setup + 1.707964 compile + 10 × 0.016389 query seconds**.

4. **Bank-fit probe — CORRECT.** J2 reproduces **3,146.9923 s** projected bank fit, **660.681 s** head dependency and **3,863.6733 s** total offline cost. A100 validation gives **10,698.9735 s** versus **10,684.65 s** recorded: **0.1341%** discrepancy, below 5%.

5. **Timing validity — CORRECT; compilation table — WRONG.** Accepted panels pass observation-count, identity, basis-identity and drift checks. Sentinel ratios span **0.99478–1.01085**; A2/A1 spans **0.99259–1.01993**. J6’s incomplete 4096² panel is excluded; J6b supplies it. All twelve fresh compile probes have backend events without suspected cache hits.

   However, [make_report.py:253](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-multimesh/experiments/imp-multimesh/make_report.py:253) selects the first qualified arm rather than the fresh probe’s arm. Consequently, PM-1024’s in-process comparisons incorrectly use CPR-256:
   
   - r128/k4: printed **0.826375 s**, correct **0.891221 s**.
   - r384/k8: printed **0.790358 s**, correct **0.762845 s**.

   The decision model’s compile assignments are unaffected.

6. **Controls and gates — CORRECT in retained evidence.** S1c’s unmutated fixture passes; budget-zero fails with **300/300 nonstationary steps**. Mutation controls and construction certificates support acceptance. Failed competitor configurations are excluded. No additional numerical gate failure was identified.

7. **Stop-rule readings — CORRECT; reporting — NEEDS-RESTATEMENT.** Independent exhaustive enumeration reproduces all fifteen verdict tables and both adverse compile assignments:
   
   - Primary: **126 MATCH**.
   - Casewise: **84 MATCH / 42 NO-MATCH**.
   - Rollout: **126 NO-MATCH**.
   
   The sensitivity NO-MATCH cells lack accuracy-admissible competitors; they do not establish a cost advantage.

No repository files or lab logs were changed; no jobs were submitted. Scratch copies were deleted.

**Required fixes:**

- Complete A5.2’s A_sel union-minus-deployed accounting, or explicitly disclose the deviation. The current paragraph includes only target configuration runs/setup, omitting non-deployed construction, J5 qualification, historical non-producing head work and the registered stand-in.
- Correct the two arm-mismatched compile comparisons.
- Report query-time medians alongside means and outlier counts.
- Relabel “timed configurations” as timing blocks, or deduplicate A1/A2: the distinct configuration counts are **16/24/27**, rather than **22/30/33**.