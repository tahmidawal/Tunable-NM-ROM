**Overall verdict: the numerical claims reproduce. The current report is stale, and several interpretations need qualification.** No numerical discrepancy overturned an acceptance decision or cross-check verdict.

I used CPU-only, in-memory calculations; I modified no files. I selected 24 random case-kinds per dimension with seed `20261014`, then expanded the numerical audit to **all 404 historical case-kinds**, all 16 cross-checks, all 300 available old-reference comparisons, and both calibration jobs. Independent cohort/IC reconstruction covered 63 distinct sampled/cross-check records.

Computed quantities agree with stored metadata within A6.4’s tolerance. All **844 store-manifest entries** and **863 locally available run-manifest entries** verified. Sealed ciphertext was not opened.

| Item | Assessment | Finding |
|---|---|---|
| Historical observed orders, differences, `eps_ref`, direction consistency, floor and outcome | **CORRECT** | All 404 recomputed; no mismatches. Per-step residuals, floor proxies and Newton totals also agree. |
| 3D historical acceptance | **CORRECT** | **200/200:** 100 ST and 100 S. |
| 2D historical outcomes | **CORRECT** | **191/204 accepted:** 101 ST and 90 S; 12 S unresolved-floor and one ST order-fail. |
| Refinement/tightening procedure | **CORRECT** | All 12 floor failures received tight attempts; the ST failure received spatial refinement. All remained rejected. |
| Cross-check case selection | **CORRECT** | Both eight-case lists match the smallest-viscosity/lowest-index rule. |
| A10 cross-check numbers and verdicts | **CORRECT** | **8/8 PASS in each dimension.** Recomputed spatial/time indicators, discrepancy, effectivity and PS–Richardson distances agree. |
| Original-rule verdicts | **CORRECT** | 3D **8/8 PASS**; 2D **1/8 PASS**, seven UNRESOLVED. This is the original rule **on the same ladders**, not a replay of its possible escalation procedure. |
| X1 first-order rejection | **CORRECT** | All six 2D and four 3D ladders fail the order criterion. Orders span **0.9174–0.9849** and **0.6725–0.9175**, respectively. Two 3D outcomes are labelled `peclet-fail` by precedence; their order checks also fail. |
| X1b | **NEEDS-RESTATEMENT** | The CPU test reproduces first-order rejection on both ladders and second-order acceptance. But it uses random synthetic fields, **not the stored dev6 case-0 fields specified by A2.8**. |
| Unit gates | **CORRECT** | Recorded unit gates pass. CPU re-execution reproduced order-edge, X1b and injected-Péclet tests. |
| X2 old-reference discrimination | **CORRECT** | Old reference fails on **all 16 cross-check cases**, with the required PS resolution condition satisfied. |
| X3 numerical counts | **CORRECT** | 2D ST **102/102**, 2D S **102/102**, 3D ST **96/96** exceed the stored estimate. |
| X3 interpretation | **NEEDS-RESTATEMENT** | The 2D totals include 13 rejected references whose estimates cannot support accuracy claims. Restricting to accepted references still gives **101/101 ST** and **90/90 S**. No old 3D dev4 or S reference is available. |
| X4 sensitivity probe | **CORRECT** | Doubling the middle step produces undefined order at all five times and `order-fail`. The probe detected the defect. |
| G1 | **CORRECT** | Recorded parity differences are zero in both dimensions, with matching Newton counts. |
| G2 | **CORRECT** | Recorded true orders and effectivities satisfy the amended bars. The bar applies to **true order**, not the separately reported observed order. |
| G3a–c | **CORRECT** | Recorded errors satisfy the amended A4 thresholds; all selected PS runs satisfy their individual checks. |
| G4 | **CORRECT** | Both spatial-pair orders recompute from recorded errors and pass; temporal orders and separation ratios pass. Independent direct sums reproduce the projection control. |
| G5 / A12 perturbation | **CORRECT** | Swap creates relative IC disagreement **0.9413**. The 1% middle-level perturbation makes the third order undefined and changes the outcome to `order-fail`, satisfying A12’s definedness-change predicate. |
| G6 | **CORRECT** | In-memory replay: pristine load succeeds; swapped data, perturbed data and stale status all raise. |
| G7 | **CORRECT** | All four sealed/OOD cohorts refuse load, status and score. Restricted metrics remain ineligible; mismatched/ineligible pairwise scores refuse. |
| Calibration A2.1/A3.2/A5.4 | **CORRECT** | All 24 calibration rows reproduce both measured changes and proxies; all 72 solves satisfy the acceptance rule. Refinement levels remain uncalibrated, as disclosed. |
| A9 tighter budget | **CORRECT** | Stored 2D tighter attempts use budget 8; 3D uses 60. The recorded 2D residuals support the round-off-stagnation explanation. |
| A11 provisional release | **CORRECT** | All 204 historical 2D records are marked provisional with factor 10. Final status explicitly clears provisional status and binds to the verified, passing calibration evidence with the same factor. |
| A11 expectation of no tight attempts | **NEEDS-RESTATEMENT** | That expectation did not hold: **12 production S cases required tightening**. They were recorded correctly. |
| 3D sealed/OOD QC counts | **CORRECT** | **128/128 sealed case-kinds; 32/32 OOD case-kinds accepted.** Job index sets are disjoint and complete. This verifies aggregate counts, not inaccessible fields. |
| Current E0 status | **CORRECT** | Collector: **3D PASS; 2D INCOMPLETE**, with both dimensions validated. Missing 2D sealed/OOD QC prevents a completed 2D verdict. |
| Current `report.md` | **WRONG** | It says 2D has 0/0 cases and is unvalidated, G5 failed, and no historical case-kind failed verification. All contradict the current evidence. |
| Calling `eps_ref` a bound | **WRONG** | It is a conditional Richardson estimate. Neither calibration nor eight cross-checks proves a rigorous error bound. |

G1–G4 solver trajectories were not rerun; those assessments check the recorded evidence against the amended specification, with the independent recalculations identified above.

The exact rejected 2D records are:

- **S unresolved-floor:** dev6 `{1}`; val32 `{21,24}`; test64 `{4,10,15,25,44,54,56,58,63}`.
- **ST order-fail:** test64 `{36}`. After refinement its orders are **2.15538, 2.21779, 2.29980, 2.34885, 2.37905**.

The floor failures have near-second-order differences, but their final smallest middle–fine difference reaches only **0.397–0.975 times** the required floor threshold. They cannot be relabelled accepted merely because their observed orders look good.

The independently recomputed cross-check values below are relative quantities, not percentages. **Every row is A10 PASS and X2 FAIL.**

| Dimension / case | `eps_ref` | `x_PS` | `sigma_PS` | Effectivity | Original rule |
|---|---:|---:|---:|---:|---|
| 2D / 0 | 1.36072e−6 | 1.69785e−6 | 1.45416e−8 | 1.24775 | UNRESOLVED |
| 2D / 1 | 4.08640e−6 | 4.08046e−6 | 1.30930e−12 | 0.99855 | UNRESOLVED |
| 2D / 2 | 1.95633e−6 | 1.92090e−6 | 5.86337e−9 | 0.98189 | UNRESOLVED |
| 2D / 3 | 4.83375e−6 | 4.81717e−6 | 1.29623e−12 | 0.99657 | PASS |
| 2D / 4 | 3.16772e−6 | 3.15193e−6 | 2.25558e−9 | 0.99501 | UNRESOLVED |
| 2D / 5 | 9.51955e−6 | 9.48329e−6 | 4.83390e−9 | 0.99619 | UNRESOLVED |
| 2D / 19 | 6.01517e−6 | 6.01118e−6 | 3.94324e−9 | 0.99934 | UNRESOLVED |
| 2D / 26 | 8.75256e−6 | 8.70677e−6 | 1.33493e−12 | 0.99477 | UNRESOLVED |
| 3D / 0 | 4.18133e−5 | 4.19608e−5 | 9.42134e−9 | 1.00353 | PASS |
| 3D / 1 | 2.86892e−5 | 2.85505e−5 | 1.86221e−10 | 0.99516 | PASS |
| 3D / 2 | 2.07648e−5 | 2.07433e−5 | 1.45354e−10 | 0.99896 | PASS |
| 3D / 3 | 2.20851e−5 | 2.20937e−5 | 8.77763e−10 | 1.00039 | PASS |
| 3D / 5 | 8.42161e−5 | 8.52457e−5 | 4.69267e−8 | 1.01223 | PASS |
| 3D / 33 | 8.39290e−5 | 8.43086e−5 | 7.60872e−9 | 1.00452 | PASS |
| 3D / 43 | 1.15885e−4 | 1.18427e−4 | 4.67359e−9 | 1.02193 | PASS |
| 3D / 47 | 2.10601e−5 | 2.10626e−5 | 2.33889e−10 | 1.00012 | PASS |

**2D case 0 warrants explicit qualification:** PS discrepancy is about 25% larger than `eps_ref`, although comfortably within the registered `2 eps_ref` bar. Its PS–Richardson distance is **8.416e−7**, about **0.62 eps_ref**; a blanket claim that extrapolation leaves a negligible discrepancy would overreach.

All old-versus-new distances reproduce directly from the sibling worktrees:

| Dimension / cohort / kind | Cases | Median distance (%) | Maximum (%) |
|---|---:|---:|---:|
| 2D / dev6 / ST | 6 | 0.083114 | 0.208827 |
| 2D / dev6 / S | 6 | 0.033895 | 0.106251 |
| 2D / val32 / ST | 32 | 0.069945 | 0.349118 |
| 2D / val32 / S | 32 | 0.027830 | 0.148831 |
| 2D / test64 / ST | 64 | 0.079445 | 0.305648 |
| 2D / test64 / S | 64 | 0.029975 | 0.128746 |
| 3D / val64 / ST | 64 | 0.735146 | 2.808081 |
| 3D / test32 / ST | 32 | 0.708078 | 2.695233 |

Calibration’s worst measured-change/proxy ratios are **0.0110627 / 0.000391278** for 2D production–tight / tight–tighter, and **0.0245780 / 0.0158483** for 3D. These support the registered conditional proxy on the calibration cases. “All solves OK” means residuals satisfy **2e−11**, not that every tighter target was reached. A11’s first finest-level calibration row totals **3.510 hours** across all three solves; the tight solve alone took **2.880 hours**.

The release narrative should therefore retain three limits: A10 was changed after inspecting PS self-convergence; PASS establishes conditional agreement only on the scoring lattice at the five positive output times; and the floor proxy’s stability/normalisation assumptions are calibrated rather than proven.

The immediate reporting defect is in [report.md](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-references/experiments/jcp-references/report.md:27), including its [stale G5 statement](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-references/experiments/jcp-references/report.md:89). **The stored numerical results pass this audit; that report is not ready to serve as the current results record.**