**A1 results: CORRECT. Interim report: NEEDS-RESTATEMENT.** No numerical verdict needs changing.

| Item | Assessment | Evidence / required fix |
|---|---|---|
| A1 raw-input recomputation | CORRECT | Recomputed bank/sine/POD floors for val32 cases 19, 26, 5 using raw upwind fields, `fcommon.load_ref`, and `fcommon.G`; independent direct-SVD LOO POD. Maximum absolute discrepancy across floors and D statistics: **3.45e−12**. |
| Case 19 D statistics | CORRECT | `(D_bank384, D_pod_upw, D_pod_ref)` = **(1.085186727, 0.939029482, 0.932477016)**. |
| Case 26 D statistics | CORRECT | **(0.874513738, 0.826569491, 0.810059689)**. |
| Case 5 D statistics | CORRECT | **(1.321046138, 1.053787382, 1.040220269)**. |
| POD controls | CORRECT | Reaggregated T6: positive **0.902507 < 2**, negative **0.884579 ≤ 1.25**, resolution **1.016712 < 2**. Positive and resolution controls fail. |
| Exact-D prerequisite | CORRECT | Independently obtained **c = 0.000213872447 < 0.001** at r=384. Amendment 3 explicitly requires fixture failure and INCONCLUSIVE; algebraic error **2.94e−14** cannot override this. |
| A1 verdict | CORRECT | Gates pass; controls fail. **INCONCLUSIVE** follows the registered precedence despite bank median **0.928005**. Neither smearing conclusion is permitted. |
| Input integrity | CORRECT | All **52** manifest hashes match; stored fields/residuals finite; maximum residual **9.98981e−13**. |
| Numerical export/report/plots | CORRECT | Scratch regeneration from current checks reproduced `report.md`, `EXPORT.json`, and all three PNGs **byte-for-byte**, pinning the reported generation commit. A2–A4 conclusions retain the previously audited meanings. |
| Markdown tables | WRONG | Unescaped case/configuration `|` characters break T6 and every A3 data row, shifting numerical columns. Escape them in `make_report.py`, then regenerate. |
| Title and diagram | NEEDS-RESTATEMENT | “Why … fails” implies established causation. Use “Diagnostics of … failure”; annotate diagram destinations with fidelity **deferred**, boundary arm **eligible but deferred**, F15 **STOP**. Mermaid and role colors otherwise comply. |
| A1 explanation | NEEDS-RESTATEMENT | Explicitly name the failed positive/resolution controls and the failed **c** prerequisite; “prerequisites: False” alone obscures the independently sufficient fixture failure. |
| Retraction disclosure | NEEDS-RESTATEMENT | Existing A2 disclosures are useful but incomplete. Add discarded debugging runs/A4 results seen before Amendment 1, demotion of own-field/K16/high-frequency checks to descriptive status, and Amendment 13’s two excluded GAL budget exits. |
| A2 reference curvature | NEEDS-RESTATEMENT | Label “0 (exact)” as the **continuum compatibility identity**, not a measured numerical E0 curvature. |
| Conditional/provisional labels | CORRECT | Interim status, conditional Richardson references, A2 heuristic eligibility, deferred F7 timing and absent Phase B results are explicit. |
| Standalone PNG labels | NEEDS-RESTATEMENT | Add conditional-reference captions to A3/A4 and A1’s INCONCLUSIVE status/population; these qualifications currently depend on surrounding prose. |
| LaTeX/glossary | NEEDS-RESTATEMENT | Replace prose `ε_ref` with LaTeX. Define `n`, r/K, ROM, GAL/CN/BE, K1 and S1/S3; specify reference-Laplacian normalization for curvature and pooled-time wall share. |

No repository, main or LAB-LOG writes; no jobs submitted. Scratch stayed below **1 MB** and was deleted.