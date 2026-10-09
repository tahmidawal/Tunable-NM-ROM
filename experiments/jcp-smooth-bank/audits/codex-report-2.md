Read-only audit completed; all five PNGs inspected. The report reproduces byte-for-byte in memory. Independent checks passed for 4,140 quadrature statistics, 216 thresholds, 96 projection summaries and 14 timing medians.

**CORRECT:** prior items on arithmetic, tails, representation, Sobolev effects, unidentified rollout limitation, treatment-versus-base noise, evaluation procedure, cost, and no promotion. Plot items **ladders, derivatives, e2e and spectra: CORRECT**.

Remaining problems:

| Prior item | Verdict | Concrete fix |
|---|---|---|
| Every number generated | **WRONG** | `make_report.py` still embeds numerical prose: “first 320 tests,” “257² nodes,” “256 vs 512 … within 2,” training-state count, query steps/output count, and plot cohort/sample sizes. Historical R1 comparison values also remain literal `R1_PINNED` constants. Generate these from configuration/results or pinned source evidence; assert population-wide statements. The R2b discrepancy is now correctly extracted. |
| Points/H2 summary | **NEEDS-RESTATEMENT** | The brief groups sig1/sig2/sob01 before saying “the pre-registered verdict is UNRESOLVED.” The detailed verdict is **UNRESOLVED for sig1/sig2, not met for sob01**. State that distinction explicitly. |
| Onset claim | **NEEDS-RESTATEMENT** | Delete “**which a smoother bank cannot remove**.” The restricted-vector diagnostic supports consistency with a test-frequency contribution, not impossibility of changing onset through bank smoothness. |
| Status/provisional labels | **NEEDS-RESTATEMENT** | R1 still quotes accuracy percentages without a nearby **PROVISIONAL** label. Add it directly to that bullet. |
| Glossary and mathematics | **WRONG** | “Each … step solves \(r(c)=0\)” misstates the implemented overdetermined LM least-squares solve. Write “approximately minimizes \(\tfrac12\lVert r(c)\rVert_2^2\).” Define its symbols. Convert remaining glossary equations to LaTeX and specify that projection/derivative aggregates pool **228 states, including initialization**, whereas evolved rollout errors exclude initialization. |
| `frontier.png` | **NEEDS-RESTATEMENT** | Layout and threshold semantics are fixed, but overlapping markers still hide several banks, especially in `fast`. Use bank facets or annotated grouped markers so every bank is identifiable without changing its coordinates. |

One additional caption fix: `e2e.png`’s shared **38-case** caption can imply that timing uses the same cohort. Add **“Timing: dev6 only, three repetitions in both ordering directions”** to the timing panel.