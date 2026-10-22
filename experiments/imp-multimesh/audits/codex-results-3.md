| Item | Verdict |
|---|---|
| A_sel union/deviation | **NOT FIXED** — subtraction deviation is disclosed, but “all competitor construction artefacts” excludes `snap`, `pod`, `codes`, and `trust`. The subtotal is **1,639.605 s**; all artifacts total **2,963.051 s**. Omitted codes are explicitly required by A5.2. |
| Compile-table arm mismatch | **FIXED** — PM-1024 comparisons reproduce **0.891221 s** and **0.762845 s**. |
| Query-time medians | **FIXED** — checked against saved observations; means and outlier counts remain reported. |
| Timing-block label | **FIXED** — **22/30/33** correctly count blocks. |

**New numerical reporting defect:** the A_sel construction subtotal is misrepresented as a complete total and an upper bound. The displayed **0.198** ratio reproduces its implemented formula, but inherits that accounting defect. Primary **FIRES** remains unchanged.

No files changed or jobs run.

**Required fix:** include the omitted construction categories, or explicitly disclose their omission and describe the ratio as a partial-cost scenario; remove the unsupported completeness/upper-bound claim.