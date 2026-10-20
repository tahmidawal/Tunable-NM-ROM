**C1 RESULTS ACCEPTED.** All four Round 1 reporting fixes are **FIXED**.

Line references below are within `experiments/jcp-fom2d/`.

| Fix | Verdict | Evidence |
|---|---|---|
| PS stability and anchor disclosure | **FIXED** | `report.md:223–227`; `make_report.py:402–428`. Reports 0/38 stability passes, both steps, both-valid status, anchor distance, and diagnostic-only meaning. |
| Finality and extension scoped to P1 preview | **FIXED** | `report.md:3,221`; `make_report.py:216–222,396–400`. Base ladder completion is distinguished from provisional matching; extension must be recomputed for FRONT or the authorized fallback. |
| “Lowest median error” qualification | **FIXED** | `report.md:8–10`; `make_report.py:245–251`. No overall accuracy-superiority claim. |
| Screening-cost wording, earlier-comparison scope, median-only blow-up disclosure | **FIXED** | `report.md:11,192,229,239`; `make_report.py:256–258,376–377,430–439,493–497`. Earlier speed-up comparison awaits C2; N16’s six median-only selections are explicitly distinguished from joint matches. |

**Numerical check: PASS.** Every printed C1 value agrees with its JSON source at the displayed precision. Scratch-only regeneration reproduced `report.md` byte-for-byte. Independently verified all **170 admissible configurations’** screening costs, median errors and worst errors against archived results, and reconstructed the **26/26/22** Pareto fronts. Target values, smoke results, ledger totals, PS diagnostics and selection counts also agree.

No repository, main, lab-log or cluster changes; writes were confined to the authorized scratch directory.