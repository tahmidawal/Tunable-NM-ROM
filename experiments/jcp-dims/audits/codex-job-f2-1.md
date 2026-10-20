**f2a: SUBMIT. f2b: SUBMIT.** No blocking defect found. Resource estimates need the qualification below.

Audited staged commit `f8242f3d04c18bed97c844d19167b584b7863f47`.

| Item | f2a | f2b | Evidence |
|---|---|---|---|
| 1. Batch script | CORRECT | CORRECT | `gpu`, one H200, GPU preflight exits 42 on CPU fallback, f64/highest precision, correct `jd_f2a`/`jd_f2b` names, separate paralab directories and logs. Limits match reservations: 2.75 / 1.5 h. |
| 2. Committed staging and manifest | CORRECT | CORRECT | Each contains 160 source files matching HEAD and provenance hashes; all 164 manifest entries verify, with no uncovered files. Config hashes match certificates. |
| 3. Configuration and scientific coverage | CORRECT | CORRECT | Exactly 21 / 5 distinct `(r,M)` cells, 168 / 40 settings; no missing or extra cells. Numerical identity manifests agree. Details below. |
| 4a. Resource certificate arithmetic | CORRECT | CORRECT | Independently reproduced projections: **1.801351 / 0.887893 h**. Limits exceed 1.5× projections, rounded to quarter-hours. |
| 4b. Resource-model interpretation | NEEDS-RESTATEMENT | NEEDS-RESTATEMENT | Smoke-derived estimates are plausible, but the scaling model does not establish conservative completion bounds. |
| 4c. Memory feasibility | CORRECT | CORRECT | Accepted smk2 exercised R512/M4096 through gauss384, including successor tables, at **29.661 GB** peak, below the 120 GB gate. |
| 5. Ledger and concurrency controls | CORRECT | CORRECT | Pending inventory validates; both attempts are covered. Lane/account caps and shared campaign submission lock are implemented. |

The configurations implement the requested LSPG-CN-R slice of DESIGN §2.1/A1.6/A1.7:

- All eight Gauss rungs, `dt = 0.005`, explicit `M`, frozen `F4096` prefixes.
- Production and tight solves on all38; tighter solves on dev6.
- `rho_only` on every nonterminal rung, with `qp` exactly the next rung. Gauss384 correctly has neither.
- Q3 step-Jacobian diagnostics at gauss256 on all κ cells.
- Successor and check rungs share the same ordered all38 population. Thus these outputs support ladder Q2 and A4 selection over gauss64–gauss256; the top two rungs are not qualified candidates.
- Together, the jobs supply A3 accuracy comparisons and decomposition corners. **A3 cost conclusions still require the separate timing panel**; `dacc` elapsed times are resource measurements.

The resource-model qualification is substantive but nonblocking: `resources.py` scales from the largest-work smoke measurement using \(mMr\), with the smallest measured per-case time as its floor. It underpredicts some smaller smoke units and does not explicitly model successor work, iteration variability, or compilation amortization. The maximum setup allowance and scheduler margins provide headroom, but “conservative” is not demonstrated. The certificates already correctly say **estimate, not bound**.

At the live queue check, **jd_smk3 (5536382)** was running: lane count **1**, account count **4**. Adding both jobs gives **3 / 6**. Neither remote attempt directory existed. Ledger totals were **0.1747 h spent + 2 h reserved + 30.25 h pending = 32.4247 h**, below 55 h; submission moves each pending allowance into a live reservation without double-counting.

Read-only audit completed: no submissions, file changes, scratch copies, or LAB-LOG writes. I computed `ledger.py show`’s totals in memory because its actual implementation rewrites the ledger.