**New tables pass; several prose claims remain wrong or unsupported.** Read-only, offline; no files changed. Line numbers below refer to the [report](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-01-quadrature-study/reports/2026-10-01-burgers2d-offmesh-quadrature.md).

**Numerical verification**

In-memory regeneration matches exactly: **151,899 characters**.

| New table | Rows checked | Result |
|---|---:|---|
| Frozen post-hoc confirmation | 3 | All values/verdicts match raw test results and summaries |
| Mesh invariance on test64 | 27 | All maxima, ratios and B2 verdicts match |
| Per-case mesh invariance | 24 | All populations, spreads and moved-case lists match raw development/test results |

Confirmed outcomes:

- **Accurate Gauss96 / fast Fibonacci1597: yes.**
- **Head Gauss32: no.** B3 = **0.585/1.105/1.258%**, exceeding **0.50%** everywhere.
- Head `gref` B2 = **1.9074**, failing **1.02**; largest per-case spread **4.715496 pp**.

**Previous findings**

| Previous finding | Status | Evidence |
|---|---|---|
| Blanket “every number comes from data” | **NOT FIXED** | L3; generator still hard-codes dimensions, thresholds, timing counts and numerical prose. |
| Wrong “18 majors”; ambiguous submission chronology | **FIXED** | L1442 removes the count and distinguishes ROM submission from already-running reference jobs. |
| Blanket mesh invariance | **NOT FIXED fully** | Head exception and failures now explicit, but “one mesh-invariant solution” still exceeds error-spread evidence; see below. |
| Dense improvement used mismatched populations | **FIXED** | L49/57 disclose populations. Independent common-six-case maxima decrease at every mesh in all settings. |
| Off-mesh superiority too broad | **FIXED** | L49/57 qualify resolved rules, cohort-worst metrics and absence of casewise dominance. |
| “Agree against S” confused errors with solutions | **FIXED** | Now “worst errors are close,” with off-mesh slightly worse against S. |
| Favourable B1 failure overgeneralized | **FIXED** | Restricted to resolved linear arms. |
| Approximately matched points; different rho targets | **FIXED** | L51/58 explicitly distinguish both. |
| Early/wide characterization | **FIXED** | L53/60 distinguish argmax from tail timing and acknowledge exceptions. |
| Head spread attributed exclusively to solver | **NOT FIXED** | L131 retains “not from the quadrature”; cause remains unisolated. |
| Missing head confirmation failure | **FIXED** | L62–68 explicitly reports **no**. |
| Missing test B2 table | **FIXED** | L70–101 includes head failures. |
| D4 dense B2 mixed populations | **FIXED** | L600 explicitly omits dense. |
| Unsupported smoke exception / calibration-kill history | **NOT FIXED** | L1443 unchanged; inspected local records still provide no corroboration. |
| “Needs ten times” / minimum quadrature requirement | **FIXED** | L1445 distinguishes certified target from demonstrated minimum. |
| Ambiguous case labels; wrong test cross-references | **FIXED** | `test64:1`; references now T5/T6. |
| D4 units | **FIXED** | Explicit percent error; per-case spread labelled pp. |
| Linear coefficients “solved directly” | **FIXED** | Diagram confines closed form to initialization; glossary specifies LM stepping. |
| Missing notation/source identification | **NOT FIXED fully** | L24 adds most definitions and Hari’s path; mesh-size \(N\) remains undefined and “paper eq. 13” lacks a specific local source. |
| Tail counts / half-step medians | **NOT FIXED fully** | Brief gives 19/32 states and fractional medians; T6 still rounds **42.5→42**, **1.5→2**. |
| B1 broken LaTeX | **FIXED** | Replaced by readable threshold prose. |

Previously correct timing, reference-coverage, GPU and historical-test-reuse qualifications remain. Freeze chronology was not independently re-audited this pass.

**Remaining WRONG claims in the requested sections**

- **L50:** development B2 range **1.0001–1.0060**, supposedly excluding only accurate Gauss32, omits **head Gauss32 = 1.006824**. The generator excludes `gauss32` in *every* setting.
- **L49/57/131:** nearly unchanged **error magnitudes** do not establish identical **solutions**. L131 also drops “resolved”: accurate Gauss32 has per-case spreads **1.968861 pp development / 0.407649 pp test**. Even the strict development **≤0.02 pp** assertion narrowly fails for selected fast Fibonacci1597: **0.020004912 pp**.
- **L131:** “not from quadrature” remains an unsupported causal exclusion.
- **L1447:** “six cases **per cohort**” is literally wrong: development dense at \(4096^2\) contains **six dev6 cases and zero val32 cases**; test contains six test64 cases.

The smoke-history claim remains **unverified**, rather than demonstrated false.