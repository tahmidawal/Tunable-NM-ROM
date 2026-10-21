**All audited result items are CORRECT under Amendments 1–13. No unsupported interpretation found in the JSON verdicts.**

Independent CPU recomputation used four in-range cases for A2/A3, four C1 cases for A4—including all reported worst-case contributors—and all 38 cases for the A3 fixture. Aggregate checks covered every JSON row.

| Item | Assessment | Evidence |
|---|---|---|
| A2 tail ratio and wall share | CORRECT | Raw ratios: 1939.5456, 2602.1831, 151.1955, 2058.6247; respective ω: 0.957249, 0.952243, 0.966715, 0.956641. Match stored values. |
| A2 controls | CORRECT | Recomputed ω+: 0.813035 ≥ 0.5; ω−: 0.046884 ≤ 0.25. Envelope/sine identities pass. Independent bank-path gap 0.491574 < bound 1.409004. |
| A2 RUN decision | CORRECT | Reaggregated median ratio **2165.2114 ≥ 3**, median ω **0.950571 ≥ 0.5**; availability 38/38. RUN follows the amended conjunction. |
| A3 primary Δ-worst, K=9216 | CORRECT | Both maxima occur at val32\|19: 0.0177526910 → 0.0151545010; Δ = **−0.00259819008**, or −0.259819 percentage points. |
| A3 controls and validity | CORRECT | Independent fixture Δ = **0.00500016341456 > 0.001**. Across nine configurations, recorded identity/reproduction discrepancies are ≤7.72e−17/1.12e−16; reproduction checked against source analysis. Two GAL budget exits are excluded consistently with Amendment 13. |
| A3 verdict | CORRECT | CPU screen passes; timing remains deferred. K=16 trips for all configurations but is descriptive under Amendment 1. |
| A4 C1 worst floors | CORRECT | Independent projections reproduce **H=4.352161%, S1024=5.850373%, B512=6.583686%**; sampled discrepancies ≤2.6e−15. |
| A4 controls | CORRECT | Containment holds at every sampled time. Across stored rows, null-enrichment/bank discrepancy ≤1.12e−16; null is stopped. High-frequency enrichment also stops, though now descriptive. |
| A4 STOP F15 | CORRECT | H beats B512 but exceeds half the sine worst floor, **2.925187%**. The required twofold improvement fails. |

A2 RUN establishes **budget eligibility under a heuristic**, not causation or permission to launch. A3 establishes no timing gain; A4 tests representation floors only. Reference accuracy remains conditional on the stated Richardson estimates.

A1 was skipped. No repository/main/lab-log writes or jobs submitted; scratch deleted.