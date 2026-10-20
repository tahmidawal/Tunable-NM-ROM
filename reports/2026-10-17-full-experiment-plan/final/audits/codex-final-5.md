OVERALL: NOT-CLEAN

No sampled numerical mismatch was found. The remaining issues concern qualification definitions, glossary completeness, one omitted correction, and provenance for operational metadata.

I made no file changes, submitted no jobs, ran no generator, and used no scratch storage.

Below, `F` means `reports/2026-10-17-full-experiment-plan/final/`. Lane paths mean `worktrees/<date>-jcp-<lane>/experiments/jcp-<lane>/`.

| Requested item | Verdict |
|---|---|
| 1. Independently recompute ≥40 numbers | **CORRECT** — substantially more than 40 checked; no mismatch at printed precision |
| 2. Provenance and hand-typed numbers | **NEEDS-RESTATEMENT** — result records pass; heartbeat metadata bypasses registration |
| 3. Verdict fidelity | **CORRECT** for all 36 verdict rows; qualification glossary needs correction |
| 4. Retractions | **NEEDS-RESTATEMENT** — named withdrawals preserved; head-rule correction absent from consolidation |
| 5. Brief coverage | **NEEDS-RESTATEMENT** — required sections present; glossary remains incomplete |
| 6. Generator correctness | **CORRECT** for current numerical aggregation and filters; metadata-registration omission remains |

**Remaining findings**

1. **NEEDS-RESTATEMENT — Q1/Q2 definitions omit material qualification conditions.**

   [F/report.md:478](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-17-full-experiment-plan/final/report.md:478) defines Q1 through least-squares stationarity without the GAL exception. However, the owning qualification lane explicitly uses **Galerkin residual certification for GAL**, rather than that stationarity criterion: `jcp-qualify/report.md:65,498`.

   The report acknowledges this change in its corrections at `F/report.md:379`, making its glossary internally inconsistent.

   Likewise, `F/report.md:479` omits the requirement that the successor rule itself converge against the next rule. The owning definitions include this at `jcp-qualify/report.md:499,520`. Its definition of numerical qualification also includes the reproduction gate where applicable (`:502`).

   Restore those exceptions and conditions. This does **not** change the checked qualification counts.

2. **NEEDS-RESTATEMENT — important table terms remain undefined.**

   The closing glossary does not explain:

   - **“stable”** in the long-horizon table, `F/report.md:321`. The lane’s actual criterion checks validity and kinetic energy **at recorded output times only**, with no between-output check (`jcp-budget/report.md:463`).
   - **“no error growth”** in that table. This is a bootstrap-interval classification with eligibility and minimum-population conditions, not simply a comparison of two displayed errors (`jcp-budget/report.md:458`).
   - **TH06**, used at `F/report.md:165`: the owning glossary defines it as the θ-method with θ = 0.6, a first-order control (`jcp-qualify/report.md:495`).
   - **C-A / C-E** and **EQ**, used in deferred items at `F/report.md:413,416`.

   These omissions matter under the explicit requirement to define every term and column for a reader opening the report cold.

3. **NEEDS-RESTATEMENT — the head quadrature-policy correction is missing from the consolidated corrections.**

   The heads lane explicitly contrasts the earlier **post-hoc Gauss 32²** rule with the incumbent’s **Gauss 96²** outcome under the new successor-agreement policy at 1024²:

   - [jcp-heads/report.md:643](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/report.md:643)
   - `LAB-LOG.md:19240`

   The compilation’s head correction at `F/report.md:387` covers the initial-fit tolerance and same-seed replay, but omits this scientific qualification change. The source-section pointer at `:404` makes it recoverable; it does not state it in the consolidation.

   Add the contrast, retaining the lane’s qualification that 96² is a **policy outcome, not a universal requirement**. I found no revival of the old 32² rule as a valid result.

4. **NEEDS-RESTATEMENT — pending-lane metadata is not covered by the claimed provenance contract.**

   [make_final.py:943](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-17-full-experiment-plan/final/make_final.py:943) reads heartbeat files directly with `json.load`, then prints their contents without registering their paths, values or hashes.

   Consequently, the heartbeat timestamps and the quoted **14.87 GPU-h** in `F/report.md:341,345` have no corresponding heartbeat records in `PROVENANCE.json`, despite the universal assertion at `:5`.

   These values are **read from data, not hand-typed or invented**. Register the heartbeat fields, or explicitly narrow the provenance assertion to exclude operational status quotations.

**Independent numerical checks — CORRECT**

I read lane JSONs directly without importing or executing `make_final.py`.

| Report evidence | Independently recovered values |
|---|---|
| `F/report.md:80–87`, kernel fits | All **24** exponents and interval endpoints agree. Fitted exponents: 2D **1.174032, 1.084166, 1.924877, 1.235084**; 3D **1.011199, 1.031497, 1.326162, 1.034708**. |
| `:111–118`, mesh cost | All **8** max/min ratios recomputed from mesh medians; all **8** upper interval endpoints agree. Largest upper endpoint **1.0403859222**. |
| `:122–125`, 3D timings | All **16** millisecond conversions agree: **522/522/525/523**, **60.7/64.9/61/61.4**, **28.4/28.9/28.7/28.6**, **29.2/31.9/30.3/29.7**. |
| `:138–147`, accuracy and qualification | All **16** percentage conversions agree; qualification independently counts **98/112**. |
| `:41–44`, extended verdicts | H-acc-N: **21 pass, 4 inconclusive, 5 unqualified/inconclusive**. Through r=384: **14 monotone, 10 inconclusive, 3 not monotone**; through r=512: **20 not monotone, 7 inconclusive**. |
| `:188–199`, FOM ladder | **185 configurations, 170 admissible**. At each mesh, primary scope: UB **16 ≥2×, 2 <2×, 3 unresolved**; CC **19 <2×, 2 unresolved**; PS/union **21 <2×**. |
| `:55,61`, union extrema | 2D **0.0493062645–0.7055166351**; 3D **0.0078554128–0.1582392454**. |
| `:167,180–182`, 3D front/H2q | Width counts **18,25,32,6,2**; H2q **1.1491377674, 2.9528017141, 9.8741349644**, comparator Q2 **pass/pass/fail**. |
| `:221–246`, POD | Domination counts **8/8, 8/8, 2/4 qualified 3D, 5/5 diagnostic 3D**. All **20** displayed time ratios reproduce as medians of exported per-case ratios. |
| `:262–279`, rule families | Finite point counts **9216,4181,24577,16384,28657,32768,32768** agree; all-draw qualification interpretation retained. |
| `:288–316`, test space/error budget | Blocked square-cell verdict and low-frequency comparison agree. Cross-mesh budget labels count **7 same, 3 inconclusive**; sampling claim remains LSPG-only. |
| `:323–327`, horizons | Qualified/stable/no-growth counts **9/10/10**, **9/11/11**, **12/12/12**; T=1 retains **11 provisional W3/W2 labels without verdicts**. |
| `:335`, head comparison | **1.4402459%**, **10.1713102%**, **0.26480846×**, **1.0953379%**, **0.59100019×**. |
| `:354–359`, P1g | All **42** displayed measurements agree, including separate E0 and first-order spreads. |
| `:372–373`, retractions | Historical H5 **13.6447877× / 17.7057367×**; all six withdrawn-arm distances agree, with worst case **val32\|19**. |
| `:429–442`, ledger | All ten spends, caps and entry counts agree. Finished total **107.4971 GPU-h**; all-lane total **144.8293 GPU-h**, correctly printed **107/145**. |
| `:451–460`, lane list | All **20** current head/mirror identifiers match the compilation. |

Sections 3.11–3.12 remain explicitly pending. Neither lane currently has an EXPORT; no scientific results are invented.

**Provenance, verdicts and code — checked**

- **CORRECT:** All **437 direct JSON-value records** resolve and match.
- **CORRECT:** All **57 referenced disk-file hashes**, both historical H5 blob hashes, and all **33 PNG hashes** match.
- **CORRECT:** The two wildcard-path exceptions are the documented optional `state`/`reaches_2x` branches.
- **CORRECT:** All **36 FINAL-EXPORT verdict strings** appear in the report. Their classifications preserve primary scope, finite-ladder upper bounds, developmental/in-sample status, exploratory heads evidence, deployment-mode policy and unqualified 3D comparators.
- **CORRECT:** Primary LSPG filtering, baseline batch separation, baseline/ours ratio direction, growth-window separation and ledger summation.
- **CORRECT:** Visually inspected family and union summary plots contain the intended datasets.
- **CORRECT:** Generator syntax parses in memory.
- **CORRECT:** Required major sections, Mermaid, both readings of k, dimensional/family H-comp, mesh-specific H-base, both K2 outcomes, P1g, deferrals and merge status are present. No prohibited heat comparison or pre-September Burgers speed-up was found.

**Previous WRONG / NEEDS-RESTATEMENT items rechecked**

| Prior finding | Current verdict and evidence |
|---|---|
| Invalid growth path; ambiguous derived paths/batch filters/optional keys | **CORRECT — fixed/documented**; direct paths resolve and current derivations preserve filters. |
| Literal 3D success count, failed-reference count and worst-case identifier | **CORRECT — fixed**; `make_final.py:1000–1002`, `report.md:213,329,373`. |
| Audit 4’s literal “one label” and “two out-of-memory paths” | **CORRECT — numerical specificity removed**; `report.md:383–384`. |
| Universal E0 claim; first-order P1g labels | **CORRECT — fixed**; `report.md:3,353,474`. |
| Overbroad baseline population; evolution-only cost scope | **CORRECT — fixed**; `report.md:25,33`. |
| Historical P1g qualification | **CORRECT — fixed**; `report.md:354–359`. |
| H2q qualification/timing scope; head exponent exception; baseline budget qualifiers | **CORRECT — fixed**; `report.md:62–70,97,489`. |
| Previously named missing corrections, including P1a fixture and dims A14.5 | **CORRECT — included**; `report.md:381–386`. |
| Faulty exhaustive quotation selector; quotation-number provenance | **CORRECT — quotation appendix removed**; explicit source pointers replace it. |
| Union-only figure labelled per-family; broken table pipes | **CORRECT — fixed**; `report.md:205–209,227–246`. |
| T=1 growth conflation; instantaneous ledger implication | **CORRECT — fixed**; `report.md:321–327,440`. |
| Family “none” interpretation; blanket slower-kernel claim | **CORRECT — fixed**; `report.md:259,89`. |
| Previously identified non-LaTeX quotation expressions | **CORRECT — offending appendix removed**; H2q and deferred-mesh expressions fixed. |
| Audit 4’s Jacobian/normal-matrix/Cholesky/FSP omissions and qualification exception | **CORRECT — addressed**; `report.md:471–472,480`. |
| Overall glossary completeness | **NEEDS-RESTATEMENT** — remaining operational definitions and terms listed above. |

No checked numerical result needs changing. The remaining corrections concern the report’s definitions and completeness claims.