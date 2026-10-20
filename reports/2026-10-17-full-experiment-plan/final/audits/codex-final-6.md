OVERALL: NOT-CLEAN

The sampled numbers and provenance pass. Two substantive presentation defects remain: an incorrect QN solver definition and an unlabeled population change in the error-budget table. The glossary also still omits statistical definitions needed to interpret columns.

I made no file changes, submitted no jobs, ran no generator, and used no scratch storage.

Below, `F` means `reports/2026-10-17-full-experiment-plan/final/`. All line numbers refer to the current files.

| Requested audit item | Verdict |
|---|---|
| 1. Independently recompute at least 40 numbers | **CORRECT** — over 150 checked; no mismatch at printed precision |
| 2. Provenance, hashes, hand-typed results | **CORRECT** for checked records; no new hand-typed measured-number counterexample found |
| 3. Verdict fidelity | **CORRECT** for the 36 hypothesis rows; error-budget scope needs restatement |
| 4. Retractions | **CORRECT** — named withdrawals preserved and previous omissions addressed |
| 5. Brief coverage | **NEEDS-RESTATEMENT** — required sections present; glossary defects remain |
| 6. Generator correctness | **WRONG** for the QN definition; **NEEDS-RESTATEMENT** for the mixed-population table; checked arithmetic and filters pass |

**Remaining findings**

1. **WRONG — `k4` is not four iterations per step.**

   [F/report.md:471](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-17-full-experiment-plan/final/report.md:471) says `K1`, `K2`, `k1`, and `k4` are “sweep or iteration counts per step.” This conflates two different controls.

   The owning speed3d report defines QN-k as a Gauss–Newton matrix **refresh interval**, at [speed3d/report.md:506](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/report.md:506). The implementation checks `(k - last_ref) >= kr` across time steps; nonlinear iterations have a separate `qmax` limit: [qn3.py:139](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/qn3.py:139), [qn3.py:168](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/qn3.py:168).

   Fix [make_final.py:1267](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-17-full-experiment-plan/final/make_final.py:1267): distinguish fixed-sweep counts from QN refresh intervals, including forced refreshes. This matters because the reported H2q winner uses `k4`.

2. **NEEDS-RESTATEMENT — the error-budget table silently combines different populations.**

   [F/report.md:303](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-17-full-experiment-plan/final/report.md:303) gives R256/GAL/CN/1:

   `e_P | no single dominant | e_P | same`

   Each value is correctly transcribed, but “same” does **not** compare the three preceding cells on their displayed populations.

   The owning lane’s 1024² result uses 38 cases and reports “no single dominant”: [budget/report.md:129](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-budget/experiments/jcp-budget/report.md:129). Its cross-mesh comparison instead uses the shared 16-case subset, where all three labels are `e_P`: [budget/report.md:190](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-budget/experiments/jcp-budget/report.md:190), [checks/meshcmp.json:773](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-budget/experiments/jcp-budget/checks/meshcmp.json:773).

   [make_final.py:861](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-17-full-experiment-plan/final/make_final.py:861) combines `b2` labels with `b2_mesh` labels without explaining this distinction. Label each population explicitly, or display the shared-subset dominant labels alongside their cross-mesh verdict.

3. **NEEDS-RESTATEMENT — statistical table terms remain undefined.**

   The glossary now fixes the terms identified in audit 5, but still does not explain:

   - **“dominant” / “no single dominant”** in §3.8: a population-level rule based on resolved per-case winners, with completeness and minimum-population requirements.
   - **“front order … preserved”** in §3.9: no resolved pair reverses on common eligible cases—not necessarily identical raw rankings.

   These definitions are available at [budget/report.md:452](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-budget/experiments/jcp-budget/report.md:452) and [budget/report.md:460](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-budget/experiments/jcp-budget/report.md:460). Add them to satisfy the brief’s requirement to explain every term and column.

**Independent numerical checks — CORRECT**

I read lane JSONs directly, without importing or executing `make_final.py`. This inventory exceeds the required 40 numbers:

| Report evidence | Independently recovered values |
|---|---|
| `F/report.md:80–87`, kernel fits | All 24 fitted exponents and interval endpoints agree: 2D exponents **1.17, 1.08, 1.92, 1.24**; 3D **1.01, 1.03, 1.33, 1.03**. |
| `:111–118`, mesh cost | All eight max/min ratios recomputed from mesh medians; all eight upper interval endpoints agree. |
| `:122–125`, 3D timings | All 16 millisecond conversions agree: **522/522/525/523**, **60.7/64.9/61/61.4**, **28.4/28.9/28.7/28.6**, **29.2/31.9/30.3/29.7**. |
| `:138–147`, accuracy/qualification | All 16 median/worst percentage conversions agree; **98/112** configurations qualify. |
| `:159–167`, order/front | Seven temporal-order estimates agree; 3D coverage **18,25,32,6,2**. |
| `:180–182`, H2q | **1.1491377674, 2.9528017141, 9.8741349644**; comparator Q2 **pass/pass/fail**. |
| `:188–213`, FOM comparison | **185/170** configurations/admissible. At each mesh: UB **16 ≥2×, 2 <2×, 3 unresolved**; CC **19 <2×, 2 unresolved**; PS/union **21 <2×**. |
| Verdict table, union ranges | 2D **0.0493062645–0.7055166351**; 3D **0.0078554128–0.1582392454**. |
| `:221–246`, POD | Interval predicates independently recover **8/8, 8/8, 2/4, 5/5** dominations. All 20 displayed time ratios equal the median exported per-case baseline/ours ratios. |
| `:262–279`, rule families | Finite point counts **9216,4181,24577,16384,28657,32768,32768** agree. |
| `:288–316`, test space/budget | Blocked square-cell control agrees; cross-mesh labels count **7 same, 3 inconclusive**, subject to finding 2. |
| `:323–327`, horizons | Qualified/stable/no-growth counts agree; T=1 retains **11 provisional W3/W2 labels without verdicts**. |
| `:335`, head comparison | **1.4402459%, 10.1713102%, 0.26480846×, 1.0953379%, 0.59100019×**. |
| `:354–359`, P1g | All **42** measurements agree, including separate E0 and first-order spreads. |
| `:372–373`, retractions | Historical H5 **13.6447877× / 17.7057367×**; all six withdrawn-arm Q1 distances agree. |
| `:429–442`, ledger | Ten spends and entry counts agree. Totals **107.4971** finished and **144.8293 GPU-h** including pending lanes, correctly rounded to **107/145**. |
| `:451–460`, lane list | All **20** head/mirror identifiers match local git refs. |

Sections 3.11–3.12 remain explicitly pending. Neither lane has an EXPORT; their heartbeat quotations are operational metadata, not scientific results.

**Provenance, figures, verdicts, and code — CORRECT where checked**

- All **441 direct JSON records** resolve and match; **522 source-path checks** passed.
- All **63 referenced disk-file hashes**, both historical H5 blob hashes, and all **33 destination PNG hashes** match.
- Heartbeat fields now have provenance records at `PROVENANCE.json:6949` onward.
- All **36 FINAL-EXPORT key-number strings** match the report. Verdicts retain finite-ladder upper bounds, primary LSPG scope, developmental/in-sample baseline status, exploratory heads evidence, deployment-mode policy, and unqualified-comparator labels.
- Primary-scope filtering, baseline batch separation, baseline/ours time-ratio direction, separate growth windows, and ledger summation are correct.
- Visually inspected family and union summary figures contain the intended datasets.
- The generator parses successfully in memory.
- Required major sections, LaTeX equations, Mermaid, both readings of k, dimensional/family H-comp, mesh-specific H-base, both K2 outcomes, P1g, deferrals, ledger, and merge-status list are present.
- No prohibited heat comparison or pre-September Burgers speed-up was found.

**Previous audit findings rechecked**

| Previous WRONG / NEEDS-RESTATEMENT item | Current verdict and evidence |
|---|---|
| Invalid growth path; ambiguous derived paths/batch filters | **CORRECT — fixed/documented**; provenance checks pass. |
| Literal success/failed-reference counts | **CORRECT — fixed or numerical specificity removed**; `report.md:213,329`. |
| Universal E0 claim; P1g reference labels | **CORRECT — fixed**; `:3,353–359`. |
| Overbroad baseline population | **CORRECT — fixed**; `:25`. |
| H2q qualification and timing scope | **CORRECT — fixed**; `:66–70,489`. |
| Head exponent exception | **CORRECT — fixed**; `:39,97`. |
| Baseline budget qualifications | **CORRECT — fixed**; `:62–65`. |
| Missing consolidated corrections | **CORRECT — included**, with full-record pointers; `:379–407`. |
| Head Gauss 32²→96² policy correction | **CORRECT — fixed**, including “not a universal requirement”; `:387`. |
| Union-only plot labeled per-family; broken table pipes | **CORRECT — fixed**; `:205–209,227–246`. |
| T=1 growth conflation; instantaneous ledger implication | **CORRECT — fixed**; `:321–327,440`. |
| Q1 GAL exception, Q2 successor check, reproduction gate | **CORRECT — fixed**; `:478–480`. |
| Stable/no-growth, TH06, C-A/C-E, EQ definitions | **CORRECT — added**; `:507–511`. |
| Unregistered heartbeat metadata | **CORRECT — fixed**; `make_final.py:945–947`, provenance records cited above. |
| Overall glossary completeness | **NEEDS-RESTATEMENT** — findings 1 and 3 remain. |

No sampled numerical result requires changing. The remaining corrections concern solver meaning, population scope, and statistical definitions.