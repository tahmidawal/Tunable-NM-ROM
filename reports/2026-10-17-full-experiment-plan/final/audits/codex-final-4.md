OVERALL: NOT-CLEAN

The sampled numerical values agree with the lane JSONs. Remaining problems concern hand-typed empirical counts, incomplete correction summaries, and glossary completeness.

I made no file changes, submitted no jobs, ran no generator, and used no scratch storage.

Below, `F` means `reports/2026-10-17-full-experiment-plan/final/`.

| Requested item | Verdict |
|---|---|
| 1. Independently recompute ≥40 numbers | **CORRECT** — no sampled numerical mismatch |
| 2. Provenance and hand-typed numbers | **WRONG** — mechanical checks pass, but empirical counts remain literal prose |
| 3. Verdict fidelity | **CORRECT** — the 36 verdict rows preserve the relevant qualifications |
| 4. Retractions | **NEEDS-RESTATEMENT** — major withdrawals preserved; consolidation still incomplete |
| 5. Brief coverage | **NEEDS-RESTATEMENT** — required sections present; glossary remains incomplete |
| 6. Generator correctness | **CORRECT** for checked arithmetic and filters; prose-generation defects remain |

**Remaining findings**

1. **WRONG — empirical counts are still hand-typed.**

   Two examples violate the brief’s requirement that every numerical result be generated:

   - “**one label** was ‘baseline already stationary’” is literal text at [make_final.py:1056](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-17-full-experiment-plan/final/make_final.py:1056), printed at `F/report.md:383`.
   - “**two out-of-memory paths**” is literal text at [make_final.py:1061](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-17-full-experiment-plan/final/make_final.py:1061), printed at `F/report.md:384`.

   These describe observed outcomes, not registered design constants. Their documentary sources agree—the speed3d report at line 439 and LAB-LOG at line 19191—but the generator neither computes nor registers these counts individually. Section-level pointers do not satisfy the assertion at `F/report.md:5`.

   Generate and register the counts, or remove the numerical specificity.

2. **NEEDS-RESTATEMENT — the correction summary still omits documented corrections.**

   The major requested withdrawals remain clearly withdrawn. However, the consolidated list at `F/report.md:367–388` omits, for example:

   - The P1a fixture correction: truncating R384 states cannot supply the larger-width arm; the fixture was replaced with `W_R512.npz`. Evidence: [LAB-LOG.md:19149](/home/tahmid/Dev/Tunable-NM-ROM-Claude/LAB-LOG.md:19149).
   - Dims A14.5: the original timing-panel memory argument cited a pair of rungs that the smoke never held live; production evidence replaced it. Evidence: [dims/report.md:850](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-dims/experiments/jcp-dims/report.md:850).

   The pointers at `F/report.md:405–407` let readers recover these, but the consolidation itself does not state them. Add these corrections or explicitly describe §5 as selected highlights rather than the complete requested consolidation.

3. **NEEDS-RESTATEMENT — the glossary is still not self-contained.**

   The newly added statistical and quadrature definitions fix several prior omissions. Remaining examples include:

   - **passive-set QR**, **blocked downdate**, and **G-B2**, used at `F/report.md:384–385` without plain-language definitions;
   - **Jacobian**, **normal matrix**, and **Cholesky**, used to explain the central kernel table at `F/report.md:76–87`, without explanations in the closing glossary;
   - **FSP**, listed at `F/report.md:471` alongside FS and QN without explaining what distinguishes it.

   Also, [report.md:479](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-17-full-experiment-plan/final/report.md:479) says only numerically qualified configurations carry primary verdicts, while the 3D H-cost-N and H-comp rows explicitly include unqualified configurations. The table qualifications are appropriately candid; the glossary should preserve those exceptions.

**Independent numerical checks — CORRECT**

I read the lane JSONs directly without importing or executing `make_final.py`. The following inventory exceeds 40 measurements before counting the ledger, retractions, or verdict aggregates. All agree at printed precision.

| Report evidence | Independent result |
|---|---|
| `F/report.md:80–87`, kernel fits | All **24** fitted exponents and interval endpoints agree. 2D fitted exponents: **1.17403, 1.08417, 1.92488, 1.23508**; 3D: **1.01120, 1.03150, 1.32616, 1.03471**. |
| `:111–118`, mesh cost | All **8** max/min ratios recomputed from exported mesh medians; all **8** upper interval endpoints agree. Largest upper endpoint: **1.0403859222**. |
| `:122–125`, 3D timings | All **16** seconds-to-milliseconds conversions agree: C_W1024 **522/522/525/523**; C_old **60.7/64.9/61/61.4**; W_C_W768 **28.4/28.9/28.7/28.6**; W_C_old **29.2/31.9/30.3/29.7**. |
| `:138–145`, accuracy | All **16** median/worst percentage conversions agree. |
| `:147`, qualification | Independently counted **98/112** qualified configurations. |
| `:41–44`, extended accuracy verdicts | H-acc-N: **21 pass, 4 inconclusive, 5 unqualified/inconclusive**. Through r=384: **14 monotone, 10 inconclusive, 3 not monotone**; through r=512: **20 not monotone, 7 inconclusive**. |
| `:190–199`, FOM counts | At each mesh, primary scope: UB **16 ≥2×, 2 <2×, 3 unresolved**; CC **19 <2×, 2 unresolved**; PS and union **21 <2×** each. |
| `:55–61`, union extrema | 2D **0.0493062645–0.7055166351**; 3D **0.0078554128–0.1582392454**. |
| `:180–182`, H2q | **1.1491377674, 2.9528017141, 9.8741349644**; comparator qualification **pass/pass/fail**. |
| `:221–246`, POD comparisons | Dominated/total **8/8, 8/8, 2/4 qualified 3D, 5/5 diagnostic 3D**. All **20** displayed time ratios agree with medians of exported per-case ratios. |
| `:258–276`, rule families | Family results retain the all-draw intersection; checked point counts include **9216, 4181, 24577, 32768**. Individual Sobol draws can qualify despite no common qualifying size. |
| `:291–318`, test space/error budget | Blocked square-cell controls and low-frequency verdict agree. Cross-mesh budget outcomes count **7 same, 3 inconclusive**. Sampling adequacy is correctly restricted to LSPG. |
| `:324–328`, long horizons | Qualified/stable/no-growth counts: **9/10/10**, **9/11/11**, **12/12/12**; T=1 has **11** qualified and stable members and **11 provisional W3/W2 labels without verdicts**. |
| `:336`, head comparison | Head median **1.4402459%**; linear r=16 **10.1713102%**, cost **0.26480846×**; linear r=128 **1.0953379%**, cost **0.59100019×**. |
| `:355–360`, P1g | All **42** displayed measurements agree, including separate E0 and first-order spreads. |
| `:371–373`, retractions | Pinned combined H5: **13.6447877× / 17.7057367×**. Six withdrawn-arm tight-solve distances reproduce; every worst-case identifier is **val32\|19**. |
| `:429–442`, ledger | Ten spends, caps and entry counts agree. Finished total **107.4971 GPU-h**; all-lane terminal-job total **144.8293 GPU-h**, correctly rounded to **107/145**. |
| `:451–460`, lane list | **18/20** current head/mirror refs still match. Robust advanced after compilation; the remaining entries match. |

The robust refs now resolve to `08677b2c6` / `d0faa7998`, versus the compiled `3ad62e1a9` / `0f1ebd634`. This is a disclosed compile-time snapshot becoming older, not a fabricated commit.

**Provenance, verdicts, figures, and code**

- **CORRECT:** All **437 direct JSON-value records** resolve and equal their source values.
- **CORRECT:** All **57 referenced disk-file hashes** match; both pinned historical H5 blob hashes match.
- **CORRECT:** All **33 PNG hashes** match their records. I visually inspected the family and union summary figures; their content matches the intended datasets.
- **CORRECT:** The only wildcard-path exceptions are the documented optional `state` and `reaches_2x` branches for 3D FOM cells.
- **CORRECT:** All **36 FINAL-EXPORT verdict key-number strings** appear in the report. The classifications agree with owning exports and reports, including primary LSPG scope, developmental/in-sample baseline status, finite-ladder upper bounds, deployment-mode policy, and unqualified 3D comparators.
- **CORRECT:** Primary-scope filtering, baseline batch separation, baseline/ours ratio direction, W2/W1 versus W3/W2 separation, and terminal-job ledger summation behave as described.
- **CORRECT:** Generator syntax parses in memory.
- **CORRECT:** Both unfinished lanes have no EXPORT and remain pending, with no scientific results invented. Their operational heartbeat and ledger numbers are identified as such.
- **CORRECT:** Required major sections, Mermaid, dimensional/family H-comp, mesh-specific H-base, both K2 outcomes, both readings of k, deferred items, and merge status are present. No prohibited heat comparison or pre-September Burgers speed-up was found.

**Previous WRONG / NEEDS-RESTATEMENT findings rechecked**

| Prior finding | Current verdict and evidence |
|---|---|
| Invalid growth key path | **CORRECT — fixed**; direct paths resolve. |
| Hand-typed 3D success count, failed-reference count, worst-case identifier | **CORRECT — fixed**; `F/report.md:213,330,373`; worst cases are read at `make_final.py:997–1002`. |
| Derived-path notation, batch filters, optional-key semantics | **CORRECT — fixed/documented**. |
| Universal E0 claim and first-order P1g labels | **CORRECT — fixed**; `F/report.md:3,353,473`. |
| Overbroad baseline population | **CORRECT — fixed**; `F/report.md:25` says eight tested configurations per mesh. |
| Evolution-only cost scope | **CORRECT — fixed**; `F/report.md:25,33`. |
| Historical P1g qualification | **CORRECT — fixed**; `F/report.md:355–360`. |
| H2q glossary, timing contract, comparator qualification | **CORRECT — fixed**; `F/report.md:66–70,488`. |
| Head exponent exception | **CORRECT — fixed**; `F/report.md:97`. |
| Baseline budget qualifications | **CORRECT — fixed**; `F/report.md:62–65,248`. |
| Previously named missing corrections | **CORRECT — included**; `F/report.md:379–387`. |
| Faulty “exhaustive” quotation selector | **CORRECT — removed**; replaced by explicit pointers at `make_final.py:1202`. Consolidation completeness remains limited as noted above. |
| Historical quotation number-level provenance | **CORRECT — quoted numerical appendix removed**. New literal-count exceptions remain. |
| Union-only figure labelled per-family | **CORRECT — fixed**; `F/report.md:207–211`, inspected images. |
| Broken POD-table pipes | **CORRECT — fixed**; escaped configuration identifiers. |
| T=1 growth conflation | **CORRECT — fixed**; separate columns at `F/report.md:322–328`. |
| Ledger implying instantaneous consumption | **CORRECT — fixed**; terminal-job scope at `F/report.md:440`. |
| “None” conflating family and draw qualification | **CORRECT — fixed**; `F/report.md:261`. |
| Blanket slower-than-operation-count claim | **CORRECT — fixed**; kernel-specific interval interpretation at `F/report.md:89`. |
| Quoted non-LaTeX equations | **CORRECT — offending quotation appendix removed**; prior H2q and deferred-mesh instances fixed. |
| Glossary completeness | **NEEDS-RESTATEMENT — improved, but remaining examples listed above**. |

No sampled measured value requires correction. The report still needs the literal-count, correction-summary, and glossary fixes before it meets the requested clean gate.