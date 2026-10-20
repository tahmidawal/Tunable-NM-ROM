OVERALL: CLEAN

For the current compiled report, I found no remaining WRONG or NEEDS-RESTATEMENT item. **234 explicitly checked numerical entries match at printed precision**, and all checked provenance hashes match.

I made no file changes, submitted no jobs, ran no generator, and used no scratch storage.

Below, `F` means `/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-17-full-experiment-plan/final/`. Line references identify the current files.

| Requested item | Verdict | Evidence |
|---|---|---|
| 1. Independently recompute ≥40 numbers | **CORRECT** | 234 explicit numerical comparisons passed, plus independent counts and classifications; inventory below. |
| 2. Provenance, paths, hashes, hand-typed results | **CORRECT** | 471 direct JSON records match their source values; 60 referenced disk-file hashes, both historical H5 records’ blob hashes, 33 destination PNG hashes, and 20 git identifiers match. No additional hand-typed measured-result counterexample found. |
| 3. Verdict fidelity | **CORRECT** | All 36 exported verdict rows match the report. Owning-lane qualifications are retained at `F/report.md:33–70`. |
| 4. Retractions | **CORRECT** | Named withdrawals and corrections remain explicit at `F/report.md:373–390`, with source-section pointers at `:399–409`. No withdrawn claim was revived as valid evidence. |
| 5. Brief coverage | **CORRECT** | Required sections, dimensional/family verdicts, both readings of k, plots, P1g, deferrals, ledger, lane list, Mermaid and glossary are present. Pending lanes remain pending. No prohibited comparison or pre-September speed-up found. |
| 6. Generator correctness | **CORRECT for the current inputs** | Primary-scope filtering, batch separation, ratio direction, population separation, growth-window counting and ledger summation agree with independently read data. Syntax parses in memory. |

**Independent numerical checks**

These checks read lane JSONs directly; I neither imported nor executed `make_final.py`.

| Report evidence | Independently recovered result |
|---|---|
| `F/report.md:80–87` — kernel fits | All 24 fitted exponents and interval endpoints match. Exponents: 2D **1.174032, 1.084166, 1.924877, 1.235084**; 3D **1.011199, 1.031497, 1.326162, 1.034708**. |
| `:111–118` — mesh cost | All eight max/min ratios recomputed from mesh medians match, as do eight upper interval endpoints. |
| `:122–125` — 3D timing | All 16 millisecond conversions and four max/min ratios match. |
| `:138–147` — accuracy/qualification | All 16 percentage conversions match; qualification counts **98/112**. |
| `:41–44` — extended accuracy verdicts | H-acc-N: **21 pass, 4 inconclusive, 5 unqualified/inconclusive**. Through r=384: **14 monotone, 10 inconclusive, 3 not monotone**. Through r=512: **20 not monotone, 7 inconclusive**. |
| `:160–179` — order/front/H2q | Seven temporal-order values match; width counts **18,25,32,6,2**; H2q ratios **1.1491377674, 2.9528017141, 9.8741349644**. |
| `:188–210` — FOM comparisons | **185 configurations, 170 admissible**. At each 2D mesh: UB **16 ≥2×, 2 <2×, 3 unresolved**; CC **19 <2×, 2 unresolved**; PS/union **21 <2×**. |
| `:55,61` — union extrema | 2D **0.0493062645–0.7055166351**; 3D **0.0078554128–0.1582392454**. |
| `:218–246` — POD baselines | Independent interval predicates recover **8/8, 8/8, 2/4 qualified 3D, 5/5 diagnostic 3D** dominations. All 60 displayed error/ratio values match; time ratios reproduce from exported per-case medians. |
| `:262–281` — quadrature | Finite qualifying counts **9216,4181,24577,16384,28657,32768,32768** match. “None” retains the all-required-draws interpretation. |
| `:288–319` — test space/budget | Blocked square-cell verdict agrees. Shared-subset labels count **7 same, 3 inconclusive**. R256/GAL/CN/1 has **e_P/e_P/e_P** on the shared subset. |
| `:325–331` — horizons | Qualified/stable/no-growth counts match; T=1 retains **11 provisional W3/W2 labels without verdicts**. |
| `:338` — head comparison | **1.4402459%, 10.1713102%, 0.26480846×, 1.0953379%, 0.59100019×** reproduce. |
| `:357–362` — P1g | All **42** displayed measurements match, including separate E0 and first-order spreads. |
| `:373–375` — retractions | Historical H5 **13.6447877× / 17.7057367×** and all six withdrawn-arm tight-solve distances match. |
| `:431–444` — ledger | All ten spends, caps and entry counts match. Totals: **107.4971 GPU-h** finished; **144.8293 GPU-h** including pending lanes. |
| `:453–462` — lane list | All **20** head/mirror identifiers match local git refs. |

The optional wildcard provenance fields `state` and `reaches_2x` are absent only in the documented alternative branches. Their defaults and counting semantics are described in `F/make_final.py:373–383`; I found no unresolved source-path defect.

**Previous findings rechecked**

| Previous WRONG / NEEDS-RESTATEMENT item | Current verdict and evidence |
|---|---|
| QN `k4` incorrectly described as iteration count | **CORRECT — fixed.** Refresh intervals and fixed sweeps are distinguished at `F/report.md:473`; consistent with speed3d `qn3.py:139,168`. |
| Error-budget table mixed populations silently | **CORRECT — fixed.** Own-population and shared-subset columns are explicit at `F/report.md:297–312`. |
| Missing dominant/front-order definitions | **CORRECT — fixed.** `F/report.md:511–512`, consistent with budget `report.md:452,460`. |
| Invalid growth path; ambiguous derived paths, batch filters and optional keys | **CORRECT — fixed/documented.** Current provenance checks pass. |
| Literal success counts, worst-case identifier and historical quotation numbers | **CORRECT — fixed or numerical specificity removed.** `F/report.md:210,375,385–386`; quotation appendix replaced by source pointers. |
| Universal E0 assertion and P1g reference ambiguity | **CORRECT — fixed.** Exceptions at `F/report.md:3,476`; distinct spread columns at `:355`. |
| Overbroad baseline population and mesh-cost scope | **CORRECT — fixed.** Tested population and evolution-only scope at `F/report.md:25,33`. |
| H2q comparator qualification and timing scope | **CORRECT — fixed.** `F/report.md:66–70,491`. |
| Head exponent exception; baseline budget qualifiers | **CORRECT — fixed.** `F/report.md:39,97,62–65`. |
| Missing corrections, including head quadrature policy | **CORRECT — included.** `F/report.md:381–390`; Gauss 96² is explicitly a policy outcome, not universal. |
| Union-only figure mislabeled per-family; broken table pipes | **CORRECT — fixed.** `F/report.md:201–208,225–246`. Both summary figures were visually inspected. |
| T=1 growth conflation; instantaneous ledger implication | **CORRECT — fixed.** Separate windows at `F/report.md:323–331`; terminal-job accounting at `:442`. |
| Family “none” interpretation; blanket slower-kernel claim | **CORRECT — fixed.** `F/report.md:258,89`. |
| Q1 GAL exception, Q2 successor convergence, reproduction gate | **CORRECT — fixed.** `F/report.md:480–482`. |
| Missing glossary terms and mathematical formatting | **CORRECT — addressed.** `F/report.md:468–519`. |
| Unregistered pending-lane heartbeat metadata | **CORRECT — fixed.** Registered reads at `F/make_final.py:954–957`. |

Both priority-4 EXPORT files remain absent. Their sections at `F/report.md:345,349` contain explicitly labeled operational status, not invented scientific results. This verdict covers the current compilation; it does not certify future pending-lane exports or their automatic rendering.