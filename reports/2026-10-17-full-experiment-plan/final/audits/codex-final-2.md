OVERALL: NOT-CLEAN

The numerical tables check out, and most previous findings are fixed. Remaining issues concern scope, incomplete retraction coverage, a hand-typed empirical case identifier, and provenance/glossary completeness.

I made no file changes, submitted no jobs, executed no generator, and used no scratch storage.

Below, `F` means `reports/2026-10-17-full-experiment-plan/final/`. Lane paths are relative to the repository root.

| Requested item | Verdict |
|---|---|
| 1. Recompute at least 40 numbers | **CORRECT** — substantially more than 40 checked; no numerical mismatch found |
| 2. Provenance and hand-typed results | **WRONG** — hashes and direct values match, but coverage defects remain |
| 3. Verdict fidelity | **NEEDS-RESTATEMENT** — classifications match; some scope statements remain too broad |
| 4. Retractions | **NEEDS-RESTATEMENT** — major withdrawals preserved; “complete record” is incomplete |
| 5. Brief coverage | **NEEDS-RESTATEMENT** — required sections present; glossary and mathematical formatting remain incomplete |
| 6. Generator correctness | **WRONG** — correction-selection logic misses documented corrections; principal arithmetic is correct |

**1. Independent numerical checks — CORRECT**

I read the lane JSONs directly, without importing or executing `make_final.py`. The following inventory alone exceeds the requested 40 numbers. Values agree at the report’s printed precision.

| Report evidence | Independently recovered values |
|---|---|
| `F/report.md:80–87`; P1 EXPORT `detail/h_cost_r_2d`, `h_cost_r_3d` | 2D exponents and intervals: **1.17 [1.09,1.26]**, **1.08 [1.08,1.09]**, **1.92 [1.90,1.95]**, **1.24 [1.22,1.24]**. 3D: **1.01 [0.985,1.04]**, **1.03 [1.03,1.04]**, **1.33 [1.31,1.34]**, **1.03 [1.03,1.04]**. |
| `F/report.md:111–118`; P1 EXPORT `detail/h_cost_n` | All eight cost ratios and upper interval endpoints agree. Largest upper endpoint: **1.0403859222**. |
| `F/report.md:122–125`; speed3d EXPORT `m1/H41/*/seconds`, converted to milliseconds | C_W1024: **522,522,525,523**; C_old: **60.7,64.9,61,61.4**; W_C_W768: **28.4,28.9,28.7,28.6**; W_C_old: **29.2,31.9,30.3,29.7**. |
| `F/report.md:138–145`; P1 EXPORT `key_numbers/configurations` | Median/worst percentages: **1.10/4.84**, **0.450/5.20**, **0.983/3.32**, **0.209/2.52**, **0.978/3.01**, **0.198/1.78**, **0.979/3.09**, **0.199/2.17**. |
| `F/report.md:147`; qualify EXPORT `configurations` | **98/112** numerically qualified. |
| `F/report.md:47–54`; fom2d EXPORT `c2/hcomp`, primary scope only | At each mesh: UB **16 ≥2×, 2 <2×, 3 unresolved**; CC **19 <2×, 2 unresolved**; PS and union **21 <2×** each. |
| `F/report.md:55,61`; FOM union extrema | 2D **0.0493062645–0.7055166351**; 3D **0.0078554128–0.1582392454**. |
| `F/report.md:62–65`; baseline EXPORT `cells/*/X` | Dominated/total: **8/8**, **8/8**, qualified batch 2 **2/4**, diagnostic batch 1 **5/5**. |
| `F/report.md:66–68`; dims EXPORT `h2q` | **1.1491377674**, **2.9528017141**, **9.8741349644**; comparator Q2 **pass, pass, fail**. |
| `F/report.md:259–276`; dims EXPORT family outcomes | **9216**, **4181**, **24577**, **16384**, **28657**, **32768**, **32768**; reported draw counts also agree. |
| `F/report.md:321–325`; budget EXPORT and `checks/b3-b3d.json` | Qualified/stable/no-growth: **9/10/10**, **9/11/11**, **12/12/12**, **0/4/4**, **11/11/11**. W3/W2 has **11 provisional “no growth” labels without verdicts**. |
| `F/report.md:333`; heads EXPORT `lh/BE/per_K/16` | Head **1.4402459%**; linear r=16 **10.1713102%**, time ratio **0.26480846**; r=128 **1.0953379%**, ratio **0.59100019**. |
| `F/report.md:352–357`; P1 EXPORT `detail/p1g` | All **42** displayed measurements agree, including the newly separated E0 and first-order spreads. |
| `F/report.md:368–370`; pinned combined JSON and qualify analysis | Historical H5 **13.6447877× / 17.7057367×**; six tight-solve distances round to **0.000361, 0.000405, 0.000360, 0.000404, 0.000360, 0.000404**. |
| `F/report.md:571–584`; ten lane ledgers | Finished total **107.4971 GPU-h**; all-lane terminal-job total **142.6529 GPU-h**, correctly printed as **107/143**. All ten entry counts agree. |
| `F/report.md:593–602`; local git refs | All **20** head/mirror commit identifiers match. |

The categorical statements in §§3.7–3.8 also agree with the exported test-space and error-budget outcomes. Both unfinished lanes remain explicitly pending in §§3.11–3.12.

**2. Provenance — mechanically consistent, not fully compliant**

**CORRECT:** All **425 direct JSON-value records** resolve to their recorded values. All **57 referenced disk-file hashes**, both historical H5 git-blob hashes, and all **33 PNG hashes** match. No stale-hash finding occurred during this audit.

**WRONG — empirical case identifier is still hand-typed.** The statement “all on `val32|19`” is literal text in [make_final.py:997](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-17-full-experiment-plan/final/make_final.py:997), printed at `F/report.md:370`. This identifies the observed worst case; it is not a registered cohort-size or design constant. The corresponding data field exists: `configurations/<configuration>/worst_dist_tight_case`, for example qualify `checks/analysis.json:698`.

The current identifier agrees with the data, but it must be read and registered rather than asserted literally.

**NEEDS-RESTATEMENT — optional-key semantics remain undocumented.** The `s3.hcomp.counts` sources at [PROVENANCE.json:1438](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-17-full-experiment-plan/final/PROVENANCE.json:1438) request `state` and `reaches_2x` across every family cell. Union cells lack `state`; unmatched UB/UC cells lack `reaches_2x`. The implementation correctly handles this with `.get()`, but the provenance paths alone do not encode those defaults or branches. This is not an arithmetic mismatch; it prevents literal mechanical replay of that derivation.

**NEEDS-RESTATEMENT — quoted appendix numbers have section-level provenance only.** Entries beginning at `F/PROVENANCE.json:7565` register Markdown sections and line ranges, not individual numeric values and JSON paths. That is useful documentary provenance, but weaker than the report’s “every number” assertion at `F/report.md:5` and the brief’s number-level requirement. Either register those numerical sources individually or explicitly distinguish historical quotations from the machine-recomputed numerical tables.

**3. Verdict fidelity — classifications correct, scope still needs work**

All **36 verdict rows** agree with the owning exports’ classifications or the displayed aggregations. Primary-scope filtering, baseline batch separation, developmental/in-sample labels, upper-bound wording, and H2q comparator qualification are now retained.

Two scope issues remain:

- **NEEDS-RESTATEMENT — 2D mesh-independent cost means evolution cost.** The headline at [report.md:25](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-17-full-experiment-plan/final/report.md:25) says “Cost does not depend on the mesh”; the 2D verdict row at line 33 does not state the evolution-only contract. The owning P1 report explicitly limits H-cost-N to **evolution with internal-state recording**, not the full query: [P1 report.md:183](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-17-jcp-p1-scaling/experiments/jcp-p1-scaling/report.md:183). Initialization/decode are outside that verdict. Section 3.2 supplies some qualification, but the headline and verdict scope should carry it too.
- **NEEDS-RESTATEMENT — P1g historical arms lack their qualification labels.** `F/report.md:354–357` presents H64/H640/HF6765/H96 without saying these are descriptive, unqualified historical arms. P1 EXPORT `detail/p1g/<arm>/meshes/*/qualified` is false for both frozen and nominal versions; the owning report explicitly says “not run (descriptive arm)” for Q2, for example `jcp-p1-scaling/report.md:101–107`. These are legitimate descriptive rechecks, but should not appear indistinguishable from the qualified R384 rows.

**4. Retractions — major withdrawals retained; completeness claim fails**

**CORRECT:** The report preserves the requested major retractions: speed3d H2 qualification, combined H5 scope, combined finalist labels, false 1024² reuse, the coordinator’s 4096² mode-list claim, and post-data tolerance changes. I found no reinstatement of those withdrawn claims as valid qualified results.

**WRONG — the “complete record” selector misses corrections.** [make_final.py:1150](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-17-full-experiment-plan/final/make_final.py:1150) searches for `corrected` but not `correction`/`corrections`. Consequently, the appendix claiming every relevant lab-log bullet at `F/report.md:384` omits:

- [LAB-LOG.md:19124](/home/tahmid/Dev/Tunable-NM-ROM-Claude/LAB-LOG.md:19124), **“Corrections from the audits”**: the P1 run-count correction, quadrature-point IC sampling, finite-ladder interpretation, and replacement of the degenerate three-mesh sign test.
- [LAB-LOG.md:19191](/home/tahmid/Dev/Tunable-NM-ROM-Claude/LAB-LOG.md:19191), describing the ill-posed ECSW tolerance knob, NNLS cycling and replacement, and memory-path fixes. The summary mentions the tolerance knob, but does not preserve the full correction record.

The selector also omits the P1g “Correction of interpretation” bullet at log line 19179, although that particular substance is covered elsewhere. Use an explicit reviewed inventory or a broader selector with an omission check; the current implementation cannot support “complete.”

**5. Brief coverage and presentation**

**CORRECT:** Required major sections are present, including dimensional/family H-comp, mesh-specific H-base, both K2 outcomes, both readings of k, P1g, deferred items, ledger, lane list and merge status. Mermaid is present. No prohibited heat comparison or pre-September Burgers speed-up was found.

**CORRECT:** The new `summary_family_speedup.png` actually contains separate UB, CC, PS and union panels. I inspected it and `summary_union_speedup.png`; their content agrees with the filtered data. The copied union-only figure now has an accurate caption.

**NEEDS-RESTATEMENT — glossary still contradicts the corrected reference scope.** [report.md:615](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-17-full-experiment-plan/final/report.md:615) says `$R_{ST}$ ... scores every error`, despite the explicit first-order P1g and long-horizon exceptions at line 3.

The expanded appendix also introduces unexplained terms such as **NNLS, Givens, Möbius sum, GNAT, IMEX-CG, G-R2 and prefix rules**. Thus the strict requirement to define every term/column is still unmet.

**NEEDS-RESTATEMENT — mathematical formatting is only partly fixed.** The deferred `127^3` instance is fixed, but `65^3` remains plain text in verdict rows 66–68, and the copied log bullets retain non-LaTeX mathematical expressions. Apply formatting consistently, or explicitly obtain an exception for verbatim historical quotations.

**6. Generator logic**

**CORRECT:** Syntax parses in memory. Primary LSPG filtering, baseline batch-1/batch-2 counting, baseline/ours time-ratio direction, union plot selection, W2/W1 versus W3/W2 separation, and ledger summation agree with the data.

**WRONG:** `quote_record()`’s keyword selection produces an incomplete record while describing it as exhaustive, as detailed above.

**Previous audit findings, individually rechecked**

| Previous finding | Current verdict |
|---|---|
| Invalid `growth/W2\|W1/label` path | **CORRECT — fixed**, generator line 877 uses the literal `W2/W1` key |
| Hand-typed “only one 3D cell reaches 2×” | **CORRECT — fixed**, computed at generator line 773 |
| Hand-typed failed-reference count | **CORRECT — fixed**, numerical count removed at lines 897–898 |
| Ambiguous derived paths / omitted batch filters | **NEEDS-RESTATEMENT — partly fixed**; arrays and batch filters added, optional-key semantics remain |
| Universal E0 claim / first-order P1g labeling | **NEEDS-RESTATEMENT — mostly fixed** at report lines 3 and 350; glossary line 615 still contradicts it |
| Overbroad “every qualified 2D configuration” | **CORRECT — fixed**, headline now says eight tested configurations per mesh |
| H2q glossary says both sides qualified | **CORRECT — fixed**, report line 630 |
| Missing H2q timing contract / dev6 applied to 3D | **CORRECT — fixed**, report lines 66–70 |
| Head exponent “not identifiable” | **CORRECT — fixed**, report line 97 retains the 4096² exception |
| Missing H-base budget qualifiers | **CORRECT — fixed**, report lines 62–65 |
| Eight specifically identified missing corrections | **CORRECT — now included**, report lines 398, 480, 494–503, 535 and 547 |
| Retraction consolidation incomplete overall | **NEEDS-RESTATEMENT — remains incomplete**, additional omissions identified above |
| Per-family plot was union-only | **CORRECT — fixed**, new family figure and corrected copied-figure caption |
| Broken POD table pipes | **CORRECT — fixed**, escaping at generator lines 142–150 |
| Incomplete/inaccurate glossary | **NEEDS-RESTATEMENT — improved, still incomplete** |
| Mathematical formatting | **NEEDS-RESTATEMENT — partly fixed** |
| T=1 growth-window conflation | **CORRECT — fixed**, separate columns at report lines 319–325 |
| Ledger implied instantaneous total consumption | **CORRECT — fixed**, terminal-job accounting explicitly stated at report line 582 |

No sampled numerical value requires correction. The report still needs the scope, completeness and provenance fixes above before it merits **CLEAN**.