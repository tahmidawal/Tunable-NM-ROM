OVERALL: NOT-CLEAN

The numerical transcription is largely sound. The blockers are incorrect scope statements, misleading figure/table presentation, incomplete corrections, and provenance gaps.

I made no file changes, ran no jobs, executed no generator, and used no scratch storage.

Below, `F` means `reports/2026-10-17-full-experiment-plan/final/`; lane paths are relative to the repository root.

| Requested audit item | Verdict |
|---|---|
| 1. Independently recompute at least 40 numbers | **CORRECT** for sampled numerical values; labeling exceptions below |
| 2. Provenance and hand-typed results | **WRONG** |
| 3. Verdict fidelity and scope | **NEEDS-RESTATEMENT** |
| 4. Retractions and corrections | **NEEDS-RESTATEMENT** — named major retractions retained, consolidation incomplete |
| 5. Brief coverage, plots, glossary, formatting | **WRONG** |
| 6. Generator correctness | **WRONG** — presentation/provenance defects; principal aggregation arithmetic agrees |

**1. Numerical recomputation — CORRECT**

I independently read lane JSONs without importing or executing `make_final.py`. More than 100 numerical values were checked, including counts, extrema, percentage conversions, medians of per-case time ratios, P1g measurements, and ledger totals. No sampled value differed at the report’s printed precision.

This compact inventory alone includes substantially more than 40 checked numbers:

| Report evidence | Independently recovered values |
|---|---|
| `F/report.md:80–87`, kernel fits; P1 EXPORT `detail/h_cost_r_*` | 2D fitted exponents and intervals: **1.17 [1.09,1.26]**, **1.08 [1.08,1.09]**, **1.92 [1.90,1.95]**, **1.24 [1.22,1.24]**. 3D: **1.01 [0.985,1.04]**, **1.03 [1.03,1.04]**, **1.33 [1.31,1.34]**, **1.03 [1.03,1.04]**. |
| `F/report.md:122–125`, speed3d EXPORT `m1/H41/*/seconds`, converted to milliseconds | C_W1024: **522,522,525,523**; C_old: **60.7,64.9,61,61.4**; W_C_W768: **28.4,28.9,28.7,28.6**; W_C_old: **29.2,31.9,30.3,29.7**. |
| `F/report.md:138–145`, P1 EXPORT `key_numbers/configurations` | Median/worst percentages: **1.10/4.84**, **0.450/5.20**, **0.983/3.32**, **0.209/2.52**, **0.978/3.01**, **0.198/1.78**, **0.979/3.09**, **0.199/2.17**. |
| `F/report.md:147,188` | Qualification **98/112**; FOM configurations/admissible **185/170**. |
| `F/report.md:47–55`, primary-scope FOM counts | At each reported mesh: UB **16 ≥2×, 2 <2×, 3 unresolved**; CC **19 <2×, 2 unresolved**; PS and union **21 <2×** each. |
| `F/report.md:55,61`, union extrema | 2D **0.0493062645–0.7055166351**, maximum upper interval **0.7396832157**; 3D **0.0078554128–0.1582392454**. |
| `F/report.md:215–218`, baseline interval predicates | Dominated/total: **8/8**, **8/8**, qualified 3D **2/4**, diagnostic 3D **5/5**. |
| `F/report.md:224–243`, baseline time ratios | All **20** ratios match the median of the exported per-case ratios. Direction is correctly **baseline/ours**. |
| `F/report.md:321–325`, long-horizon counts | Qualified/stable/no-growth: **9/10/10**, **9/11/11**, **12/12/12**, 3D **0/4/4**, T=1 **11/11/11**; window-label issue below. |
| `F/report.md:333`, head comparison | **1.4402459%**, **10.1713102%**, cost ratio **0.26480846**; linear r=128 **1.0953379%**, ratio **0.59100019**. |
| `F/report.md:352–357`, P1g | All **30** measurements match their JSON fields. For example R384/BE: **0.000523394, 0.0103292 pp, 1.32671e−15, 0.0143259/0.0123724 pp**. Reference-label issue below. |
| `F/report.md:368–370`, retractions | Historical H5 **13.6447877× / 17.7057367×**; six Q1 distances reproduce **0.000361,0.000405,0.000360,0.000404,0.000360,0.000404**. |
| `F/report.md:403–414`, ledgers | Finished total **107.4971 GPU-h**; all-lane exported total **141.0296 GPU-h**, correctly rounded to **107/141**. |

For sections without measured-number tables, I also checked the quadrature-family outcomes (§3.6), blocked square-cell controls (§3.7), error-budget labels and sampling scope (§3.8), and pending status (§§3.11–3.12) against their exports.

All **20** reported head/mirror commit IDs match current local git refs. All **36** verdict rows’ key-number strings match `FINAL-EXPORT.json`.

**2. Provenance — mostly correct mechanically, but not complete**

- **CORRECT:** All **404 direct JSON value records** resolve to matching values. All **49 referenced disk-file hashes** match. Both historical H5 git-blob hashes match. All **32 destination PNG hashes** match provenance, including the copied figures.
- **WRONG:** A derived provenance path names a nonexistent key: `growth/W2|W1/label`. The actual key is `W2/W1`. See [PROVENANCE.json:3413](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-17-full-experiment-plan/final/PROVENANCE.json:3413), generator line 819, and `jcp-budget/checks/b3-b3d.json:6505`.
- **WRONG:** The assertion that every result number is generated has counterexamples:
  - “**Only one 3D cell reaches 2×**” is literal prose at `make_final.py:712`, not rendered from `c3`.
  - “**one validation reference failed its order check**” is literal prose at `make_final.py:831–832`, without a registered count/source.
  
  These are empirical counts, not exempt design constants. Their apparent agreement with current evidence does not satisfy the generation requirement.
- **NEEDS-RESTATEMENT:** Derived records use an undocumented mixture of wildcard, brace, and slash notation. Some descriptions also omit the batch filter actually used—for example the batch-specific H-base denominator records say merely “count of X.” Explicit key arrays and derivations would make these reliably machine-checkable.

**3. Verdicts and scope — numerical labels agree, several statements overreach**

- **WRONG: universal E0-reference claim.** [Report line 3](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-17-full-experiment-plan/final/report.md:3) says every accuracy number uses E0. Two exceptions:
  1. P1g’s spread columns are taken from `first_order_max_pp`, explicitly selected at `make_final.py:882–883`. The lane labels these “first-order” in `jcp-p1-scaling/report.md:122`; the compilation drops that qualifier at line 350.
  2. Long-horizon references beyond \(t=0.25\) are the budget lane’s own references, as stated in `jcp-budget/report.md:3`.
  
  For example the printed R384/BE frozen spread **0.0143 pp** is first-order-reference spread; its E0 counterpart is **0.0120973 pp**.

- **NEEDS-RESTATEMENT: headline baseline scope.** `F/report.md:25` says POD dominates “every qualified 2D configuration.” The evidence is **eight tested X configurations per mesh**, not the complete qualified campaign population. The owning baseline report limits its claim to X (`jcp-hr-baselines/report.md:3,9–10`). Say “all eight tested qualified configurations at each mesh.”

- **WRONG: H2q glossary.** [Report line 455](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-17-full-experiment-plan/final/report.md:455) says H2q qualifies **both sides**. C_W1024 explicitly fails Q2 in the report’s own line 68 and `jcp-dims/report.md:54,749`. Its 9.87× result is qualified-winner versus unqualified deployed comparator.

- **NEEDS-RESTATEMENT: timing contracts.** H2q rows omit the owning lane’s **full-field 65³ mesh-path, H200** scope (`jcp-dims/report.md:52–54`). Also `F/report.md:70` attaches “dev6 timing workload” to every matched FOM speed-up; speed3d Route 4 uses **four cases × two repetitions**, not dev6 (`jcp-speed3d/report.md:521`).

- **WRONG: head exponent prose.** `F/report.md:97` says the exponent is not identifiable from these timings. It **is identified at 4096²**, \(p=0.119\), seed 0 only (`jcp-heads/report.md:412`). The compilation’s own verdict row 39 correctly reports that exception.

- **NEEDS-RESTATEMENT: H-base qualifications.** The table preserves developmental/in-sample and restriction labels, but omits the budget qualifiers adjacent to the verdicts: **2D budget band unresolved; 3D unequal-budget, descriptive**. The lane explicitly carries these with every pair (`jcp-hr-baselines/report.md:24`).

The remaining verdict counts and pass/fail/inconclusive classifications agree with the owning outputs. The major speed3d qualification withdrawal is preserved.

**4. Retractions — major items retained, consolidation incomplete**

**CORRECT:** The report explicitly preserves all six major examples requested: speed3d H2 qualification, narrow H5 scope, combined finalist withdrawal, false 1024² reuse, false 4096² mode-set difference, and several post-data changes.

**NEEDS-RESTATEMENT:** The purported consolidated list at `F/report.md:364–378` omits documented corrections, including:

| Missing correction | Evidence |
|---|---|
| speed3d’s “all replay labels mixed” was wrong | `jcp-speed3d/report.md:439` |
| P1 smoke-check defects and reconstructed/mislabelled acceptance file | `LAB-LOG.md:19180` |
| Budget G-B2 bar restatement after smoke | `jcp-budget/report.md:373` |
| T=1 eligibility using the wrong horizon | `jcp-budget/report.md:379` |
| E0 last-bit/shared-IC correction | `jcp-budget/report.md:380`; `LAB-LOG.md:19286` |
| Reported case durations were cumulative timestamps | `jcp-budget/report.md:382` |
| Dims missing evidence relabelled from “disagrees” to “missing” | `jcp-dims/report.md:848` |
| Dims protected jobs missing from ledger reserve | `LAB-LOG.md:19303` |

These need explicit inclusion or a clearly linked exhaustive correction appendix.

**5. Brief coverage and presentation — WRONG**

- **WRONG: per-family figure is actually union-only.** [Report line 205](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-17-full-experiment-plan/final/report.md:205) captions `fom2d_speedup_vs_mesh.png` as speed-up over **each FOM family**. I inspected the image: its title, axis and curves are **union** speed-ups. The owning caption agrees (`jcp-fom2d/report.md:424`). The required per-family plot is therefore missing; changing the caption alone does not supply it.
- **WRONG: broken POD table.** `make_final.py:134–138` joins cells without escaping pipes. Configuration IDs in [report line 224](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-17-full-experiment-plan/final/report.md:224) onward contain literal `|`, including inside backticks. The seven-column table has **15 parsed cells per 2D row and 18 per 3D row** under GFM table rules.
- **NEEDS-RESTATEMENT: glossary incomplete and partly inaccurate.** Missing explanations include `B_acc`, H64/H640/HF6765/H96, A–B–A, elasticity, ETDRK4, rule-family abbreviations, and F/S/I/M error packages. Line 440 also calls L “grid points” although lane conventions use intervals. H-cost-N’s glossary definition drops the confidence-interval acceptance rule.
- **NEEDS-RESTATEMENT: mathematical formatting incomplete.** Mermaid and LaTeX are present, but expressions such as `127^3` at line 390 and several glossary equations remain plain ASCII/Unicode rather than the required LaTeX.
- **CORRECT:** Required major sections exist; both readings of k, dimensional/family H-comp, mesh-specific H-base, both K2 outcomes, deferred items, ledger and merge-status list are present. Both unfinished lanes remain explicitly pending. No prohibited heat comparison or pre-September Burgers speed-up appears.

**6. Generator logic — aggregation sound; specific defects remain**

**CORRECT:** Primary-scope filtering excludes diagnostic GAL configurations; 3D baseline batches are separated correctly; baseline time-ratio direction is correct; the new union summary actually plots primary union data and the four 3D union values.

**NEEDS-RESTATEMENT:** At `make_final.py:825–827`, the T=1 row counts **W2/W1**, then appends “W3/W2 provisional.” Those are separate results. W3/W2 exports explicitly say **“no verdict … provisional: no growth.”** Print separate columns and compute each from its own field.

**NEEDS-RESTATEMENT:** Ledger totals faithfully sum `spent_gpu_hours`, but robust’s field counts terminal jobs only. Its ledger also records **1.3861 + 0.5981 GPU-h** already elapsed in running jobs. Label the total as the lanes’ **accounted/terminal-job spend**, rather than implying an exhaustive instantaneous consumption total.

The generator passes an in-memory syntax check. I did not run it because it writes outputs.