The recomputed numbers pass, but five reporting fixes remain. No files were changed.

The previous audit’s three fixes are present: test pairs carry search qualifications; all nine test source paths identify the correct files; plotting uses retained A3 cells, fixed arm colours, qualification markers and deferred A4 panels.

Verified against the collected JSONs:

| Check | Result |
|---|---|
| Export | `stage: final`; nine retained cells; seven deferred cells |
| Cell tables | 186 exported entries agree with their sources |
| SPEED-STOP | **NOT FIRED**: 95 matches, 80 ≥ 0.5, 27 missing/FOM-censored; maximum 2.14378237 |
| BANK-STOP | **NOT FIRED**: 3 above, 4 below, 8 unaffordable, 3 missing |
| Crossings | 60 records, 156 points; brackets reproduce |
| A11 | All 27 rows reproduce; FFT/dense ratios 1.542507–2.017621 |
| A16 | All 28 ratios reproduce from timing arrays |
| Test-once | 164 pairs, 142 matched, including 9 with incomplete FOM searches; all 53 printed transitions reproduce |
| GPU-hours | 48.9677778; ledger rounding and report’s 49/87 agree |

Conditional references are labelled estimates, deferred cells are not presented as results, and test-once is explicitly confirmation of frozen validation pairs. Neither prohibited historical comparison appears.

Required fixes:

1. **Carry ladder-minimum qualifications into crossings.** [make_report.py:319](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-22-imp-box/experiments/imp-box/make_report.py:319) propagates FOM-search status but omits ladder-minimum labels from both the report and exported crossing points. **136 numerical crossing points** have such qualifications in their corresponding cell entries. For example, the A1 1%-median PODG crossing prints unqualified ratios despite both ladder-minimum labels. Attach those labels to each affected point and qualify the resulting crossing claim.

2. **Escape pipes inside Markdown table cells.** In [report.md:72](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-22-imp-box/experiments/imp-box/report.md:72), `spec|N10|dense|nt5` splits the comparator into extra columns, shifting the bank values under incorrect headings. The test table has the same defect starting at line 352: `podg|0.01|median` splits the final column, so rendered tables can discard the ratios and their restored qualifications. Escape literal pipes in the generator.

3. **Generate the A16 numerical summary from its arrays.** [make_report.py:186](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-22-imp-box/experiments/imp-box/make_report.py:186) hard-codes the measured “10 %” and “±3 %” statements, contrary to the requested provenance rule. The largest reduced-query ratio is **1.0314016**, slightly outside a literal ±3% claim. Generate an appropriately rounded range and identify the later A1/A2 cells accurately when describing the timing implementation.

4. **Complete the glossary and mathematical formatting.** [report.md:375](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-22-imp-box/experiments/imp-box/report.md:375) does not define several table columns and technical terms, including \(r_\times^{\rm query}\), cheapest form, `tensor-gal`/`tensor-lspg`/`point4-gal`, A16 kernel names, eligibility, FFT, POD and ETDRK4. Explain these in plain language and use LaTeX for mathematical expressions outside tables/code.

5. **State the A11 finding prominently.** The dedicated table is correct, but its introduction only explains how to read the ratio. Add a source-generated finding near the opening: the tested dense-transform implementation beats FFT in every reported screen comparison, with FFT taking approximately **1.54–2.02×** as long. Preserve the qualification that these are screening diagnostics, not paired ROM speed-ups.

VERDICT: RESTATE (5 items)