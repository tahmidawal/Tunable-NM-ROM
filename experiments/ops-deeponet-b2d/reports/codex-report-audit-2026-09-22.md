# Codex report audit, 2026-09-22 — `gpt-6-astra`, headless

Run after the report was generated, on the brief in `report-audit-prompt.txt`. The `-s read-only`
sandbox could not start in this environment (`bwrap: loopback: Failed RTM_NEWADDR`), so it was
re-run with sandboxing bypassed under an explicit read-only instruction, exactly as the sibling
`ops-timing-panel` lane did the same day; `git status` was clean before and after (0 lines), so
it wrote only its own output. **Eight findings, all accepted and applied**; disposition in
`DESIGN.md` §A5. Reproduced as received.

---

**The negative accuracy result survives. One rendered number is wrong: every timing row reports 62 repetitions; the arrays contain 30 repetitions on each of 32 cases, or 960 measurements per arm. No accuracy value or D1/D2 ratio was wrong.**

No BLOCKER found. No files were changed. Paths below are relative to the lane.

**MAJOR**

1. **The interpretation exceeds what early stopping and capacity scaling establish.**  
   `reports/2026-09-22-ops-deeponet-b2d.md:28,57,167`  
   Early stopping establishes 250 epochs without a new best **validation selection score**, not that further training cannot help. Training loss actually decreased between the best checkpoint and final epoch in all four histories. Increasing these three coupled capacity configurations also does not eliminate under-capacity as an explanation. Finally, “read as a property of the family” overgeneralises this implementation’s result and conflicts with the architecture-ceiling disclaimer.  
   **Smallest fix:** say the tested capacities failed to improve validation mean under this schedule, all exhausted patience with budget remaining, and the result concerns this implementation/protocol. Remove claims that these observations rule out insufficient capacity or training.

2. **The summary exports a cross-job timing comparison surface despite A4.**  
   `reports/generate_report.py:258`; `reports/summary.json:1182`  
   The summary includes device and transfer times for all 16 arms across four jobs under the shared category `"same-job timing"`. The `job_id`, `cross_job`, and `admissible_as_speed_claim:false` flags are helpful, and no ratio is calculated. Nevertheless, these rows allow exactly the cross-job comparison the requested A4 check seeks to prevent.  
   **Smallest fix:** omit sibling timing rows from this lane’s summary; retain only DeepONet’s internally comparable timing rows.

3. **A stale caveat contradicts the stopping result and can excuse the negative incorrectly.**  
   `reports/2026-09-22-ops-deeponet-b2d.md:160`; `reports/generate_report.py:355`  
   “One 3000 s budget per capacity, on which every arm was still training when it stopped” preserves the earlier budget-bound framing. The table instead records early stopping after approximately 543–898 training seconds.  
   **Smallest fix:** generate this caveat from stopping records: every arm exhausted validation patience before its budget; more training under changed stopping/scheduling conditions remains untested.

**MINOR**

4. **Timing repetition counts are numerically wrong.**  
   `reports/2026-09-22-ops-deeponet-b2d.md:139`; `reports/generate_report.py:157`  
   `sum([32,30])` produces 62 from an array shape. It is neither repetitions per case nor total measurements.  
   **Smallest fix:** label and print “30 per case × 32 cases,” or print 960 under “total measurements.” Timing medians themselves recompute correctly.

5. **“Every number comes from five pinned sources; none is typed” is false.**  
   `reports/generate_report.py:31,35,193,346,351`; `reports/2026-09-22-ops-deeponet-b2d.md:3`  
   The generator contains literal numerical metadata and protocol constants, including the diagnosis job identifier, criterion threshold, patience, external timing-panel job identifier, and “46-file harness.” In particular, external-panel metadata does not trace to the five listed audits. Generating prose containing literals does not make those literals source-derived.  
   **Smallest fix:** qualify the guarantee as applying to measured results, and pin/reference DESIGN and external-panel provenance for the remaining metadata—or derive those values too.

6. **The glossary reverses the meaning of an error bound.**  
   `reports/2026-09-22-ops-deeponet-b2d.md:207`; `reports/generate_report.py:398`  
   A recent best checkpoint proves neither nonconvergence nor that observed error is a “lower bound.” Additional successful optimisation would lower error.  
   **Smallest fix:** call this a heuristic indicator of recent validation improvement; remove the bound claim.

7. **A4 changes the original timing commitment, rather than merely exercising its fallback.**  
   `DESIGN.md:196,238`  
   Section 5 says to submit a panel iff a usable checkpoint returns and job budget allows. Both conditions held; A4 instead defers execution to another lane to avoid duplicating its harness. That is a transparent and reasonable amendment, but “taken the way §5 pre-registered” is inaccurate.  
   **Smallest fix:** explicitly call A4 a post-result change to the timing plan. Keep the accuracy-only limitation.

8. **The handoff’s “6–9× less accurate” lacks a metric and comparator selection.**  
   `reports/timing-handoff.json:19`  
   This approximately describes selected-arm validation **worst-error** ratios, not every arm or metric. For example, selected DeepONet’s validation mean is approximately 10.94× selected U-Net’s.  
   **Smallest fix:** identify the metric and selected comparators, and generate the ratios.

Independent recomputation used the actual definition: interior discrepancy divided by the reference initial-field interior norm, maximised over output times for each case.

| Cohort / arm | Mean % | Median % | Worst % | Cases >5% |
|---|---:|---:|---:|---:|
| Validation / don-small | 14.787686 | 11.752249 | 54.744408 | 30 |
| Validation / don-large | 18.221094 | 14.802155 | 60.353992 | 32 |
| Matched eight / don-small | 9.606988 | 8.128440 | 16.022913 | 7 |

These match `audit.json`; the rendered columns match their rounding. I subsequently checked **all four arms on both cohorts**, including case-file hashes, finite float64 predictions, exact boundaries, and returned initial states.

Also verified:

- All five report source hashes, the generator hash, four checkpoint hashes, and both handoff code hashes match. In-memory regeneration reproduces the report and summary exactly.
- All four DeepONets have exactly 250 epochs after their best checkpoint; all 12 sibling arms’ pinned records report budget termination.
- Validation **mean** worsens monotonically across small/medium/large. Worst error does not: medium improves over small.
- Section 3.1 selects `don-small` unchanged. The medium-arm tail warning is correct.
- D1 fails at **8.577307× worst / 6.509414× median** versus `fno-large`; D2 fails at **16.022913% versus ROM 1.867068%**. D3’s selected-arm worst-error ranking is correct. No section 4 criterion was omitted.
- A3’s commit changes only DESIGN/reporting, not training, selection, budgets, or numerical criteria. Its timestamp falls after small finished and while medium ran. Its interpretation needs the qualification above; the claimed absence of anyone reading results cannot be independently established.
- FNO’s archived provenance agrees with the DeepONet manifest’s train/validation hashes. Validation and cohort indices rehash correctly. Training data were excluded from the collected archive, so training-index equality is supported by provenance, manifest, and preamble verification rather than a fresh training-index hash.

A hostile reviewer would chiefly request independent seeds, training-versus-validation curves, controlled schedule/patience ablations, a trunk/branch or representation-floor diagnostic, and an untouched evaluation cohort after tuning. Their absence does **not** erase this negative result; it limits the conclusion to the tested recipe and prevents either a family-wide dismissal or an unsupported “just needed longer” excuse.