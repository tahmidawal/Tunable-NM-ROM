**NOT-READY** at `71089cea7`.

CPU-only, in-memory regeneration reproduced **report.md, EXPORT.json and all four PNGs byte-for-byte**. I viewed all four PNGs. No repository files changed; no jobs or lab-log writes.

Remaining blockers:

1. **“Every number generated” remains incomplete — prior item 2 partially resolved.** Mesh, configuration counts, resampling counts, training bounds and several thresholds now come from configuration/constants. However, [make_report.py:338](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-regimes/experiments/jcp-regimes/make_report.py:338) still embeds “five output times,” `257 × 257`, the Péclet limit `2`, and bootstrap endpoints `2.5–97.5` directly in prose. Derive these from the scoring/reference/statistical definitions. Regeneration alone does not establish their provenance.

2. **Adjacent qualifications remain incomplete — prior item 3 partially resolved.** The six main regime summaries now include coverage and unresolved counts, and GAL/C2 explicitly reports **2 invalid runs** in the label tables. But [report.md:15](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-regimes/experiments/jcp-regimes/report.md:15) still gives the N0s ratio without coverage/unresolved counts; the following representation-floor summary lacks local conditional/selection qualifications and populations. The distance-trend table also prints A+ statistics without its steepness-confounding label beside that column. Finish these local qualifications.

3. **The glossary is substantially improved but still not complete — prior item 6 partially resolved.** [Glossary](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-regimes/experiments/jcp-regimes/report.md:205) now defines the requested acronyms and several diagnostics. However:
   - Specify the error normalization unambiguously: Euclidean norm divided by the initial-field Euclidean norm, or RMS divided by initial-field RMS. “RMS … divided by the size” leaves that unclear.
   - Explain the printed Q1 fields `distance`, `tight_tighter`, `gcert`, and `offline_agrees`, and the normalization of Q2 successor/check distances.
   - Define separation-table `median`, `median_Am`, and `median_C1` as stiffness statistics, including the in-range comparison population.
   - Distinguish **fractional errors in plots** from **percentages in tables**; the current glossary says errors are reported as percentages without this exception.

Verified resolutions: **item 4’s cited LaTeX expressions are corrected; item 8’s failure conclusion is explicit; item 9’s plot-context fixes are present and readable.** Mermaid, deferrals and A7/A9/A10/A11 disclosures remain present. EXPORT agrees with the analysis and generated report, including GAL/C2’s exclusions. I found no new unsupported performance or causal claim.

Minor style only: remaining prose `log10 error` should use LaTeX; coverage-table missing outcomes render as literal `"\u2014"`.