**Verdict: ACCEPT-WITH-FIXES.** The accepted numerical findings remain supported, but the report needs corrections before switching to final.

1. **CORRECT — Regeneration reproduces the content.** Scratch regeneration matches `report.md` and `EXPORT.json` except for their commit stamps. All three PNGs reproduce **byte-for-byte**. The stored stamp is `962c150262`; current generator changes are committed at `8473bee46c`. Refresh provenance when regenerating.

2. **CORRECT — Most previously requested fixes are implemented.**
   - **A16:** standard plotted median; fresh-draw confidence bounds; failed draws; median/IQR/range summaries; Sobol diagnostic; conditional-reference caveat; E5 device/comparator scope, unit-to-unit results and **7/12** timing-anchor misses.
   - **A17/A17.4:** corrected front results, explicit R256 retraction, generated data-hash grouping and deployed-rule failure lists, and “different definitive labels.”
   - **E2 3D #7:** fixed-gl64 transitions and all six spread ratios are present.
   - **E2 3D #8:** fixed-gl64 median **and worst-error** spreads and intervals are present. W1024 FS-CN worst error spans approximately **2.95–5.64%**, versus incumbent **5.57%**.
   - **E2 3D #10:** the hash sentence is explicitly 2D; the selection table identifies diagnostic picks and unverified absences. Only seed 2’s W768 FS-CN Gauss selection is certified; none is certified at W1024.

3. **WRONG — Resource table double-counts GPU allocation.** [Report line 529](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-robust/experiments/jcp-robust/report.md:529) reports `e2t2d` as **12.5 elapsed hours / 37.6 GPU-h**. The ledger’s `elapsed_hours` already contains GPU-hours: **12.535 GPU-h**, approximately **4.178 wall hours** on three GPUs. The generator multiplies by three again. Its unrounded table calculation totals **71.4776**, whereas the correct ledger total is **46.4076 GPU-h**. Fix both column semantics and calculation; do not increase the ledger total.

4. **WRONG — Wide-width prose contradicts the report’s own verdicts.** [Line 138](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-robust/experiments/jcp-robust/report.md:138) says wide-width front and selected-rule labels are inconclusive rather than seed-dependent. Both W768 FS-CN lattice front rows are nevertheless labelled **seed-dependent**, with target-unverified qualifiers. Restrict the blanket statement to **selected-rule selection verdicts**, and explain the qualified/diagnostic status of the front labels explicitly.

5. **NEEDS-FIX — E2 3D #9 is only partially resolved.** G-E2.2 now correctly shows W768 lattice disagreement and W768 Gauss’s uncertified numerical agreement. However, its comparison table labels W1024 FS-CN Gauss/lattice **“agrees”** while suppressing their unverified-target status. These are matches of missing selections, not certified agreement on a numerical \(m^*\). Separate numerical matches, certified absences and unverified absences in the **15/16** headline and table. [G-E2.2 table](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-robust/experiments/jcp-robust/report.md:140)

6. **WRONG — The E1 verdict table is malformed Markdown.** All **25 data rows** contain unescaped `|` characters inside configuration names. The header has seven columns; data rows have nine or twelve. A renderer therefore misassigns or drops verdict cells. Escape pipes centrally in `tbl()` or replace configuration separators. [Table start](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-robust/experiments/jcp-robust/report.md:38)

7. **NEEDS-FIX — The glossary is incomplete and contains an incorrect definition.** Missing explanations include IQR, eligible cases, \(r/R/W/M\), rule names, LSPG/GAL, BE/CN-R/BDF2, FS/QN, K1/k4, FOM, CC/PS, S2, H-cost-N(2), bootstrap intervals, conditional Richardson uncertainty, and resource columns. Moreover, H-E1b incorrectly assigns the error tie margin to all rankings: **distance ranks use \(\delta=0\)**; E0-error ranks use the reference-aware margin. Define \(\rho\), \(\varepsilon\), and \(e_{\mathrm{conv}}\) too. [Glossary](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-robust/experiments/jcp-robust/report.md:589)

8. **NEEDS-FIX — The summary is numerically correct but insufficiently qualified.** Add the central interpretations beside its counts:
   - E1 fragility concerns **successor distance**, while tested accuracy remains insensitive.
   - E2 2D selected-rule accuracy is seed-robust, but rule selection varies; deployed R128 fails on every retrain, while R384 qualification varies.
   - E2 3D wide-width selection is sparse; distinguish fixed-gl64 evidence from selected-rule evidence.
   - E5 concerns normalized ratios and limited hypotheses on observed devices with frozen H200-selected comparators.

   These qualifications mostly appear later, but the summary alone invites broader conclusions.

9. **NEEDS-FIX — Some material qualifications and summaries remain omitted.** Explicitly report non-blocking **G-E2.1’s incumbent R256 mismatch** and the unavailable historical replay gates for E1’s supplementary configurations. Also print the 2D transition spread ratios—particularly selected BE 256→384: **2.531, 4.561, 2.585** across the three meshes. The generator silently truncates the 2D spread table to **40 of 60** entries; remove the truncation or disclose and justify the subset. [Generator line 375](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-robust/experiments/jcp-robust/make_report.py:375)

10. **NEEDS-FIX — “Every number generated” is not fully satisfied.** The E1 reading paragraph still hard-codes the empirical **rank 16/16** and associated outcome claims. They are accurate now, but are not generated from the evidence. Derive them or assert them against the source data. The repaired hash grouping and failure-seed lists do satisfy this requirement. [Generator line 107](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-robust/experiments/jcp-robust/make_report.py:107)

11. **NEEDS-FIX — EXPORT is consistent but incomplete as a handoff.** All **183 2D and 129 3D** exported E2 verdicts match their checks; E5 ratios/intervals/agreement match; all **54 ledger entries** match exported job identities. However, export omits the E1 descriptive summaries, 2D error-spread statistics, G-E2.2 comparison and E5 anchor/unit-to-unit details. Its 3D selection records also omit `target_certified`, losing the distinction between certified and unverified “none.” Carry these material qualifications and key numbers explicitly.

12. **CORRECT — Plots reflect their source data.** Visual inspection and regeneration confirm E1 distance/median-error distributions, E2 selected-rule 2D versus fixed-gl64 3D medians, and E5 ratio intervals. Minor presentation fixes: E1 includes an empty diagnostic Sobol category under its “16 random draws per configuration” title; E2’s 3D tick labels crowd together. Label E1 supplementary configurations directly in the figure.

13. **NEEDS-FIX — `--final` leaves contradictory statuses.** A scratch execution changes the opening status and export to final but leaves **three section-level “provisional until … accepted” statements**. Make status handling consistent throughout. Finalization cannot currently be accomplished by the promised status-only switch.

14. **CORRECT — Basic structure and prohibited comparisons.** The report has a level-1 title, status line, Mermaid diagram and LaTeX equations. The incorrect R256 front claim is plainly retracted. No heat POD-128 comparison or pre-September Burgers speed-up is imported.

No repository or lab-log files were changed; no cluster/SSH commands or jobs were run. Audit scratch occupied approximately **884 KiB**, with no arrays copied.