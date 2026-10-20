**NOT CLEAN.** The core comparison and order formulas are correct, but eligibility, masking, labelling, and missing-data handling need fixes before publication.

Read-only audit completed, including in-memory fault probes with file and plot writes disabled. No files changed.

1. **WRONG — failed audits still permit FOM matches and speedups.**  
   [report_extra.py:41](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-time2/experiments/jcp-time2/report_extra.py:41), lines 84–127: `ok` changes the introductory text and returned metadata; it does not gate comparisons or plots. A synthetic failed-audit, one-case result still produced a **2×** matched speedup while the prose claimed 38 cases. Both sections also bind audits only by job ID and commit, allowing stale audits of different result bytes.

   **Fix:** require a passing audit bound to the exact result hash and attempt, complete results, and the registered cohort before emitting claims. Otherwise emit explicitly unavailable diagnostics. Preserve that eligibility in the analysis JSON.

2. **CORRECT — the matched-comparison arithmetic implements A9.4, conditional on eligible inputs.**  
   [report_extra.py:84](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-time2/experiments/jcp-time2/report_extra.py:84): candidates satisfy `FOM worst ST <= ROM worst ST`; `min(...ms)` selects the cheapest; speedup is `FOM ms / ROM ms`. Both timings and accuracy records come from this job. `_timing` uses median candidate times and pooled flanking baseline times.

   Calibration exclusion currently depends on `fomrun.py` omitting unresolved configurations. **Fix:** explicitly require an accepted calibration status, complete matching case inventories, and finite positive timings in the report’s eligibility filter.

3. **WRONG — unverified trajectories appear as ordinary accuracy/cost results.**  
   [report_extra.py:50](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-time2/experiments/jcp-time2/report_extra.py:50), lines 104–113 and 192–202, 244–245: reference-error aggregates include unverified rows; both plots draw them without failure markings. Verification counts in tables do not qualify standalone plotted points.

   **Fix:** mask ineligible configurations from comparison plots and accepted-result tables. Retain their raw errors in explicitly labelled failure diagnostics; do not silently recompute cohort-worst errors over only successful cases.

4. **WRONG — A13’s required reference-limitation labels are absent.**  
   [report_extra.py:144](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-time2/experiments/jcp-time2/report_extra.py:144), lines 178–182, 236–239, 255: “PROVISIONAL” appears in the introduction and plot axis, but **“reference-limited (BE reference at Δt₀/4)”** appears nowhere. H1-3D/H2-3D are not labelled reference-dependent.

   **Fix:** put the A13 label beside the vendor reference errors, reference-error table columns, and reference-error plot; label both hypotheses **reference-dependent, provisional**. State that these errors cannot rank first- against second-order schemes.

5. **CORRECT formula; WRONG case association and unresolved presentation.**  
   [report_extra.py:156](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-time2/experiments/jcp-time2/report_extra.py:156): `unc()` computes the requested sum, requires all four anchor/replay trajectories verified, and resolution uses `>= 3U`.

   However, line 193 uses `enumerate(L)` after missing cases have been removed. A gap can associate a trajectory with another case’s uncertainty. Also, unresolved discrepancies enter `anc` and receive ordinary filled plot markers. The probe retained an anchor median with **0/1 resolved**.

   **Fix:** use `x['case']`; validate every uncertainty component and anchor value; visibly distinguish unresolved discrepancies. Keep raw distances only with explicit unresolved labelling.

6. **WRONG — reproducible missing-data crashes.**  
   [report_extra.py:20](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-time2/experiments/jcp-time2/report_extra.py:20):

   - Missing FOM timing: `None` formatted with `:.1f`/`:.3f`, before matching.
   - An invocation with `ratio=None`: `_timing()` raises `TypeError`.
   - Missing 3D timing: `A_ms=None` formatted at line 179.
   - An anchor uncertainty component equal to `None`: addition raises at line 163.
   - Empty vendor rows: `ve.max()` raises.

   **Fix:** validate timing and metric inputs; render unavailable values as `—` and exclude them from claims. Incomplete invocations should invalidate timing eligibility, not silently shrink the timing sample. An entirely absent anchor run is already handled safely by `unc()` returning `None`; preserve that behavior.

7. **CORRECT — 16-case order-claim logic matches `Setting.order_claim`.**  
   [report_extra.py:165](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-time2/experiments/jcp-time2/report_extra.py:165), lines 204–230: the primary and adjacent triples, six-trajectory verification, strict `10×` sensitivity tests, order bands, and 80% tests agree. For 16 cases, the thresholds are **13 primary-valid** and **8 both-valid**. Below eight, the qualified primary claim is allowed; at eight or above, a failed adjacent-band test withholds it.

   **Fix needed around eligibility:** require the registered 16-case cohort for reportable claims. Reading arbitrary `cohort_count` currently allows small diagnostics to generate claims.

8. **CORRECT comparator and inequalities; incomplete output gating.**  
   [report_extra.py:232](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-time2/experiments/jcp-time2/report_extra.py:232): H1-3D/H2-3D use `vendor_rows`, hence deployed fixed-sweep BE, and H2 uses candidate/deployed paired ratios. The accuracy and step-factor inequalities are correct.

   **Fix:** require full eligible comparator/candidate inventories and finite timing. On audit failure, suppress candidate lists and encode hypotheses as unavailable in the analysis JSON; currently qualifying lists survive despite the prose saying “unavailable.”

9. **NEEDS-RESTATEMENT — the call sites make global 2D explanations misleading for 3D.**  
   [make_report.py:549](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-time2/experiments/jcp-time2/make_report.py:549): both additions precede globally worded explanations describing six timing cases, 257² anchor nodes, and hollow unresolved markers. The 3D section uses four timing cases, 65³ nodes, and no hollow markers.

   **Fix:** scope those statements explicitly to 2D and supply corresponding 3D definitions. Add section-local provisional labels to the new tables and matched-comparison output.

10. **NEEDS-RESTATEMENT — FOM-CN’s required diagnostic label is missing.**  
    [report_extra.py:65](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-time2/experiments/jcp-time2/report_extra.py:65): “no (A11.1: reported only)” does not implement A11.1’s required out-of-band label.

    **Fix:** label out-of-band CN orders **“pre-asymptotic (consistent with a stiff-mode transient)”** and avoid implying established second order.