**NOT CLEAN.** Read-only review with in-memory fault probes; file and plot writes disabled. No files modified.

| Round-1 item | Verdict | Finding |
|---|---|---|
| 1 | **STILL-WRONG** | Hash/job/commit/attempt binding and matched-speedup suppression are fixed. Failed-audit configurations still enter ordinary plots: the probe returned `eligible=False`, yet plotted two FOM/ROM series. |
| 2 | **RESOLVED** | Matching arithmetic remains correct; accepted calibration is now required. The bound audit checks inventories and timing validity. |
| 3 | **STILL-WRONG** | Unverified configurations are excluded from plot series, but 3D reference errors remain in the ordinary table without explicit failure-diagnostic labelling. A `15/16` verified probe printed normal worst/median errors. Both sections’ plots also ignore section-level audit eligibility. |
| 4 | **RESOLVED** | A13 reference-limitation labels and reference-dependent hypothesis labels are present, including the prohibition on ranking schemes using these errors. |
| 5 | **RESOLVED** | Uncertainty uses the actual case ID, checks finite components, and displays resolved-only anchor medians with explicit labels. |
| 6 | **RESOLVED** | The five previously identified missing-data crashes are addressed. One additional crash remains below. |
| 7 | **RESOLVED** | Order claims now require the registered 16-case cohort and seed, completeness, and a passing bound audit. |
| 8 | **RESOLVED** | Hypothesis candidate lists are suppressed when ineligible; analysis JSON retains section eligibility. Bound audits validate comparator inventories and timing. |
| 9 | **RESOLVED** | The identified explanations are scoped to 2D; corresponding 3D definitions and local provisional labels are supplied. |
| 10 | **RESOLVED** | Out-of-band FOM-CN orders receive the required “pre-asymptotic (consistent with a stiff-mode transient)” label. |

Remaining masking defects are at [report_extra.py:151](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-time2/experiments/jcp-time2/report_extra.py:151), [259](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-time2/experiments/jcp-time2/report_extra.py:259), and [307](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-time2/experiments/jcp-time2/report_extra.py:307). Plot eligibility must include the section gate; unverified table errors need masking or explicit failure-diagnostic presentation.

**New blocking defect:** [report_extra.py:107](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-time2/experiments/jcp-time2/report_extra.py:107) formats every FOM order with `:.2f`. An unavailable order represented by `None` raises `TypeError`, even with a failed audit. The preceding finite-value check handles this value, but the row renderer does not. Reproduced in memory; render unavailable orders as `—`.