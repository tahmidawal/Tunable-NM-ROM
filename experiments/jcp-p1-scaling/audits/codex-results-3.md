**NEEDS-FIX** — audited HEAD `9a077adb0`. Primary numerical results are unchanged from round 2; remaining issues concern provenance and definitions.

| Round-2 item | Disposition | Evidence |
|---|---|---|
| Memory plot inventory/clipping | **RESOLVED** | Now sums only `main`, `qp`, `qpp`: **0.183561, 1.168146, 1.487975, 5.234552 GB**. Inspected PNG; title fits. |
| Provenance statement/hard-coded values | **PARTIALLY** | Offline costs, smoke failures and recording ratios now come from sources. But `analyze.py:25–29` still hard-codes historical values; `:425` copies them into analysis.json. **report.md:3 falsely says these are read from cited files.** |
| Complete EXPORT.json | **PARTIALLY** | Scope/identity limitations, fitted exponents, intervals, classifications, accuracy statistics, transitions, P1e and P1g are present and agree with analysis.json. However, `accuracy_common_population` labels values only `sub16`; it never identifies that population as **dev6 + val32[0:10]**, nor identifies S-eligibility exclusions. |
| Conditional accuracy labels | **RESOLVED** | Explicit opening qualification, inline configuration-table qualification, and conditional accuracy plot labels. |
| Glossary | **PARTIALLY** | Most requested definitions added; Q2 population, fixed-panel M and error normalization corrected. Remaining errors/omissions below. |
| LaTeX | **RESOLVED** | Margin and error-difference formulas now use inline LaTeX. |
| Scientific wording | **RESOLVED** | Monotonicity explicitly means passing every transition’s majority rule; operation-model labels explicitly disclaim causal interpretation. Practical-equivalence and P1g qualifications remain. |
| Plot caption | **RESOLVED** | report.md:227 explains population medians versus paired resolved-case verdicts. |
| Forbidden historical comparisons | **RESOLVED** | Neither prohibited comparison appears. |
| Mermaid omission | **RESOLVED** | No diagram requiring conversion. |

End-to-end reread found these additional glossary problems:

- **Wrong tie definition**, [report.md:257](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-17-jcp-p1-scaling/experiments/jcp-p1-scaling/report.md:257): “otherwise a tie” includes unqualified cases. `p1stats.py:138–139` counts those separately; ties require qualification.
- **Undefined statistic population:** verdict-table “median change” means **median over resolved cases**, while “worst change” covers all paired cases. State this explicitly.
- Still undefined: **P256/P4096**, **W512/W1024**, **lat8192/lat32768**, **gl56/gl64**, and **fused query / A**. The glossary’s evolution-only B/A definition does not define the fused-query column.

No newly changed numerical result was found: analysis.json differs from round 2 only in four added provenance/support groups. No files changed, jobs submitted, or scratch/archive copies created.