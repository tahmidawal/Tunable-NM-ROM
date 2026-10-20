**Overall: NEEDS-FIX — numerical checks pass; presentation fixes remain.**

| Previous must-fix | Verdict | Evidence |
|---|---|---|
| Dominance formula | RESOLVED | `report.md:269` matches DESIGN §6. |
| Stopping disclosure | RESOLVED | Report and JSONs agree: three rounds, `round_limit_reached: true`, 16 exposed identities, 146 open candidates. |
| Adjacent E0 caveats | RESOLVED | Present beside headline accuracies, tables and accuracy-plot axes. |
| Plot readability | PARTIALLY | Inspected all eight PNGs. R384 clears the axis, but **R128·2 and R128·1 overlap** in the 4096² panel of [front2d_worst.png](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-qualify/experiments/jcp-qualify/plots/front2d_worst.png). R128 annotations still touch markers elsewhere. |
| Glossary/LaTeX | RESOLVED for cited fixes | Lines 494–520 now define BDF2, $\bar\varepsilon_X$, $C_{\rm old}$, distinguish exploratory outcomes, and use LaTeX for the requested formulas. |

Remaining whole-report fix: the correction bullets still use plain-text `2.5·10⁻⁴`, `>` and `ρ` outside math delimiters (`report.md:462–469`). Convert those expressions to LaTeX.

Rechecked **329 result rows and 16 job rows** against source JSONs; EXPORT projections and frozen FRONT hash match. No untraceable empirical number, conflation of qualification/role/front membership, or additional substantive glossary gap found. Mermaid is color-coded; no apparent syntax issue.

No files changed, scripts run, GPU used or cluster jobs launched.