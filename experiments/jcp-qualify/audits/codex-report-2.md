**Overall: NEEDS-FIX — numerical tables check out; presentation fixes remain.**

| Previous must-fix | Verdict | Evidence |
|---|---|---|
| Dominance formula | **RESOLVED** | `report.md:269` now gives the registered, error-scaled tolerance formula. |
| Stopping disclosure | **RESOLVED** | Report summary and 3D section disclose three rounds exhausted and 16 exposed candidates. `analysis.json` records `round_limit_reached: true` and their identities; EXPORT matches, including 146 open candidates. |
| Adjacent E0 caveats | **RESOLVED** | Conditional wording accompanies summary accuracies, each B5 table, and the inspected `unqualified.png`. |
| Plots | **PARTIALLY** | Inspected all eight PNGs. Missing error-versus-N and conditioning-versus-r plots are present; unqualified annotations fixed. In `front2d_worst.png`, the 4096² R384·1 label touches the bottom axis; R128 labels remain crowded against points/lines. Move annotations clear. |
| Glossary/LaTeX | **PARTIALLY** | Requested cohort, solver, rule and dimensional definitions added; temporal-range labels converted. Remaining issues below. |

Remaining glossary fixes in [report.md](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-qualify/experiments/jcp-qualify/report.md:490):

- **Line 515 contradicts the table:** sensitivity is described as “reported without a verdict,” but lines 188–217 print pass/inconclusive outcomes. Distinguish exploratory outcomes from registered verdicts.
- **LaTeX remains unfinished:** formulas/inequalities at lines 494–495, 498, 506, 508–509 and 520 still use Unicode/plain text.
- Define **BDF2**, **$\bar\varepsilon_X$**, and **$C_{\rm old}$**; these remain unexplained.

Independent read-only comparison matched **329 result rows and 16 job rows** to source JSONs. EXPORT’s configuration and 3D projections match analysis; frozen FRONT-2D hash matches. No untraceable empirical number or conflation of numerical qualification, role and front membership found. Mermaid is present and color-coded.

No files changed, scripts regenerated, scratch written, GPU used or cluster jobs launched.