**NEEDS-FIX — plot readability remains.**

| Previous must-fix | Verdict | Evidence |
|---|---|---|
| Dominance formula | RESOLVED | `report.md:269` retains the corrected tolerance and strict-improvement conditions. |
| Stopping disclosure | RESOLVED | Report, analysis and EXPORT agree: three rounds, `round_limit_reached=true`, 16 exposed untested candidates, 146 open. |
| Adjacent E0 caveats | RESOLVED | Conditional-reference caveats remain beside accuracy summaries, tables and plot axes. |
| Plot readability | PARTIALLY | Median-front overlaps fixed. **Regression:** [front2d_worst.png](experiments/jcp-qualify/plots/front2d_worst.png), 4096² panel: GAL `R128·2` and BE `R128·1` annotations overlap. Separate them. |
| Glossary/LaTeX | RESOLVED | Definitions and corrected LaTeX remain; no substantive new gap or Mermaid problem found. |

Inspected all eight PNGs. **329 result rows and 16 job rows match source JSONs**; stopping fields and frozen FRONT checksum agree. No additional untraceable empirical number or conflation of numerical qualification, descriptive role and front membership found.

No files changed; no GPU or cluster work.