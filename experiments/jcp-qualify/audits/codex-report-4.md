**NEEDS-FIX — plot readability remains.**

| Previous must-fix | Verdict | Evidence |
|---|---|---|
| Dominance formula | RESOLVED | `report.md:269` matches DESIGN §6. |
| Stopping disclosure | RESOLVED | Report, analysis and EXPORT agree: three rounds, limit reached, 16 exposed untested candidates, 146 open. |
| Adjacent E0 caveats | RESOLVED | Present beside accuracy summaries, tables and plot axes. |
| Plot readability | PARTIALLY | Worst-front overlap fixed. In [front2d_median.png](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-qualify/experiments/jcp-qualify/plots/front2d_median.png), the 4096² legend title overlaps **R256·1**; the 256² **R128·5** annotation overlaps y-axis tick text. |
| Glossary/LaTeX | RESOLVED | Previously cited definitions remain; correction expressions now use LaTeX (`report.md:464–471`). |

Inspected all eight PNGs. **329 result rows and 16 job rows match source JSONs**; EXPORT projections and frozen FRONT checksum agree. No additional untraceable empirical number, conflation of qualification/role/front membership, substantive glossary gap or Mermaid problem found.

No files changed; no GPU or cluster work.