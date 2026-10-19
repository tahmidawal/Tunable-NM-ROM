ACCEPT-WITH-CORRECTIONS

**One low finding remains:** [report.md:45](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-dims/experiments/jcp-dims/report.md:45) puts `M=4r` inside backticks rather than LaTeX; repeated at line 596. Render the mathematical expressions as `$M=4r$` and `$M=2048$`, keeping rule identifiers in backticks. The restored LMF heading should likewise use `$(r,M)=(384,1536)$`.

Verified:

- LMF setting correctly restored.
- Viewed H2q PNG: E0-provisional qualification is visible.
- Screening/confirmation counts derive correctly from configs: 4 × 2 and 8 × 3.
- Rule-name formulas use LaTeX.
- Full regeneration produces **byte-identical report, EXPORT, and all ten PNGs**. No other regression found within scope.

No repository files changed or jobs submitted. Scratch usage: approximately 4 MB.