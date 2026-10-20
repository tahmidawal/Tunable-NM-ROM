**Verdict: ACCEPT.**

1. **CORRECT — Glossary (#4).** \(m\), \(m^*\), and \(M\) now distinguish quadrature points, the smallest qualifying rung, and test modes. BDF2 is expanded correctly; Richardson’s asymptotic-refinement assumption is explained.

2. **CORRECT — Generated summary (#5).** Direct JSON checks confirm deployed R128 fails on all three retrains, R256 qualifies on all three, and R384 fails on seeds 1–2 but qualifies on seed 3. These facts appear beside the selected-rule accuracy conclusion.

3. **CORRECT — Sobol diagnostic (#6).** All 16 `sob0`–`sob15` draws contribute. Independently recomputed target-distance median/range are **0.00135565 / 0.000843648–0.00300186**; case-median E0 median/range are **0.01093897 / 0.01083119–0.01105845**. Report rounding matches. The diagnostic is explicitly non-verdict evidence, and the empty plot category is removed.

4. **CORRECT — Reproduction.** Scratch regeneration matches committed `report.md` and `EXPORT.json` except for commit stamps (`e07d63db54` versus current `10a21b7ecd`). All three PNGs reproduce **byte-for-byte**.

5. **CORRECT — No previously accepted item regressed.** Analysis inputs are unchanged; every previous export field is unchanged apart from provenance. Report differences are limited to A21’s fixes and the stamp. Previously accepted tables, qualifications, retractions, resource accounting and E5 findings remain intact.

6. **CORRECT — Finalization.** Scratch `--final` changes exactly four report status lines and the export status; substantive content and plots remain unchanged. Repository finalization was not performed.

No repository or lab-log changes, cluster/SSH commands, or job submissions. Scratch usage: **1.3 MB**.