**Seven items are fixed; LaTeX compliance and glossary completeness remain unfinished.** I visually inspected both PNGs. No files were modified and no GPU work was run.

| Item | Verdict | Finding |
|---|---|---|
| S errors provisional inline | **FIXED** | All four 2D tables now say “S worst, PROVISIONAL.” |
| LaTeX in glossary/prose | **NOT FIXED** | The coordinate and logarithm formulas are fixed, but bare mathematical notation remains: `R′ = 512`, `M = 2052`, Unicode powers throughout the narrative, and bare `h` in the glossary. See [report.md:17](experiments/jcp-mechanism/report.md:17), [235](experiments/jcp-mechanism/report.md:235), and [250](experiments/jcp-mechanism/report.md:250). |
| Glossary completeness | **NOT FIXED** | The previously enumerated slope, survival, mesh, LM, setting, bank and gate entries were added. However, `fib121393`, `gref`, own-mesh states, certification draws, and `dev6`/`val32` still lack explicit explanations or mappings. The gap formula’s $N_h$ and $N$ are also not explicitly defined. |
| Clipped A1 caption | **FIXED** | The entire caption is visible and wrapped. Its overlap explanation explicitly applies to the top row. |
| “Every number” wording | **FIXED** | The opening now distinguishes generated measured results from DESIGN protocol constants. |
| Scientific notation for tiny distances | **FIXED** | The answer now reports `4.56e-04 %` and `7.27e-05 %`, consistent with the detailed tables. |
| Finite-window caveat | **FIXED** | The answer explicitly disclaims established asymptotic rates. |
| Causal scope | **FIXED** | The added paragraph limits R to sufficiency within the declared margins and preserves the possibility of placement/training effects elsewhere. |
| A2 plot caption | **FIXED** | Visually readable; identifies dashed curves as C-pos upwind/central, excludes the coarsest mesh from fitting, and states that the nodes curve is not a fit. |

Additional problems:

- **Incorrect new glossary definition:** [“leading norms”](experiments/jcp-mechanism/report.md:257) describes the first error **term**, but the implementation reports leading **coefficient** norms, without the factors $h$ and $h^2$. The agreement statistic also compares error **vectors**, rather than simply their scalar gaps. See [a2_gap.py:200](experiments/jcp-mechanism/a2_gap.py:200) and [217](experiments/jcp-mechanism/a2_gap.py:217).
- **Solver-status wording is inconsistent:** the [glossary](experiments/jcp-mechanism/report.md:247) applies provisional status to any label lacking sensitivity testing, while the 2D X0 rows say “no” despite the explicit absence of sensitivity reruns. DESIGN restricts this qualification to R/N; the glossary should say so, and X0 cells should preferably say “N/A.”
- **Previously identified reporting omissions remain:** registered A1 p90 quadrature summaries and per-case mesh spreads are still absent.
- Minor wording: “All slopes … on fixed reached states” also grammatically includes the immediately preceding manufactured-control slopes. Restrict that sentence to reached-state slopes.

Remaining WRONG: incomplete LaTeX compliance; incomplete glossary; incorrect leading-norm definition; inconsistent solver-provisional definition.