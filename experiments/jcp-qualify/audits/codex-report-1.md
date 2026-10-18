**Overall: NEEDS-FIX. Numerical results check out; report fixes remain.**

| Item | Verdict | Evidence |
|---|---|---|
| Round-3 passes | **CORRECT** | Independently recomputed from coefficients on CPU: all successor/finer distances and all 64×6 rho values agree with job evidence; offline checks agree. Reproduction errors ≤1.86e-15. |
| All 55 labels | **CORRECT** | Rounds contain 17/20/18 configurations; passes 0/1/3. All labels and round selections match. |
| Qualified-subset fronts/open candidates | **CORRECT** | Recomputed over the 436-candidate universe: 3 median-front members, 2 worst-front members, 146 open candidates. |
| Stopping disclosure | **NEEDS-FIX** | Three rounds exhausted; 16 untested candidates remain exposed on the next front. “Round limit reached” is absent from analysis label, report and EXPORT. |
| Report numbers | **CORRECT** | All 329 result-table rows and 16 job rows match sources. Summary counts, 4.1338 GPU-h and 85/85 controls agree. No hand-typed empirical results found. |
| Qualification/role/front separation | **CORRECT** | Distinct labels and columns maintained. |
| Corrections | **CORRECT** | Combined CN-R fast arms fail Q1; both speed3d H2 winners fail Q2-3D; P1 has no numerical retraction. |
| Order/subset wording | **CORRECT** | Order-2 claims explicitly finite-range; 3D never claims completeness. |
| EXPORT/frozen FRONT-2D | **CORRECT** | EXPORT projections agree. Frozen front’s hash, candidates, labels, errors, timings and fronts match final analysis; SHA starts `a51ac1a250f00d4b`. |

Round-3 raw recomputation:

| Configuration | Successor distance | Finer distance | rho max |
|---|---:|---:|---:|
| W256 lat8192 QN BDF2 .025 k4 | 2.04729e-4 | 8.93834e-6 | .00423928 |
| W256 gl24 FS CN .0125 K1 | 1.50979e-4 | 6.95220e-6 | .00446134 |
| W512 lat16384 QN BDF2 .025 k4 | 2.00785e-4 | 1.79531e-5 | .00282965 |

**Ranked must-fix:**

1. **WRONG — dominance formula.** [Report line 269](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-qualify/experiments/jcp-qualify/report.md:269) prints $\max(2\varepsilon,2\%)$. Registered/code formula is $\max(2\max(\bar\varepsilon_X,\bar\varepsilon_Y),0.02\max(e_X,e_Y))$. Computed fronts are unaffected.
2. **NEEDS-FIX — stopping disclosure.** Explicitly state three-round limit reached; identify/link the remaining exposed candidates.
3. **NEEDS-FIX — adjacent E0 caveats.** Summary accuracy numbers, B5 tables and `unqualified.png` lack nearby conditional wording; the preamble alone is insufficient.
4. **NEEDS-FIX — plots.** Unqualified-panel annotations overlap badly; worst-front labels also collide. DESIGN §9’s error-versus-N and conditioning-versus-r plots are missing.
5. **NEEDS-FIX — glossary/LaTeX.** Missing definitions include `n`, P4096/sensitivity, dev6/val32, FS/QN/K/k and 3D bank/rule identifiers. Existing error/Q2/reproduction definitions describe only 2D. Glossary formulas and `dt0..dt0/4` labels need LaTeX.

Mermaid is present and color-coded. No prohibited historical Burgers speed-up or heat comparison found. No files changed, scratch written, GPU used or cluster jobs launched.