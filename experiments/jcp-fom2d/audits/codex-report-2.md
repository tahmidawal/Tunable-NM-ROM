**NOT READY — reporting/provenance fixes remain.** Numerical spot-checks still support **K2 firing under the registered dev6 workload rule**. No production rerun is indicated.

I inspected all three PNGs and read the report, generator, export and analysis JSONs. No files were changed, no scratch was created, and no cluster commands were run.

References below use [report.md](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-fom2d/experiments/jcp-fom2d/report.md) and [make_report.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-fom2d/experiments/jcp-fom2d/make_report.py).

| Round-1 item | Round-2 status | Evidence |
|---|---|---|
| Adjacent conditional E0 labels | **FIXED** | Summary lines 8–10; Pareto introduction 74; resolution table introduction 165; Figure 1 caption 198; target table introduction 204. |
| Timing scope and upper-bound labels | **FIXED** | Same-mesh summary 17 and table introduction 343 now carry the bound/workload qualifications. Figure 3 caption 424 distinguishes the finite-ladder bound from descriptive bootstrap intervals. |
| Primary versus diagnostic C1 targets | **FIXED** | Lines 204–240 explicitly identify GAL-CN rows. Plot shading now uses only LSPG targets; generator lines 408–413 enforce that filter. |
| Both deferred PS resolutions | **FIXED** | Line 431 explicitly names \(1024^2\) and \(4096^2\), “not run,” scope deferral and no comparative claim. |
| H5 conclusion change | **FIXED** | Line 430 explicitly says the earlier restricted advantage does not establish an advantage over the expanded ladder; “C2 above” is corrected. |
| Remove hard-coded historical measurements | **PARTIAL** | Generator lines 105–112 now scrape **rounded prose from another worktree’s report**, rather than reading run JSON. Literal constants are gone, but the repository’s JSON-source requirement remains unmet. Read the historical JSON or omit the measurements. |
| Complete closing glossary | **PARTIAL** | Substantially expanded, but several requested definitions remain absent or insufficient; details below. |
| Truncated audit verdicts | **FIXED** | Lines 435–455 contain complete first sentences and superseded/current labels; generator lines 86–102 remove the truncation. |

Items already correct in round 1 remain intact: title/status (1–3), LaTeX metric (64), finite-ladder bound direction (3, 429), prepared-input timing disclosure (255, 369), primary/diagnostic verdict separation (263, 312, 339–341), K2 prominence (12, 339), and absence of the prohibited historical comparisons.

The glossary check against the round-1 column inventory is:

| Round-1 table/term group | Status | Remaining issue or supporting lines |
|---|---|---|
| Job table | **FIXED** | Definitions at 483. |
| Gates, controls, smoke and memory | **PARTIAL** | Line 484 distinguishes pass/fire and defines drift/veto. “Smoke,” workload pilot and peak-versus-capacity memory remain unexplained. |
| Family counts, admissibility and overlapping reasons | **FIXED** | Lines 464, 478; overlap explicitly acknowledged. |
| Configuration syntax and solver terms | **PARTIAL** | Lines 461–466 decode configurations. Line 477 names Newton/BiCGStab without explaining their roles in plain language. |
| Resolution and cheapest configuration | **PARTIAL** | The correct “tested minimum, no accuracy match” explanation is at 165, but missing from the closing glossary. |
| Targets, family-joint columns and union | **FIXED** | Lines 461–475. |
| Target variants, stability label and anchor | **PARTIAL** | Lines 481–482 help; “distance” still lacks its norm/normalisation definition. |
| Panel validity, pilots and candidate count | **PARTIAL** | Line 485 repeats “pilot valid” without explaining the pilot; candidate count is not defined. |
| H-comp intervals and no-match status | **PARTIAL** | Lines 488–489 explain intervals and unresolved outcomes; “no match” is not distinguished explicitly from an unresolved timing interval. |
| Same-mesh versus single-statistic matching | **FIXED** | Lines 473–474. |
| Prepared-input query and absolute seconds aggregation | **PARTIAL** | Line 487 explains excluded preparation. It does not define the absolute-seconds aggregation: `c2analyze.py:125` pools valid stage-2 `tB` observations, unlike the normalised case-median cost. |
| Crossover and audit-table terminology | **FIXED** | Lines 490, 493. |
| Richardson, bank, quadrature | **FIXED** | Lines 460, 468. |
| “Bitwise reproducible timings” error | **FIXED** | Line 478 now correctly says repeated screening **outputs**. |

I looked at all three images:

| Plot | Status | Assessment |
|---|---|---|
| Pareto | **FIXED in the report** | Title now qualifies “tested”; caption 198 explains separate single-statistic envelopes, conditional E0 errors, primary-only shading and the omitted point. The image is legible. |
| Cost by resolution | **FIXED in the report** | Caption 202 explicitly says tested admissible configurations with differing accuracy, not an equal-accuracy comparison. |
| Speed-up | **PARTIAL** | Caption 424 supplies workload, prepared-input, upper-bound, interval and offset disclosures. However, curves sharing rank and scheme still cannot be mapped to their individual time steps: referring to the table does not identify which curve is which. Add a time-step encoding or explicit curve labels. |

These caption fixes travel with the report, **not with standalone PNG exports**.

Numerical spot-checks passed:

| Check | Recomputed source value |
|---|---|
| Configurations / admissible | 185 / 170 |
| UB best-median member: median %, worst %, seconds | 0.1495008373 / 0.6874363738 / 13.04130997 |
| CC best-median member | 0.0232327591 / 0.0624053002 / 4.91485137 |
| PS best-median member | 0.0003763507 / 0.0025095260 / 0.05089976 |
| First primary union cell, report 267 | 0.4709015524 [0.4550168041, 0.4866651872] |
| First absolute-latency row, report 373 | ROM 0.0249092942 s; FOM 0.0117232790 s |
| K2 universe / speed-up range | 21; 0.0493062645–0.7055166351 |
| Largest primary union interval upper endpoint at \(4096^2\) | 0.7396832157 |
| Pareto omissions, median / worst panels | 0 / 1 |
| GPU-hours | 4.594 |

Exported C1 fronts and union selections equal `match-final.json`; exported C2 `k2`, `hcomp`, `crossover` and `meshes` equal `c2-analysis.json`. The strengthened claim at report 430 is supported on the tested, qualified primary cells: union interval upper endpoints remain below one at all three meshes.

Two additional inconsistencies need correction:

- **Provenance documentation:** `make_report.py:1–2` still claims every value comes from JSON, contradicting its new `HIST_H5()` implementation. The unpinned sibling-worktree dependency also makes regeneration depend on mutable external prose.
- **PS resolution definition:** report 465 calls \(N\) the number of sine modes while simultaneously saying there are \(N-1\) modes. `psjax2d.py:25` uses modes `1,…,N−1`; define \(N\) consistently as the resolution parameter.

**Final verdict: NOT READY.** Finish the historical JSON provenance, remaining glossary definitions and time-step identification in Figure 3. The audited numerical conclusion remains supported.