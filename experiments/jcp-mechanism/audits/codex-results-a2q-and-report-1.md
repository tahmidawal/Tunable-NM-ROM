**A2’s numerical conclusions are CORRECT. The report and plots still have presentation and completeness defects.** No files were modified; no GPU was used.

I independently recomputed A2 statistics with NumPy, without calling `make_report.py`. I also independently reconstructed the manufactured field, derivatives, quadrature, and stencil sums using separable NumPy formulas.

**I. A2 raw results**

**CORRECT — provenance and execution.** All three output checksums pass. Job 5016189’s log records `jax_backend=gpu`, `x64=True`, `precision=highest`, `A2 COMPLETE`, and `ALL-DONE`; stderr is empty.

**CORRECT — median/worst gaps.** Every reported fixed-population gap matches the independent calculation at the report’s displayed precision. Below, entries are **median / worst**, not percentages.

| Panel | Mesh | Upwind | Central | Nodes |
|---|---:|---:|---:|---:|
| 3D R512 | 33 | 2.690970e-1 / 3.884671e-1 | 2.121991e-2 / 4.990729e-2 | 6.700782e-5 / 2.396023e-3 |
| | 65 | 1.370844e-1 / 1.964743e-1 | 5.370796e-3 / 1.292915e-2 | 3.973189e-6 / 1.243571e-4 |
| | 129 | 6.908315e-2 / 9.855850e-2 | 1.346896e-3 / 3.261909e-3 | 2.446787e-7 / 7.474548e-6 |
| | 257 | 3.460639e-2 / 4.932903e-2 | 3.369878e-4 / 8.173485e-4 | 1.522969e-8 / 4.626017e-7 |
| 3D R256 | 33 | 2.695685e-1 / 3.858236e-1 | 2.102450e-2 / 4.693416e-2 | 4.944481e-5 / 1.258326e-3 |
| | 65 | 1.371680e-1 / 1.950733e-1 | 5.317462e-3 / 1.202365e-2 | 2.986439e-6 / 6.793769e-5 |
| | 129 | 6.909621e-2 / 9.784384e-2 | 1.333260e-3 / 3.024567e-3 | 1.837011e-7 / 4.113212e-6 |
| | 257 | 3.467550e-2 / 4.896892e-2 | 3.335594e-4 / 7.573155e-4 | 1.143659e-8 / 2.551001e-7 |
| 2D acc | 128 | 6.845449e-2 / 2.007988e-1 | 2.040903e-3 / 5.750527e-2 | 3.493570e-4 / 6.853418e-2 |
| | 256 | 3.436822e-2 / 8.611544e-2 | 5.116909e-4 / 1.881418e-2 | 1.039465e-4 / 1.344273e-2 |
| | 512 | 1.721518e-2 / 3.763116e-2 | 1.279537e-4 / 4.903392e-3 | 9.607914e-6 / 9.657944e-4 |
| | 1024 | 8.616216e-3 / 1.757502e-2 | 3.199224e-5 / 1.247078e-3 | 2.489808e-7 / 2.791759e-5 |
| | 2048 | 4.310314e-3 / 8.498799e-3 | 7.998847e-6 / 3.133646e-4 | 1.884922e-9 / 7.390157e-7 |
| | 4096 | 2.155704e-3 / 4.180276e-3 | 1.999769e-6 / 7.844173e-5 | 1.359030e-10 / 5.997564e-8 |
| 2D fast | 128 | 6.859430e-2 / 1.586607e-1 | 2.047450e-3 / 4.880941e-2 | 2.934445e-4 / 7.710910e-2 |
| | 256 | 3.439319e-2 / 6.812501e-2 | 5.127139e-4 / 1.554099e-2 | 1.155827e-4 / 1.625572e-2 |
| | 512 | 1.722537e-2 / 3.032054e-2 | 1.281489e-4 / 3.950128e-3 | 6.969251e-6 / 1.363209e-3 |
| | 1024 | 8.618444e-3 / 1.477770e-2 | 3.203923e-5 / 1.005401e-3 | 2.111524e-7 / 4.400716e-5 |
| | 2048 | 4.311147e-3 / 7.389585e-3 | 8.010025e-6 / 2.525700e-4 | 2.284713e-9 / 4.089334e-7 |
| | 4096 | 2.156050e-3 / 3.694958e-3 | 2.002520e-6 / 6.321646e-5 | 2.215647e-10 / 4.305362e-8 |

The separately reported own-mesh median gaps also match; those populations are correctly excluded from the slope fits.

**CORRECT — screening and survival.** I applied both strict inequalities, separately for every state at every window mesh:

\[
g(h;c)>100\rho_{\mathrm{check}}(c),\qquad g(h;c)>10^{-12}.
\]

| Panel | Upwind survivors | Central survivors | Nodes survivors | Nodes verdict |
|---|---:|---:|---:|---|
| 3D R512 | 200/200 | 200/200 | 6/200 = 3% | Unresolved |
| 3D R256 | 200/200 | 200/200 | 10/200 = 5% | Unresolved |
| 2D acc | 300/300 | 300/300 | 1/300 = 0.333% | Unresolved |
| 2D fast | 300/300 | 300/300 | 1/300 = 0.333% | Unresolved |

Nodes survivors at successive window meshes are respectively `[188,114,6]`, `[191,150,10]`, `[300,300,300,64,1]`, and `[300,300,300,62,1]`. Every nodes population fails the 25% requirement.

**CORRECT — all three slope statistics and bars.** Fits use precisely the requested windows and spacings.

| Panel | Stencil | Slope of median gap | Slope of maximum gap | Median per-state slope | Registered bar |
|---|---|---:|---:|---:|---|
| 3D R512 | Upwind | 0.992977 | 0.996916 | 0.994234 | Pass [0.8,1.2] |
| | Central | 1.997184 | 1.991766 | 1.997281 | Pass [1.7,2.3] |
| 3D R256 | Upwind | 0.991978 | 0.997039 | 0.994276 | Pass |
| | Central | 1.997361 | 1.994418 | 1.997474 | Pass |
| 2D acc | Upwind | 0.998750 | 1.087580 | 0.999211 | Pass |
| | Central | 1.999828 | 1.977983 | 1.999765 | Pass |
| 2D fast | Upwind | 0.998971 | 1.044584 | 0.999213 | Pass |
| | Central | 2.000026 | 1.985027 | 1.999822 | Pass |

The bars apply to the slope of the median gap. No registered nodes bar exists.

**CORRECT — independent-family check on every state set.** All maxima are below \(10^{-2}\).

| Panel | Fixed | Own coarse | Own middle | Own fine |
|---|---:|---:|---:|---:|
| 3D R512 | 4.493581e-4 | 4.821863e-5 | 6.080727e-5 | 6.922684e-5 |
| 3D R256 | 6.444827e-5 | 2.713024e-5 | 3.370319e-5 | 3.792247e-5 |
| 2D acc | 2.269628e-5 | 2.591771e-6 | 2.626427e-6 | 3.901162e-6 |
| 2D fast | 8.192899e-6 | 1.989417e-6 | 2.313586e-6 | 2.376138e-6 |

Own meshes are 65/129/257 in 3D and 256/1024/4096 in 2D. Fixed-population Gauss-check maxima are \(7.268802\times10^{-8}\), \(1.178039\times10^{-8}\), \(4.320803\times10^{-8}\), and \(3.240186\times10^{-8}\).

**CORRECT — manufactured controls actually behave as designed.**

| Quantity | 3D | 2D | Requirement |
|---|---:|---:|---|
| Upwind slope | 1.001424 | 1.000165 | Within 0.15 of 1 |
| Central slope | 1.999010 | 1.999970 | Within 0.15 of 2 |
| Finest upwind leading-term discrepancy | 0.00576927 | 0.000486799 | ≤0.1 |
| Finest central leading-term discrepancy | 0.000263949 | 0.00000372256 | ≤0.1 |
| Relative upwind leading norm | 5.495769 | 5.144444 | ≥0.001 |
| Relative central leading norm | 8.122000 | 10.257872 | ≥0.001 |
| C-neg-a gap, every mesh | 0.01 | 0.01 | Constant injected defect |
| C-neg-a slope | Approximately zero | Approximately zero | Absolute value <0.05 |
| Central×1.01 slope | −0.058204 | −0.003033 | Descriptive only |

Independent manufactured reconstruction agrees with the stored leading-term discrepancies to absolute differences below \(9.5\times10^{-13}\) in 3D and \(6.1\times10^{-10}\) in 2D. The latter difference concerns a discrepancy already only \(3.7\times10^{-6}\), with no effect on acceptance.

The positive control recovers the expected orders and nonzero leading coefficients. The negative control produces the constant error plateau, rather than a spurious convergent slope. C-neg-a tests normalization and fitting; it does **not** independently validate stencil assembly.

Manufactured nodes slopes are 4.021345 and 4.004397. Additional NumPy Gauss comparisons gave target discrepancies \(8.17\times10^{-15}\) and \(1.26\times10^{-14}\), supporting resolution of these manufactured gaps.

**II. Report and plots**

| Item | Verdict | Finding |
|---|---|---|
| Generated numerical results | **CORRECT** | Report sections preceding historical reproducibility reproduce byte-for-byte in memory from the generator and raw inputs, with plotting and writes disabled. The closing glossary also matches its source. Historical numerical entries have computation paths in the generator; I did not independently rerun the historical field reconstruction. No hand-entered measured result was found. |
| “Every number … from job outputs” | **NEEDS-RESTATEMENT** | Literally too broad: protocol thresholds, dates, reference resolutions, and glossary metadata include source-code literals. Say “Every reported measured result is generated from the outputs; protocol constants come from DESIGN.” |
| Inline provisional labels | **WRONG** | All four 2D tables label **ST** provisional but leave **S worst** unqualified. S also uses a first-order refined reference. Change the S column heading to “S worst, PROVISIONAL.” See [report.md:101](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-mechanism/experiments/jcp-mechanism/report.md:101). |
| LaTeX mathematics | **WRONG** | Compliance is incomplete: the glossary contains bare `x = k/64` and `log(median gap)` / `log h`; narrative mathematical notation also uses Unicode superscripts outside math delimiters. The actual displayed gap and recovered-fraction formulas are correctly written in LaTeX. See [report.md:239](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-mechanism/experiments/jcp-mechanism/report.md:239). |
| Closing plain-language glossary | **WRONG** | Present, but not exhaustive. Missing explanations include the worst-gap slope versus median per-state slope, `bar`/`consistent`, the 25% survival condition, normalized leading norms and leading-term discrepancy, central×1.01, \(n,L,h\), and what “LM its” counts. `acc`/`fast`, bank, tests, and several gate terms also need explanations for a cold reader. |
| A1 causal scope | **CORRECT** | The answer uses the prescribed “recovers … within the declared margins” wording and does not assert that placement causes the gain. The four R labels support sufficiency only. Explicitly adding that limitation would improve the report. |
| 2D X0 interpretation | **CORRECT** | Median separations are 0.882898%, 0.882940%, 0.224658%, and 0.224675%, all below 1%. Worst-case separations are explicitly descriptive, not mechanism labels. Bootstrap bounds are N/A. |
| Rounded zero distances | **NEEDS-RESTATEMENT** | The answer’s 1024² nodes distances print as 0.000%, although they are **0.000455534%** and **0.0000726656%**. Use scientific notation, as the detailed tables already do. |
| Nodes A2 slope unresolved | **CORRECT** | All four labels follow the screen; the manufactured fourth-order observation is explicitly separate and has no registered acceptance bar. |
| Finite-window interpretation | **NEEDS-RESTATEMENT** | Add the DESIGN limitation explicitly: these are finite-window observations on fixed reached states, not established asymptotic rates. |
| Amendments 1–6 | **CORRECT, with reporting omissions** | Current populations, screens, family checks, control thresholds, R/X0 decision order, bootstrap treatment, 128³ solver qualification, and amended 2D derivative convergence check are consistent. However, registered p90 quadrature summaries and per-case mesh spreads are absent; the report is not exhaustive coverage of all registered deliverables. |
| A1 plot numerical content | **CORRECT** | Curves agree with the report; backward-difference tensor versus sign-upwind dense is correctly distinguished. The caption identifies refined errors as provisional and 2D errors as ST. |
| A1 plot legibility | **WRONG** | The caption extends beyond the image’s right edge and is clipped. Its overlap explanation is also ambiguous unless restricted to the top row; bottom-row selected and nodes curves are distinct. Wrap the caption. |
| A2 plot numerical content | **CORRECT** | Plotted medians, percentile bands, slope labels, and manufactured positive-control curves agree with the raw arrays. Nodes legends correctly say unresolved. |
| A2 plot standalone labeling | **NEEDS-RESTATEMENT** | Mark the fitting window: the displayed coarsest point is excluded. Identify the dashed curves specifically as C-pos upwind/central; negative controls are not plotted. Consider marking unresolved nodes points so the connected curve cannot be mistaken for an accepted slope fit. |

For clarity, the 3D R-label median recovered fractions range from **0.999995265 to 0.999999191**, with bootstrap lower bounds **0.999993829 to 0.999998877**. Printing 1.000 is rounding, as the report correctly discloses.

**Remaining WRONG: missing inline provisional labels on S errors; incomplete LaTeX compliance; incomplete glossary; clipped A1 plot caption.**