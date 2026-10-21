Read-only review completed; no files modified. I reproduced all **115 entries in `VALUES.json`**, checked their inherited-value generator, recomputed the CPU convergence fits, reran all PCG iteration cases, and spot-checked the two spatial solvers. Reproducible arithmetic does **not** establish that the comparisons or predictions are valid.

Below, `report.md`, `values.py`, and `analysis/` refer to `reports/2026-10-20-spectral-investigation/where-we-win/`.

1. **BLOCKER — The principal 3D forecast uses an explicitly unqualified ROM.**  
   **Evidence:** `report.md:247` calls `W_C_W768` the “fastest qualified” configuration. However, `reports/2026-10-17-full-experiment-plan/final/report.md:34` explicitly records its Q2 failure; line 460 retracts qualification of the speed3d winners. Its 28.8 ms cost anchors `values.py:227` and the geometry forecasts.  
   **Fix:** Retract the qualification label and rebuild the model around a Q1/Q2-qualified configuration, timed under the same output contract. Do not substitute the jcp-dims winner’s timing directly: that result uses a different path/contract.

2. **MAJOR — “Geometry forces resolution” does not justify a speed-up against the proposed strongest FOMs.**  
   **Evidence:** `report.md:68` assumes global resolution \(L\) from hole radius; line 196 turns the resulting uniform-grid ratios into a prediction of ≥3× against embedded FD **and high-order FE**. Nothing establishes that either competitor needs a globally fine grid. Local boundary refinement, curved elements, adaptive octrees and cut-cell methods invalidate that implication. Even the required number of cells across a hole is assumed, not accuracy-tested.  
   **Fix:** Treat the table as a uniform-grid scenario only. Build matched-error cost curves for locally refined, curved-geometry FOMs before predicting a win against their union. Report DOFs, temporal resolution and output error, not just global \(L\).

3. **MAJOR — The ROM cost extrapolation is not calibrated for the proposed changes.**  
   **Evidence:** `values.py:117` fits an intercept using **R128 BE versus R384 CN-R**; those points change solver scheme as well as dimensions. `values.py:229` then transports a descriptive **2D full-query** elasticity to 3D and assumes exactly 4096 additional points per hole. Neither identifies geometry’s effect on bank width, tests, conditioning or iteration count.  
   **Fix:** Remove the interpretation of the two-point intercept as measured fixed overhead. Fit controlled same-scheme experiments, and present sensitivity to \(r,M,m\), solver iterations and geometry setup. Include scenarios in which qualification requires a larger rule or bank.

4. **MAJOR — The 3D timing extraction selects the wrong panel and changes the speed-up statistic.**  
   **Evidence:** `values.py:136–141` overwrites `tB[L]` for each ROM; its final value comes from **C_W1024**, while provenance at line 149 says **W_C_W768**. Line 150 divides aggregate latencies instead of using the stored paired `ratio`. In `analysis-pm.json`, `m1.per_L[L].panels.W_C_W768.same_mesh[...]`, the paired ratios are **8.136, 62.385, 268.460**; the report gives **7.89, 60, 261**.  
   **Fix:** Key both timings by configuration and mesh. Preserve the registered paired statistic for measured speed-ups; label any ratio-of-aggregate-times model separately. The panel overwrite has a small numerical effect here, but its provenance is false.

5. **MAJOR — The \(f_{\rm geo}\) thresholds are necessary conditions under restrictive assumptions, not evidence of a win.**  
   **Evidence:** `values.py:103–104` correctly computes \(2/S_{\rm CC}\), but `report.md:66` uses these values to characterize geometry’s prospects. The source \(S\) values are upper bounds against screening-selected comparators, and the calculation freezes ROM cost. The claim at line 8 that a box-fast non-box FOM means “we do not win” also overstates matters: some reported CC point estimates exceed one.  
   **Fix:** Write the conditional relation explicitly:
   \[
   S_{\rm new}=S_{\rm box}\frac{f_{\rm FOM}}{f_{\rm ROM}}.
   \]
   Distinguish “faster” from “≥2× faster,” retain uncertainty, and state that the necessary slowdown can be larger against a better FOM.

6. **MAJOR — FD iteration counts are presented as evidence for a spectral time-step cost.**  
   **Evidence:** `analysis/varcoef_pcg.py:41–52` implements conservative second-order FD diffusion with an FD-eigenvalue DST preconditioner. Line 54 manufactures a solution containing white noise and starts PCG from zero. This is neither the spectral operator nor a warm-started Burgers trajectory. Nevertheless, `report.md:89–97` transports its counts into spectral-FOM forecasts.  
   **Fix:** Measure the actual spectral discretization and realistic stage right-hand sides, with previous-step/predictor initial guesses. Sweep solver tolerances against trajectory error. Compare coefficient-aware multigrid, improved preconditioners, IMEX variants and reusable solver setup. A single zero-start synthetic solve cannot determine per-step iteration counts.

7. **MAJOR — The assumed 0.065 ms PCG iteration cost has no measured implementation behind it.**  
   **Evidence:** `values.py:199–221` divides screening PS query time by 20 steps and 12 supposed transform pairs, then charges three pairs per iteration. That query includes output synthesis; the CPU FD operator itself needs a stencil plus one preconditioner transform pair. A spectral operator has a different transform workload. The model also adds this charge to an ETDRK4 query while proposing replacement by an IMEX/CN solver.  
   **Fix:** Benchmark the complete candidate step on one GPU, including reductions, operator application, preconditioner and stage count. Retune timestep and resolution. Until then, replace the 1.3–3.1 prediction with a sensitivity table over explicitly hypothetical iteration costs.

8. **MAJOR — Break-even calculations are internally inconsistent and are not established lower bounds.**  
   **Evidence:** `values.py:240–241` omits the hole-related ROM slowdown used in the preceding geometry model. Retaining its other assumptions, eight holes changes \(Q^*\) to approximately **65,000** and **6,400**, rather than 54,000 and 6,200. More fundamentally, a 2D bank fit is not a proven lower bound for a new 3D training pipeline. `report.md:205` gives \(10^4\)–\(10^5\) queries for varying geometry without a corresponding model. The variable-coefficient 300,000 estimate uses the optimistic high-cost FOM scenario.  
   **Fix:** Compute break-even separately for each defined workload, using complete incremental offline costs and consistent online costs. Show “no break-even” when the saving is nonpositive. Call borrowed training costs illustrative assumptions, not lower bounds. Restrict “POD breaks even first” to the measured fixed-box setting.

9. **MAJOR — The penalisation experiment does not eliminate competitive FFT-based geometry solvers.**  
   **Evidence:** `report.md:12` labels approximately 650 points/unit “measured”; `analysis/penalized_spectral.py:100–102` actually **extrapolates** it. Line 134 generalizes one binary-mask Brinkman scaling to immersed/embedded FFT competitors. Higher-order active penalty methods retain Fourier discretizations, and Fourier continuation directly targets general domains. See [Shirokoff–Nave](https://arxiv.org/abs/1303.5681) and [Lyon–Bruno](https://www.sciencedirect.com/science/article/abs/pii/S0021999110000215).  
   **Fix:** Label 650 and 4,400 as extrapolations for this specific manufactured problem and penalty scaling. Retain tuned active penalty, corrected immersed-interface and Fourier-continuation methods as unresolved competitors; do not dismiss the class using this control.

10. **MINOR — The advertised “spectral accuracy” smooth control converges approximately second order.**  
    **Evidence:** `analysis/penalized_spectral.py:79–92` uses \(u=e^x\sin(\pi x)\sin(\pi y)\). My rerun gives relative errors **0.00616, 0.00167, 0.000436, 0.000111, 0.0000281** for \(N=8,16,32,64,128\). The chosen function does not have the boundary compatibility required for a smooth odd periodic extension.  
    **Fix:** Use a compatible sine-series control, or explain the observed algebraic convergence. Both PCG analyses should also persist convergence status and recomputed residuals rather than silently returning an iteration-cap result.

11. **MAJOR — The L-shape computation is valid, but the comparison with the historical ROM is not established.**  
    **Evidence:** `analysis/lshape_fd.py` correctly solves its two harmonic manufactured problems; the reported orders and \(N=64\) error reproduce. However, `report.md:130` uses this particular nonhomogeneous-boundary solution to conclude that a much coarser grid reaches the historical lane’s ROM error on its different problem family. It also says the singularity slows convergence “only” in max norm and gradients, although its own \(L^2\) order falls from roughly two to 1.33.  
    **Fix:** Restrict the conclusion to the manufactured example, then run a coarse FOM ladder on the historical lane’s actual cases and continuum reference. Correct the “only” statement. Label the sparse-LU timing as a single factorization-plus-solve measurement excluding assembly.

12. **MAJOR — Varying topology is not an established exclusive advantage, and its online setup is undercounted.**  
    **Evidence:** `report.md:12,25,36` says the FOM must remesh and implies POD cannot share a representation. Yet line 204 includes a fixed-grid embedded FOM and a continuous linear-subspace competitor. [LiCROM](https://dgp.toronto.edu/~changyue/LiCROM/) explicitly demonstrates cutting and hole punching. Moreover, line 203 makes the bank geometry-dependent: bank/gradient evaluations and reduced matrices may need rebuilding per query, beyond quadrature construction.  
    **Fix:** Remove the categorical structural-win claim. Test common-background/extended POD, local or component reduced bases, and the proposed continuous linear subspace. Charge geometry-dependent basis evaluation, operator assembly and initialization to both sides.

13. **MAJOR — Workload and accuracy contracts do not support the broader conclusions.**  
    **Evidence:** The source fom2d report at lines 74 and 255 distinguishes screening on dev6 cases 0–1 from stage-2 timing on dev6, with matching on development/validation errors. The reviewed report largely drops these cohort restrictions. Its 3D PS/CC-small-grid constants are screening costs (`values.py:154–158`), whereas the geometry model uses m1 panels. Flagship 2 adds gradients and wall flux (`report.md:202`) but retains field-error-oriented qualification.  
    **Fix:** Attach GPU, cohort, timing stage, prepared-input boundary and output contract to each table. Require separate gradient/flux accuracy checks; exact autodiff of an approximate bank does not establish derivative accuracy. For many-query claims, benchmark **batched FOMs as well as batched ROMs**, including reusable setup and output costs. The existing H200 paired box comparisons remain useful within their original scope.

14. **MINOR — Generation and project-rule compliance are incomplete.**  
    **Evidence:** An in-memory regeneration differs from `report.md` at lines 7 and 11: the source contains timestep qualifications missing from the generated file. Line 3 says “reviewed,” while line 229 says “Not yet reviewed.” Literature performance numbers are literal text in `report.src.md:167–180`, outside `VALUES.json`. Provisional labels generally exist, but the penalisation extrapolation and inherited POD restrictions do not travel consistently. The glossary omits terms including DST, UQ, KL and AMG; some mathematical notation remains plain text, such as `4096^2` at line 99.  
    **Fix:** Regenerate and check equality automatically; correct review status; register literature numerical claims with precise sources; carry limitations inline; complete the glossary and LaTeX notation. Registered design constants and explicitly labelled planning budgets should be distinguished from measured results.

**Overall verdict: REJECT.**

The top-three candidates are defensible as research questions, but their ordering is not defensible as an evidence-based ranking of likely wins. The first depends on an unqualified ROM and an unproved global-resolution requirement; the second overstates exclusivity and omits geometry-dependent setup; the third has the clearest inexpensive falsification path, but its speed-up model is uncalibrated. The proposed execution sequence—variable coefficient, then 2D geometry, then conditional 3D—is more defensible than the headline ranking.