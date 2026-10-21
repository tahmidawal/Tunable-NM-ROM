**Verdict: REJECT.** The numerical corrections largely reproduce, but several round-1 findings are only partially resolved. The report is suitable as an exploratory shortlist after further qualification; its ranking and remaining performance predictions are not yet supported.

Read-only review completed; no files modified, including the lab log. I verified:

- In-memory regeneration matches `report.md` and all **146 used `VALUES.json` entries byte-for-byte**.
- All **48 PCG cases** reproduce their iteration counts and convergence flags.
- L-shape solver spot checks reproduce; fitted relative-\(L^2\) orders are **1.32675** and **2.00495**.
- Penalisation spot checks reproduce; the extrapolated resolutions are **646.58** and **4374.74** points/unit.
- Raw m1 timing repetitions reproduce C_old’s paired ratios: **3.51927, 28.80684, 122.59360** at \(L=128,256,384\). Its recorded median/worst errors reproduce as **1.57474% / 4.46053%**.
- The revised geometry table and variable-coefficient sensitivity arithmetic reproduce.

Round-1 disposition:

| R1 finding | Round-2 assessment |
|---|---|
| 1: unqualified 3D anchor | Fixed: C_old replaces W_C_W768; the latter is explicitly unqualified. |
| 2: global resolution implies win over adaptive FOMs | Fixed in the 3D forecast: explicitly restricted to uniform grids. |
| 3: uncalibrated ROM extrapolation | Partial: assumptions disclosed and sensitivity added; conditioning/iteration sensitivity remains absent. |
| 4: wrong panel and speed statistic | Fixed: independently verified from raw repetitions. |
| 5: slowdown thresholds | Substantially fixed: conditional relation and necessary-condition language present. |
| 6: FD counts transported to spectral trajectories | Partial: operator identified, but synthetic warm starts are promoted to “realistic.” |
| 7: invented iteration cost | Fixed as a hypothetical sensitivity; solver-stage interpretation remains unresolved. |
| 8: break-even | Substantially fixed: hole slowdown included, borrowed cost labelled, nonpositive savings handled. Minor consistency issue remains. |
| 9: penalisation excludes FFT competitors | Substantially fixed in discussion; competitor enforcement remains incomplete. |
| 10: invalid smooth control/convergence recording | Fixed: control removed; convergence and recomputed residuals recorded. |
| 11: L-shape comparison | Fixed: conclusion restricted to manufactured example; historical comparison explicitly unresolved. |
| 12: topology exclusivity/setup | Partial: setup and competitors added, but unsupported geometry and multi-mesh advantage claims remain. |
| 13: workload/accuracy contracts | Partial: better disclosure; output mismatch and incomplete flux acceptance persist. |
| 14: generation/report rules | Partial: generation/status fixed; literal literature numbers and other rule gaps remain. |

Below, file references are within [the reviewed directory](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-20-spectral-investigation/where-we-win/report.md).

1. **MAJOR — Flagship 2 still predicts a win without a supporting cost model.**

   **Evidence:** `report.md:213` predicts “online parity to a few ×” against embedded FD, and a per-query win against FE once meshing is counted. No measured geometry-dependent bank evaluation, assembly, initialization, FOM meshing or FOM solve cost supports either prediction. Counting meshing does not itself establish that FE loses. The added setup accounting at line 211 is a requirement for future measurements, not evidence.

   The multi-mesh row at line 126 also claims “yes, the one consistent edge” over POD while explicitly naming continuous/interpolated POD as a competitor. That is inconsistent with the report’s more careful opening. LiCROM demonstrates discretization-independent linear subspaces and geometry changes; it does not establish performance on this Burgers workload, but it defeats exclusivity. [LiCROM paper](https://arxiv.org/abs/2310.15907).

   **Fix:** Replace both geometry speed predictions with “unknown.” Describe F14b as an untested amortization hypothesis against continuous/interpolated POD, including transfer and setup costs.

2. **MAJOR — The 3D table remains a comparison of different output workloads, and its sensitivity range is incomplete.**

   **Evidence:** `report.md:83` explicitly acknowledges scoring-lattice ROM output versus full-field FOM output. Disclosure is an improvement, but these ratios still feed the headline scenario, ranking and break-even calculation. They do not represent one common delivered query.

   The revised eight-hole arithmetic is correct: **2.2375 / 18.3148** for the smaller rule and unchanged bank, and **0.5421 / 4.4376** for the larger rule and doubled bank/tests. However, these endpoints vary only assumed quadrature size and transported 2D elasticities. They do not vary nonlinear iteration count, conditioning, timestep requirements or initialization. The report itself records an **8.1×** wider-bank configuration, whereas its isolated doubled-bank/test multiplier is only **2.268×**. Those configurations are not controlled comparisons, so neither multiplier validates the other.

   **Fix:** Define a common output contract and separate evolution from output costs before using the ratios for payoff estimates. Keep the present table only as an explicitly incomplete algebraic scenario. Add independent sensitivity to solver iterations and timestep count; do not present its endpoints as a plausible performance range established by evidence.

3. **MAJOR — The warm-start sensitivity is still interpreted more strongly than the experiment permits.**

   **Evidence:** `analysis/varcoef_pcg.py` manufactures
   \[
   u_*=\mathrm{sm}+0.05\,\mathrm{sm2},\qquad x_0=\mathrm{sm},\qquad b=A u_*.
   \]
   This is neither a previous Burgers timestep nor a predictor from one. The headline comparison also changes **three things simultaneously**: solution smoothness, initial guess and tolerance. Thus “3 versus 31 iterations” does not isolate the benefit of warm starting.

   Nevertheless, `report.md:102` calls the warm column “the realistic one.” The cost model also adds one CG solve per inherited PS timestep to an existing ETDRK4 query cost without identifying a replacement integrator’s stage count or retuned timestep. The hypothetical-cost label fixes the invented benchmark, but not this algorithmic interpretation.

   Finally, “no win” is too broad: the warm column at **0.2 ms/iteration predicts 1.4×** for its best configuration. What the table consistently predicts is **no ≥2× win**.

   **Fix:** Call every column synthetic. Separate tolerance, RHS and initialization effects using the existing cases. Reserve realism claims for actual trajectories with error-tuned tolerances. Specify complete stage costs and timestep counts for each candidate integrator, and distinguish \(S>1\) from \(S\ge2\).

4. **MAJOR — Strong competitors remain missing from, or optional in, the actual decision rules.**

   **Evidence:** Flagship 1 lists Fourier continuation as competitor (e), but its kill criterion tests only (a)–(c). Section 4.2 retains corrected Fourier methods for flagships 1 **and 2**, yet flagship 2’s execution list omits them. Its kill criterion tests speed only against embedded FD, despite the proposed broader competitive story.

   For variable coefficients, losing diagonal sine-basis diffusion does **not** force a CG-based implicit stage. Full-operator exponential actions using **Krylov/KIOPS or Leja interpolation** remain untested competitors. These algorithms approximate exponential actions without requiring diagonalization. Their competitiveness is problem-dependent, so they are candidates to screen, not presumed winners. [Deka, Tokman and Einkemmer](https://arxiv.org/abs/2211.08948). Reused sparse factorizations for fixed per-query diffusion and fixed timestep also deserve a low-resolution screening arm.

   **Fix:** Make the tested competitor union consistent across the ranked table, experiment list and kill criterion. Screen full-operator exponential integration and reusable factorization for #3. If Fourier continuation or another named strong method cannot be run, explicitly limit any resulting “best FOM” claim to the tested set.

5. **MAJOR — Flagship 2’s acceptance rule does not cover all promised outputs.**

   **Evidence:** `report.md:210` promises field, gradient and wall-flux outputs, each with its own accuracy check. But line 214’s speed criterion covers only field and gradient error; wall flux disappears. The bank-floor criterion remains unspecified as to norm and cohort. A field-\(L^2\) projection floor does not certify boundary-normal derivatives or flux.

   **Fix:** Register separate field, gradient and flux tolerances and references, including boundary integration error. Require the timed comparison to satisfy **all deployed output tolerances**. Specify whether floor and generalization gates use medians, maxima or another statistic.

6. **MINOR — Break-even arithmetic is repaired, but its timing convention is not fully consistent.**

   **Evidence:** The reported **85,000** and **6,500** reproduce as approximately **85,194** and **6,503**, with the stated 2.97-hour illustrative cost. However, `values.py:229,246` uses the \(L=128\) C_old latency for both meshes, while the speed-up table uses each mesh’s own paired panel. C_old is **64.450 ms** in the \(L=128\) panel and **60.665 ms** at \(L=256\); “64.4 ms at every mesh” is therefore an approximation, not the observed value. Using the latter panel gives approximately **6,480** queries, unchanged at the displayed precision.

   The excluded training-data and qualification costs are now honestly disclosed. Consequently these are illustrative partial-cost crossovers, not deployment break-even estimates.

   **Fix:** Use one declared timing convention consistently, preferably paired per-invocation savings for amortization. Say “approximately mesh-independent.” Retain the partial-cost qualification beside headline crossovers and include FOM setup in any eventual total-cost comparison.

7. **MINOR — The proposed KL coefficient family does not yet have the claimed contrast.**

   **Evidence:** `report.md:219` defines \(\kappa=\nu e^{sg(x;\mu)}\) for a random KL field and calls its contrast \(e^s\). In general,
   \[
   \frac{\kappa_{\max}}{\kappa_{\min}}
   =e^{s(g_{\max}-g_{\min})}.
   \]
   The CPU script explicitly normalizes its single deterministic field to approximately unit range; the flagship’s random family does not specify that normalization.

   **Fix:** Define the normalization or parameter scaling and record realized coefficient extrema. Otherwise “contrast 100” does not identify the proposed experiment.

8. **MINOR — CLAUDE.md compliance and the review-closure statement remain incomplete.**

   **Evidence:** Generation equality, the opening status, Mermaid diagram and expanded glossary now pass. But `report.src.md:173–186` still hand-types literature performance numbers—**2.6×, 11.7×, 109×, 89×, 199×, 79%**, among others—outside the value registry. Attribution does not satisfy the explicit “never hand-type a number” rule. The glossary still leaves terms such as RHS, DOFs and CN-R’s Rannacher start unexplained. The filename `report.md` also does not follow the literal dated descriptive filename rule.

   Most importantly, `review_record.md` says **“All addressed”**, although findings 3, 6, 12–14 remain partially open.

   **Fix:** Register literature numerical claims with precise source locations, complete the glossary, resolve the filename convention, and record each finding’s actual disposition. Preserve the distinction between an error fixed, a claim withdrawn and an experiment still unperformed.

The top three are defensible **research questions**, but their exact ordering is not defensible as an evidence-based ranking of expected wins. #1’s payoff rests on a uniform-grid scenario whose strongest competitors remain unmeasured; #2 retains unsupported speed predictions; #3 offers a useful falsification test but currently has no calibrated forecast. The execution order—**#3, then a 2D geometry pilot, then conditional 3D**—is more defensible. I would make #3 initially a FOM-only trajectory benchmark and make the 2D geometry pilot pass matched-output, adaptive-FOM and qualification gates before committing to variable-topology training or 3D.