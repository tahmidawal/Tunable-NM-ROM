VERDICT: NOT-CLEAN

`I` = `reports/2026-10-20-improvement-ideation`. Paths below are repository-relative.

**Verification:** all **213** `VALUES.json` entries reproduce; rebuilding the report in memory matches `report.md` exactly. All **seven CPU analyses** reproduce exactly or to roundoff. This establishes reproducibility, not correctness of interpretation. No files were written; no GPU, cluster, or state-changing Git commands ran.

**Round-1 disposition:** resolved: **1, 5–9, 11, 13–14, 18, 20**. Partially resolved: **4, 10, 12, 15–16, 19, 21–23**. Still materially unresolved: **2, 3, 17**. Details follow.

1. **[major] F5’s claimed second-order method is not second order under the implied least-squares implementation.**

   **Evidence:** `I/report.md:227`; `I/values.py:85–88`. Here \(A\) is rectangular, \(M\times r\), so the displayed equation generally has no exact solution. The cost model’s \(J^\top J\)/Cholesky implies least squares. Second-order Rosenbrock accuracy for a square ODE does not transfer automatically.

   Counterexample: \(A=(1,0)^\top\), \(F(c)=(c^2,c^2)^\top\). The limiting projected ODE is \(c'=-c^2\), but the proposed least-squares update gives
   \[
   \delta=-\frac{h c^2(1+2hc)}{1+2hc+2h^2c^2}
          =-hc^2+O(h^3),
   \]
   missing the exact \(h^2c^3\) term. CPU integration to \(T=0.25\), \(c(0)=1\), gives errors **0.00350773, 0.00176927, 0.000888573, 0.000445283** for 10/20/40/80 steps: orders **0.987, 0.994, 0.997**.

   **Required fix:** define the square projected ODE and linearise that, or derive the actual least-squares method’s order. Rebuild its cost model. Define the residual, Jacobian, weighting, regularisation and startup explicitly.

2. **[major] The revised “ideal dynamics” impossibility claim remains unsupported.**

   **Evidence:** `I/report.md:8,88,369`. `worktrees/2026-10-18-jcp-budget/experiments/jcp-budget/ana2d.py:326` computes distance to the **projected reference**, not total reference error. The JSON key `members["R384|LSPG|CNR|1"].summary.worst_e_dyn_star` is **0.0153852632**; its worst projection floor is **0.0128738004**.

   The report acknowledges the definition but still concludes that **no time-integrator or nonlinear-solver change** can close the gap. A converged trajectory does not bound finite-step trajectories: discretisation error can cancel model error. Furthermore, this ideal endpoint changes tests, quadrature and initialisation as well as time integration; it is not the deployed \(M=1536\) system. The comparison also remains 256² versus 1024².

   **Required fix:** say that convergence to this particular ideal endpoint does not achieve the target. Remove the universal exclusion of integrator/solver changes; compare total errors on the same mesh, cases, initialisation and reduced formulation.

3. **[major] The new “non-span” decomposition is misnamed and cannot support the causal conclusion.**

   **Evidence:** `I/report.md:9,97–107,273`; `I/values.py:326–329`. For a ROM field \(v\) in the bank span,
   \[
   \|v-u\|^2=\|(I-P)u\|^2+\|v-Pu\|^2.
   \]
   The floor is the component **outside** the span; the remainder lies **inside** it. Subtracting squares of separately aggregated medians/maxima does not recover that remainder.

   The reported **4.5% / 13.3%** reproduce as arithmetic on medians, but do not establish numerical-error contributions. The remainder also contains reduced-model error and initialisation error. Changing the span changes the dynamics, so “a perfect span alone” improving C1 by less than 2× does not follow.

   **Required fix:** compute the decomposition per case and time before aggregation. Rename the remainder “distance to projected reference”; distinguish model error from numerical error using converged rollouts. Remove the counterfactual gain claim.

4. **[major] OOD scores silently include rejected references, including in calibration.**

   **Evidence:** `I/analysis/ood_indicator.py:58–65` checks validity and finite error but ignores `eligible`. In `worktrees/2026-10-19-jcp-regimes/experiments/jcp-regimes/checks/analysis.json`, cases **Ap|4, Ap|8, N0|4, N0|15, B2|7** have `eligible:false` in every configuration.

   Filtering these gives **153**, not 158, scored rows and **52**, not 54, calibration rows per configuration. Retaining invalid ROM runs as mandatory alarms, recomputed target-A ranges become:

   | Indicator | AUC | Detection |
   |---|---:|---:|
   | `dist_qp` | **0.903617–0.986678** | **0.584270–0.808824** |
   | steepness | **0.918539–0.988169** | **0.674157–0.912281** |

   For R384 CN-R, the rule-distance alarm threshold changes from **1.773876e-5** to **1.881621e-5**.

   **Required fix:** separate reference eligibility from solver failure. Exclude unresolved truth labels from primary accuracy classification, report their deployment outcomes separately, and regenerate both target tables. Publish per-configuration and per-regime results, not only ranges. The disclosed in-sample calibration and post-hoc indicator selection remain limitations.

5. **[major] F16’s proposed envelope does not enforce the claimed compatibility.**

   **Evidence:** `I/report.md:73,244,266`. For \(u=e(x)n(x)\),
   \[
   u''=e''n+2e'n'+en''.
   \]
   Setting \(e=e''=0\) at a wall leaves \(2e'n'\). For example, \(e=\sin(\pi x)\), \(n=e^x\) gives **\(u''(0)=2\pi=6.283185\)**. A sine envelope or sine inputs to an unrestricted network do not guarantee a smooth odd extension.

   The current envelope permits incompatible curvature; it does not force nonzero curvature for every coefficient combination. The report supplies no boundary-curvature measurement establishing that this causes the observed tail.

   **Required fix:** specify parity-constrained features or actual sine-series outputs and prove their boundary properties. First measure boundary derivatives and tail convergence. Keep the causal claim hypothetical; F7 need not wait for retraining.

6. **[major] The decode impossibility claim survives its stated retraction.**

   **Evidence:** `I/report.md:69` still says “no solver or quadrature change suffices at r ≥ 384”; line 370 says that claim was retracted. The corrected illustrative ratios **0.809 / 1.98 / 1.73** reproduce, but combine different jobs and fixed existing FOM matches. Improved ROM accuracy can change the matching FOM.

   Timing inclusion is correct: ROM decode is inside `jcp-combined/t2core.py:265–272`; spectral synthesis is inside `jcp-fom2d/psjax2d.py:61–68`.

   **Required fix:** restrict the statement to the measured decoder, workload and existing accuracy target. Establish a ceiling within one job before treating it as a bound. Remove the unsupported extension to all \(r\ge384\).

7. **[major] F2/F3 can pass their kill criteria while being slower or less accurate.**

   **Evidence:** `I/report.md:203–205,353`. Q2 establishes quadrature agreement, not accuracy of the changed projection. F2 permits \(m=36864\):
   \[
   6mr=\mathbf{84{,}934{,}656}\text{ flops},
   \]
   versus **75,497,472** for current uncontracted advection and **37,748,736** for F1’s precontracted version. Thus a passing F2 can require **2.25×** F1’s advection work.

   F3 excludes dev6 from fitting and now includes the full Q2 contract—good—but neither stop rule requires a useful accuracy/cost result. Combining projection and fitted-rule changes also obscures attribution.

   **Required fix:** cross projection and rule choices in separate arms. Require within-form solver qualification, common-reference accuracy and paired end-to-end timing. Retain the **5.33–10.67×** node ratios as descriptive only; account for stencil gathers and fitted-rule setup.

8. **[major] W2 still confounds attribution and applies the wrong success test to the boundary arm.**

   **Evidence:** `I/report.md:263–280,352`. Each retraining arm also receives adaptive stepping/quadrature, but no incumbent-bank-plus-adaptivity control is specified. A C1 improvement could therefore come entirely from numerical changes.

   Conversely, the F16 arm targets quadrature cost and spectral tails yet must deliver the same **3× N0s floor / 2× C1 ROM** improvements as coverage retraining. It can achieve its stated mechanism and still be killed. The Step-0 floor-ratio test cannot uniquely identify upwind smearing; generator matching and temporal error remain alternatives.

   **Required fix:** include the unchanged bank with identical adaptive numerics. Give fidelity, coverage and boundary compatibility separate primary endpoints. Treat Step 0 as a diagnostic, not a decisive causal gate.

9. **[major] F1’s experiment and conclusions remain stronger than its evidence.**

   **Evidence:** `I/report.md:173,178–181,351,357`. “F1 helps only if the baseline is denied the same integrator” is unproved: relative gains depend on both implementations. Conversely, comparing ETDRK4 Galerkin only against LSPG CN-R confounds integrator, projection and precontraction. A passing experiment does not establish “the time error removed.”

   The **0.836/0.464 ms** models reproduce, but are assumptions, not a measured performance interval. Their omitted coefficient preparation, stage operations and overhead are now disclosed.

   **Required fix:** add a precontracted GAL-CN control and time complete matched queries for both bank and POD. Measure time-error reduction using refinement; say “reduced,” not “removed.” Report H-base outcomes rather than predicting that a fair integrator necessarily preserves them.

10. **[major] F17 compares different error quantities in its kill criterion.**

    **Evidence:** `I/report.md:339`. The gap between \(e_{\rm dyn}^\*\) and the floor is not a decomposition: those are different components, maximised separately. Requiring **total** error 15% below the **dynamic-component** value produces a target of **1.307747%**, only **0.020367 percentage points** above the **1.287380%** worst floor.

    **Required fix:** choose either total error or distance to projected reference, use the same quantity for baseline and candidate, and derive the threshold from paired per-time data. Define separate closure and windowed-LSPG arms and include their online cost.

11. **[major] The historical wide-bank floor is still mislabelled.**

    **Evidence:** `I/report.md:258` groups **3.23→1.29%** under “both against first-order references.” But `worktrees/2026-10-08-jcp-wide-bank/experiments/jcp-wide-bank/report.md:113–133` defines these as floors on **native-grid bank-validation fields**, seed **923751**, **96 cases**, six times **including \(t=0\)**, with **each snapshot normalised by its own norm**.

    They are not refined-reference rollout scores and do not use this report’s five-time/\(\|u(t_0)\|\) convention.

    **Required fix:** carry the actual population and norm convention beside these numbers. Use the lane’s separately reported reference-based floors when arguing about floor-versus-rollout behaviour.

12. **[minor] Cost estimates still omit work or overlook cheaper exact alternatives.**

    **Evidence:** `I/values.py:85–88` omits gradient formation \(J^\top r\) from the **1.31 ms** one-pass sum. `jcp-p1-scaling/p1comp.py:150–154` confirms the Cholesky kernel receives an already formed gradient. `I/report.md:86` calls full-query time divided by 50 a “per-step” measurement; it includes initialisation and decode.

    The tensor bandwidth arithmetic is correct, but F4 omits exact symmetry packing. At \(r=384\), storing the symmetric quadratic input pair needs **0.227082 GB** for Galerkin or **0.908329 GB** for \(M=1536\), rather than **0.452985/1.811939 GB**. This changes the screening estimate, though not necessarily runtime.

    **Required fix:** complete the one-pass operation inventory; label query-amortised step costs accurately. Include precontracted, symmetry-packed tensors in the microbenchmark. Keep bandwidth-only times explicitly separate from achieved query speed.

13. **[minor] The stability proxy is now caveated, but two mathematical labels remain wrong.**

    **Evidence:** `I/report.md:157,163`; `I/analysis/explicit_stability.py:4–5,60–69`. The exact RK4 imaginary-axis intercept is \(2\sqrt2\), not 2.8. For negative-real diffusion eigenvalues the intercept is approximately **2.78529356**. With stored \(\rho_d=1567.11327346\), the corresponding threshold is **0.0017773403**, versus the reported **0.0017867247**.

    The script’s header still says selected continuum/discrete mode sets agree up to ties, contradicting the report’s correction. The common test normalisation cancels correctly; there is no missing-\(L\) factor.

    **Required fix:** call 2.8 a heuristic constant, remove “would need” as a method-independent diffusion bound, and correct the script documentation. Test actual frozen operators and reached stages before accepting a timestep. Positive Jacobian eigenvalues alone do not diagnose loss of nonlinear energy conservation.

14. **[major] Several remaining kill criteria are not decisive or fully specified.**

    **Evidence:** `I/report.md:301,305,312–325,347–354`.

    - **F12:** bootstrap intervals are merely “reported”; acceptance does not specify whether interval bounds must satisfy detection/false-alarm/AUC bars. Cohort size and mixture are unfixed.
    - **F13:** fewer accepted steps can conceal extra stages, rejected steps, rule evaluations and coefficient recomputation. No complete-query cost test is required.
    - **F14a:** “L-shaped or annular” and “largest registered mesh” leave the actual pilot unresolved.
    - **F14b/W4:** break-even needs an explicit comparator and workload definition. Continuous POD should also get a reusable-rule arm where feasible; forcing refits can manufacture the bank’s advantage.
    - **F14f:** a manifold oracle win can reflect decoder capacity, unequal training or failed oracle optimisation. It does not establish online hyper-reduced performance.

    **Required fix:** distinguish cheap screening gates from claim-acceptance gates. Freeze populations, workloads, uncertainties and failure accounting. Use end-to-end cost for adaptive methods and matched training/optimisation controls for nonlinear decoders.

15. **[minor] Missing controls and prior-art integration still distort priorities.**

    **Evidence:** `I/report.md:159,189,238–249,314–316`. Several useful ideas are named but not integrated into the tests:

    - Run F7’s CPU truncation check immediately; its validity does not depend on F16.
    - Retain same-form residual sketching as a control for reducing \(M\), rather than treating it like deleting test modes. [Balabanov–Nouy](https://arxiv.org/abs/1910.14378) provides minimal-residual sketching and verification machinery.
    - Compare local-bank/local-rule approaches directly with global coverage retraining. [Grimberg et al.](https://arxiv.org/abs/2008.02891) already extend ECSW to Petrov–Galerkin and local reduced bases.
    - The Kirsten–Simoncini citation supports exponential reduced integration, but also a missing **matrix-oriented/two-sided POD-DEIM** comparator relevant to separability and storage. [Primary paper](https://arxiv.org/abs/2006.13289).

    **Required fix:** add these bounded controls or give evidence-based reasons for deferral. Replace the undocumented literature-search absence claim at line 72 with a scoped statement and searchable bibliography.

16. **[minor] Report-rule compliance remains incomplete.**

    **Evidence:** title/status, LaTeX displays, coloured Mermaid and expanded glossary are present. However:

    - `I/report.src.md:87,196` hand-types measured \(m^\*\) outcomes; these are not design constants and bypass `VALUES.json`.
    - Provisional `refdv` qualifications do not travel with the hybrid results at `report.md:331`.
    - The glossary still omits terms such as n-width, CGL, CROM, SNF-ROM and GNAT.
    - The folder-layout explanation does not amend the literal filename rule in `CLAUDE.md:142`.
    - `report.md:69` proposes single precision despite `AGENTS.md:42`; F3’s fitting description needs reconciliation with the decoder-output snapshot rule at `AGENTS.md:121`.

    **Required fix:** generate measured counts, attach reference limitations locally, finish the glossary and distinguish proposed protocol changes from currently authorised methods. Document an actual naming-rule exception or use the required filename.