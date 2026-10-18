Overall verdict: **WRONG as written.** The core approach is viable, but several factual errors, fairness violations and nonfunctional controls prevent this from being a ready pre-registration.

Evidence shorthand: `D` = audited `experiments/jcp-hr-baselines/DESIGN.md`; `C` = supplied combined-lane directory; `R` = supplied references-lane directory; `S` = supplied speed3d directory; `P` = supplied full-experiment-plan directory. Line references below refer to those files. No files were changed and no jobs were submitted.

1. **CORRECT — Central-FD residual and ECSW gradient target are mathematically consistent.**

   **Evidence:** D:49–52, 64–66, 103–108; R/refcore.py:64–87, 125–143.

   With fixed nonnegative weights, minimizing
   \[
   \tfrac12\sum_{i\in E}\xi_i\mathcal R_i(c)^2
   \]
   has gradient \(\sum_{i\in E}\xi_iJ_i^\top\mathcal R_i\). Training against the **all-node** sum is correct; training against only the candidate-node sum would not be. Petrov–Galerkin ECSW is an established extension, including finite-difference models ([primary paper](https://arxiv.org/abs/2008.02891)).

   **Fix:** Retain these equations. Describe this specifically as weighted strong-residual LSPG; the ECSW name does not itself establish energy conservation for this Burgers implementation.

2. **WRONG — The `none` Jacobian formula omits the temporal and diffusion factors.**

   **Evidence:** D:52 defines \(F_h\), D:66 defines \(\mathcal R\), but D:101 gives \(J=J_F\Phi_r\).

   Under that notation, the required Jacobian is
   \[
   J=\left[a_0I+\Delta t\,b_0\left(N_h'(\Phi_rc)-\nu\Delta_h\right)\right]\Phi_r.
   \]
   The stated expression is correct only if \(J_F\) is silently redefined as the *time-discrete residual* Jacobian.

   **Fix:** Print the explicit expression above and use it consistently in ECSW training and FD tests.

3. **NEEDS-RESTATEMENT — Q-DEIM’s reduction is sound before truncation; “exact linear terms” is not guaranteed afterward.**

   **Evidence:** D:120–127.

   The DEIM reconstruction and the identity \(\|Bz\|=\|SV^\top z\|\) are correct for a complete thin SVD. Dropping nonzero singular values changes the objective. Moreover, \(B=[\Phi,\Delta_h\Phi,U]\) mixes blocks with very different scales; a relative cutoff against the largest singular value is not an error certificate for the coefficient-dependent residual.

   **Fix:** Preserve the complete numerical range using a stable factorization, or explicitly label the truncation approximate and bound its residual/Jacobian effect over relevant states. Add comparison against the independently assembled **full DEIM residual**, including cancellation-sensitive cases.

4. **WRONG — The 2D snapshot-history claim is false.**

   **Evidence:** D:81–85 claims the twelve listed steps are “the bank’s snapshot steps.” The R512 JSON records `T=51`, `n_states_total=29376`, `n_states_trained=16384`, and `n_early_states_in_pick=3456` at lines 5545, 5559–5561. `experiments/separable-decoder/sep_burgers_r3.py:321–332` selects all early states and randomly samples remaining states. In contrast, C/inputs/model_W1024/training.json:14–26 contains exactly the twelve listed 3D steps.

   The 2D trajectory count and seed are supported; the snapshot-selection claim is not. The proposed 6,912-state POD training set gives the baseline substantially different temporal coverage.

   **Fix:** Reproduce the 2D bank’s state-selection recipe, or explicitly register a smaller, different snapshot set and justify its adequacy through a training-only saturation check.

5. **NEEDS-RESTATEMENT — Regenerating central-FD snapshots is reasonable, but violates the literal generator-reuse instruction.**

   **Evidence:** D:81–86 versus P/campaign/briefs/jcp-hr-baselines.md:27–31. `engines.py:41–51` and the vendored 3D `common.py:3–5` confirm sign-upwind generation.

   Matching snapshots to the projected central operator can strengthen the baseline. It is nevertheless **not reuse of the bank’s trajectory generator**: only the parameter draws are reused.

   **Fix:** Obtain a documented amendment distinguishing “same parameter distribution” from “same trajectory generator.” Report the advantage and changed offline cost explicitly; do not describe this as identical training data.

6. **WRONG — The design does not supply the same 3D time-stepping options as its comparator set.**

   **Evidence:** D:68–71 restricts baselines to BE/CN at 0.01; D:218 admits every qualified full-field speed3d configuration. S/DESIGN.md:60–63 includes BE, CN, BDF2 and four step sizes.

   A baseline restricted to two schemes at one step cannot support “same options as ours” against that unrestricted export.

   **Fix:** Either give baselines the corresponding scheme/step ladder, including BDF2 history handling, or restrict X to a frozen BE/CN-at-0.01 subset and narrow the claim accordingly.

7. **WRONG — Equal tuning budgets are replaced by disclosure of unequal budgets.**

   **Evidence:** D:12–14, 180–184; lane brief:27; P/PLAN-v3-amendments.md:89–93.

   A “tuned with less budget” label does not satisfy equal budgets. The accounting is also asymmetric: baseline snapshot/POD construction is included while ours’ bank construction is treated separately as sunk cost. Selected combined/P1/speed3d jobs are not a demonstrated complete inventory of selection effort.

   **Fix:** Freeze comparable accounting categories and allocations before tuning. Separate model construction, hyper-reduction construction, validation search and numerical certification. If equality is infeasible under the cap, explicitly change the hypothesis to an unequal-budget descriptive comparison.

8. **NEEDS-RESTATEMENT — The cohort comparison is developmental, but not symmetric in selection exposure.**

   **Evidence:** D:31–35, 186–196; C/DESIGN.md:864–871, 887–946, 1012–1015.

   Baselines select on six cases; ours benefited from broader development/validation evidence, including post-hoc selection. Thus `val32`/cases 6–63 are outside **baseline** selection, not outside selection for both methods. Conversely, baseline selection directly targets E0 error, whereas combined’s original 2D selections were frozen against earlier references.

   **Fix:** Label the primary result a developmental comparison with unequal selection histories. Prefer the same allowed validation cohort for both selection procedures. Reserve any generalization claim for a later jointly frozen cohort; do not call the secondary view held out for both methods.

9. **WRONG — Q1’s normalization is not exactly the inherited contract.**

   **Evidence:** D:163–166 uses \(\|F_0^{\mathrm{tight}}\|\). S/DESIGN.md:122–125 uses the physical initial-field norm; S/score3.py:138–147 constructs that norm from `initial_interior(65, ...)`.

   For POD, the tight solution’s initial field is \(\Phi\Phi^\top u_0\), whose norm generally differs from \(\|u_0\|\). This changes the threshold by rank and typically makes it stricter for a poorer basis.

   **Fix:** Normalize by the same canonical scoring-lattice physical IC norm used by the inherited certificate. Explicitly require valid, verified tight and tighter anchors; agreement between two stalled solves is insufficient.

   The fresh-Jacobian stationarity formula, pooled 1% threshold and every-case distance requirement otherwise match COMMON-BRIEF §3. Also fix the nonexistent `t2core.tol_of` reference: the supplied definition is C/c2common.py:32–36.

10. **NEEDS-RESTATEMENT — ECSW training has undefined zero-target behavior and weaker-than-advertised coverage.**

    **Evidence:** D:104–116.

    Scaling every block by \(1/\|b(s)\|\) is undefined when \(b(s)=0\) and unstable when projected states nearly satisfy stationarity. Candidate restriction and the support cap remove the all-ones feasibility guarantee. A missing tolerance crossing currently makes a configuration “not exist,” potentially hiding a baseline construction failure.

    **Fix:** Freeze a scale-aware block-normalization floor and treatment of zero targets. Report every attempted configuration, achieved tolerance and cap failure. Define a training-only fallback or classify the cell as incomplete if ECSW cannot be constructed adequately. Clarify duplicate startup states: adding both CN-R startup states to a set already containing step 1 changes weighting unless deduplicated.

11. **WRONG — The independent residual control cannot be applied literally to every arm.**

    **Evidence:** D:256; D:103 and 125–127.

    ECSW residual rows are weighted physical rows. Q-DEIM’s online residual consists of compressed coordinates, not physical sample rows. Those coordinates cannot equal `refcore` residual rows. At interpolation nodes, raw DEIM advection can match exactly, but that checks only part of the implementation.

    **Fix:** Split G-fd into:
    - raw stencil residual/Jacobian checks against an independent implementation;
    - weighted-row checks for ECSW;
    - full reconstructed DEIM residual and compressed objective/gradient checks for Q-DEIM;
    - directional FD checks of each actual online residual.

    Use nonzero, scale-aware perturbations; \(10^{-6}\|c\|\) degenerates at \(c=0\).

12. **WRONG — The wrong-gradient negative control need not fail.**

    **Evidence:** D:262; C/t2core.py:64–65, 76–99.

    Replacing \(J\) by \(J/2\) preserves the zeros of \(J^\top r\), and the stationarity ratio is invariant to this scaling. It changes step length, but LM can still converge to the correct stationary point. Failure is not guaranteed.

    **Fix:** Inject a gradient-direction error, such as dropping a demonstrably active nonlinear contribution, while certifying with the correct independently recomputed Jacobian. Freeze a nonstationary fixture and finite budget on which the mutation demonstrably triggers the intended gate.

13. **NEEDS-RESTATEMENT — The under-sampling control neither guarantees degradation nor fully tests the required failure.**

    **Evidence:** D:261 versus lane brief:32–34.

    Q-DEIM retains exact linear terms and can remain accurate with few nonlinear modes on diffusion-dominated cases. Under-sampled ECSW can instead achieve tiny sampled residuals with an underdetermined coefficient vector. Its principal failure may be distance to the tight/full target, which the parenthetical Q1-failure alternatives omit.

    **Fix:** Preselect representative difficult fixtures without using verdict outcomes. Include the complete Q1 failure predicate, explicitly distinguish solver failure from approximation degradation, and require the brief’s degraded-accuracy evidence. Failure of an arbitrary under-sampling threshold is not automatically an implementation bug.

14. **NEEDS-RESTATEMENT — Full-rank recovery and several other controls have incompatible or unjustified thresholds.**

    **Evidence:** D:254, 258, 260, 263, 266, 269–270.

    - G-full needs an explicitly completed \(n\)-dimensional basis and nonlinear basis, not merely available snapshot rank. A production stationarity tolerance of \(10^{-3}\) does not imply \(10^{-9}\) field agreement.
    - NNLS solutions/supports need not be unique; greedy early stopping need not reproduce SciPy’s fully converged support.
    - ST↔S swaps need not change a score by \(10\varepsilon_{\rm ref}\); S is prohibited for 3D scoring.
    - Halving synthetic errors does not ensure non-overlapping intervals unless uncertainties are also specified.

    **Fix:** Specify complete fixtures and tolerances. Compare NNLS objective/KKT feasibility rather than universal support identity. Test reference-kind rejection directly. Set synthetic uncertainty magnitudes explicitly. Keep small-mesh solver recovery separate from real-size implementation checks.

15. **CORRECT — The basic dominance directions and case-kind eligibility treatment follow the lane brief.**

    **Evidence:** D:37–43, 228–243; lane brief:38–46; R/refstore.py:132–170.

    Strict non-overlap in time and worst error, separate offline/median reporting, conditional-Richardson labeling and exclusion of ineligible case-kinds are appropriate. Endpoint propagation through maximum and median is mathematically valid conditional on the individual error bounds.

    **Fix:** Clip lower error endpoints at zero and explicitly call these *conditional reference-error envelopes*, not statistical confidence intervals for population worst-case error.

16. **NEEDS-RESTATEMENT — “Matches” does not establish equivalence of the declared worst-error objective.**

    **Evidence:** D:239–241.

    Equivalence of the **mean standardized paired error difference** permits cancellation: several improvements can hide one severe deterioration. It therefore does not imply equivalent worst errors. The denominator also depends on observed X errors; its treatment during resampling is unspecified. The timing clause does not explicitly say that cases, with all repeats, remain the resampling unit.

    **Fix:** Name the estimand precisely—“mean standardized error equivalence”—and avoid presenting it as worst-error matching. If overall matching is intended, add an explicit worst-error equivalence requirement. Freeze denominator handling, bootstrap construction and label precedence when equivalence and superiority tests both pass.

17. **NEEDS-RESTATEMENT — H-base has unresolved selection, multiplicity and empty-set cases.**

    **Evidence:** D:189–196, 235–247.

    - Two selected picks do not exhaust the baseline front; another candidate may dominate on the verdict cohort.
    - Tie-breaks are incomplete across HR methods, schemes and solvers.
    - No explicit result covers empty eligible sets, zero equivalence margins, missing X results, failed picks or both picks being `none`.
    - Many X/pick tests use unadjusted intervals, and the same cases influence selection and verdict.
    - Selection noise can create `none` or exclude a useful candidate.

    **Fix:** Freeze a total configuration ordering and an exhaustive decision table. Interpret “no selected pick dominates” narrowly; it is not proof that no baseline dominates. Either control the declared family of statistical claims or label intervals and verdicts exploratory and unadjusted. `none`, failed qualification and empty eligibility must yield “not assessed/incomplete,” never support for H-base.

18. **WRONG — Memory and disk bounds omit major mandatory arrays.**

    **Evidence:** D:120–127, 296–307.

    At \(1023^2\), \(r=384,p=1536\), f64 storage is approximately:
    - \(\Phi\): 3.21 GB;
    - \(\Delta_h\Phi\): 3.21 GB;
    - \(U_p\): 12.86 GB;
    - concatenated \(B\): 19.29 GB, before SVD workspace.

    Thus “\(\Phi\) and HR data ≤5 GB per cell” is not a valid general bound. The 12.9 GB ECSW contribution matrix is correctly calculated for 96 states at rank 512, but the design adds startup states in 2D. NNLS cross-products, factorization workspaces, nonlinear snapshots, QR copies and full-size Jacobian workspaces are also missing.

    **Fix:** Provide phase-specific live-array and disk inventories, including deletion/reuse points. Distinguish temporary full \(U/B\) storage from compact online artifacts. Measure maximum-rank offline phases, not only a reduced-training smoke. Specify how the thin-SVD fallback fits host/device memory.

19. **NEEDS-RESTATEMENT — The GPU-hour estimate is not supported by an executable workload inventory.**

    **Evidence:** D:176–178, 300–305; S/resources.py:4–12, 66–93.

    The stated grid alone implies 392 production configurations across three cells and approximately **3,920 selection trajectories including anchors**, before verdict runs, FOM floors, controls and training. Snapshot generation adds 2,688 trajectories. Compilation multiplicity depends on rank, support size, reduced rank and solver shapes, but no compile inventory is given.

    The smoke-certificate requirement is good; the 13 GPU-hour estimate is presently only a guess. Dropping common-solver runs after front selection also cannot generally save the runs needed to determine that front.

    **Fix:** Enumerate each job’s trajectories, stages, certification passes, factorizations and compiled shapes. Include maximum-rank POD/NNLS pilots and all anchors. Freeze a staged defer policy whose dependencies are executable; if required comparisons exceed 30 GPU-hours, declare the registered scope incomplete.

20. **NEEDS-RESTATEMENT — Several claims overstate baseline strength or disadvantage it without an adequate diagnostic.**

    **Evidence:** D:55–59, 146–153, 303–305.

    - An unreduced FOM is the full-rank limit, **not a guaranteed lower error bound**: reduced-model error can cancel discretization error.
    - Global spectral differentiation does not make every sampling-based reduced treatment impossible; dense derivative actions on a fixed trial basis can be precomputed.
    - Undamped GN with 30 iterations and a previous-state predictor is a fragile sole “native” solver specification. The common solver has a much larger budget and better predictor.
    - The common trust rule uses 0.01 in both dimensions, while S/DESIGN.md:57 uses 0.05 for 3D.
    - Dropping `none` targets can remove the proposed HR-error diagnostic.

    **Fix:** Rename the FOM “full-rank limiting comparator”; justify spectral deferral by scope/cost. Provide a robust, budgeted native-solver option and report convergence sensitivity. Match dimension-specific common-solver parameters or label the difference. Preserve sufficient `none` anchors to diagnose whether failure comes from POD, hyper-reduction or optimization.

**Top blockers:** incorrect 2D snapshot provenance; unequal tuning and 3D stepping options; Q1 normalization mismatch; nonfunctional controls; incomplete H-base matching/selection semantics; and unsubstantiated memory, disk and runtime bounds. Correct these and re-audit before implementation or production release.