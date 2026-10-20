**A4 fixes audit 4’s WRONG items 12 and 16. The overall design still NEEDS-RESTATEMENT**, principally around offline POD certification, timing-based pruning, and incomplete control-fixture specifications.

This was read-only: no files changed, no jobs submitted, and no draft lane implementation audited. Resource feasibility and actual control firing remain smoke-stage questions; I assess whether failure is visible and blocks the appropriate conclusion.

Evidence shorthand: **D** = [baseline DESIGN.md](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-hr-baselines/experiments/jcp-hr-baselines/DESIGN.md); **Brief/Common** = the supplied campaign briefs; **R/C/S** = the supplied references/combined/speed3d experiment directories. Items 1–29 correspond to audit 4.

1. **CORRECT — Central-FD residual and ECSW gradient-contribution target.**

   **Evidence:** D:52,66,103–108,338–343; R/refcore.py:64–87.

   For fixed nonnegative weights,
   \[
   \nabla_c\frac12\sum_i\xi_i\mathcal R_i(c)^2
   =\sum_i\xi_iJ_i(c)^\top\mathcal R_i(c).
   \]
   Training against the all-node sum is mathematically consistent. Petrov–Galerkin ECSW explicitly accommodates finite-difference discretizations. [Primary ECSW paper](https://arxiv.org/abs/2008.02891).

   **Fix:** None to the formula. Retain the weighted strong-residual label and avoid claiming energy conservation or demonstrated baseline strength before results.

2. **CORRECT — The amended residual Jacobian.**

   **Evidence:** D:340–343; R/refcore.py:64–87; C/t2core.py:35–38,165–175.

   The mass coefficient, nonlinear derivative, diffusion sign, and \(\Delta t b_0\) factor are correct. History is held fixed during differentiation.

   **Fix:** Preserve this expression for each startup/main-step equation and independent checks.

3. **CORRECT — Q-DEIM preserves the linear terms under the governing amendment.**

   **Evidence:** D:119–127,345–352,529–534.

   The governing construction is **untruncated Householder QR**, replacing the original truncated SVD. Algebraically,
   \[
   \|Bz(c)\|_2=\|R_Bz(c)\|_2.
   \]
   Only the nonlinear interpolation is approximate. Pivoted QR of each \(U_p^\top\) is the standard Q-DEIM selection mechanism. [Primary Q-DEIM paper](https://arxiv.org/abs/1505.00370).

   **Fix:** Describe the final method as QR compression; retain the cancellation-scaled numerical checks.

4. **CORRECT — Snapshot inadequacy has an appropriate consequence.**

   **Evidence:** D:354–365,536–542.

   Training saturation and actual-cell projection checks are registered. Failure blocks claims that ours is not dominated, while allowing observed baseline domination to be reported.

   **Fix:** Retain the limitation: these checks establish the registered adequacy criteria, not exhaustive POD optimality or training coverage.

5. **CORRECT — The generator deviation is handled honestly.**

   **Evidence:** D:544–554; Brief:27–31; engines.py:41–51; vendored common.py:3–5.

   The design distinguishes identical physical draws from different trajectory fields, explicitly acknowledges departure from the brief, and registers a probe and ratification fallback.

   **Fix:** Record the coordinator’s decision before production. Keep the actual generator beside every result, including fallback results.

6. **CORRECT — Time-stepping coverage matches the stated comparator ladder.**

   **Evidence:** D:367–370,556–558; C/t2core.py:35–38; S/DESIGN.md:60–63; R/refcore.py:202–217.

   The 3D ladder includes BE/CN/BDF2 and all four step factors. The distinction between two full BE startup steps and four half steps is accurate. Deferred options exclude corresponding comparators.

   **Fix:** Export those exclusions explicitly.

7. **CORRECT — A4 honestly handles the pending budget-band ratification.**

   **Evidence:** D:675–685,788–791; Brief:27; PLAN-v3-amendments.md:89–93.

   A4 withdraws the unsupported assertion that expenditure within \([0.8,1.25]\) is automatically “equal budget.” Training-only and validation-informed costs are separated, and the actual ratio is reported.

   **Fix:** At release, record ratification or retain explicitly descriptive, non-equal-budget wording. Pin the task ledger, including historical escalation and qualification work.

8. **CORRECT — Developmental comparison is defensible, but selection histories remain unequal.**

   **Evidence:** D:385–394,572–576; C/DESIGN.md:887–902,931–945; C/report.md:3.

   Stage A screens six cases; Stage B selects and judges on the full development cohort. That is legitimate exploratory, in-sample comparison. It does not reproduce ours’ historical full-cohort and post-hoc selection exposure.

   **Fix:** Retain the unequal-history disclosure and restrict coverage claims to configurations actually carried through screening. See item 28 for a new timing-related pruning problem.

9. **CORRECT — The production Q1 predicate matches the binding contract.**

   **Evidence:** D:158–169,396–405,578–582; Common:49–60; PLAN-v4-amendments.md:16–25.

   Fresh per-step certification, pooled nonstationarity, every-case solver distance, physical-IC normalization, and tight/tighter agreement are specified. The residual-tolerance exception has been withdrawn.

   **Fix:** Keep termination and certification separate; report invalid anchors distinctly.

10. **CORRECT — ECSW startup coverage and zero-target handling.**

    **Evidence:** D:407–418,584–591.

    Startup equations and BDF2 histories are included. The positive scaling floor, deduplication, and all-dropped outcome are explicit.

    **Fix:** Generate actual retained state/block counts and use the final normalization consistently.

11. **CORRECT — Independent checks cover all baseline formulations.**

    **Evidence:** D:424–426,529–534; Brief:32–34.

    Raw-stencil checks, directional finite differences, weighted ECSW rows, and independently assembled DEIM objectives/gradients cover distinct errors.

    **Fix:** Preserve independence: the reference assembly must not call the sampled-row or compression routine being tested.

12. **CORRECT — A4 repairs the mathematical defect in N-grad.**

    **Evidence:** D:773–780.

    A4 no longer assumes that a linearized perturbation guarantees nonstationarity for a quadratic residual. It uses a bounded ladder, both signs, a nonzero absolute scale, and an oracle. Zero singular values and unavailable discriminating fixtures are acknowledged.

    **Fix:** Retain oracle agreement as the binding test. Explicitly keep Q1 acceptance blocked when the required discriminating fixture is unavailable; see item 30.

13. **CORRECT — Under-sampling failure blocks the relevant HR family.**

    **Evidence:** D:433,600–602; Brief:32–34.

    Finite accuracy degradation is required. Solver failure alone does not count. An unfulfilled control prevents acceptance of comparisons using that HR family.

    **Fix:** Preserve this consequence. Actual firing belongs to the smoke; the design need not prove it now.

14. **NEEDS-RESTATEMENT — A4 fixes N-swap’s main identity ambiguity, but its fallback is incomplete.**

    **Evidence:** D:793–797.

    Rank, tolerance, scheme, timestep, generator, and case are now specified. However, the first fallback is triggered by nonexistence **or failure to certify**, whereas the terminal condition says only “if neither exists.” A `none` fallback that exists but fails certification has no explicit outcome.

    **Fix:** Require a certified-valid fallback; if neither candidate is certified-valid, report “fixture unavailable.” State whether the reference-only control permits scoring acceptance or whether the ROM control remains blocking.

15. **CORRECT — Reference envelopes and eligibility.**

    **Evidence:** D:37–43,441–443,647–648; R/refstore.py:132–173; R/verdict.py:148–151.

    Clipped endpoints propagated through maxima and medians give the registered conditional envelopes. Ineligible cases remain visible without entering rankings.

    **Fix:** Require matching case IDs, metric, reference version, and eligibility masks for both arms.

16. **CORRECT — A4 fixes both identical-arm regimes.**

    **Evidence:** D:617–619,782–786.

    For d1, the clipped worst envelopes are \([0,10^{-4}]\); their difference envelope is \([-10^{-4},10^{-4}]\), which fails strict inclusion in the stated margin. “Unresolved” is correct.

    For d2, the difference envelope is \([-2\times10^{-6},2\times10^{-6}]\), comfortably inside the relative margin. With identical timing, “matches” is correct.

    **Fix:** Retain both fixtures and no identical-arm override.

17. **CORRECT — Decision precedence, multiplicity disclosure, and denominators are substantially repaired.**

    **Evidence:** D:452–464,644–655,709–713,827–829.

    Dominance, crossing objectives, equivalence, and unresolved outcomes have explicit precedence. Failed picks, empty fronts, missing eligible data, and incomplete coverage are visible. Duplicate picks count once.

    Unadjusted, development-selected intervals cannot establish familywise confirmation; the governing wording now avoids claiming that.

    **Fix:** Retain “observed domination” and report assessed, unresolved, and blocked counts together.

18. **CORRECT — The resource inventory now names the important maximum branches.**

    **Evidence:** D:466–485,623–625,715–725.

    The larger ranks, ECSW buffers, QR matrices, and `none` Jacobians are represented. The stated maximum matrix sizes are consistent with their shapes.

    **Fix:** The generated certificate must include coexistence, workspaces, full snapshot-count Gram/eigensolve phases, and the revised fallback in item 27. Reduced-snapshot pilots alone must not certify snapshot-count-dependent peaks.

19. **CORRECT — A4 makes feasibility conditional and failure visible.**

    **Evidence:** D:627–631,727–730,799–802; Brief:9–15; S/resources.py:4–12.

    The unsupported aggregate estimate is withdrawn. The 3D base count of 1,008 configurations and 6,048 Stage-A production runs is correct. Actual Stage-B membership must be cost-certified before execution; failure means incomplete scope.

    **Fix:** Include all construction, compilation, controls, retries, retained artifacts, and contingency in the certificate. Neither the 30 GPU-h cap nor the smoke allowance is established yet, but that is not a present design defect.

20. **CORRECT — Native/common solver coverage and comparator naming.**

    **Evidence:** D:511–521,633–635; C/c3common.py:95–100; C/t2core.py:74–105.

    Backtracked GN is a credible native solver, and common-solver runs are no longer pruned by native results. The FOM is correctly described as a limiting comparator rather than a strict error floor. Spectral projection is deferred for scope.

    **Fix:** Preserve both solver arms and report their failures separately.

21. **CORRECT — The order test now distinguishes unresolved order from implementation failure.**

    **Evidence:** D:637–642,732–735,804–808.

    A bounded refinement ladder replaces the demand that the original step sizes necessarily exhibit asymptotic order. Finest-level failure blocks that scheme pending investigation; the independent step-equation check remains binding.

    **Fix:** List the exact \(h,h/2,h/4\) triples in the executable fixture manifest to avoid ambiguity in “pairs.”

22. **CORRECT — Timing population, doubled-work control, and retries.**

    **Evidence:** D:737–742,827–829.

    N-time executes the same case twice. Every timing case must retain a valid measurement; equal case weighting avoids survivor-population bias. Retry limits now apply per original triple, and rejected measurements persist.

    **Fix:** Retain all raw measurements and rejection reasons. Actual control firing remains a smoke requirement.

23. **CORRECT — Imported comparators require their own Q1 and Q2 certificates.**

    **Evidence:** D:744–745; Common:57–60; C/report.md:8–10,346–348.

    Historical finalist status is insufficient, as the source report itself demonstrates.

    **Fix:** Require certificates for the exact imported configuration and population; otherwise report no assessed comparator.

24. **CORRECT — The registered 3D rank grid reaches 1024.**

    **Evidence:** D:715–718; Brief:21–23; S/DESIGN.md:50–54.

    The previous equal-rank omission is repaired.

    **Fix:** Distinguish registered, constructed, completed, and qualified rank coverage.

25. **CORRECT — ECSW construction failures restrict favorable claims.**

    **Evidence:** D:747–749.

    Failed boundary constructions produce an incomplete-coverage label and prevent claims of adequately tuned or exhaustively tested ECSW.

    **Fix:** Include numerical failures, rank rejection, support exhaustion, and unresolved KKT conditions in those coverage records.

26. **CORRECT — The corrected dependency/provenance claims agree with the supplied files.**

    **Evidence:** D:354–361,751–754; training JSON:5,19,33–36,5544–5561,5582; burgers2d_film.py:160–185; sep_burgers_r3.py:321–332; C/inputs/model_W1024/training.json:3–27; R/cohorts.py:35–37,82–93,134–150.

    Verified: training draws/seeds, 2D early-state sampling, 3D snapshot steps, nodes-versus-intervals conventions, dimensional timesteps, cohort construction, and IC conventions. The 2D bank-fit time is approximately 10,684.65 seconds.

    **Fix:** Compare physical parameter columns, excluding `sample_params`’ normalized float32 descriptors. Keep bank-fit and whole-job costs distinct.

27. **NEEDS-RESTATEMENT — A4 improves NNLS robustness, but does not fully repair POD numerical certification.**

    **Evidence:** D:87–90,608–610,756–764,810–818.

    Solving the passive fallback on original \(G_S\) columns is correct. Checking banned candidates at KKT termination also addresses the earlier omission; a feasible \(\tau\)-stopped approximation need not be the global NNLS optimum.

    The POD fallback remains underspecified:

    - A global projection-error check within 1% need not detect erroneous tiny singular values near the \(10^{-10}\sigma_1\) rank cutoff.
    - A rank rejected before that check may never trigger fallback.
    - A truncated randomized SVD does not provide the full discarded spectrum used by the registered tail formula. Reusing its approximate spectrum is not an independent optimality certificate.
    - The random seed and target sketch dimension are not frozen.

    **Fix:** Trigger a stable factorization on numerical-rank uncertainty before rejecting a requested rank. Specify sketch rank, seed, and an independently justified error/rank certificate, or retain a stable direct/blocked SVD fallback. Define the NNLS gradient sign explicitly so A2.11 and A4.7 cannot be implemented with opposite conventions.

28. **NEEDS-RESTATEMENT — A4 repairs B_cost’s threshold but leaves incompatible front pruning.**

    **Evidence:** D:187–195,572–576,820–822.

    B_cost now uses paired ratios, but the inherited front and near-front filter still use median absolute seconds. That filter can eliminate a baseline that would dominate under the new timing estimand.

    For six cases, consider:
    \[
    t_X=(1,1,100,100,100,100),
    \]
    \[
    t_B=(0.5,0.5,50,50,50,50),\qquad
    t_C=(2,2,3,3,200,200).
    \]
    Absolute medians are \(50\) and \(3\), so C can prune B if C has slightly better error. Yet median paired ratios are \(0.5\) for B and \(2\) for C. B can dominate X while C cannot. B also fails the factor-1.5 absolute-time near-front condition.

    **Fix:** Preserve the union of X-specific fronts using paired ratios, and select roles from those fronts. Alternatively, explicitly retain absolute-time screening as a limitation and stop describing the entire selection procedure as using one timing estimand.

29. **CORRECT — G-full now has a valid small-mesh metric.**

    **Evidence:** D:428,605–606,824–825; R/refcore.py:222–235.

    Comparing on the fixture’s own nodes avoids the incompatible E0 lattice restriction. Physical-IC normalization and the recovery threshold are explicit.

    **Fix:** Retain the dedicated recovery tolerances and all-points constructions for every baseline.

30. **NEEDS-RESTATEMENT — The global smoke fixture and unavailable-control consequences remain incompletely frozen.**

    **Evidence:** D:433–434,593–598,773–780,793–797; Common:20–23; Brief:13–15.

    A4 freezes N-swap’s rank, but N-under still refers to “the smoke rank,” and N-grad refers to “the frozen fixture run” without a complete shared fixture ID. A4’s new “fixture unavailable” outcome correctly avoids accusing a certifier of a mathematical impossibility, but it does not explicitly state the resulting acceptance block.

    **Fix:** Register complete per-dimension fixture IDs, including rank, HR tolerance, generator, solver, timestep, case selection, and reduced-training subset. State that unavailable required controls leave the relevant acceptance gate unfulfilled; they are neither passes nor implementation defects.

**Overall verdict: NEEDS-RESTATEMENT.**

Top blockers are **POD rank/optimality certification (27), paired-ratio selection versus absolute-time pruning (28), and complete control-fixture/fallback semantics (14, 30)**. A4 resolves the earlier WRONG findings 12 and 16. The generator and budget-band deviations are honestly pending ratification; resource feasibility and real-data control firing appropriately remain for the registered smoke.