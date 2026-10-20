**A3 repairs several major defects, but the design is still WRONG.** The remaining blockers are control expectations that are not mathematically guaranteed, an incorrect identical-arm test, incomplete numerical-failure handling, and a new relaxation of “equal budget.”

This was a read-only design audit. I made no changes, submitted no jobs, and did not audit the lane’s draft implementation. Passing controls and fitting the resource cap remain unverified.

Evidence shorthand: **D** = [baseline DESIGN.md](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-hr-baselines/experiments/jcp-hr-baselines/DESIGN.md); **Brief/Common** = the supplied campaign briefs; **R/C/S** = the supplied references/combined/speed3d experiment directories. Items 1–27 correspond to audit 3.

1. **CORRECT — The central-FD residual and ECSW gradient target remain mathematically consistent.**

   **Evidence:** D:52,66,103–108,338–343; R/refcore.py:64–87.

   For fixed nonnegative weights,
   \[
   \nabla_c\frac12\sum_i\xi_i\mathcal R_i(c)^2
   =\sum_i\xi_iJ_i(c)^\top\mathcal R_i(c).
   \]
   The all-node target is correct. Petrov–Galerkin ECSW applies to finite-difference models; it does not require a structural finite-element formulation. The design appropriately avoids claiming energy conservation. [Primary ECSW paper](https://arxiv.org/abs/2008.02891).

   **Fix:** Retain the construction and explicit weighted strong-residual label. Describe baseline strength as something the experiments must establish.

2. **CORRECT — The amended Jacobian is correct.**

   **Evidence:** D:340–343; C/t2core.py:35–38,165–175; R/refcore.py:64–87.

   It includes the mass coefficient, current-step nonlinear derivative, diffusion, and \(\Delta t b_0\). History is fixed during the current solve.

   **Fix:** Preserve the same expression in training, online evaluation, startup handling, and independent certification.

3. **CORRECT — The governing Q-DEIM compression is untruncated QR, not truncated SVD.**

   **Evidence:** D:345–352,529–534,765.

   With \(B=Q_BR_B\), minimizing \(\|R_Bz(c)\|\) preserves the reconstructed full-field DEIM least-squares objective algebraically. Linear terms remain exact; only nonlinear interpolation is approximate. A3 correctly downgrades the cancellation fixture to a *candidate*. Pivoted QR of each \(U_p^\top\) is the standard Q-DEIM selection construction. [Primary Q-DEIM paper](https://arxiv.org/abs/1505.00370).

   **Fix:** Keep the cancellation-scaled checks and report the observed cancellation ratio. Do not describe the superseded singular-value truncation as the implemented design.

4. **CORRECT — Snapshot inadequacy blocks the relevant favorable conclusion.**

   **Evidence:** D:354–365,536–542.

   Both the training saturation comparison and actual-cell projection check have consequences: failure withholds statements that ours is not dominated.

   **Fix:** Retain that consequence. Passing these tests supports the registered snapshot schedule; it does not establish exhaustive POD adequacy.

5. **CORRECT — A2.3 remains an honestly flagged, pending generator deviation.**

   **Evidence:** D:544–554; Brief:27–31; engines.py:41–51; vendored common.py:3–5.

   The design distinguishes reused parameter draws from changed trajectory fields and registers a probe and ratification fallback. I do **not** re-flag the deliberate deviation.

   **Fix:** Record the coordinator’s decision before production and retain the generator label beside results.

6. **CORRECT — Time-stepping coverage and option deferral are appropriately linked.**

   **Evidence:** D:367–370,556–558; C/t2core.py:35–38; S/DESIGN.md:60–63; R/refcore.py:202–217.

   The 3D options match speed3d’s BE/CN/BDF2 ladder. The distinction between two full BE startup steps and four half steps is correct. Deferring an option excludes the corresponding X from assessment.

   **Fix:** Carry that exclusion into exports and cell summaries.

7. **NEEDS-RESTATEMENT — A3 fixes the accounting categories but introduces an unauthorized definition of “equal.”**

   **Evidence:** D:675–685; Brief:27; PLAN-v3-amendments.md:89–93.

   Training-only versus all validation-informed work is the right distinction. Including selection, escalation, anchors, and qualification on both sides repairs audit 3’s central finding.

   However, baseline expenditure anywhere from **0.8 to 1.25 times** ours is now called “equal budget.” The binding brief does not establish that tolerance. This is a new substantive relaxation, not merely clearer bookkeeping.

   **Fix:** Call this “budget within the registered factor-1.25 band,” report the actual ratio, and obtain ratification if it is to satisfy the equal-budget requirement. Otherwise retain “unequal-budget, descriptive.” Freeze the actual task ledger, including all historical selection/escalation work.

8. **CORRECT — Stage B preserves the screening front; the comparison remains developmental.**

   **Evidence:** D:385–394,572–576; C/DESIGN.md:887–945.

   Every screening-front member survives, and the additional near-front selection is explicit. Final picks use the full development cohort.

   This is defensible in-sample comparison, **not symmetric selection history**: ours benefited from historical full-cohort and post-hoc decisions; a baseline eliminated on six cases never reaches full-cohort evaluation.

   **Fix:** Preserve the unequal-history disclosure and restrict conclusions to the screened/constructed configurations. Do not claim complete full-cohort frontier coverage.

9. **CORRECT — The production Q1 predicate matches the binding contract.**

   **Evidence:** D:158–172,396–405,578–582; Common:49–60; PLAN-v4-amendments.md:16–25.

   Fresh per-step Jacobians, pooled stationarity, physical-IC normalization, every-case tight-solve distance, and anchor agreement are specified. A2 withdraws the residual-tolerance exception.

   **Fix:** Keep solver termination distinct from certification. Report anchor invalidity separately from approximation error.

10. **CORRECT — ECSW startup coverage and zero-target handling are defined.**

    **Evidence:** D:407–418,584–591.

    Startup equations and BDF2 histories are represented. Deduplication is explicit; the positive floor and all-dropped outcome avoid division by zero.

    **Fix:** Consolidate the surviving normalization as division by \(\max(\|b(s)\|,\phi)\), with the stated dropping rule. Generate retained counts rather than inheriting “96 states.”

11. **CORRECT — The independent residual/Jacobian checks cover the actual formulations.**

    **Evidence:** D:424–426,529–534.

    Raw stencil checks, online directional finite differences, ECSW weighted-row checks, and independently assembled DEIM objective/gradient checks address distinct failure modes.

    **Fix:** Ensure independent assembly does not reuse the sampled-row/compression routine under test. Automatic differentiation of the same erroneous residual is insufficient by itself.

12. **WRONG — A3 fixes history propagation but still does not guarantee N-grad fires.**

    **Evidence:** D:687–693.

    Isolated tuples correctly eliminate the audit-3 history defect. But
    \[
    \|J(c)\delta\|=100\|\hat r(c)\|
    \]
    only motivates the claimed \(\gamma\) for a **locally linear** residual. Burgers residuals are quadratic, and the certifier evaluates both residual and Jacobian at \(c+\delta\). No bound makes the neglected nonlinear terms small.

    There is also a direct degenerate case: if \(\hat r(c)=0\), the prescribed perturbation can be zero and cannot produce non-stationarity. A zero leading singular value likewise leaves the prescription undefined.

    **Fix:** Register a bounded deterministic perturbation ladder, including both signs and a nonzero absolute scale. Use the independent oracle to establish a finite discriminating tuple before testing the certifier. If none exists, report “fixture unavailable”; do not call a correct certifier defective.

13. **CORRECT — Under-sampling now blocks acceptance per HR family.**

    **Evidence:** D:433,600–602; Brief:32–34.

    Finite accuracy degradation is required; solver failure is recorded separately. An unfulfilled family control withholds acceptance of that family’s comparisons.

    **Fix:** Preserve this gate. Demonstrate it in the smoke; it cannot be declared successful from the design alone.

14. **NEEDS-RESTATEMENT — G-pod is fixed, but the control fixtures are not fully frozen.**

    **Evidence:** D:604–615,695–703.

    G-pod now has the correct relative Frobenius normalization. Hand-derived N-dom labels are an improvement. N-swap now identifies reference and ROM fixtures.

    However, the ROM fixture still says “the smoke rank” and “first certified ECSW native BE run,” without fixing rank, \(\tau\), generator, or the outcome when no such run certifies. A successful reference fixture does not prove the ROM fixture discriminates. N-dom(d) also has the defect in item 16.

    **Fix:** Freeze complete fixture IDs and expected numerical inequalities. Define unavailable-fixture handling separately from implementation failure.

15. **CORRECT — Reference envelopes and eligibility are appropriate.**

    **Evidence:** D:37–43,228–234,441–443,647–648; R/refstore.py:132–173; R/verdict.py:148–151.

    Clipped endpoints propagated through maxima and medians are valid conditional envelopes. They are correctly distinguished from population confidence intervals. Missing eligible cases invalidate assessment.

    **Fix:** Require identical case IDs, metric, reference version, and eligibility mask for both arms.

16. **WRONG — A3 overgeneralizes the identical-arm degeneracy and gives N-dom an incorrect unconditional label.**

    **Evidence:** D:617–619,701–707.

    The registered margin is
    \[
    \delta_W=\max(2\max_i\varepsilon_i,\;0.1L_X).
    \]
    Equal arms with equal \(\varepsilon\) are unresolved through the worst-error guard **only when the reference-uncertainty term controls the margin**, subject also to clipping.

    For example, identical errors \(0.01\) and \(\varepsilon=10^{-6}\) yield a difference envelope \([-2\times10^{-6},2\times10^{-6}]\), strictly inside a margin approximately \(10^{-3}\). With identical timing, the registered table says **matches**, not unresolved.

    **Fix:** Make N-dom(d) explicitly reference-limited and specify its numbers. Add a second identical-arm fixture where the relative margin dominates and the expected label is “matches.” Do not add an identical-arm override to the decision table.

17. **CORRECT — Cell accounting now distinguishes incomplete and unresolved evidence.**

    **Evidence:** D:644–655,709–713.

    Attempted, assessed, unresolved, invalid, deferred, control-blocked, and failed-pick counts repair the denominator problem. The unadjusted, developmental wording appropriately limits the multiplicity claim.

    **Fix:** Define assessed pairs as those with valid inputs receiving a substantive decision; unresolved is a reported subset. Deduplicate a configuration selected for both roles. Retain “observed domination,” not formal familywise falsification.

18. **CORRECT — A3 repairs the specific maximum-branch shape omissions.**

    **Evidence:** D:715–725.

    The 2D extension, rank-1024 3D branch, amended ECSW state counts, and support-retry buffer are now represented. The stated decimal-GB matrix sizes are consistent with their shapes.

    **Fix:** Preserve the requirement to cover **every** allowed phase, including full snapshot Gram/eigensolve, thin-SVD fallback, pivoting, workspace, and coexistence. Reduced-snapshot pilots alone cannot establish the peak of snapshot-count-dependent phases.

19. **NEEDS-RESTATEMENT — Feasibility remains conditional, and “40” is not a conservative Stage-B bound.**

    **Evidence:** D:627–631,727–730; Brief:13–15; S/resources.py:4–12.

    A screening front can contain every candidate. “\(|F|\le40\), otherwise recertify” is a planning assumption with an abort policy, not a bound.

    The enlarged 3D base grid alone contains \(6\times7\times12\times2=1008\) configurations before construction failures or deferrals: 6,048 Stage-A production runs, plus anchors and certification. No measured evidence establishes completion within 30 GPU-h or the phase-I one-hour allowance.

    **Fix:** Label the preliminary projection conditional. Before Stage B, certify its actual frozen membership; before production, include all snapshot-dependent phases, fallbacks, retained artifacts, compilations, controls, and retries. If it does not fit, report incomplete scope.

20. **CORRECT — The common-solver ablation is no longer pruned by native outcomes.**

    **Evidence:** D:511–521,633–635; C/c3common.py:95–100; S/DESIGN.md:56–58.

    Both solvers run for retained configurations. Backtracked GN is a credible native baseline. The FOM is correctly called a limiting comparator, and spectral projection is deferred for scope rather than declared impossible.

    **Fix:** Preserve both solver arms through any rank/option deferral and report their failures separately.

21. **NEEDS-RESTATEMENT — The order fixture is specified, but the prescribed acceptance ranges remain an unverified assumption.**

    **Evidence:** D:637–642,732–735; engines.py:33–38.

    Mesh, case, viscosity, final-time norm, and self-convergence estimator are now frozen. This fixes the missing-fixture specification.

    But a real Gaussian trajectory—especially one with wall values forcibly zeroed—does not guarantee asymptotic order within these particular step sizes. A correct implementation could miss the narrow bands. The independent step-equation check remains valuable.

    **Fix:** Add a smooth manufactured semidiscrete fixture with independently specified forcing, or a bounded predeclared refinement ladder. Distinguish “not yet asymptotic” from an incorrect scheme.

22. **CORRECT — A3 fixes the two concrete timing defects from audit 3.**

    **Evidence:** D:737–742.

    Running the same case twice removes the unequal-work error in N-time. Requiring all six timing cases and using equal case weighting removes survivor-population bias.

    **Fix:** Freeze whether “three re-measurements” applies per original triple, and retain rejected measurements and retry counts. Resolve the remaining estimator issue in item 28.

23. **CORRECT — Imported comparators now require Q1 and own-state Q2.**

    **Evidence:** D:744–745; Common:57–60; PLAN-v3-amendments.md:73.

    This directly fixes the missing 3D quadrature qualification. Q1-only configurations are diagnostic.

    **Fix:** Verify that each certificate identifies the exact imported configuration and required population. If none qualifies, report no assessed comparator; do not substitute historical finalist status.

24. **CORRECT — The registered 3D rank grid now includes 1024.**

    **Evidence:** D:715–718; Brief:21–23; S/DESIGN.md:50–54.

    The missing equal-rank branch is registered.

    **Fix:** Distinguish registered coverage from completed, constructible, Q1-qualified coverage. A rank deferred or rejected numerically does not count as completed rank coverage.

25. **CORRECT — ECSW construction failures now restrict family-level claims.**

    **Evidence:** D:747–749.

    Failed boundary constructions produce an explicit incomplete-coverage label and prohibit “adequately tuned ECSW” or exhaustive non-domination wording.

    **Fix:** Export coverage by rank, tolerance, and failure reason, including numerical failures and banned-column termination under A3.13.

26. **CORRECT — The corrected provenance claims agree with the supplied files.**

    **Evidence:** D:354–361,751–754; 2D training JSON:5,19,33–36,5544–5561,5582; C/inputs/model_W1024/training.json:3–27; burgers2d_film.py:160–185; R/cohorts.py:35–37,82–93,134–150.

    Verified: training seeds/counts, 2D selected-state count, 3D snapshot indices, nodes versus intervals, dimensional timesteps, cohort construction, and IC conventions. The 2D bank-fit time is approximately 10,684.65 seconds.

    **Fix:** Compare physical parameter columns from `sample_params`, whose return also contains normalized float32 descriptors. Keep bank-fit and total-job costs separately attributed.

27. **NEEDS-RESTATEMENT — A3 adds failure reporting but does not fully repair offline numerical robustness.**

    **Evidence:** D:87–90,756–764.

    The DEIM backward-error check is useful. However:

    - A numerical-rank cutoff does not specify a stable advection-POD fallback. Singular values near \(10^{-10}\sigma_1\) cannot reliably be classified through a squared-condition-number Gram calculation.
    - “Minimum-norm least-squares solve” does not identify whether it operates on \(G_S\) or the normal matrix. The latter retains the conditioning problem.
    - Permanently banning an immediately dropped index can prevent a later valid NNLS entry; reporting the ban does not establish optimality over the original candidate pool.

    **Fix:** Apply a stable direct/blocked SVD fallback explicitly to both PODs. Perform passive-set fallback on the original contribution matrix. Check KKT conditions against all original candidates and classify unresolved violations as incomplete construction.

28. **NEEDS-RESTATEMENT — The timing population is now common, but selection and verdict still use different estimands.**

    **Evidence:** D:187–194,208–209,737–742.

    Role selection compares median absolute query times. The verdict uses the median of paired case ratios. In general,
    \[
    \frac{\operatorname{median}_i t_{B,i}}
         {\operatorname{median}_i t_{X,i}}
    \ne
    \operatorname{median}_i\frac{t_{B,i}}{t_{X,i}}.
    \]
    Consequently, “equal cost” selection and “faster” judgment can disagree even without noise. A3’s statement that the same estimator is used obscures this distinction.

    **Fix:** Either use paired ratios for the X-specific cost role, or explicitly label role selection as median-seconds screening and the verdict as median paired relative time.

29. **NEEDS-RESTATEMENT — Small-mesh recovery needs an explicit comparison grid.**

    **Evidence:** D:428,605–606; R/refcore.py:222–235.

    G-full uses \(L=16\) and \(L=8\) but specifies “lattice distance.” The inherited scoring-lattice restriction requires \(L\) to be a multiple of 256 or 64; it cannot directly score these small meshes.

    **Fix:** Define G-full on the small mesh’s own nodal field with a fixed normalization, or specify a common interpolation operator. This is a recovery gate, so E0-lattice scoring is unnecessary.

**Overall verdict: WRONG.**

Top blockers: **N-grad’s unsupported guaranteed failure; N-dom’s incorrect identical-arm expectation; the new “equal-budget” tolerance; and incomplete offline factorization/NNLS failure policies.** A3 successfully repairs audit-3 items **18, 22, and 23** at design level, but resource feasibility and actual control firing still require the registered smoke evidence.