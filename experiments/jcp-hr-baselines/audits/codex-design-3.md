**Overall verdict: WRONG.** A2 fixes much of audit 2, including the residual/Jacobian definitions, the Q1 predicate, startup coverage, screening-front preservation, and several failure policies. Remaining blockers concern validation-budget accounting, invalid control expectations, resource coverage, and qualification of imported comparators.

Items **1–22 below correspond to audit 2**. Later items are additional findings. These are design findings; I did not run solvers or establish that the proposed GPU controls pass. No files were changed and no jobs were submitted.

Evidence shorthand:

- **D** = [baseline DESIGN.md](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-hr-baselines/experiments/jcp-hr-baselines/DESIGN.md).
- **Brief/Common** = [lane brief](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-17-full-experiment-plan/campaign/briefs/jcp-hr-baselines.md) / [COMMON-BRIEF.md](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-17-full-experiment-plan/campaign/briefs/COMMON-BRIEF.md).
- **R/C/S** = the supplied `jcp-references`, `jcp-combined`, and `jcp-speed3d` experiment directories.
- **V3/V4** = the supplied plan amendments.

1. **CORRECT — Strong-residual LSPG and the ECSW target remain mathematically consistent.**

   **Evidence:** D:66, 103–108, 338–343; R/refcore.py:64–87.

   For fixed nonnegative weights,
   \[
   \nabla_c\frac12\sum_i\xi_i\mathcal R_i(c)^2
   =\sum_i\xi_iJ_i(c)^\top\mathcal R_i(c).
   \]
   Training against the full-node sum is appropriate. Petrov–Galerkin ECSW is applicable to finite-difference models; it is not restricted to structural finite elements. The withdrawal of an energy-conservation claim is appropriate. See the [primary ECSW paper](https://arxiv.org/abs/2008.02891).

   **Fix:** Retain this construction and its explicit weighted strong-residual label. Do not infer baseline strength merely from its mathematical validity.

2. **CORRECT — The amended Jacobian has the correct temporal, nonlinear, and diffusion factors.**

   **Evidence:** D:340–343; C/t2core.py:35–37; R/refcore.py:64–87.

   The derivative is
   \[
   J=\left[a_0I+\Delta t\,b_0
   \{\operatorname{diag}(D_hu)+\operatorname{diag}(u)D_h-\nu\Delta_h\}\right]\Phi.
   \]
   History terms are constant during the current step.

   **Fix:** Use this expression consistently in online residuals, ECSW training, certification, and startup tests.

3. **CORRECT — A2 repairs the cancellation-sensitive Q-DEIM comparison.**

   **Evidence:** D:345–352, 529–534.

   The governing construction is now **untruncated QR**, not the original truncated SVD. It preserves the norm of the reconstructed DEIM residual algebraically, including all linear terms. The new tolerances scale with assembled terms rather than a potentially vanishing residual or gradient.

   Q-DEIM’s pivoted QR of each \(U_p^\top\) is the standard selection construction. See [Drmač–Gugercin](https://epubs.siam.org/doi/abs/10.1137/15M1019271).

   **Fix:** Retain the amended checks. Describe the eight-iteration fixture as a cancellation *candidate* and record its actual cancellation ratio; eight iterations do not guarantee cancellation.

4. **CORRECT — Snapshot inadequacy now blocks the relevant favorable claim.**

   **Evidence:** D:354–365, 536–542.

   A2 adds an actual-cell projection check and withholds statements that ours is not dominated when either adequacy check fails. This fixes the previous policy of continuing with a knowingly restricted baseline while merely attaching a label.

   **Fix:** Keep the failure consequence. Describe passing as evidence about the registered snapshot schedule, not proof that the POD training set or rank ladder is sufficient. If the all-step fallback is used, its formerly “held-out” steps are no longer held out.

5. **CORRECT — The generator deviation is honestly handled, subject to ratification.**

   **Evidence:** D:544–554; Brief:27–31; `engines.py`:41–51; vendored `common.py`:3–5.

   A2 explicitly distinguishes reused parameter draws from changed trajectory fields, supplies a scientific rationale, registers a generator probe, and makes the primary choice subject to coordinator ratification. This is adequate handling of a pending deviation; it is not appropriate to repeat audit 2’s objection as though the departure were concealed.

   **Fix:** Record ratification before production. Clarify that the probe compares generators **on the lane’s meshes and snapshot schedule**, rather than reproducing the bank’s complete historical training dataset. See item 26.

6. **CORRECT — Deferred time options now restrict the comparator set.**

   **Evidence:** D:367–370, 556–558; C/t2core.py:35–37; S/DESIGN.md:60–63; R/refcore.py:202–217.

   The 3D ladder matches speed3d, and A2 prevents claiming complete coverage against an X whose option was deferred. The distinction between two full BE startup steps and four half BE startup stages is correct.

   **Fix:** Carry the option-deferred status into exports and every cell-level statement.

7. **WRONG — The ledger still excludes validation-driven tuning from the equal-budget comparison.**

   **Evidence:** D:560–570; Brief:27; V3:89–93; C/report.md:181–195; C/DESIGN.md:887–945.

   A2 assigns quadrature-rule **selection and escalation** to C-hr, then grants equal allocation only for C-search. Those rule decisions used validation trajectories and materially selected ours’ deployed method. Calling their cost “HR construction” does not remove it from validation tuning.

   Additionally, case (b) explicitly permits exceeding the equal allocation. Disclosure makes the comparison honest, but does not make its budget equal. Historical certification work is included in ours’ C-search when inseparable, while new baseline certification can be excluded, creating another accounting asymmetry.

   **Fix:** Separate training-only construction from **all validation-informed selection**, including rule selection, fallback qualification, Stage B, and relevant probes. Freeze comparable measured totals or explicit bounds. Label both budget overruns and shortfalls “unequal-budget, descriptive”; equal allocation should never be reported as equal expenditure.

8. **CORRECT — A2 fixes the particular Stage-B pruning defect.**

   **Evidence:** D:572–576, 385–394; C/report.md:3.

   The near-front relation is now algebraic, every screening-front member survives, and extra slots preserve both fast and accurate candidates. The full-cohort, in-sample developmental interpretation is defensible.

   It is **not selection-symmetric** in the stronger sense: ours has historical full-cohort and post-hoc selection, while the baseline first screens on six cases. A configuration discarded on six cases can perform well on the remaining cases.

   **Fix:** Retain the unequal-history disclosure and restrict conclusions to the configurations actually compared. “Complete screening-front coverage” must not become “complete full-cohort frontier coverage.”

9. **CORRECT — The production Q1 predicate now matches the common contract.**

   **Evidence:** D:158–172, 396–405, 578–582; Common:49–60; V4:16–21.

   The residual-tolerance exception is withdrawn. Certification uses fresh Jacobians at every saved step in the same job; stationarity is pooled over steps; distances use the physical IC normalization; invalid anchors fail closed.

   **Fix:** Preserve the separation between solver termination and certification. Report anchor invalidity separately from poor accuracy or unsuccessful hyper-reduction.

10. **CORRECT — ECSW startup coverage and the all-zero-target outcome are now defined.**

    **Evidence:** D:407–418, 584–591.

    The amended schedule includes actual startup states, BDF2 history, deduplication, and a strictly positive scaling floor. An all-dropped training set explicitly fails construction.

    **Fix:** State the final retained-block normalization formula explicitly—presumably division by \(\max(\|b(s)\|,\phi)\)—and generate counts after deduplication and dropping. The old 96-state memory estimates no longer apply; see item 18.

11. **CORRECT — Independent checks cover the actual baseline formulations.**

    **Evidence:** D:424–426, 529–534.

    Raw stencil checks, directional finite differences of each online residual, weighted ECSW row comparisons, and independently assembled DEIM objective/gradient comparisons address different possible errors. They satisfy the brief’s required per-baseline residual/Jacobian check at design level.

    **Fix:** Ensure independent assembly does not call the same sampled-row or compression routine being tested. Successful automatic differentiation alone cannot establish independence.

12. **WRONG — The new N-grad expectation mishandles temporal history.**

    **Evidence:** D:593–598; residual definition D:66; C/t2core.py:165–175.

    Replacing a saved \(c_k\) changes more than the equation whose returned state is \(c_k\). It also changes the history of the next step, and another step for BDF2. Consequently, an “unperturbed” later state can correctly become non-stationary. Requiring the certifier to flag **none** of those states is an invalid expected outcome.

    A random perturbation with norm \(0.01\|c_k\|\) also does not guarantee \(\gamma>10^{-3}\).

    **Fix:** Test isolated residual tuples with unchanged original histories, or independently recompute expected labels for every equation affected by the mutated trajectory. Establish discriminating perturbations with an independent oracle before production. Keep the wrong-gradient solver as a reported mutation rather than assuming it must fail.

13. **CORRECT — The under-sampling control now has the required blocking consequence.**

    **Evidence:** D:433, 600–602; Brief:32–34.

    If no registered under-sampling level demonstrates degraded accuracy, acceptance of comparisons using that HR family is withheld. Solver failure is correctly distinguished from measured accuracy degradation.

    **Fix:** Preserve the per-family gate. A failed or non-finite trajectory cannot supply the required finite error-increase evidence. Passing still needs demonstration in the smoke.

14. **NEEDS-RESTATEMENT — Most control repairs are sound, but several specifications remain defective.**

    **Evidence:** D:604–615, 428–438.

    - **G-full:** a dedicated recovery tolerance and separation from Q1 fix the earlier logical problem.
    - **G-nnls:** the two-sided support KKT condition is correct.
    - **G-pod:** the right-hand side in A2.11 is a **relative** Frobenius error, while the left is described simply as Frobenius projection error. Specify
      \[
      \frac{\|S-S\Phi_r\Phi_r^\top\|_F}{\|S\|_F}.
      \]
    - **N-swap:** cases and time indices are frozen, but the tested field/configuration is not. I checked the archived finest references: using case 0’s reference as the fixture, case-swap/time-shift changes are approximately **1.123/0.129 in 2D** and **0.684/0.168 in 3D**, comfortably above the corresponding two-case \(10\max\varepsilon\) thresholds, approximately **\(5.05\times10^{-5}\)** and **\(4.18\times10^{-4}\)**. This demonstrates a usable reference-based fixture, not that every ROM trajectory discriminates.
    - **N-dom:** expected labels “computed by the frozen decision table” are not an independent test of that table.

    **Fix:** Normalize G-pod explicitly; freeze N-swap’s field and comparison population; give N-dom analytically predetermined labels and independently checked fixtures.

15. **CORRECT — Error-envelope propagation and per-case eligibility are appropriate.**

    **Evidence:** D:37–43, 228–234, 441–443; R/refstore.py:132–173; R/verdict.py:148–151.

    Clipped endpoints propagated through maxima and medians give conditional reference-error envelopes. They are correctly distinguished from confidence intervals for a population worst case.

    **Fix:** Require identical eligible case IDs, reference version, and metric for both arms. Missing eligible cases must invalidate the comparison, not shrink either maximum.

16. **CORRECT — A2 propagates uncertainty from both worst errors.**

    **Evidence:** D:617–619.

    The difference envelope
    \[
    [L_B-U_X,\ U_B-L_X]
    \]
    fixes the point-estimate-only guard. Worst-case ties no longer need special selection.

    There is a conservative degeneracy: identical arms need not “match” when the margin is determined by reference uncertainty. For equal worst errors and uncertainty \(\varepsilon\), the difference envelope can be exactly \([-2\varepsilon,2\varepsilon]\), which is not strictly inside the registered margin.

    **Fix:** Document this as an intentional “unresolved” outcome and test it. Do not silently change strict to non-strict inclusion after observing results.

17. **NEEDS-RESTATEMENT — Pair decisions are substantially repaired; cell-level failure accounting remains incomplete.**

    **Evidence:** D:644–655, 458–464; Brief:37–45.

    Unique IDs, failed-pick handling, explicit invalid-input outcomes, and developmental/unadjusted labels repair most of audit 2. The two-objective dominance definition is faithful to the lane brief: offline cost and median error are reported alongside it.

    However, “no observed domination among the N configurations compared” needs a precise denominator. Failed picks, unfulfilled controls, drift-invalid pairs, missing cases, deferred pairs, and unresolved comparisons must not be counted as successfully assessed non-dominations.

    **Fix:** Export counts for attempted, assessed, unresolved, invalid, deferred, and control-blocked pairs. Permit the negative cell statement only with those counts and the appropriate incompleteness label. The unadjusted wording is adequate for exploratory evidence, not formal familywise inference.

18. **WRONG — The declared maximum-size smoke does not cover the allowed branches.**

    **Evidence:** D:621–625, 665–667, 379–381, 584–589.

    A2 promises coverage of every branch, but explicitly lists smaller smoke shapes:

    - Listed 2D QR smoke: \((r,p)=(384,1536)\), \(B\approx19.29\) GB.
    - Allowed 2D extension: \((512,2048)\), \(B\approx25.72\) GB.
    - A1’s 3D rank-768 extension is not explicitly withdrawn, while the listed NNLS smoke stops at rank 512.
    - The amended ECSW schedule has **120 states in 2D** and **102 in 3D**, before dropping blocks. Thus the base \(G\) buffers are approximately **12.08 GB** and **13.69 GB**, not the old 9.7/12.9 GB. Rank 768 raises the latter to approximately **20.54 GB**.
    - A2.13’s 32,832-column buffer is not explained by a 16,384-support cap.

    These figures exclude workspace and other live arrays.

    **Fix:** Generate the maximal permitted shapes first, then smoke those shapes, including support retries, rank extensions, factorization workspaces, and host/device coexistence. Explicitly explain buffer padding and artifact sharing.

19. **NEEDS-RESTATEMENT — The unsupported GPU-hour estimate is withdrawn, but feasibility is still conditional.**

    **Evidence:** D:627–631, 572–576, 550–554; Brief:13–15; S/resources.py:4–12, 66–93.

    Withdrawing the approximately 13-hour estimate is correct. A manifest-based certificate is an appropriate design commitment.

    However, Stage B now has an outcome-dependent uncapped front; the generator probe includes another full snapshot generation/POD; fallback construction may repeat HR setup; and the listed smoke must also fit the brief’s one-hour phase-I allowance. There is currently no demonstrated 30-hour completion argument.

    **Fix:** Budget a conservative Stage-B upper bound before screening, then regenerate the certificate after its membership freezes. Include both generators, ratification fallback, saturation, all controls, warmups, certification, compilation, retries, and retained artifacts. Report incomplete scope if this does not fit; do not present the cap itself as evidence of feasibility.

20. **CORRECT — The common-solver ablation is no longer pruned using native outcomes.**

    **Evidence:** D:511–521, 633–635; C/c3common.py:95–100; S/DESIGN.md:56–58.

    Running both solvers for every retained configuration removes the previous asymmetric pruning. Calling the FOM a limiting comparator rather than an error floor is correct. The spectral deferral is now a scope choice rather than a false impossibility claim.

    **Fix:** Preserve both solvers when deferring ranks/options. Report numerical failures separately for native and common solvers.

21. **NEEDS-RESTATEMENT — Central BDF2 implementation is registered, but the temporal-order fixture remains unspecified.**

    **Evidence:** D:637–642; R/refcore.py:202–217; C/fom2.py:29–50; C/t2core.py:35–37.

    A2 correctly acknowledges the need for a central generic-LMM FOM and adds history/startup residual checks. Neither inherited central solver supplies BDF2; the existing generic `fom2` uses upwind advection.

    The order check still lacks its case or manufactured solution, mesh, norm, error statistic, and order estimator. An arbitrary Burgers trajectory at the listed steps need not already be in the asymptotic regime.

    **Fix:** Freeze a smooth manufactured temporal fixture, independently written coefficients/history equations, the measurement norm and times, and the order estimator. Keep the real-trajectory residual checks as a separate test.

22. **WRONG — Timing population repair is incomplete, and doubled-work N-time is incorrectly specified.**

    **Evidence:** D:661–667, 206–211, 437.

    Two distinct calls on cases \(j\) and \(j'\) cost approximately
    \[
    t_A(j)+t_A(j'),
    \]
    not necessarily \(2t_A(j)\). The expected ratio is \(1+t_A(j')/t_A(j)\); requiring it to lie in \([1.8,2.2]\) can reject a correct harness.

    Also, after drift rejection the verdict may use only three of the six cases, while selection used all six. Unequal retained repeat counts further change case weighting. A2 therefore does not fully preserve its claimed common timing population.

    **Fix:** Execute and synchronize the **same case twice** through two host calls, or normalize against separately measured costs of both cases. For primary verdicts, restore complete case coverage through a bounded retry policy or mark the pair unassessed. Explicitly define case weighting.

23. **WRONG — Imported 3D comparators can enter without the required Q2 qualification.**

    **Evidence:** D:215–219; Common:57–60; V3:73.

    The design imports every speed3d configuration reported **Q1-qualified**, whereas the governing context requires reduced-rule arms to pass their own Q2 check before entering a front. Speed3d’s solver grid does not automatically provide that qualification.

    This can admit an under-resolved version of ours as a cheap comparator.

    **Fix:** Require Q1 plus the applicable own-state Q2 certificate for each imported X. Keep Q1-only configurations diagnostic until Q2 is supplied.

24. **NEEDS-RESTATEMENT — The permitted POD rank range does not fulfill the literal “rank r and smaller ranks” scope for all imported X.**

    **Evidence:** Brief:21–23; D:91, 218, 379–381; S/DESIGN.md:50–54.

    Ours includes rank 1024, but the baseline stops at 512 in the base grid and at 768 in the inherited extension. No rank-1024 POD arm is registered.

    Equal-rank comparisons are not necessary for every cost–accuracy comparison, but the lane cannot claim to have completed the brief’s stated rank coverage or ruled out a higher-rank POD competitor.

    **Fix:** Register the missing rank with its resource consequences, or obtain an explicit restricted-rank scope amendment and keep the limitation beside negative findings about POD.

25. **NEEDS-RESTATEMENT — ECSW construction failures can remove the strongest candidates without establishing family coverage.**

    **Evidence:** D:113–116, 415–418, 590–591, 652–655.

    A fixed candidate pool, pooled training options, and capped NNLS path are legitimate implementation choices. But an unreachable \(\tau\) is failure of that construction, not evidence that ECSW cannot attain the corresponding accuracy/cost region. A nonempty weaker ECSW front does not resolve this limitation.

    **Fix:** Report attempted versus successfully constructed \((r,\tau)\) coverage separately. Withhold “adequately tuned ECSW” or exhaustive non-domination conclusions when registered boundary constructions fail, unless an independently specified training-only fallback resolves them.

26. **NEEDS-RESTATEMENT — Most provenance claims check out, but mesh terminology and one offline-cost attribution do not.**

    **Evidence:** D:75–85, 278–279, 354–361; 2D training JSON:5, 19, 33–36, 5544–5561, 5582; C/inputs/model_W1024/training.json:3–27; vendored `common.py`:91–106.

    Verified: the 2D draw count/seed, 16,384 selected states, early-state selection, 3D seed/count and twelve snapshot indices, dimensional \(\Delta t_0\), scheme definitions, and IC families.

    Two corrections remain:

    - The historical 2D training JSON’s `N=256` means **256 nodes**, with 254² interior values. The lane’s \(L=256\) means **256 intervals**, with 255² interior values. Reusing the parameter draws does not reproduce the historical bank mesh.
    - D:278–279 ambiguously associates the W1024 cost “1.85 H200-h job total / 1.14 h bank fit” with both banks. The supplied 2D JSON records a different bank-fit time: **10,684.65 seconds**.

    **Fix:** State nodes versus intervals explicitly and report each bank’s own training and job-total costs from its provenance.

27. **NEEDS-RESTATEMENT — Numerical failure policies for the offline factorizations remain underspecified.**

    **Evidence:** D:87–90, 120–127, 134–138, 430–431.

    The method-of-snapshots fallback is specified for the state POD, but not clearly for the advection POD at large \(p\). Greedy NNLS uses normal equations and Cholesky without a registered response to a singular or numerically dependent passive set. A finite DEIM interpolation constant alone does not establish an accurate interpolation solve.

    These failures can selectively eliminate high-rank or tight-tolerance baseline candidates.

    **Fix:** Register numerical-rank handling for both PODs, stable passive-set fallback for NNLS, and residual/backward-error checks for DEIM solves. Report construction failure explicitly rather than silently dropping modes, regularizing, or removing candidates.

**Overall: WRONG.** The baseline equations are credible, and A2’s generator-deviation handling is honest. The top blockers are **validation-budget accounting; N-grad’s invalid history expectation; N-time’s unequal-case doubled-work expectation and survivor population; missing maximum-branch resource coverage; and Q2 eligibility for imported 3D comparators**. Resolve those before treating this as a complete, fair C3-minimal pre-registration.