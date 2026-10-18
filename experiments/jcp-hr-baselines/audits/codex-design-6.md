**A5 resolves audit 5 items 14, 28 and 30 at design level. Item 27 remains NEEDS-RESTATEMENT.** I found no unambiguous remaining WRONG in the governing online formulas or decision rules, but the new POD fallback still has unspecified numerical and dimensional branches.

No files changed or jobs submitted. Dependency code was read only to verify design claims; the lane implementation was not audited. A2.3/A4.3 ratification, resource feasibility, and actual control firing are not re-flagged.

Evidence shorthand: **D** = [baseline DESIGN.md](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-hr-baselines/experiments/jcp-hr-baselines/DESIGN.md); **R**, **C**, **S** = the supplied references, combined, and speed3d experiment directories; **Brief/Common** = the supplied campaign briefs.

1. **CORRECT — A5 resolves item 14’s N-swap fallback ambiguity.**

   **Evidence:** D:793–797,835–838.

   Both the ECSW fixture and `none` fallback must now be certified-valid. If neither qualifies, the design explicitly permits scoring acceptance through the reference fixture and G-score, with the unavailable ROM fixture disclosed. This supplies the missing terminal outcome.

   **Fix:** None. Treat A5.1 as the specific exception for the ROM-swap fixture; it does not waive the remaining required scoring controls.

2. **NEEDS-RESTATEMENT — Item 27: the revised POD certificate has an undefined roundoff branch.**

   **Evidence:** D:840–849; inherited construction at D:87–90.

   The exact-arithmetic tail identity is correct:
   \[
   \tau_r^2=\frac{\|S\|_F^2-\sum_{i\le r}\lambda_i}{\|S\|_F^2}.
   \]
   But subtracting nearly equal computed quantities can produce a negative radicand precisely in the newly introduced “saturated” regime. The specification does not say whether that triggers fallback, clipping, or rejection. Recomputing the basis alone does not repair a certificate that still uses the same Gram tail.

   A small read-only NumPy check demonstrated this: a rank-three \(20\times60\) snapshot matrix, evaluated at requested rank ten, produced a normalized radicand of approximately \(-7.34\times10^{-16}\). This is an arithmetic edge case, not a GPU feasibility question.

   Furthermore, “far below any E0 error” is not established by a training-set Frobenius projection certificate. That certificate does not bound rollout error or derivative error.

   **Concrete fix:** Specify an explicit roundoff allowance for the squared tail. Clip negative values only inside that allowance; larger violations must trigger a stable recomputation or a certificate failure. Define the zero-tail/saturated branch through the directly measured projection residual and orthogonality gate. Replace the E0 claim with “below the registered training-projection certification tolerance.”

3. **NEEDS-RESTATEMENT — Item 27: the randomized fallback’s sketch size is ambiguous for advection POD.**

   **Evidence:** D:119–127,715–719,810–814,846–849,865.

   A4 applies the fallback to **both** PODs. A5 freezes its sketch as \(r_{\max}+64\), but elsewhere \(r_{\max}\) denotes the state rank and the advection basis needs \(p_{\max}=4r_{\max}\).

   Read literally, the maximum sketches are therefore 576 and 1,088 columns, insufficient to construct the registered advection bases of 2,048 and 4,096 columns. If “\(r_{\max}\)” instead means the maximum requested rank of whichever matrix is being factorized, say so.

   The reduced 3D smoke also has 4,104 snapshots, so \(p_{\max}+64=4,160\) needs an explicit dimension cap. Removing the Gram numerical-rank rejection additionally leaves the outcome for genuinely rank-deficient requested bases unclear.

   **Concrete fix:** Define, separately for state and advection matrices,
   \[
   \ell=\min(q_{\max}+64,\ n_{\rm snapshots},\ n_{\rm spatial}),
   \]
   with \(q_{\max}=r_{\max}\) or \(p_{\max}\). Specify what happens when the snapshot span cannot supply the requested rank, including whether completion is forbidden or explicitly allowed and labelled. Ensure non-finite method-of-snapshots factors trigger fallback before division by unusable singular values.

4. **CORRECT — A5 repairs the other substantive parts of item 27.**

   **Evidence:** D:815–818,841–852.

   The seed and power-iteration procedure are frozen. Gram numerical rank no longer silently rejects requested ranks before fallback. The tail is derived from the original snapshot matrix’s energy and Gram eigenvalues, rather than incorrectly treating a truncated randomized spectrum as the full discarded spectrum.

   The NNLS sign convention is now consistent:
   \[
   w=G^\top(b-G\xi),\qquad
   w_i=0\ \text{on positive support},\quad w_i\le0\ \text{off support},
   \]
   subject to the registered tolerances. Solving the passive fallback on original contribution columns avoids squaring its conditioning again.

   **Fix:** Retain these changes while resolving findings 2–3.

5. **CORRECT — A5 resolves item 28’s specific pruning counterexample.**

   **Evidence:** D:572–576,737–742,820–822,854–857.

   Stage B now includes the union of the X-specific paired-ratio fronts. A baseline cannot be removed solely because an absolute-seconds front prefers another configuration with a worse median ratio to X. The absolute-time front is additional screening coverage.

   **Fix:** Preserve all X-specific front members without applying the near-front cap to them. Apply the full-cohort selection requirement of A1.6 when constructing the final role picks.

6. **CORRECT — A5 resolves item 30 at design level.**

   **Evidence:** D:433–434,593–598,773–780,859–877.

   The smoke now names meshes, training subsets, ranks, HR settings, schemes, timesteps, cases, solver coverage and comparator adapters. Required unavailable controls explicitly leave acceptance unfulfilled.

   I independently checked the viscosity ordering from the supplied draw functions: the two smallest-viscosity screening cases are **2,3 in 2D** and **5,0 in 3D**, as registered.

   **Fix:** None to the design. Generate concrete fixture IDs from this specification and retain unavailable outcomes. Actual firing remains a smoke obligation.

7. **CORRECT — Central-FD residual, Jacobian and ECSW target are mathematically consistent.**

   **Evidence:** D:52,66,103–108,338–343; R/refcore.py:64–87.

   The governing Jacobian includes the mass term, nonlinear product derivative, diffusion sign and timestep coefficient. For fixed nonnegative weights,
   \[
   \nabla_c\frac12\sum_i\xi_i\mathcal R_i(c)^2
   =\sum_i\xi_iJ_i(c)^\top\mathcal R_i(c).
   \]
   Thus the all-node gradient-contribution target is appropriate for this weighted LSPG construction.

   Petrov–Galerkin ECSW includes finite-difference models in its stated scope. [Primary ECSW paper](https://arxiv.org/abs/2008.02891).

   **Fix:** None. Retain “weighted strong-residual LSPG” and the explicit absence of an energy-conservation claim.

8. **CORRECT — Governing Q-DEIM compression preserves linear terms.**

   **Evidence:** D:119–127,345–352,529–534.

   The original truncated SVD has been superseded by **untruncated Householder QR**. With \(B=Q_BR_B\),
   \[
   \|Bz(c)\|_2=\|R_Bz(c)\|_2
   \]
   algebraically. Rank deficiency does not require inversion of \(R_B\). Nonlinear DEIM interpolation remains the approximation.

   Pivoted QR of \(U_p^\top\) is the standard Q-DEIM selection construction. [Primary Q-DEIM paper](https://arxiv.org/abs/1505.00370).

   **Fix:** Describe the final method as QR compression, not SVD compression. Preserve the mixed, cancellation-scaled checks.

9. **CORRECT — The baseline is credible and reasonably protected against artificial handicaps; strength remains empirical.**

   **Evidence:** D:354–365,511–521,536–542,633–635,715–718,747–749.

   The design includes smaller and matching ranks, both solver families, backtracked native GN, snapshot adequacy checks and explicit construction-failure coverage. It correctly calls the FOM a limiting comparator rather than an error floor. Spectral projection is deferred for scope rather than declared impossible.

   These provisions support a credible standard baseline; they do not establish that the finite grid contains the best possible POD method.

   **Fix:** Retain the incomplete-coverage and basis-adequacy restrictions on claims favorable to ours. Do not upgrade “no observed domination” into exhaustive baseline superiority.

10. **CORRECT — Fairness accounting and developmental selection are now honestly stated.**

    **Evidence:** D:385–394,544–554,675–685,788–791; Brief:27–31; C/DESIGN.md:887–902,931–945.

    Stage A screens six cases; Stage B selects and judges on the full development cohort. This is defensible as an **in-sample developmental comparison**, with unequal historical selection exposure explicitly disclosed. It is not independent validation.

    Validation-informed qualification, rule selection and escalation are included in the budget definition; training-only construction and offline cost remain visible. The generator and budget-band decisions remain the already identified ratification items.

    **Fix:** None beyond implementing the frozen accounting and preserving the labels. Do not revive the withdrawn “out-of-selection” description.

11. **CORRECT — Time-stepping coverage and startup conventions match the intended comparator ladder.**

    **Evidence:** D:367–370,556–558,637–642; C/t2core.py:36–50,165–175; C/c2common.py:24; C/c3common.py:35; S/DESIGN.md:60–63; R/refcore.py:202–217.

    The dimensional base timesteps, 3D BE/CN/BDF2 ladder and four timestep factors are consistent. Two full BE startup steps in `t2core.CNR` are correctly distinguished from the reference solver’s four half steps. Deferring an option excludes corresponding X comparisons.

    **Fix:** None. Keep startup equations and excluded comparator options explicit in exports.

12. **CORRECT — Q1 matches COMMON-BRIEF §3 and V4-2.**

    **Evidence:** D:158–169,396–405,578–582; Common:49–60; PLAN-v4-amendments.md:16–25; C/c2common.py:32–36.

    The governing specification uses fresh per-step Jacobians in-job, the exact stationarity ratio, pooled nonstationarity, every-case tight-solve distance, physical-IC normalization and tight/tighter checks. A2 withdraws the residual-tolerance certification exception.

    Imported X configurations require their own Q1 and own-state Q2 certificates; historical finalist status is insufficient.

    **Fix:** None. Keep solver termination, anchor validity and numerical qualification distinct.

13. **CORRECT — H-base’s objectives, precedence and equivalence rules follow the lane brief.**

    **Evidence:** D:228–243,441–464,617–619,644–655,782–786; Brief:37–46.

    Dominance requires resolved improvement in **time and worst error**; offline cost and median error accompany the label. This follows the lane brief’s explicit interpretation of the broader plan objectives.

    “Matches” requires the registered paired-error and timing equivalence tests plus the conservative worst-envelope guard. Crossing resolved objectives receive “neither dominates”; overlap alone gives “unresolved.” Both identical-arm regimes now have correct expected labels.

    **Fix:** None. Preserve strict boundaries and the declared precedence, including dominance inside equivalence margins.

14. **CORRECT — Degenerate cases and multiplicity are handled conservatively.**

    **Evidence:** D:453–464,617–619,644–655,709–713,827–829.

    Deterministic IDs settle ties. Empty fronts, failed picks, missing eligible data, zero margins and inadequate timing coverage cannot become favorable evidence. Duplicate picks count once.

    Unadjusted intervals and development-selected comparisons cannot establish familywise confirmation, but the governing text explicitly restricts them to developmental, in-sample, unadjusted observations.

    **Fix:** None. Report assessed, unresolved, blocked and deferred counts together; retain conditional-reference-envelope terminology.

15. **CORRECT — The brief’s required control families are present and block the appropriate claims.**

    **Evidence:** D:424–436,600–610,773–780,824–825,875–877; Brief:32–35.

    Independent residual/Jacobian checks cover the actual `none`, ECSW and DEIM formulations. Full-rank/all-points recovery uses the fixture’s own mesh and dedicated tolerances. Under-sampling requires finite accuracy degradation; solver failure alone is insufficient. An unfulfilled under-sampling control blocks acceptance for that HR family.

    N-grad’s binding test is now agreement with an independent oracle, with a required discriminating tuple, rather than an unsupported claim that a particular mutation must fail.

    **Fix:** None at design stage. Preserve independence of reference assembly and tested routines.

16. **CORRECT — Resource feasibility is appropriately conditional, not established.**

    **Evidence:** D:623–631,715–730,799–802,854–865; Brief:9–15; S/resources.py:4–12.

    The unsupported 13 GPU-hour estimate is withdrawn. The registered 3D base grid count is correctly **1,008 configurations and 6,048 Stage-A production runs**, before anchors and certification. Maximum matrix shapes and major live buffers are identified.

    A5 can enlarge Stage B further through the X-specific union. Its actual membership must therefore enter the post-screening certificate. Reduced-snapshot pilots do not by themselves establish full snapshot-count Gram/eigensolve peaks.

    **Fix:** Include the corrected POD fallback shapes, complete Stage-B union, full-count factorization phases, compilation, retries, controls, retained disk artifacts and contingency in the smoke-derived certificate. No present feasibility verdict is justified.

17. **CORRECT — The amended provenance claims checked against the supplied dependencies agree.**

    **Evidence:** D:354–361,751–754; training JSON:5,19,33–36,5544–5561,5582; `burgers2d_film.py`:160–185; `sep_burgers_r3.py`:321–332; C/inputs/model_W1024/training.json:3–27; R/cohorts.py:35–37,91–93,134–150; vendored `common.py`:3–5,91–106.

    Verified: physical training draws and seeds; 2D early-state sampling; 3D snapshot schedule; nodes-versus-intervals conventions; IC conventions; cohort mapping; dimensional timesteps; and the approximately **10,684.65-second 2D bank-fit time**. The 2D initialization description also agrees with C/t2core.py:259–263.

    **Fix:** None. Compare physical draw columns, excluding the normalized float32 descriptors, and keep bank-fit time separate from whole-job cost.

**Overall verdict: NEEDS-RESTATEMENT.**

**Top blockers:** A5.2 needs a defined roundoff/saturation branch for its Gram-tail certificate and separate, dimension-capped fallback sketch sizes for state and advection POD, including rank-deficient outcomes. Audit 5 items **14, 28 and 30 are resolved**; **27 is only partially resolved**. Resource feasibility and real-data control firing remain properly deferred to the registered smoke.