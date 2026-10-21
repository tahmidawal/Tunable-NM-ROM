**Overall verdict: NEEDS-RESTATEMENT — A2 does not yet close every item.** The reduced equations and integrators are sound. The remaining problems concern qualification endpoints, stability classification, competitor selection and timing. In particular, A2 introduces a timing rule that can discard two of six registered cases and still pass.

I reviewed only E5’s design, audit 3, the binding briefs and the referenced supporting code. I did not inspect `fscore.py`, change files, submit jobs or write to the lab log.

1. **Diffusion-coordinate algebra — CORRECT.**

   A1.1 correctly replaces the original rectangular equation with
   \[
   Q^\top[A\dot c+N(c)+\nu\Lambda Ac]=0.
   \]
   Substituting \(A=QR_m\), \(Q^\top\Lambda Q=VDV^\top\) and \(z=V^\top R_mc\) gives
   \[
   \dot z=-V^\top Q^\top N(R_m^{-1}Vz)-\nu Dz.
   \]
   Thus \(B_z,D_z,P_z\) reproduce `qcore.tested_value('point')` exactly, and orthonormality gives
   \[
   \|z\|_2=\|R_mc\|_2=\|Ac\|_2.
   \]
   This matches the GAL residual in `p1core`. It is equivalence to that **point-form** operator, not to a different upwind operator. The triangular solves and table diagnostics remain appropriate.

2. **ETDRK4 coefficients and recurrence — CORRECT.**

   A1.2 matches `psjax2d`: \(L=-\nu D\), four nonlinear evaluations, and the update contains \(2f_2(N_a+N_b)\). `pscore.etdrk4_coefs` already includes the timestep in \(Q,f_1,f_2,f_3\); the design does not multiply it twice.

   Coefficient construction and coordinate maps are charged to evolution. Stage finiteness and independent reconstruction cover the substantive implementation risks.

3. **One-pass CN order and Q1 analogues — CORRECT.**

   For smooth \(g\),
   \[
   \delta=-h(I+\tfrac h2J_g)^{-1}g
          =-hg+\tfrac{h^2}{2}J_gg+O(h^3),
   \]
   matching the second-order Taylor expansion of \(\dot z=-g(z)\). The specified first step preserves this result.

   The certificates are honest analogues of COMMON-BRIEF Q1:

   - GAL Newton: fresh root residual plus tight/tighter trajectory comparisons.
   - ETDRK4: independently reconstructed discrete updates on every case and step.
   - One-pass: fresh Jacobian and residual of the **saved increment**.

   These certify the discrete algorithm, not small temporal error; the design now distinguishes those claims. **A2.1 closes the earlier wording defects**, including the unjustified round-off-floor assertion.

4. **Q2 and POD successor policy — NEEDS-RESTATEMENT.**

   [A2.2](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-faststep/experiments/imp-faststep/DESIGN.md:403) closes missing-checkpoint ordering, successor validity/reconstruction and denominator handling. All-case rollouts and same-main-state force comparisons are appropriate.

   Two endpoints remain:

   - **Duplicate checkpoints:** unchanged [`build.ecsw_weights`](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-faststep/experiments/jcp-hr-baselines/build.py:339) fills unreached support targets with the same KKT-certified optimum. Multiple nominal κ members can therefore contain identical nodes and weights. Comparing these “successive rules” gives zero force and rollout differences without testing quadrature convergence.
   - **All-zero indicators:** excluding and counting both-zero force comparisons does not specify the label when every indicator is excluded. `p1score` explicitly treats no remaining indicators as unverified.

   **Fix:** choose the next available **distinct rule**, identified by nodes/weights hash; skip duplicates and ultimately compare with full integration. Explicitly adopt the all-trivial “unverified” outcome. State that `HR-free` satisfies the absence of quadrature error for numerical-label aggregation while still requiring Q1.

5. **Stability monitor and diagnostic classification — NEEDEDS-RESTATEMENT.**

   The operational stop monitor is correctly described as a conservative amplitude-growth threshold, not an energy theorem. A2.3 also closes the wrong-reference-scheme and undefined stiff-mask issues.

   However, its classification still overclaims near the boundary. For example,
   \[
   G_8=1.049,\qquad G_{16}=1.053
   \]
   satisfies \(|G_8-G_{16}|\le0.005\), yet A2 labels the trip “integrator-induced” although the finer trajectory also trips. Non-finite or uncertified refined trajectories have no explicit classification.

   **Fix:** require valid, certified refinements; classify threshold-straddling results and results within the stated convergence uncertainty of 1.05 as unresolved. Use “growth absent in the tested refinements” rather than claiming exclusive integrator causation. This affects diagnostics, not the conservative stop predicate.

6. **Order tests and reference convergence — CORRECT.**

   **A2.4 closes audit 3’s substantive requests:** finite-range slopes are no longer proofs of formal order; an unconverged required reference cannot disappear from the median or yield a passing F5 result; an undemonstrated control blocks production.

   Keep F5’s outcome visibly separate from E5. Passing E5 does not override an unresolved or failed F5 recommendation.

7. **Stop rule and numerical qualification — CORRECT, with a required label distinction.**

   The quoted stop rule matches the lane brief, apart from formatting. A2.5 explicitly retains both the all-38 and eligible-subset worst-error checks, resolving the previous precedence ambiguity.

   This is a **conservative strengthening** of the literal all-case worst-error rule: an arm can improve the overall maximum while worsening the eligible-subset maximum. Report those two predicates separately so the reason for stopping is visible.

   The median paired evolution statistic is a legitimate preregistered interpretation of the speed requirement; it must remain described as median performance. E5 “fixed” correctly requires Q1+Q2 in addition to the three role conditions. Deferred cells correctly make the full-grid conclusion incomplete. The timing-population defect in item 11 still needs correction before this predicate is executable faithfully.

8. **POD retraining, selection and competitor fairness — NEEDS-RESTATEMENT.**

   Galerkin-specific contributions, their own importance distribution, positive block scaling, cross-rank tuning, both timesteps and native historical competitors are appropriate. A2.6 fixes the unavailable-κ diagnostic fallback.

   Remaining issues:

   - **“Validation-matched” is tested on the union, not validation.** Suppose dev worst errors are \(0.10\) for X and \(0.09\) for B, while val worst errors are \(0.02\) and \(0.025\). A2 calls B validation-matched because its union maximum is smaller, although B fails the val32 match.
   - The matching definition does not explicitly require the frozen competitor to remain valid and stable on the claimed validation population.
   - The 1% timing-tie relation is not transitive. Three times can have adjacent pairs within 1% while the endpoints differ by more than 1%. Selection can depend on traversal order.
   - The κ tie-breaker is undefined for Q-DEIM and full integration.

   **Fix:** report dev6, val32 and all-38 match statuses separately; name the population beside every claim and require validity/stability there. Define a tie set relative to the global minimum, then apply a complete ordering ending in a unique configuration ID. Preserve the numerical labels separately, as A2 already requires.

   Excluding POD at 4096² is acceptable **for the explicitly limited E5 scope**, with no H-base conclusion there. Lack of an existing basis and the budget justify the exclusion; avoid presenting the historical 576-trajectory recipe as the only conceivable way to construct a competitor.

9. **Must-fail controls — CORRECT as a gated plan; not yet demonstrated.**

   **A2.7 closes the central defect:** a proposed mutation is no longer assumed to fire. Real smoke witnesses are mandatory, escalation is bounded and failure to exercise a detector blocks production.

   C19 and C20 address the requested missing-checkpoint and matching-label paths. Their fixtures need explicit preconditions: removing κ=4 must actually change a successor, and C20 should exercise the actual population-specific matcher.

   Add controls for the remaining defects identified here:

   - duplicate KKT-filled checkpoints;
   - growth maxima straddling 1.05;
   - timing-case loss and identity mismatch;
   - union-matched but val32-unmatched competitors.

   No real-data execution occurred in this audit, so none of C1–C20 is certified as demonstrated.

10. **GPU-hour estimate and defer order — NEEDS-RESTATEMENT; the earlier retry error is fixed.**

   A2.8 correctly budgets two attempts per rank:
   \[
   2(240+360+600)=2400\text{ seconds}.
   \]
   The expanded POD ladder and full-node certificates are now explicitly included in the required cost certificate. The defer order matches the brief, and the 4096² memory pilot remains necessary.

   However, **0.67 h is not a hard NNLS worst-case wall bound**. [`nnls.py`](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-faststep/experiments/jcp-hr-baselines/nnls.py:344) documents deadline overshoot; it also computes final residuals, checkpoint residual vectors and KKT evidence after the iteration loop.

   **Fix:** call 0.67 h the sum of configured attempt budgets, and price overshoot/finalization in the measured certificate. The 3.1 h total is a planning estimate, not established feasibility. Reserving the 1.25 h job limit protects the cap; it does not guarantee usable ECSW artifacts. If mandatory work still cannot fit after the prescribed deferrals, return an incomplete result rather than dropping certificates or competitors silently.

11. **Frozen selection and full-query timing — NEEDS-RESTATEMENT, with a substantive new loophole.**

   A2.9 correctly freezes the dev-selected stepper and preserves its validation failure.

   Its timing rule is not yet acceptable:

   - **Four of six cases can now pass.** This overrides A1.7’s missing-timing-case failure rule and changes the registered six-case statistic. Losing two slower or harder cases can change the median and the E5 verdict.
   - **Identity mismatch is treated like ordinary drift.** An output mismatch means the timing may describe a different computation; discarding that triplet and retaining enough other cases is insufficient.
   - The minimum valid repetition count per case and the common case set for comparing two subjects are unspecified.
   - Separate arm-1–X–arm-1 and arm-1–B–arm-1 measurements yield an **indirect common-anchor estimate**. They are not a direct paired X–B–X comparison under COMMON-BRIEF §5.

   **Fix:** require all six registered cases and a fixed repetition rule for headline statistics; incomplete coverage makes the result incomplete. Treat identity mismatch as a correctness failure requiring resolution. Use direct fused-query X–B–X triplets for the frozen matched competitor and the F5 comparison, retaining common-anchor timings for screening. Define best-stepper timing ties against the global minimum as in item 8.

**A2 closure summary:** A2.1, A2.4, A2.5 and the witness-gating part of A2.7 close their items. A2.2, A2.3, A2.6, A2.8 and A2.9 need the corrections above. The mathematical proposal is ready; the decision procedure is not yet ready for production.