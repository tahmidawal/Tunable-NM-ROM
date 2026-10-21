**Overall verdict: NEEDS-RESTATEMENT. No WRONG remains.** A3 closes the substantive qualification, matching, timing and budget findings from audit 4. One diagnostic definition remains incomplete: A3.2 does not assign a stability classification to every possible refinement pair.

I audited only E5’s `DESIGN.md`, applying A3 > A2 > A1 > §§1–11, against the prior audit, binding briefs, specified ideation sections and supporting code. I did not inspect `fscore.py` or `fsrun.py`, modify files, submit jobs or write to the lab log.

1. **Diffusion-coordinate precontraction — CORRECT.**

   The governing equation is the projected GAL equation from A1.1:
   \[
   Q^\top[A\dot c+N(c)+\nu\Lambda Ac]=0.
   \]
   With \(A=QR_m\), \(Q^\top\Lambda Q=VDV^\top\) and \(z=V^\top R_mc\),
   \[
   \dot z=-P_z^\top[(B_zz)\odot(D_zz)]-\nu Dz.
   \]
   The transformed tables reproduce `qcore.tested_value('point')`, and orthonormality gives \(\|z\|_2=\|Ac\|_2\). This is equivalence to the inherited **point-form GAL operator**, not an assertion of equivalence to upwind advection.

2. **ETDRK4 coefficients — CORRECT.**

   A1.2 matches `psjax2d` with \(L=-\nu D\), four nonlinear evaluations and the factor \(2f_2(N_a+N_b)\). `pscore.etdrk4_coefs` already includes the timestep in its coefficients. Coefficient construction and coordinate maps remain charged to evolution; independent reconstruction checks the recurrence.

3. **One-pass CN and observed order — CORRECT.**

   For the smooth autonomous reduced ODE,
   \[
   \delta=-h(I+\tfrac h2J_g)^{-1}g
   =-hg+\tfrac{h^2}{2}J_gg+O(h^3),
   \]
   so the specified method, including its first step, is second order. A2 correctly distinguishes finite-range observed slopes from formal order and prevents an unconverged reference from producing a passing F5 outcome.

4. **Q1 analogues — CORRECT.**

   The certificates check the algorithms actually proposed:

   - GAL Newton: fresh root residual, valid tight anchors and trajectory agreement.
   - ETDRK4: independently reconstructed complete updates on every case and step.
   - One-pass: fresh Jacobian and residual of the actual saved increment.

   These are honest discrete-algorithm certificates. They do not claim to certify temporal accuracy. The retained twin checks and separate time-refinement evidence preserve that distinction.

5. **Q2, including A3.1 — CORRECT.**

   All-case rollout comparisons and force comparisons at the same main-rule states implement the registered Q2 contract. A3.1 closes the real duplicate-checkpoint problem in `build.ecsw_weights`: KKT completion can populate multiple κ names with the same rule.

   Skipping identical rules, falling back to full integration, marking all-trivial indicators “unverified,” and treating full integration as Q2-satisfied while retaining Q1 are appropriate. Successor certification requirements from A2 remain binding.

6. **Stability monitor and A3.2 — NEEDS-RESTATEMENT.**

   The stop monitor remains correctly described as a conservative empirical rejection threshold. A3 removes unjustified causal attribution and handles invalid refinements and threshold straddling.

   However, [A3.2](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-faststep/experiments/imp-faststep/DESIGN.md:481) leaves cases unclassified. For example,
   \[
   G_8=1.057,\qquad G_{16}=1.054.
   \]
   Their difference is below 0.005, they do not straddle 1.05, and \(G_8\) is outside the uncertainty band. Yet they are neither both above 1.055 nor both below 1.045. The reversed pair receives “unresolved,” making the gap asymmetric.

   **Concrete fix:** define an exhaustive ordered classifier:

   - Invalid/uncertified refinement or \(|G_8-G_{16}|>0.005\): unresolved.
   - Both \(>1.055\): growth present in the tested refinements.
   - Both \(<1.045\): growth absent in the tested refinements.
   - Otherwise: unresolved.

   This includes boundary equalities and straddles. It changes only diagnostics, not the stop predicate.

7. **Stop rule and A3.3 — CORRECT.**

   The quoted rule preserves the brief’s wording apart from formatting. The operational reading correctly retains:

   - separate all-38 and eligible-subset worst-error predicates;
   - the registered median paired evolution speed statistic;
   - failure/incompleteness handling;
   - numerical qualification as an additional requirement for E5 “fixed”;
   - separate F5 outcomes;
   - an incomplete full-grid conclusion when cells are deferred.

   The eligible-subset predicate is a conservative strengthening of the literal all-case rule. A3.3 now makes its effect visible.

8. **POD retraining, matching and A3.4 — CORRECT.**

   Galerkin-specific contributions and their own importance distribution correctly replace the inherited LSPG-gradient training target. The rank/rule/timestep ladder and native historical picks provide meaningful competitor tuning.

   A3.4 closes the union-versus-validation matching error, requires validity/stability on each claimed population, and replaces nontransitive timing ties with a global-minimum tie set and complete configuration ordering. The selected competitor remains frozen before validation assessment.

   The 4096² exclusion is now an explicit scope limitation, with no H-base conclusion there. Numerical qualification labels and the finite-ladder speed-up caveat must remain beside the comparisons.

9. **Must-fail controls and A3.7 — CORRECT as a gated plan; not demonstrated.**

   The inherited bounded escalation and production-blocking rules prevent assumed detector sensitivity from being treated as evidence. C21–C24 address duplicate rules, threshold straddling, timing coverage/identity and population-specific matching. C19 now requires an effective fixture; C20/C24 exercise the production matcher.

   **Follow-through for item 6:** extend C22 with the uncovered pair above, its reversal and boundary values. The current straddle fixture alone cannot detect the remaining classifier gap.

   No real smoke execution occurred in this audit; none of these controls is certified as demonstrated.

10. **GPU-hour budget and A3.5 — CORRECT as a conditional plan.**

    A3 correctly calls 0.67 h the sum of configured NNLS attempt budgets, not a hard wall-time bound. Overshoot, finalization, full-node POD evaluations, certificates and timing must enter the measured production projection.

    The 3.1 GPU-h figure remains an estimate. Reservations, the memory pilot, the prescribed defer order and an explicit incomplete outcome protect the 4 GPU-h cap without silently deleting required work. Feasibility remains to be measured.

11. **Timing, selection and A3.6 — CORRECT.**

    A3 closes the previous timing loopholes: all six cases and three valid repetitions are mandatory; identity mismatch is a correctness failure; selected competitor and F5 headline comparisons use direct paired fused-query triplets.

    Common-anchor measurements now serve screening only. Selection remains dev-only, frozen and independently recomputed offline. Reporting should distinguish **dev6 timing** from the separately assessed dev6/val32/all-38 accuracy-match status.

**A3 closure:** A3.1 and A3.3–A3.6 close their items. A3.7 supplies the requested controls, with C22 needing the boundary coverage described above. A3.2 needs one exhaustive fallback rule. **No WRONG remains; overall NEEDS-RESTATEMENT for that diagnostic gap.**