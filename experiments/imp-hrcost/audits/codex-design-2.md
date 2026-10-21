**Overall verdict: WRONG for the claimed full-Q2 stop-rule conclusions; otherwise substantially improved, with several NEEDS-RESTATEMENT items.** A1’s projected certificate is mathematically defensible and its deviation is disclosed. However, passing it can still produce a “not fired” outcome under a brief requiring the inherited full Q2 contract.

This was a read-only source audit. I changed no files, submitted no jobs, copied no arrays, and did not execute controls. **Real-data firing remains unestablished.**

1. **Four-arm formulas — CORRECT.**

   A1 preserves the correct contractions, signs and inverse actions. Bank testing changes both the projection and the diffusion discretisation; that qualification remains necessary.

   P3 now adds definiteness checks and a finite construction ceiling. These improve construction safeguards without proving exact continuum integration.

2. **Arithmetic-cost statement — CORRECT.**

   [A1.1](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-hrcost/experiments/imp-hrcost/DESIGN.md:240) correctly restricts \(6mr\) to leading advection-value arithmetic and acknowledges changes in conditioning and nonlinear iterations. Separate evolution, query, Jacobian and iteration reporting resolves the prior overclaim.

3. **Stop rules — NEEDS-RESTATEMENT.**

   A1 correctly distinguishes conditional error ties, a 1.5× point estimate, selection-sensitive intervals, and incomplete evidence. Recording both quadrature-only and jointly qualified minima is also correct.

   The outcome categories still need explicit precedence:

   - **F3:** one fully evaluated successful candidate establishes “not fired” even if another support is unreached. “Incomplete evidence” applies when no success is established and some candidates remain unresolved.
   - **F2:** an unresolved smaller candidate prevents claiming the registered minimum. A larger successful candidate can establish that a useful rule exists, but not identify \(m^*\).
   - **Unqualified R:** A1.10 adds a diagnostic outcome outside A1.2’s three categories. Represent this explicitly, rather than forcing it into “tested failure.”
   - **Invalid successor/check runs:** distinguish failure to establish quadrature accuracy from demonstrated quadrature error. A1.4 currently labels an invalid successor “fail,” which can otherwise become misleading “tested failure.”

   If Fibonacci is deferred, a Gauss-only result must remain a restricted result, not completion of the original combined search.

4. **Q1 and Q2 — WRONG for inherited full-Q2 qualification; Q1 largely resolved.**

   **Q1 algebra — CORRECT on ordinary CN steps with positive normalisation.** With identical operators, rule, timestep and states,
   \[
   Q^{\mathsf T}\mathcal R_n=R_m\rho_n,\qquad
   \|\rho_n\|_{A^{\mathsf T}A}=\|R_m\rho_n\|_2,\qquad
   \|Ac_n\|_2=\|R_mc_n\|_2.
   \]
   Thus A1.3 restores the inherited sine-arm \(\eta\) algebraically. This is mathematical equality, not bitwise equality between floating-point evaluation paths.

   Two qualifications need restatement:

   - \(A^{\mathsf T}A\) is the Gram matrix of the **retained sine projection** of the bank. It is not generally the full physical \(L^2\) Gram of \(G\). The bank-arm \(\mathcal M\) has that latter interpretation, up to integration error.
   - Zero-scale behaviour differs: A1 rejects every zero denominator, while `qcert.py`/the amended qualification design can certify zero scale with exactly zero residual. Therefore the equality of acceptance rules is not unconditional.

   In-job reconstruction, fresh Jacobians, G-cert and valid tight anchors address the principal prior gaps.

   **Q2 mathematics — CORRECT as a new projected diagnostic.** A1.4 indeed removes the inverse-\(R_m\) weighting:
   \[
   \|v_q-v_{q^+}\|_{A^{\mathsf T}A}
   =\|Q^{\mathsf T}(N_q-N_{q^+})\|_2.
   \]
   It is not automatically satisfied by fitted rules: training fits do not enforce it on validation trajectories. Ignoring \(\ker Q^{\mathsf T}\) is appropriate when assessing this fixed Galerkin dynamics.

   **The remaining invalid conclusion:** [A1.4](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-hrcost/experiments/imp-hrcost/DESIGN.md:281) permits an arm with inherited \(\rho>0.116\) to become numerically qualified and satisfy the stop rule. The binding lane brief and ideation W3 require full Q2. A visible “passes Galerkin-tested \(\rho\) only” flag does not make that the inherited contract.

   Also, the historical full-vector gate applied to GAL configurations too; “because its LSPG arms use all \(M\) tests” is incomplete justification for changing it.

   **Resolution:** retain the inherited gate for binding sine-arm outcomes and report the projected qualification separately, or obtain an explicit change to the binding contract and name the resulting outcomes accordingly.

   Finally, specify **all indicators excluded → unverified**. A1 handles individual zero denominators but omits this aggregate case; `qscore.q2_label` explicitly prevents an empty set from passing.

5. **Fitting and leakage — NEEDS-RESTATEMENT.**

   The frozen draw construction, descriptor checks, replacement limit, tiny-block handling, support-targeted naming, no retries and deadline are substantial corrections. No prohibited validation-to-fitting leakage is specified.

   Remaining gaps:

   - Freeze/hash the actual selected replacement rows, snapshot arrays and parent configuration, not just the original draw table.
   - State the QR-drift acceptance policy. `nnls.py` records excessive drift and returns `drift_ok`; it does **not** automatically reject affected checkpoints.
   - Record/check each retained support’s finite positive weights, support size and independently recomputed fitting residual.
   - Define the all-blocks-dropped outcome as “not built.”
   - Reconcile “two fits in total” with a smoke fit followed by production: reuse the frozen smoke fit or explicitly budget and identify recomputation.

   These omissions do not alone invalidate a rule that subsequently passes independent qualification, but they leave the claimed fitting procedure incompletely specified.

6. **Controls — NEEDS-RESTATEMENT overall.**

   None can presently be credited as demonstrated on real data. Their design-level assessments are:

   | Control | Verdict | Assessment |
   |---|---|---|
   | **C1** | **NEEDS-RESTATEMENT** | Valid coarse runs need not fail both bars. Selecting the coarsest valid fixture does not establish sensitivity. A defective fixture must block the corresponding qualification claim until replaced and demonstrated; “investigate before any verdict” is insufficiently explicit. |
   | **C2** | **NEEDS-RESTATEMENT** | A 50% weight-vector change does not guarantee a large operator change: contributions can cancel or be nearly redundant. Validate actual Q2 failure. Its revised claim—weight–point association sensitivity—is correct. |
   | **C3** | **NEEDS-RESTATEMENT** | The corrected dependency pattern is right. But the fixed perturbation does not mathematically guarantee both adjacent residuals cross the threshold. Require an initially certified fixture and independent verification that each intended residual crosses. One or two flagged steps also need not fail the pooled 1% Q1 gate. |
   | **C4** | **CORRECT** | NaN injection through the actual production validity path protects the stated claim. |
   | **C5** | **NEEDS-RESTATEMENT** | Moving to Jacobian timing fixes the query-overhead error. A 2× workload still does not guarantee ≥1.5× measured time. Ensure synchronization, runtime-varying inputs/consumed outputs and no compile-time elimination. Failure can mean an unsuitable fixture. |
   | **C6** | **CORRECT** | Properly narrowed to an A–A sanity check. |
   | **C7** | **CORRECT** | Descriptor-level injection tests the intended leakage check. |
   | **C8** | **CORRECT** | Keeping requested identity independent of the mutation repairs the key loophole. |
   | **P1** | **CORRECT** | Tests the fitted-rule path at parent weights; P4 supplies the independent reference it previously lacked. |
   | **P2** | **CORRECT, conditional** | Appropriate for a generalised E5 stepper. Does not validate a fallback built without E5. |
   | **P3** | **NEEDS-RESTATEMENT** | Valid finite-rule convergence and definiteness checks. Rename “exact linear terms” to terms converged under the registered check. |
   | **P4** | **CORRECT** | Independent uncontracted RHS comparison plus analytic/AD Jacobian comparison addresses the shared-\(P\) weakness. Specify coverage across arm types for the Jacobian comparison. |
   | **P5** | **NEEDS-RESTATEMENT** | Useful independent operator checks, but finite-difference stiffness has discretisation error; its threshold is a fixture assumption. Add a direct bank mass-form RHS comparison to catch sign/diffusion assembly mistakes beyond checking \(P\). |
   | **P6** | **NEEDS-RESTATEMENT** | One Newton iteration is not inherently invalid. Independently establish missed tolerance and budget exit before requiring rejection. Also test invalid successor and check-rule anchors through Q2. |

7. **Timing, fairness and predicates — CORRECT within the stated ablation scope.**

   A1 now addresses post-fitting burn-in, repetition persistence, outliers, identity, output agreement, common eligibility and aggregation. The exploratory interval labels address selection sensitivity honestly.

   The internal fixed-stepper comparison is appropriate for the lane-specific ablation. It supports no tuned POD/FOM superiority claim. A later cross-job juxtaposition with E5’s POD result must not become a derived speed ratio.

   Implementation must still verify the timed outputs and their validity for the actual timed invocations.

8. **E5 dependency and resources — NEEDS-RESTATEMENT.**

   Polling, deadline, mandatory handoff reconciliation, full-support smoke fitting and measured cost projection resolve much of the previous issue. The E5 handoff was still absent at the audited path.

   Remaining issues:

   - A deadline-triggered locally implemented R changes the prescribed E5 comparison. It can support a labelled local ablation, but cannot establish equivalence to E5 without the missing reconciliation.
   - A1.9 gates production, but the smoke itself needs a bounded reservation covering its worst permitted fit duration and mandatory controls.
   - Measure **peak live memory**, including QR \(R\), refactorisation temporaries, matrix copies, compilation and resident bank tables—not merely \(C+Q\).
   - The NNLS deadline is cooperative: the source permits overshoot and performs final residual/checkpoint/KKT work after the loop. Reserve that measured tail.
   - F3 and F2+F3 fitting costs can differ. One measured path is not automatically a certified bound for both.
   - If mandatory work cannot fit after permitted deferrals, stop with incomplete evidence; do not silently reduce qualification coverage.

9. **Other ambiguities — NEEDS-RESTATEMENT; mostly resolved.**

   A1 correctly bounds candidate claims, sets the integration ceiling, expands identity records, handles an unqualified R, renames the optional sketch and repairs references.

   Remaining edge cases are the missing outcome precedence, all-trivial \(\rho\), snapshot provenance and the unvalidated E5 fallback discussed above.

**Required before accepting conclusions:** resolve the full-Q2 versus projected-Q2 contract mismatch; make outcome precedence explicit; and require demonstrated, independently checked fixtures before controls count as passed. The amended design is much stronger, but a successful projected-Galerkin experiment must not be reported as satisfying the unchanged inherited full-Q2 stop rule.