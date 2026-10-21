**Overall verdict: WRONG — one remaining F2 outcome rule can produce an unsupported conclusion.** A2 resolves audit 2’s principal full-Q2 objection. The remaining fitting and resource issues are **NEEDS-RESTATEMENT**, not mathematical failures.

This was read-only: no files changed, jobs submitted, or arrays copied. Controls were assessed as specifications; their real-data firing remains unestablished.

1. **Four-arm formulas — CORRECT.**

   The contractions, signs and inverse actions remain sound. Bank testing changes both the projection and the diffusion discretisation; conclusions must retain that qualification. A2 correctly replaces “exact linear terms” with convergence under the registered finite-rule check.

2. **Arithmetic-cost statement — CORRECT.**

   A1’s restriction of \(6mr\) to leading advection-value arithmetic remains correct. Reporting nonlinear iterations, Jacobian evaluations, evolution time and complete query time prevents an arithmetic reduction from being mistaken for an end-to-end gain.

3. **Stop rules and outcome precedence — WRONG for F2; CORRECT for F3.**

   [A2.3](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-hrcost/experiments/imp-hrcost/DESIGN.md:401) correctly handles F3’s existential success, unresolved candidates, unqualified R, and a deferred Fibonacci ladder.

   **F2 still substitutes the smallest *observed* qualified candidate for the registered minimum.** It acknowledges that \(m^*\) may be unidentified, but nevertheless evaluates accuracy and speed on the smallest qualified candidate and permits “tested failure” or “not fired.”

   Counterexample:

   - Gauss \(64^2\) is unresolved because its successor run was invalid.
   - Gauss \(80^2\) qualifies but is slower than R.
   - A2 reports **fired (tested failure)**.
   - A valid rerun could establish that \(64^2\) qualifies, preserves accuracy and is faster. The registered F2 stop rule would then **not fire**.

   Conversely, success at a larger candidate does not establish the registered outcome at an unresolved smaller minimum; accuracy need not be monotone in rule size.

   **Required correction:** when a smaller candidate remains unresolved, report the existence and measured properties of the larger qualified rule, but keep the overall \(m^*\)-based F2 verdict **incomplete evidence**. Evaluate definitive predicates (b)/(c) only after identifying \(m^*\), or explicitly amend the binding selection rule.

4. **Q1 and Q2 — CORRECT, with a scope clarification.**

   **Q1:** A2.4 resolves the projected-Gram interpretation and zero-scale behavior. The sine-arm equivalence is algebraic for matching step equations, operators and history. Startup/interface differences remain subject to mandatory A3 reconciliation.

   **Q2:** [A2.1](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-hrcost/experiments/imp-hrcost/DESIGN.md:379) restores the inherited full-\(M\)-vector gate for R/F3. Passing only the projected indicator can no longer satisfy a binding stop rule.

   The bank-tested vector
   \[
   T_q(c)=G_q^{\mathsf T}W_q[(G_qc)\odot(G_sc)]
   \]
   is the faithful analogue: it contains every advection test functional before applying the mass inverse. Its binding indicator is
   \[
   \rho_T(c)=\frac{\|T_q(c)-T_{q^+}(c)\|_2}{\|T_{q^+}(c)\|_2}.
   \]
   Together with own-state evaluation, successor validity, rollout distance and finer-rule convergence, this preserves the inherited contract for the changed test space.

   This is **not numerical equivalence between the two projections’ indicators**. The bank-vector Euclidean norm depends on the fixed bank basis and scaling; the borrowed threshold is not newly calibrated by changing tests. That limits interpretation, but does not invalidate this frozen-bank experiment.

   All-trivial indicators now correctly yield unverified. Invalid successor/check runs cannot pass.

5. **Fitting and leakage — NEEDS-RESTATEMENT; mathematical construction is CORRECT.**

   The three targets are sound:

   - F3a fits \(Q^{\mathsf T}N\), matching **Q2-projected**.
   - F3b fits \(N\), matching **binding sine Q2**.
   - F2+F3 fits \(T\), matching **binding bank Q2**.

   Block normalization makes the squared fitting objective a sum of squared relative training-block errors. It does not certify the maximum validation-state error; independent Q2 correctly remains mandatory.

   Frozen training draws, bounded replacements, hashes of actual snapshots and parent configurations, and no validation-driven refitting address leakage. F3a/F3b use different snapshot schedules, so their comparison is a comparison of fitted procedures, not an isolated test of target norm.

   Two documentation fixes remain:

   - A2.2’s blanket “target norm is matched to the gate norm” must distinguish F3a’s **non-binding** target from the binding targets.
   - [The inherited NNLS checkpoint records](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-hrcost/experiments/jcp-hr-baselines/nnls.py:385) contain no checkpoint-specific drift status; `drift_ok` is returned for the whole path. Specify either conservative rejection of all checkpoints from a drift-failed path, or recording drift-event chronology to implement A2.5’s “after a drift event” policy.

   Neither issue by itself invalidates an independently qualified fitted rule.

6. **Controls — CORRECT at design level; firing is not yet demonstrated.**

   A2.6’s explicit blocking rule resolves the central audit-2 concern: an unsuitable fixture cannot be silently credited.

   | Control | Verdict | Assessment |
   |---|---|---|
   | C1 | CORRECT | Both Q2 bars must demonstrably fail; otherwise the protected claim remains blocked. |
   | C2 | CORRECT | Requires actual Q2 failure; weight permutation alone receives no credit. |
   | C3 | CORRECT | Certified baseline and independent residual checks repair the fixture; part (c) tests configuration-level rejection. |
   | C4 | CORRECT | Exercises production non-finite validity handling. |
   | C5 | CORRECT | Varying inputs, consumed outputs and synchronization address benchmark elimination. Its scope is the Jacobian microbenchmark. |
   | C6 | CORRECT | Remains an A–A sanity check, not proof of absent timing bias. |
   | C7 | CORRECT | Tests descriptor-level leakage detection. |
   | C8 | CORRECT | Preserves requested identity independently of the mutation. |
   | P1 | CORRECT | Checks the fitted-rule evaluation path at parent weights. |
   | P2 | CORRECT, conditional | Checks E5 equivalence when its generalised stepper is used. |
   | P3 | CORRECT | Claims finite-rule convergence and definiteness, not exact integration. |
   | P4 | CORRECT | Jacobian coverage now explicitly includes both projections and a fitted rule. |
   | P5 | CORRECT | Independent mass-form RHS comparison addresses sign/assembly errors; the mesh comparison appropriately acknowledges discretisation error. |
   | P6 | CORRECT | Establishes missed tight tolerance before rejection, and exercises invalid successor/check handling. |

   Minor **NEEDS-RESTATEMENT**: after splitting F3, identify whether C2’s “κ = 8 F3” fixture is F3a or F3b, and identify P4’s fitted fixture before execution.

7. **Timing, fairness and predicates — CORRECT within the internal-ablation scope.**

   Same-job pairing, post-fitting burn-in, saved repetition arrays, common eligibility, output agreement and conditional E0 error comparisons remain appropriate.

   Update A1.2’s “three κ are searched” explanation to acknowledge **six F3 candidates across two families**. The existing exploratory, in-sample, selection-unadjusted interval label remains appropriate.

   Timed-invocation validity still requires implementation verification. Nothing here establishes superiority over tuned POD or FOM competitors.

8. **E5 dependency and resources — NEEDS-RESTATEMENT.**

   The fallback label now correctly says that equivalence to E5 is unestablished. Such results support a local ablation, not completion of the prescribed E5 comparison. The handoff was absent at the specified path when checked.

   **The smoke reservation arithmetic is inconsistent.** [A2.7](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-hrcost/experiments/imp-hrcost/DESIGN.md:454) says 1.25 hours covers three fits with their deadlines, tails and controls. A1.5 still permits **40 minutes per fit**: three serial deadlines already total **120 minutes**, exceeding the **75-minute** reservation before other work.

   The hard reservation bounds consumption, and the explicit incomplete-evidence fallback protects conclusions. Therefore this is not an unbounded experiment, but it is not a certified completion budget.

   Specify a consistent allocation for fitting, cooperative-deadline overshoot, final NNLS work and mandatory controls. Separately measured fit costs and peak device memory resolve the main previous omissions; host construction copies and a fallback when peak-memory counters cannot reset should also be covered.

9. **Other ambiguities — NEEDEDS-RESTATEMENT only, apart from the F2 defect above.**

   Identity records, all-trivial indicators, reference-unqualified precedence, actual snapshot provenance, the integration ceiling and restricted fallback interpretation are substantially resolved.

   Remaining wording should distinguish:
   
   - binding versus projected fitting targets;
   - an identified minimum versus a qualified upper bound;
   - bounded runtime versus guaranteed completion;
   - local fallback evidence versus E5-equivalent evidence.

**A2 fixes the previous full-Q2 failure. The remaining conclusion-invalidating issue is F2’s use of an unidentified minimum to issue definitive stop-rule outcomes.**