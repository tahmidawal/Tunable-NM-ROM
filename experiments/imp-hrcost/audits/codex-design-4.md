**Overall verdict: NEEDS-RESTATEMENT.** A3 resolves audit 3’s conclusion-invalidating F2 defect. I found no remaining **WRONG** mathematical construction or authorized success path when the amendments are read together. Two documentation gaps remain: exhaustive outcome precedence and the interpretation of fallback memory measurements.

This audit was read-only. No files changed, jobs submitted, arrays copied, or lab-log entries written. Control firing remains unverified experimentally.

1. **Four-arm formulas — CORRECT.**

   Audit 3’s conclusion stands. The contractions, inverse actions and signs are consistent. Bank testing changes both the test space and the diffusion discretisation; F2 therefore measures their combined effect. Finite-rule convergence does not establish exact integration, as A2.6 now acknowledges.

2. **Arithmetic-cost statement — CORRECT.**

   The \(6mr\) statement concerns leading advection-value arithmetic. A1.1 correctly permits projection-dependent conditioning, iteration counts and total runtime. Separate evolution and complete-query measurements prevent a point-count reduction from automatically becoming a speedup claim.

3. **Stop rules and outcome precedence — NEEDS-RESTATEMENT; audit 3’s substantive defect is resolved.**

   [A3.1](experiments/imp-hrcost/DESIGN.md:466) prevents either success or tested failure at an unidentified minimum. Audit 3’s counterexample—an unresolved smaller rule and a slower qualified larger rule—now yields **fired (incomplete evidence)**. The larger rule supplies only an upper bound and descriptive predicates.

   F3’s existential success rule remains sound: one fully evaluated successful candidate suffices; absent success, unresolved candidates prevent a universal tested-failure conclusion. Reference-unqualified precedence remains intact.

   **Make two remaining branches explicit:**

   - No candidate qualifies, but some remain unresolved: **fired (incomplete evidence)**. A3’s phrase “any smaller candidate” lacks an explicit comparator when no qualified candidate exists.
   - \(m^*\) is numerically identified, but its accuracy or timing evidence is missing: **fired (incomplete evidence)**, not the literal A3.1 “otherwise” tested failure.

   A1.2’s construction/budget rule and A3.5’s incomplete-work rule already protect these cases. This is an ambiguity in assembling the decision table, not a demonstrated invalid conclusion.

4. **Q1 and binding Q2 — CORRECT.**

   A2.1 restores the inherited tested-vector contract for both projections:

   \[
   \rho(c)=\frac{\|V_q(c)-V_{q^+}(c)\|_2}{\|V_{q^+}(c)\|_2},
   \qquad
   V=N\ \text{or}\ T.
   \]

   R/F3 use the full sine-tested \(M\)-vector. F2/F2+F3 use
   \[
   T_q(c)=G_q^{\mathsf T}W_q[(G_qc)\odot(G_sc)],
   \]
   containing all bank-test advection functionals **before** the mass inverse. This is a faithful analogue.

   Own-state evaluation, the 16-case rollout check, dev6 finer-rule convergence, successor/check validity and all-trivial handling preserve the inherited structure. Passing Q2-projected alone cannot satisfy a binding stop rule.

   The bank Euclidean norm remains basis- and scaling-dependent; identical thresholds do not establish equivalent calibration between projections. The frozen-bank scope and borrowed-threshold qualification are essential.

   Q1’s Gram-norm identity agrees algebraically with `qcert.py` for matching step equations and history. Startup and E5-interface equivalence remain subject to **E5R**.

5. **Fitting, matched norms and leakage — CORRECT.**

   A3.2 resolves the target-label problem:

   | Family | Fitted target | Matching indicator |
   |---|---|---|
   | F3a | \(Q^{\mathsf T}N\) | Non-binding Q2-projected |
   | F3b | \(N\) | Binding sine Q2 |
   | F2+F3 | \(T\) | Binding bank Q2 |

   Block normalization produces a sum of squared relative training-block errors. It does **not** certify the maximum validation-state error; independent Q2 remains necessary.

   Frozen training draws, descriptor disjointness, bounded replacements, snapshot/parent hashes and prohibited validation-driven refitting provide a sound leakage boundary. Different snapshot schedules mean F3a/F3b compare fitted procedures, not target norms in isolation; A3 now says so.

   A3.3 matches `nnls.py`: drift evidence is path-level, so rejecting every checkpoint from a drift-failed path is implementable and conservative.

6. **Controls — CORRECT at specification level; firing not demonstrated.**

   A3.4 resolves the unnamed fitted fixtures. The controls now protect appropriately limited claims:

   - **C1/C2:** require actual binding-Q2 failures; unsuitable fixtures receive no credit.
   - **C3/C4/P6:** test saved-state certification, pooled rejection, non-finite validity and invalid anchors/successors/checks.
   - **C5/C6:** test Jacobian-microbenchmark sensitivity and A–A timing sanity. Neither proves general absence of timing bias.
   - **C7/C8:** test descriptor separation and scoring identity.
   - **P1/P2:** test parent-path and conditional E5 equivalence.
   - **P3–P5:** test finite-rule convergence, definiteness, derivatives and independent operator assembly.

   C1’s coarse rules may be invalid; shuffled weights may not fail; P5’s mesh tolerance may be unsuitable. These are possible fixture failures, not reasons to presume successful controls. A2.6 correctly blocks the protected claim until an amended fixture demonstrably works.

7. **Timing, fairness and predicates — CORRECT.**

   A3.2 correctly counts six F3 candidates and retains exploratory, in-sample, selection-unadjusted intervals. Same-job pairing, immediate burn-in, retained repetitions, common eligibility and output agreement are appropriate.

   The F3 criterion remains a **1.5× point estimate plus evidence of some speedup**, not statistically established speedup of at least 1.5×. Conclusions remain internal ablations, not superiority over POD or spectral FOM competitors.

8. **E5 dependency and resources — NEEDS-RESTATEMENT; audit 3’s reservation inconsistency is resolved.**

   Three 12-minute cooperative fit deadlines fit arithmetically within a 90-minute smoke reservation, leaving additional work subject to measured overshoot, tails and the hard reservation. Completion is expressly not guaranteed. The cap and incomplete-evidence policy bound the experiment.

   **Memory wording needs qualification:** sampled `nvidia-smi memory.used` is an observed sampled maximum, not a guaranteed transient peak. `resource.getrusage().ru_maxrss` is a process-lifetime host high-water mark, not an isolated measurement of construction copies. Record those limitations and units; do not present either as precise phase-attributed peak allocation.

   The E5 handoff was absent at the specified path when checked. The fallback remains correctly labelled **local ablation; equivalence to E5 unestablished**, and cannot complete the prescribed E5 comparison.

9. **Other audit-3 ambiguities — CORRECT.**

   A3.6 addresses the requested distinctions: binding/projected targets, identified minimum/qualified upper bound, bounded runtime/guaranteed completion, and fallback/E5-equivalent evidence. Renaming the reconciliation amendment **E5R** removes the amendment-name collision.

**Acceptance:** no remaining WRONG finding; restate the two missing outcome branches and memory-measurement semantics before implementation acceptance. Real-data smoke evidence is still required for every protected qualification claim.