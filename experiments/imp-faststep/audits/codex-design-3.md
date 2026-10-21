**Overall verdict: WRONG as a fully closed pre-registration.** A1 fixes the principal mathematical and Q1 defects, but leaves executable gaps in POD selection and Q2, an incorrect NNLS budget assumption, and controls whose promised detection is still unestablished.

I reviewed only the **imp-faststep** design, its previous audit, the applicable briefs, and supporting code. The multimesh questions about DST resampling, bank-fit projections and PM lower bounds are unrelated to E5 and were not audited. No files were changed, jobs submitted or numerical experiments run.

References below are to [the audited DESIGN.md](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-faststep/experiments/imp-faststep/DESIGN.md), with A1 taking precedence.

1. **A1.1 — Notation and table checks: CORRECT.**  
   The projected equation fixes the rectangular-ODE mistake. Triangular solves, rank/conditioning records, orthogonality diagnostics, eigenpair residuals and positive diffusion eigenvalues address the previous requests. The diffusion-coordinate algebra remains sound. Recording a condition number does not itself establish numerical safety, but the independent trajectory certificates provide additional protection; this is not a remaining mathematical blocker.

2. **A1.2 — ETDRK4 recurrence and timing: CORRECT.**  
   The frozen recurrence agrees with `pscore.etdrk4_coefs` and its timestep-factor convention. Charging coefficient construction and coordinate maps to evolution closes the timing omission. Recording every stage’s finiteness also closes the endpoint-only validity gap. The coefficient oracle introduced in A1.3 is appropriately independent of the contour calculation.

3. **A1.3 — Q1 analogues: CORRECT, with minor wording corrections.**  
   Reconstructing every ETDRK4 step on every case closes the substantive recurrence-certification hole. Checking the one-pass equation using the actual saved state difference closes the “correct solve, wrong update” hole. Valid tight/tighter anchors and separate Newton tolerances are appropriate. Two statements should be corrected: arm 4 **does** perform an algebraic linear solve, which its residual certificate checks; and quadratic Newton convergence does **not** guarantee that \(10^{-13}\) lies above the attainable floating-point residual floor for every conditioned problem. Neither undermines the conservative acceptance tests as written. This certifies implementation of the discrete schemes, not temporal accuracy, which A1 now correctly distinguishes.

4. **A1.4 — Q2: NEEDS-RESTATEMENT.**  
   All-case successor rollouts, same-main-state force comparisons and valid bank-ROM successors close the original Q2 defects. The new POD definition is incomplete when NNLS checkpoints are absent: if κ=4 exists but κ=8 does not, “next-larger κ” does not specify whether to skip to κ=16, use full integration, or mark qualification unavailable. A1.8 explicitly permits this situation. Specify a deterministic successor policy, require valid and Q1-certified POD successors, and define the full-integration endpoint’s label. Also state the denominator and zero/non-finite handling for the new POD force ratio. Without these rules, two implementations can assign different Q2 labels to identical artifacts.

5. **A1.5 — Stability semantics: NEEDS-RESTATEMENT.**  
   Recasting 1.05 as a conservative amplitude-growth rejection threshold correctly closes the false energy-theorem justification. The diagnostic classification remains too strong: agreement of two refined trajectories at the five scoring outputs does not establish convergence of an **every-step maximum** near the 1.05 boundary. Moreover, model growth and discretization-induced growth can coexist; “reference also trips” does not distinguish them exclusively. Specify convergence checks for the monitored maximum and an “unresolved/mixed” classification. The phrase “any tripped run” also needs scope: an ETDRK4 Galerkin reference is not automatically a same-ODE reference for the incumbent LSPG arm. Finally, applying p1core’s alternation index to \(z\) requires defining its stiff-mode mask and amplitude threshold in those coordinates.

6. **A1.6 — Order references: NEEDS-RESTATEMENT.**  
   The refinement comparison and per-case reporting substantially improve the order tests. However, reference convergence alone does not establish that the three tested timesteps lie in an asymptotic range; the prior audit explicitly requested that distinction. Retain the registered coarse-grid F5 criterion if desired, but describe its slope as an observed finite-range slope rather than proof or disproof of formal order. Define the outcome when any required case has an unconverged reference: it must not disappear from the median or leave F5 implicitly passing. A1.6’s listed reference tests also omit the stability-reference use in A1.5, which needs its own convergence contract.

7. **A1.7 — Stop rule and scoring: CORRECT.**  
   The quoted stop rule matches the lane brief apart from wrapping. A1 separates the strict all-case stop predicate from eligible, reference-resolved accuracy rankings; removes the ambiguous maxima-based tie statement; handles missing results; limits deferred conclusions to completed scope; and preserves F5’s separate outcome. The median paired evolution statistic remains a legitimate preregistered interpretation, provided it is described as median performance rather than a threefold gain on every case. E0 ST, conditional reference uncertainty, per-case eligibility and developmental cohorts remain appropriately specified. For implementation clarity, explicitly say whether A1 replaces the earlier additional eligible-subset stop predicate; the natural reading is that it does.

8. **A1.8 — POD competitors: NEEDS-RESTATEMENT.**  
   The new Galerkin contribution target, importance distribution, block floor, cross-rank ladder, native historical picks and carried-forward restrictions close most fairness objections. Three gaps remain. First, the retained diagnostic fallback still says “otherwise κ=16,” although A1 explicitly permits κ=16 to be unavailable. Second, the fastest-member selection needs a deterministic timing-tie rule. Third, \(B^*(X)\) is accuracy-matched on **dev6 only**: its frozen validation result may fail that match. Such a result cannot be presented unconditionally as an accuracy-matched validation speedup or an upper bound against validation-matched competitors. Report match status on the claimed population and limit the finite-ladder upper-bound statement accordingly. Numerical qualification should remain visibly separate from validity/stability, particularly for descriptive 5L runs with limited anchor coverage.

9. **A1.9 — Controls: NEEDS-RESTATEMENT.**  
   The revised C10 and C11 aggregate definitions are sound, and C13–C18 target the important new certificate paths. Nevertheless, several “must fire” assertions still depend on unshown witnesses: negative viscosity need not cross the threshold within the horizon; gauss16 need not rescue C7; uniform weights need not be ten times worse; one Newton iteration need not invalidate an anchor; and another case can have the same viscosity. C3m’s “always” is also unjustified without a lower bound on the mutation’s effect relative to its certificate tolerance. Require explicit preconditions and a demonstrated real fixture, with production blocked if a required detector is not exercised. C11 additionally assumes an available κ=8 support despite the new timeout policy. C9 still lacks a declared outcome after an out-of-band self-timing check. Add controls for an unavailable POD successor/checkpoint and for a dev-matched competitor that fails its validation match.

10. **A1.10 — Budget and feasibility: WRONG.**  
    The assertion that unchanged `ecsw_weights(..., max_seconds=600)` imposes a **per-rank** 600-second wall cap is false. In [build.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-faststep/experiments/jcp-hr-baselines/build.py:339), the initial attempt receives that limit and a banning-triggered retry receives it again. Three ranks can therefore consume approximately **3,600 seconds in NNLS attempts alone**, before state generation, contribution assembly, compilation and transfers. This does not prove the 1.25-hour job limit impossible, but invalidates the claimed cap and weakens the 0.8-hour estimate. Use a shared remaining-time deadline across retries or budget both attempts. The actual 4096² memory pilot and queued/running reservations are good fixes. The cost certificate must also cover the expanded POD ladder—up to 30 ETDRK4 configurations before the two native picks—and their all-case reconstruction/Q2 work, including full-node evaluations. The revised totals alone do not establish feasibility.

11. **A1.11 — Selection, timing and claims: NEEDS-RESTATEMENT.**  
    Dev-only best-stepper selection, explicit \(M\) override, retained raw timing records, full-field query contracts and incumbent-versus-competitor labels close the main earlier defects. Define the outcome when the dev-selected stepper fails validation: retain it as the failed frozen selection rather than silently replacing it with a validation winner. Timing also needs an explicit paired **full-query versus full-query** statistic for competitor claims. The cited `p1time` fused triplet brackets a B fused query with A **evolution**, which is useful for drift monitoring but is not itself a full-query speed ratio. Specify the comparator triplets, aggregation, invalid-triplet handling and deterministic selection ties.

The remaining issues are chiefly about whether the registered procedure has a unique, executable outcome under plausible failures—not whether the proposed reduced equations are correct.

| Amendment | Verdict | Closure status |
|---|---|---|
| A1.1 | CORRECT | Algebra and table diagnostics closed |
| A1.2 | CORRECT | Recurrence and timing omissions closed |
| A1.3 | CORRECT | Substantive Q1 gaps closed; wording corrections |
| A1.4 | NEEDS-RESTATEMENT | Missing POD successor policy |
| A1.5 | NEEDS-RESTATEMENT | Growth classification overclaims |
| A1.6 | NEEDS-RESTATEMENT | Order-test scope and unresolved outcomes |
| A1.7 | CORRECT | Stop/scoring defects substantially closed |
| A1.8 | NEEDS-RESTATEMENT | Missing checkpoints and validation-match claims |
| A1.9 | NEEDS-RESTATEMENT | Required witnesses and fallback outcomes |
| A1.10 | WRONG | NNLS timeout is per attempt, not per rank |
| A1.11 | NEEDS-RESTATEMENT | Frozen-selection outcome and paired fused timing |

Required changes:

- Correct the NNLS deadline/retry accounting and certify the expanded workload.
- Define unavailable-checkpoint handling for POD selection, Q2 and C11.
- Qualify POD successors and specify force-ratio edge cases.
- Restrict accuracy-matched speedup claims to populations where matching holds.
- Define converged stability diagnostics and unresolved order-test outcomes.
- Require demonstrated, nondegenerate mutation fixtures before production.
- Freeze selection ties, validation-failure handling and paired full-query timing.