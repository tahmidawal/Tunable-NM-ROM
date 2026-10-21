**Overall: WRONG — Amendment 2 improves the design, but qualification precedence, policy transitions and cost certification remain blocking.** Reviewed both prior audits, the amended design, both briefs and the inherited selector/solver code. No files changed, jobs submitted, scratch created or protected cohorts accessed.

| Audit-2 item | Verdict | Fixed? |
|---|---|---|
| A1 | NEEDS-RESTATEMENT | Algebra fixed; fixture degeneracies remain |
| A2 curvature | NEEDS-RESTATEMENT | Estimator corrected; zero-scale comparisons undefined |
| A2 decision edge cases | NEEDS-RESTATEMENT | Tail floor added; zero field/application ambiguous |
| B0 | WRONG | Qualification improved; precedence contradicts stop/OR rules |
| B1(c) | CORRECT | Explicitly deferred pending another amendment |
| B2 Q1 fixture/anchors | NEEDS-RESTATEMENT | Anchors fixed; nonlinear fixture not guaranteed |
| B3 | CORRECT | Development references and split counts explicit |
| B5 | WRONG | Successor mapping and exceptional transitions incomplete |
| B7 | CORRECT | K1 and relevant-family timing specified; omission disclosed |
| B8 | NEEDS-RESTATEMENT | S1 arithmetic correct; complete-path funding unsupported |
| B9 | NEEDS-RESTATEMENT | Control inventory improved; expensive-path certification incomplete |

**A1 — NEEDS-RESTATEMENT.** The exact-D algebra is correct: orthogonality gives bank residual norms \(|\alpha|\) and \(\sqrt{\alpha^2+\gamma^2c^2}\), while both sine residual norms equal \(|\alpha|\). Thus \(\gamma=\sqrt{99}|\alpha|/c\) gives \(D=10\); swapping gives \(0.1\). However, \(c=0\), \(\alpha=0\), zero \(t_0\), or activated floor clipping invalidate the construction as written.
**Fix:** freeze seed, nonzero normalization field and \(\alpha\); assert \(c\) is sufficiently nonzero and all four floors exceed the clipping threshold. Specify deterministic failure/fallback if those prerequisites fail.

**A2 curvature — NEEDS-RESTATEMENT.** Renaming the estimator and adding refinement fixes the central objection. “Relative change” and independent-path “relative agreement” still lack explicit denominators and zero handling; compatible fields can have exactly zero curvature.
**Fix:** define norm-based comparisons with a frozen absolute tolerance scaled by the interior Laplacian, plus relative tolerance. Treat zero interior scale explicitly. Apply refinement to the independent value-difference path too, or label its fixed-spacing discrepancy as an unresolved discretization check.

**A2 decision edge cases — NEEDS-RESTATEMENT.** For an identically zero field, the new test evaluates \(0<10^{-24}\cdot0\), which is false, leaving \(\omega=0/0\). Also, Amendment 1 calls \(\omega(v-Sv)\): applying the relative-tail threshold to that already high-pass input defeats a threshold intended relative to \(v\).
**Fix:** reject zero/nonfinite total energy explicitly; compute availability from the original field and its directly retained tail coefficients. Define the decision as \(\omega(v)\), and state that unavailable control statistics force INCONCLUSIVE.

**B0 — WRONG.** Full-population arm qualification fixes the finite-median loophole. But precedence still fails two concrete cases: a known safety-guard violation plus missing endpoint control returns INCONCLUSIVE instead of STOP; an arm-(c) certified \(m^*\) success plus unavailable tail control can also return INCONCLUSIVE, contradicting “either route suffices.” The listed population minima are not scoped per arm.
**Fix:** after NOT RUN, let an evaluable violated safety guard force STOP. Scope minima to the applicable endpoint and guard. Evaluate arm-(c) routes independently as PASS/FAIL/UNAVAILABLE: either PASS suffices with a passed guard; both FAIL imply STOP; otherwise INCONCLUSIVE. Enumerate finite control prerequisites explicitly.

**B1(c) — CORRECT as a deferral.** The missing architecture choices are not supplied, but implementation is now explicitly prohibited until an amendment supplies them. This does not constitute approval to implement arm (c).

**B2 Q1 fixture/anchors — NEEDS-RESTATEMENT.** FI/FC1 test IDs 4 and 5 fix the missing anchor specification. The singular-vector argument only describes the local linearization; it does not guarantee stationarity failure under the fresh Jacobian at the perturbed state. Counterexample: \(r(w)=(\epsilon,w-w^2)\), \(w^*=0\), \(\delta=1/2\). The initial point is stationary and \(\|J\delta\|\ge100\|r^*\|\) for small \(\epsilon\), but the perturbed Jacobian is zero, giving \(\gamma=0\). A zero residual also permits zero perturbation under the stated inequality.
**Fix:** freeze the selected smoke step, nonzero perturbation magnitude/sign and bounded search procedure; recompute the actual nonlinear residual and fresh Jacobian, requiring measured \(\gamma>10^{-3}\). Failure to find a valid fixture must block certification.

**B3 — CORRECT.** Both development references are expressly permitted and separated from final scoring. The amended counts give 28 test cases and 16 development/validation cases.

**B5 — WRONG.** `succ = rule j+1` contradicts the registered successor map: fib6765’s next escalation candidate is gauss96, but its Q2 successor is fib10946; gauss96’s next candidate is fib10946, but its successor is gauss128. Missing prod, missing half, unavailable/nonfinite Q2 statistics and nonfinite step distances also lack explicit transitions.
**Fix:** separate `next_candidate[j]` from `q2_successor[j]`; retain the exact registered mapping, adding gauss256→gauss384. Explicitly handle missing/nonfinite evidence and define transition precedence. Preserve both own-state \(\rho\) evaluation and rollout-distance checks; a shared initial state alone does not establish the former.
The **10-round bound is mathematically correct** for eight candidate positions and monotone transitions: \(1+7+2=10\). However, “round 10 reached” overlaps ordinary terminal outcomes. Evaluate round 10 fully before applying the cap; the bound does not itself establish funding.

**B7 — CORRECT for the revised, conditional deliverable.** K1 is explicit; timing targets flagged steep families with the declared accuracy target. If timing is omitted, fallback cost remains unmeasured and E9 remains incomplete on that requested deliverable.

**B8 — NEEDS-RESTATEMENT.** S1 correctly totals **14.6 GPU-h**; S2 totals **14.2**. Removing arm-(a) training and selection saves **4.85**, not exactly 4.9, absent another specified saving. More importantly, “typical paths elsewhere” cannot support “the whole path is funded”; FN0s/FI can also escalate. Tight/tighter solves and comparison replays lack an explicit inventory.
**Fix:** generate exact schedule totals from a complete solve inventory, including all permitted paths, reference refinement, compilation and optional timing. Identify any extra S3 savings. Count spent smoke once. If worst-case funding fails, preregister budget-exhaustion outcomes instead of claiming complete coverage.

**B9 — NEEDS-RESTATEMENT.** Synthetic transition tests and both full-data generators repair substantial omissions. Round-1 FC1 solves still do not measure every expensive rule/factor/anchor phase, and two natural reference runs need not exercise the 12288 refinement path. A contingency multiplier cannot certify an omitted phase.
**Fix:** cost every distinct expensive phase using measured or explicitly justified conservative rates, including finer factors, largest rules, tighter anchors and reference refinement. Exercise the repaired exceptional transitions synthetically, and impose a smoke-specific resource ceiling before smoke submission.