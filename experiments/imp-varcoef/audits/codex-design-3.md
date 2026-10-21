Read-only audit completed; no files changed, jobs submitted, or arrays copied. A2 takes precedence. Implementation claims are assessed as design requirements, not verified code.

1. **CORRECT — Gate quotation.** Binding wording remains intact.

2. **CORRECT — Gate outcomes.** A2.2 resolves mixed outcomes and makes reference failure override costs.

3. **CORRECT — Historical ROM costs.** Previously verified values remain unchanged. This does not establish current numerical qualification; see item 12.

4. **CORRECT — ROM population/checker contract.** A2.4 requires the missing gates, controls, identities and exact repetition coverage. Implementation verification belongs to the code audit.

5. **WRONG — Independent reference convergence.** A2.5 adds spatial/time/quadrature comparisons but no solver-refinement certificate. Its fixed convergence bars and `max(2.5e-4, 3ε_ref)` also permit cross-check uncertainty much larger than a small reference uncertainty. **Fix:** certify solver error, bound the combined cross-check uncertainty relative to `ε_ref`, and register spatial convergence evidence beyond a single difference. Label any departure from E0’s cross-check recipe explicitly.

6. **CORRECT — Eligibility/coverage.** Fallback precedes freezing, exclusions are reported, and conclusions are conditional on eligibility.

7. **CORRECT — Viscosity/extrema.** A2.7 uses the discrete operator’s face minimum and distinguishes sampled extrema from continuous bounds.

8. **CORRECT — PSI SPD.** The positive-weight quadratic-form proof remains valid independently of quadrature accuracy.

9. **CORRECT — MG fallback.** A2.9 removes the unjustified FCG switch. Repairing the preconditioner or dropping it resolves the identified issue; any omitted MG competitor must remain disclosed.

10. **WRONG — Temporal/stability qualification.** A2.10 merely reports a two-step comparison without an acceptance threshold. Extensions trigger only when *no configuration is valid*, missing finite but temporally inaccurate rows. It expressly permits a positive gate with an unresolved family. **Fix:** prescribe convergence/stability acceptance criteria, trigger bounded refinement for accuracy-limited rows too, and make unresolved required-family qualification block a positive gate unless an explicit competitor-omission exception is authorized.

11. **CORRECT — Competitor limitations.** Finite-ladder labels and the full-operator exponential check before training remain required.

12. **WRONG — ROM qualification.** A2.12 explicitly declines fresh Q1/Q2, contradicting COMMON-BRIEF §3: earlier labels are prior evidence only. Matching trajectories or GPU types does not waive that requirement. **Fix:** certify fresh stationarity, tight/tighter agreement and own-trajectory quadrature qualification, or obtain an explicit binding-brief exception before using these denominators.

13. **CORRECT — Timing identity.** A1.13 retains subject-level fingerprints and separately synchronized doubled executions. Scoring and timed executions must use the same full-output program.

14. **NEEDS-RESTATEMENT — Timing decomposition.** The setup-inclusive gate contract is now clear, but setup-only timing is not a separate kernel measurement; subtracting it is unreliable under compilation/fusion. **Fix:** register a prepared-setup kernel timing separately, alongside total query and setup-only costs.

15. **CORRECT — Selection/inference separation.** Family and union winners are frozen before stage 2.

16. **CORRECT — Estimand disclosure.** A2.16 explicitly defines the ratio of median anchor-normalized costs and reports absolute-time medians separately. Call the gate statistic “anchor-normalized cost ratio,” not a ratio of median query seconds.

17. **CORRECT — Pruning.** Eligible-dev error pruning, all-dev validity checks and the empty-eligible-dev case are covered.

18. **CORRECT — Phase-1 final evaluation.** A2.18 restores a frozen test-once FOM evaluation without test-based selection. Phase-2 reuse needs the new correction below.

19. **WRONG — Control power.** A nonzero initial residual does **not** imply a relative residual exceeding `10 tol`. Reference mutations also do not guarantee the named outcome if an earlier predicate already fails. **Fix:** freeze a real-case fixture with a measured residual margin above the rejection threshold; start reference-predicate tests from an accepted fixture, preserve earlier predicates, and assert the intended predicate fails. Preserve shared initial data when modifying later snapshots.

20. **NEEDS-RESTATEMENT — Aggregate budget.** A2.20 compares remaining work with 8 GPU-h, while the cap includes already-spent smoke work. **Fix:** require `spent + all remaining certified work ≤ 8`, including qualification, extensions, fallback, final testing and timing retries; stop or apply the registered defer order if it cannot fit.

21. **CORRECT — Extension selection order.** Dev-only provisional winners precede validation and qualification, removing the circular dependency.

22. **CORRECT — CCv tolerance tuning.** Configuration-specific Newton thresholds replace the inappropriate reference ceiling; tight-distance qualification still applies.

23. **WRONG — Q2 remains incomplete.** A2.23 checks rollout distance only, on eligible cases only, and unilaterally dismisses the binding residual-quadrature bar. A Galerkin FOM still has a quadrature-dependent tested operator whose discrepancy can be evaluated on its reached states. **Fix:** define and audit an applicable own-state residual/operator discrepancy certificate, retain rollout agreement, and cover every qualification case—or obtain an explicit exception to COMMON-BRIEF §3.

24. **CORRECT — Normalization cost.** Extrema are explicitly prepared case parameters, their preparation cost is reported, and the same convention applies to Phase 2.

25. **NEEDS-RESTATEMENT — Eligible timing replacements.** Eligibility is fixed, but replacement cases can have indices outside ROM dev6, leaving A1.16’s index pairing ambiguous. **Fix:** assign six workload slots; map each slot explicitly to one eligible FOM case and one ROM/anchor dev6 case, and bootstrap those frozen slot records.

26. **CORRECT — Stage-2 failures.** Incomplete coverage or veto makes the target inconclusive; no replacement is selected from inference data.

**NEW blocking problem**

27. **WRONG — Phase-2 test independence.** A2.18 opens the existing test split in Phase 1, superseding its reservation for Phase 2. The design leaves no unopened final cohort for the subsequently trained ROM. **Fix:** designate the current split as Phase-1-only and reserve a separately seeded, unopened Phase-2 test cohort before training or tuning; carry this requirement into the mandatory Phase-2 amendment.

**Overall verdict: WRONG — not CLEAN; unresolved design blockers remain.**