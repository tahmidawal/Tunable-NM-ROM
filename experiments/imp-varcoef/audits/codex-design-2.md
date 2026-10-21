Read-only audit completed; no files changed, jobs submitted, or arrays copied. Items 1–20 correspond to round 1; subsequent items are new findings.

1. **CORRECT — Gate quotation.** §9 retains the binding gate verbatim. No fix needed.

2. **NEEDS-RESTATEMENT — Gate interpretation.** A1.2 identifies the operational choices and imports Phase-2 conditions, but leaves “one target fails, the other is inconclusive” unspecified. **Fix:** pass if any eligible target passes; fail only if both decisive targets fail; otherwise unresolved. Reference failure overrides all costs.

3. **CORRECT — ROM costs.** Both source hashes match; all 21 saved rows reproduce with exact six-case, three-repetition coverage. The 3% and 10% winners remain 32.4192 ms and 14.7285 ms; no qualified 1% ROM exists.

4. **NEEDS-RESTATEMENT — ROM population/checker.** Both qualified collections are now searched correctly. However, `rom_box.py` still does not require C2 gates/controls to pass or verify exact case/repetition identities; `complete` merely means execution finished. The current input passes those checks independently. **Fix:** enforce them in the extractor.

5. **WRONG — Reference cross-check remains uncertified.** A single PSI run at \(N=256\) and one timestep does not establish independent convergence. **Fix:** register spatial, temporal, quadrature and solver refinement checks, then compare against the conditional reference uncertainty. Also use the actual API: `floor_proxy(res, n_stages)` and `steps_ok(res, n_stages)`.

6. **CORRECT — Eligibility and coverage.** A1.6 freezes the mask before scoring, reports exclusions and requires 24/32 eligible cases. **Clarification:** finish all permitted fallback attempts before freezing; describe conclusions as conditional on the eligible population.

7. **NEEDS-RESTATEMENT — Viscosity scope.** The incorrect Péclet rationale is retracted appropriately. However, a finer-grid minimum is not a conservative continuous lower bound. **Fix:** distinguish sampled extrema from certified bounds; use appropriate face/node minima for discrete checks or a certified continuous bound.

8. **CORRECT — PSI SPD proof.** Positive weights, positive diffusivity and the exact transpose establish SPD. The midpoint-rule change requires the qualification correction in item 23.

9. **WRONG — MG fallback.** The fixed symmetric cycle is substantially better specified, but failed symmetry or positivity does not justify switching automatically to FCG(1). **Fix:** repair/prove SPD for CG-family methods; use an appropriate nonsymmetric method such as FGMRES when necessary, with true-residual checks.

10. **NEEDS-RESTATEMENT — Stability and ladder extensions.** Reading errors across the ladder is not a registered temporal-convergence test. Cheap-edge extensions do not address instability at the smallest timestep. **Fix:** prescribe fixed-case convergence/stability checks and a bounded smaller-step extension; unresolved accuracy/stability must not support a positive gate.

11. **CORRECT — Missing competitors.** A1.11 supplies the required finite-ladder qualification and requires a full-operator exponential competitor before training. Its implementation, tuning and gate rerun must remain inside the remaining budget.

12. **WRONG — Qualification remains incomplete.** FOM residual and tight-solve checks improve the contract, but ROM error reproduction and hashes do not re-establish in-job Q1/Q2. CCv and quadrature problems also remain; see items 22–23. **Fix:** require fresh ROM stationarity, tight/tighter agreement and own-trajectory quadrature qualification before using its denominator.

13. **CORRECT — Timing identity.** A1.13 restores subject-level identity evidence and separately synchronized doubled executions. **Implementation requirement:** compare each timed execution with a scoring execution of the same full-output program.

14. **NEEDS-RESTATEMENT — Timing scope.** A1.14 explicitly charges FOM setup, but drops the requested separate C2-compatible kernel measurement. Coefficient normalization is also unaccounted for; see item 24. **Fix:** report both kernel and setup-inclusive costs and explicitly identify the gate’s cost contract.

15. **CORRECT — Selection/inference separation.** A1.15 freezes family and union winners before stage 2. No reselection using stage-2 results is permitted.

16. **NEEDS-RESTATEMENT — Cost estimand.** Balanced anchor indices fix the weighting mismatch, and the bootstrap label is clearer. But a ratio of medians of case-normalized times is generally **not** a ratio of median query times: case-dependent anchor costs do not cancel through medians. **Fix:** use a fixed anchor or gate on drift-adjusted absolute workload costs; otherwise name the normalized estimand explicitly.

17. **CORRECT — Pruning.** A1.17 restricts error pruning to eligible dev cases while retaining validity checks on every dev case. Define the empty eligible-dev case explicitly.

18. **NEEDS-RESTATEMENT — Final test policy.** Developmental labeling answers round 1’s proposed alternative, but does not itself override COMMON-BRIEF §2’s mandatory final test-once evaluation. **Fix:** obtain an explicit brief exception or reserve a frozen final FOM test if Phase 1 ends the lane. Historical labels are now carried forward.

19. **WRONG — Controls still lack guaranteed power.** Zero CG iterations need not create \(>10^{-3}\) trajectory error; one Newton iteration can satisfy the residual threshold. T-ref-a/b do not independently exercise every reference predicate. **Fix:** specify fixtures with demonstrably nonzero initial residuals and known violations; test each predicate separately. Moving T-kappa to diagnostics and adding I-mms are correct.

20. **NEEDS-RESTATEMENT — Budget.** Smoke certificates and fallback reservation are improvements, but the 6.2 GPU-hour estimate has not been reconciled with added cross-checks, tight/tighter runs, qualification, extensions and possible final testing. **Fix:** certify an aggregate remaining-work ledger before production, including explicit worst-case fallback and stopping allocations.

21. **NEW — WRONG: extension selection is circular.** A1.10 requests the cheapest *qualified* configuration after the dev ladder, whereas A1.12 qualifies timing candidates using pooled dev+val data. **Fix:** define dev-screened provisional winners for extension, then run validation and qualification before freezing timing candidates.

22. **NEW — WRONG: CCv tolerance tuning is effectively defeated.** A1.12 applies E0 `steps_ok`, whose fixed residual ceiling is \(2\times10^{-11}\), to CCv’s inherited tolerance ladder \(10^{-8},10^{-6},10^{-4}\). Normally converged ladder solves therefore fail validity unless they overshoot substantially. **Fix:** use configuration-specific true-residual thresholds plus tight-distance qualification for competitors; retain E0’s strict threshold for references.

23. **NEW — WRONG: quadrature qualification is missing across the ladder.** I-quad checks only \(N=64\), while PSI/ETD use \(N=16\)–128 with exponential variable coefficients. That cannot establish Q2 for each candidate’s reached states. **Fix:** compare successive rules on each timing candidate’s own trajectories, applying the common residual and rollout-distance bars; qualify the spectral cross-check too.

24. **NEW — NEEDS-RESTATEMENT: coefficient normalization cost is undefined.** Evaluating \(\kappa\) requires the descriptor-dependent extrema on the \(1025^2\) normalization grid. A1.14 lists face/quadrature evaluation but neither includes this reduction nor declares extrema prepared inputs. **Fix:** explicitly charge normalization or define it as shared input preparation, report its cost and use the same convention in Phase 2.

25. **NEW — WRONG: pooled coverage can exclude the entire timing cohort.** All 24 validation cases could qualify while all eight dev references fail; the 24/32 gate would still admit costs measured on six unscorable dev cases. **Fix:** require reference eligibility for every primary timing case, or freeze a deterministic eligible-case replacement rule before timing.

26. **NEW — NEEDS-RESTATEMENT: stage-2 failures lack a disposition.** A frozen union winner can be vetoed or exhaust timing retries, yet no rule specifies missing repetitions or replacement. **Fix:** declare that target inconclusive; any replacement requires a predeclared ranking and fresh inference measurements, never selection from stage-2 costs.

**Overall verdict: WRONG — A1 makes substantial repairs, but reference convergence, numerical qualification, controls, timing semantics and gate completeness still block production.**