**NOT READY: A12.6 resolves both round-six findings, but its non-finite discrepancy handling introduces a timing-acceptance hole.**

Read-only audit completed: no files changed, jobs submitted, GPU work, smoke, or test suite run. Small CPU snippets checked inherited hashes and discrepancy/acceptance helpers. All **17 recorded hashes and 16 source-commit blob identities match**. The regenerating `checks/test_core.json` is not counted as missing.

Paths below are relative to `experiments/jcp-hr-baselines/`. Severities on CORRECT items retain the significance of the audited issue.

1. **MAJOR — WRONG: matching non-finite outputs can now pass timing acceptance. New regression.**

   `smoke.py:82–92` intentionally returns zero discrepancy for matching NaNs or signed infinities. That is appropriate for the grid’s invalid-run accounting, but the same helper is used indiscriminately by N-time (`:603–605`) and ours’ A–B–A checks (`:635–637`).

   `smoke_accept.py:112–141` receives only scalar discrepancies and timings, so it cannot distinguish finite agreement from matching non-finite outputs. The adapter acceptance checks a finite score **only for case 0** (`:470–471`); non-finite outputs on cases 1–5 have no corresponding rejection.

   **Small CPU reproduction:** extracting the actual helpers, two all-NaN arrays produced `disc = 0.0`. Supplying that discrepancy for all outputs in an otherwise complete six-case timing record made `timing_ok` return `True`, retaining all 18 ratios. This demonstrates the acceptance path, not an observed failure of the registered numerical fixtures.

   This is **not forgery-only**: deterministic numerical failure can produce matching non-finite outputs in the separately evaluated and timed calls. No evidence editing is needed.

   **Fix:** preserve NaN-aware comparison for invalid grid runs, but require finite outputs for every timing call and its comparison field. Persist that evidence for every output—including both N-time B outputs—and enforce it offline. A failed timing fixture must remain unfulfilled.

2. **MAJOR — CORRECT: A12.6 resolves the small-mesh G-pod registration mismatch.**

   `DESIGN.md:1059` explicitly replaces the former rank threshold and reconstruction tolerance. `tests/test_core.py:350–353` now uses the amended \(10^{-8}\sigma_1\) threshold, **without the former rank cap**, and the amended \(10^{-6}\) reconstruction tolerance.

   Production certification remains separate at `pipeline.py:235–251`.

   **Fix:** none required for round-six item 1.

3. **MINOR — CORRECT: matching non-finite grid outputs reach invalid-run accounting.**

   `smoke.py:294` uses the repaired discrepancy helper. It distinguishes NaNs, positive infinity, negative infinity, finite-entry differences, and shape mismatches (`:82–92`).

   Offline acceptance uses NaN-aware scheduled-row equality only for independently invalid runs (`smoke_accept.py:273–275`), then requires non-finite outputs to be classified invalid and carry no usable score (`:288–295`). Q1 still rejects their non-finite distances.

   **Fix:** none for the grid repair; restrict its timing use as described in finding 1.

4. **MAJOR — CORRECT: residuals, Jacobians, startup, and history match the governing equations.**

   `hrcore.py:37–44` matches the relevant `t2core.SCHEMES` entries. The FOM advances both histories at `:315–342`. The three ROM arms implement A1.1/A1.2 at `rom.py:54–110`; the stepper selects startup coefficients and advances histories at `:240–263`.

   CN includes the old nonlinear term; CNR uses two full BE steps; BDF2 uses one BE startup step and the older coefficient thereafter. ECSW training selects the corresponding coefficients at `build.py:278–280`.

   **Fix:** none identified.

5. **MAJOR — CORRECT: certification and Q1 follow the amended predicates.**

   `rom.py:280–295` recomputes a fresh Jacobian using each step’s own history. `:324–351` counts non-finite gamma as non-stationary, provides no residual exception, rejects damping saturation and reason 5, and implements the restricted A9/A11 anchor-stall exception.

   `pipeline.py:254–274` requires complete cohorts, the first-two-case tighter subset, physical `n0`, pooled stationarity, and every-case distance bounds. Offline reconstruction uses freshly loaded reference normalizers (`smoke_accept.py:198–206,263–270`).

   **Fix:** none identified.

6. **MAJOR — CORRECT: ECSW construction and NNLS retain the registered safeguards.**

   `build.py:230–255` implements option assignment and deduplicated startup-inclusive training schedules. `:268–311` computes the all-node \(J^\top r\) target, candidate contributions, importance sampling, and positive scaling floor. Online rows use square-root weights (`rom.py:84–88`).

   `nnls.py:24–62` performs passive-set stepping/removal; `:119–124` checks checkpoint residuals directly; `:160` uses original contribution columns for fallback. `:65–79,171–173` checks KKT over all candidates, including banned indices.

   **Fix:** none identified in these numerical paths.

7. **MAJOR — CORRECT: POD and Q-DEIM construction match the later amendments.**

   Snapshot POD, reorthonormalization, and bounded randomized fallback appear at `build.py:81–127`. Projection certificates use independently accumulated snapshot energy (`:139–153`); `pipeline.py:76–98,116–154` selects one basis by certified-rank count and rejects uncertified constructions.

   `build.py:197–202` correctly limits the LAPACK pivot-equivalence claim to the registered fixtures. `:209–225` forms untruncated Householder \(R_B\), checks the registered interpolation-solve backward error, and gathers stencil rows.

   **Fix:** none identified.

8. **BLOCKER — CORRECT: the previous discarded-Jacobian autodiff allocation remains removed.**

   Residual-only evaluation is implemented at `rom.py:115–137` and used by real-size G-fd (`smoke.py:362–379`), N-grad’s oracle (`:485–509`), native backtracking, and the common predictor.

   **Fix:** none for the former blocker. Minor cleanup remains: CPU directional-FD checks still evaluate `rJ(...)[0]` at `tests/test_core.py:160,186`, despite A12.5’s broader residual-only wording.

9. **MAJOR — CORRECT: scoring and inherited adapters are consistent with their phase-I roles.**

   Store metadata pins are checked at `score.py:28–36`; canonical IC and `n0` checks at `:66–74`; the time-max metric and four-comparison shift at `:79–90`; S restrictions at `:48–54`.

   Adapters call inherited query interfaces at `ours.py:31–37,60–62`. The smoke explicitly labels their execution check as **not G-ours** (`smoke.py:641–643`).

   **Fix:** finding 1 for timing finiteness. Export replay and imported Q1+Q2 admission remain phase-II obligations.

10. **MAJOR — CORRECT: statistics implement the amended estimands and decision table.**

    `stats.py:17–22` clips lower error envelopes. `:34–46` computes the median of case medians and bootstraps cases, including the interval for \(\log T\). `:62–110` implements strict worst-error equivalence, six-case coverage, label precedence, and equivalence annotations.

    Paired-ratio picks and X-specific Stage-B fronts appear at `:164–186`.

    **Fix:** none in these helpers. Production callers must enforce complete eligible-case intersections and qualification before supplying inputs.

11. **MAJOR — CORRECT, except finding 1: registered smoke controls and retry machinery are present.**

    G-fom, G-fd/N-fd, G-all, N-under, N-grad, and N-swap are implemented at `smoke.py:164–179,326–583`. Offline acceptance reconstructs Q1, scores, control inventories, and outcomes at `smoke_accept.py:252–457`. Missing or failed required phases prevent acceptance (`:207–210`).

    A–B–A burns, synchronization, three permitted remeasurements, retained attempts, and per-output comparisons are at `smoke.py:112–132`. Offline retry/cardinality checks are at `smoke_accept.py:112–141`.

    **Fix:** reject non-finite timing outputs independently of equality. Actual control firing remains an empirical smoke outcome.

12. **MINOR — NEEDS-RESTATEMENT: resource feasibility remains conditional on cluster measurements.**

    `build.py:299–312` retains contribution blocks while constructing scaled blocks and their concatenation. `nnls.py:109,158–160` creates indexed temporaries and potentially substantial fallback operands. `ours.py:39–41` copies full 2D fields before restriction.

    **Fix:** include simultaneous buffers, factorization workspace, compilation variants, retries, and host/device transfers in phase-II resource accounting. The CPU dry run does not establish maximal GPU feasibility.

13. **MINOR — CORRECT, with an existing concurrency limitation: cluster safeguards are present.**

    `cluster/stage.py:77–119` provides GPU partition selection, paralab paths, x64/highest precision, preflight exit 42, reference-store configuration, manifest validation, and staged-import checking. `cluster/submit.sh:13–27` checks queues, caps, and unique attempt directories.

    Its lock is lane-local (`submit.sh:17`), so independent lanes can race the account-wide cap check.

    **Fix:** use a shared campaign submission lock for atomic account-cap enforcement.

14. **MINOR — CORRECT; noted (forgery-only): residual KKT acceptance hardening is nonblocking.**

    `smoke_accept.py:240–243` partly trusts `kkt['ok']` instead of independently comparing the saved off-support bound. Honest `nnls.py:171–173` prevents a failed KKT check from retaining reason `kkt`.

    **Fix:** optionally compare both saved gradient bounds directly. This is not a launch blocker under COMMON-BRIEF §7.

Phase II still needs production orchestration, generated workload/resource inventories, complete eligibility enforcement, frozen selection, export pins and replay, imported Q1+Q2 admission, budget/generator ratification, GPU-hour accounting, and checksum-verified collection/cleanup. Their absence is outside this audit’s phase-I scope.

**Overall verdict: NOT READY.** The remaining launch blocker is **finding 1: timing acceptance can mistake matching non-finite outputs for successful execution**. Both original round-six findings are resolved; no new residual-formula or comparable autodiff-memory blocker was identified.