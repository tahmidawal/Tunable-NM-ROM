**READY — GO for preparing and running the phase-I cluster smoke.** Round 7’s sole open **WRONG** finding is fixed. I found no additional code blocker in this targeted review. This is permission to test feasibility and controls, not acceptance of their eventual results.

Read-only verification used small CPU snippets only. No files changed, jobs submitted, GPU work, or full test suite run. All **17 inherited hashes and 16 source-commit blob comparisons match**.

Paths below are relative to `experiments/jcp-hr-baselines/`. Severity on CORRECT items indicates the significance of the checked issue.

1. **MAJOR — CORRECT: A12.7 closes the matching-non-finite timing hole.**

   `smoke.py:95–100` makes the timing discrepancy non-finite if either comparison array contains NaN or infinity. N-time uses it for A and **both B outputs** at `:611–612`; the adapter comparisons use it at `:643–644`. `smoke_accept.py:130–133` rejects every non-finite discrepancy, including rejected timing attempts.

   CPU checks using the actual extracted functions confirmed that matching NaNs and signed infinities are rejected. Injecting a non-finite discrepancy into every case/output slot, separately, made acceptance fail for both timing cardinalities; finite controls passed.

   **Fix:** none required. Round 7 finding 1 is closed; no other open WRONG from round 7 remains.

2. **MAJOR — CORRECT: residuals, Jacobians, startup and history follow A1.1/A1.2.**

   `hrcore.py:37–38` matches the relevant `t2core.SCHEMES` entries at `t2core.py:36–37`. FOM history advancement is at `hrcore.py:315–342`; ROM residuals and Jacobians are at `rom.py:54–110`; stepping is at `:240–263`.

   CN includes the previous nonlinear term, CNR takes two BE startup steps, and BDF2 takes one before using the older state. ECSW training selects startup coefficients consistently at `build.py:278–280`.

   **Fix:** none identified.

3. **MAJOR — CORRECT: certification and Q1 implement the amended rules.**

   `rom.py:280–295` evaluates a fresh Jacobian with each step’s own history. `:324–351` counts non-finite gamma as non-stationary without a residual exception, rejects invalid exits/damping, and limits the anchor-stall exception as amended.

   `pipeline.py:254–274` requires complete cohorts, the first-two-case tighter subset, physical `n0`, pooled stationarity and every-case distance bounds. Offline acceptance reconstructs these predicates at `smoke_accept.py:69–105`, with independently loaded normalizers at `:198–206`.

   **Fix:** none identified.

4. **MAJOR — CORRECT: ECSW construction and NNLS retain the registered safeguards.**

   `build.py:230–255` supplies the deduplicated, startup-inclusive training schedule. `:268–312` constructs all-node \(J^\top r\) targets, candidate contributions and the positive block-scaling floor. Online weighting uses square-root weights at `rom.py:84–88`.

   `nnls.py:24–62` implements passive-set stepping/removal; `:119–124` checks tolerance checkpoints against directly evaluated residuals; `:160` supplies original-column least squares for fallback. KKT includes banned candidates at `:65–79`, and failed KKT cannot retain an optimum claim at `:171–173`.

   **Fix:** none identified in these paths.

5. **MAJOR — CORRECT: POD and Q-DEIM follow the later amendments.**

   `build.py:81–127` implements snapshot POD, reorthonormalization and the bounded randomized fallback. `:139–153` uses independently accumulated snapshot energy for the tail comparison; `pipeline.py:76–98,116–154` chooses one basis and excludes uncertified ranks.

   `build.py:197–202` correctly narrows pivot-sequence equivalence to the registered fixtures under A12.3. `:209–225` forms untruncated Householder \(R_B\), checks interpolation-solve backward error and gathers the required stencil rows.

   **Fix:** none. Do not describe the pivot implementation as generally identical to LAPACK.

6. **MAJOR — CORRECT: the former autodiff memory blocker remains removed.**

   `rom.py:115–137` provides residual-only evaluation. Real-size FD/autodiff and N-grad use it at `smoke.py:370–387,493–517`; native backtracking and the common predictor also use residual-only evaluation.

   Large FOM spectral data and ECSW construction arrays enter compiled functions as arguments (`hrcore.py:311,350–354`; `build.py:268–275`).

   **Fix:** optional cleanup: replace residual-only uses of `rJ(...)[0]` in the small CPU FD tests at `tests/test_core.py:160,184` with `make_res`.

7. **MAJOR — CORRECT: scoring and inherited adapters are consistent with phase I.**

   Store pins are enforced at `score.py:28–36`; IC and `n0` checks at `:66–74`; the time-max metric and four-comparison shift at `:79–90`; S restrictions at `:48–54`.

   Adapters invoke inherited query interfaces at `ours.py:31–37,60–62`. Their smoke execution check is explicitly **not G-ours** at `smoke.py:649–651`.

   **Fix:** none for phase I. Export replay and imported Q1+Q2 admission remain mandatory for phase II.

8. **MAJOR — CORRECT: statistics use the amended estimands and decision table.**

   `stats.py:17–22` clips lower envelopes. `:34–46` computes the median of case medians and resamples cases, including the interval for \(\log T\). `:62–110` implements strict worst-error equivalence, complete timing coverage and label precedence; `:164–186` implements paired-ratio picks and X-specific fronts.

   CPU checks confirmed both identical-arm regimes and the unequal-repeat example whose correct estimand is 1.25.

   **Fix:** production callers must enforce complete eligible-case intersections and qualification before supplying inputs.

9. **MAJOR — CORRECT: smoke failures and unavailable required controls cannot silently complete phase I.**

   Required phases are enumerated at `smoke_accept.py:24–26` and checked at `:207–210`. Acceptance reconstructs grid/Q1/scoring evidence and controls at `:252–457`. N-under requires qualified accuracy degradation; N-grad checks the perturbation inventory and independent oracle; N-swap follows the amended fallback semantics.

   Timing burns, synchronization, retained attempts and bounded retries are at `smoke.py:120–140`; offline inventory and drift checks are at `smoke_accept.py:112–141`.

   **Fix:** none identified. Actual control firing remains an empirical smoke requirement.

10. **MINOR — NEEDS-RESTATEMENT: maximal-shape feasibility remains unproved.**

    Simultaneously live ECSW blocks and concatenations occur at `build.py:299–312`; indexed NNLS temporaries and fallback operands at `nnls.py:109,158–160`; full-field host transfer before restriction at `ours.py:39–41`.

    **Fix:** phase-II resource accounting must include simultaneous buffers, factorization workspace, compilation variants and transfers. Treat the smoke as a measurement, not proof that every production branch fits.

11. **MINOR — CORRECT, with an existing concurrency limitation: cluster safeguards are present.**

    `cluster/stage.py:77–119` supplies GPU partition selection, paralab paths, x64/highest precision, preflight exit 42, store configuration, manifest validation and staged-import isolation. `cluster/submit.sh:13–27` checks queues, caps and unique attempt directories.

    Its lock is lane-local at `:17`; independent lanes can race the account-cap check.

    **Fix:** use a shared campaign submission lock for atomic account-wide enforcement.

12. **MINOR — CORRECT; noted (forgery-only): KKT acceptance could be hardened.**

    `smoke_accept.py:240–243` partly trusts the saved KKT `ok` flag rather than independently comparing the off-support bound. Honest `nnls.py:171–173` prevents a failed KKT check from retaining reason `kkt`.

    **Fix:** optionally compare both saved gradient bounds directly. This is nonblocking under COMMON-BRIEF §7.

The required `checks/test_core.json` is currently absent. Regenerate and commit the complete CPU gate evidence before staging: `cluster/stage.py:58–59` requires staged inputs to match HEAD, and `smoke_accept.py:165–170` binds that evidence to source hashes. I did not run its file-writing generator.

Phase II still needs production orchestration, generated workload/resource certificates, frozen selection, eligibility enforcement, export replay and qualification, budget accounting, and checksum-verified collection/cleanup.

**Overall: READY / GO for the phase-I cluster smoke after regenerating its required CPU gate artifact and completing the job-specific audit. Top code blockers: none identified; round 7’s timing blocker is resolved.**