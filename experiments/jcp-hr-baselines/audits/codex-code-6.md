**NOT READY under the current pre-registration.** The round-5 memory blocker is fixed. I found a remaining mismatch in the required POD test and an incomplete repair of invalid-run acceptance.

No files changed, jobs submitted, GPU work, smoke, or test suite run. Checks used small CPU snippets and allocation-free shape tracing. All **17 inherited hashes** and **16 source-commit identity claims** matched. The regenerating CPU-gate file is not counted as missing.

Paths below are relative to `experiments/jcp-hr-baselines/`.

1. **MAJOR — WRONG: the small-mesh G-pod gate is weaker than registered.**

   **Evidence:** `tests/test_core.py:350–353` uses a numerical-rank threshold of `1e-8 * sig[0]`, caps the reconstruction basis at 200, and accepts reconstruction error up to `1e-6`.

   A2.11 specifies rank \(\rho=\#\{\sigma_i>10^{-10}\sigma_1\}\); A1.9’s small-mesh reconstruction tolerance is `1e-8`. The later Frobenius-certificate amendments do not explicitly replace this separate reconstruction test. Consequently, an honestly generated CPU “PASS” can certify a different, weaker gate. This is not forgery-only.

   **Fix:** implement the registered reconstruction test, using a stable factorization where necessary, and remove the unexplained rank cap. Alternatively, explicitly amend the small-mesh fixture, rank definition, and tolerance before treating its PASS as the registered gate.

2. **MINOR — WRONG: matching non-finite runs still fail acceptance before the intended invalid-run accounting.**

   **Evidence:** the round-5 W/C comparison is repaired at `smoke_accept.py:273–275`: invalid runs receive NaN-aware equality. However, `smoke.py:281` still calculates `max(abs(warm - scored))`. Matching NaNs—or matching infinities—produce NaN. `smoke_accept.py:281–283` then records an evidence error before `:288–295` counts the run as invalid and unscored.

   Thus the specific W/C defect is **CORRECT**, but the intended A12.1 handling remains incomplete.

   **Fix:** retain numerical discrepancies for finite outputs. For non-finite outputs, separately record matching non-finite locations and finite-entry discrepancies; accept that evidence only when the trajectory is independently classified invalid and has no usable score. Never admit it to Q1 or timing comparisons.

3. **BLOCKER — CORRECT: round-5 item 27’s discarded analytic-Jacobian tangent is eliminated.**

   **Evidence:** `rom.py:115–137` implements residual-only evaluation. Its consumers include native backtracking (`:155–182,224–226`), the common-LM predictor (`:250–255`), G-fd (`smoke.py:349–366`), and N-grad’s oracle (`:472–496`).

   Small CPU checks covered **96 arm/dimension/scheme/startup combinations**: `make_res` and `rJ` residuals agreed exactly; the largest analytic-versus-autodiff Jacobian discrepancy was approximately **5.6e-17**.

   Abstract traces at both registered maximal shapes found **no `(r,n,r)` intermediates and no captured constants** in these residual Jacobians. The remaining full-field cancellation check differentiates a residual directly (`smoke.py:379–394`), rather than an analytic Jacobian.

   **Fix:** none for the blocker. Minor cleanup: CPU directional-FD checks still evaluate `rJ(...)[0]` at `tests/test_core.py:157,183`; use `make_res` there too, as A12.5 states.

4. **MAJOR — CORRECT: residual equations, Jacobians, startup, and histories match the amended design.**

   **Evidence:** `hrcore.py:37–44` agrees with the relevant `t2core.SCHEMES` entries. FOM history advances at `hrcore.py:315–342`; ROM history and startup selection are at `rom.py:240–261`; certification uses the corresponding isolated history at `:280–295`.

   BE, CN, two-full-BE-step CNR, and one-BE-step BDF2 are consistent. The three arm formulas at `rom.py:54–110` implement A1.1/A1.2, including both CN nonlinear history terms and BDF2’s older coefficient.

   **Fix:** none identified.

5. **MAJOR — CORRECT: certification and Q1 implement the governing amended predicates.**

   **Evidence:** `rom.py:268–295` recomputes a fresh Jacobian per step. `:324–351` counts non-finite gamma as non-stationary, has no residual exception, rejects damping saturation and reason 5, and limits the anchor stall exception to A9/A11.

   `pipeline.py:254–280` requires explicit cohort coverage, the prescribed tighter subset, physical `n0`, every-case distances, and pooled stationarity. Offline reconstruction checks saved trajectory shapes and per-step evidence at `smoke_accept.py:69–105`.

   **Fix:** none identified beyond item 2’s treatment of invalid output evidence.

6. **MAJOR — CORRECT: ECSW construction and NNLS implement the amended numerical contract.**

   **Evidence:** `build.py:230–253,278–312` implements option-assigned trajectories, deduplicated startup-inclusive schedules, both histories, the all-node \(J^\top r\) target, candidate selection, and the positive scaling floor. Online weighting uses square-root weights at `rom.py:84–91`.

   `nnls.py:24–62` performs passive-set stepping/removal; `:119–124` verifies tolerance checkpoints directly; `:160` falls back to least squares on original contribution columns. `:65–79,171–174` checks support stationarity and every off-support candidate, including banned indices.

   **Fix:** none identified in these paths.

7. **MAJOR — CORRECT: Q-DEIM and production POD construction match the later amendments.**

   **Evidence:** `build.py:81–127` implements snapshot POD, reorthonormalization, and the bounded randomized fallback. `:139–153` obtains the tail using the independently accumulated snapshot norm. `pipeline.py:76–98,116–154` selects one basis by certified-rank count and enforces constructibility.

   `build.py:209–225` forms untruncated Householder \(R_B\), checks the prescribed `T3E` backward error, and gathers stencil rows. Pivoting is correctly described as modified Gram–Schmidt with the Businger–Golub rule—not universal LAPACK Householder equivalence—at `:197–204`, consistent with A12.3.

   **Fix:** repair the separate CPU gate in item 1; no additional production-formula defect identified.

8. **MAJOR — CORRECT: scoring pins, physical normalization, and kind restrictions are implemented.**

   **Evidence:** `score.py:28–36` pins the metadata directory actually used by the inherited loader, its manifest, and STATUS. `:63–77` checks canonical IC and `n0`; `:82–99` implements the registered time-max metric and four-comparison shift. `:48–55` restricts S to 2D BE at `0.005`.

   N-swap’s certified-valid ECSW/`none` fallback and offline reselection appear at `smoke.py:548–567` and `smoke_accept.py:423–457`.

   **Fix:** none identified. Phase-II callers must still enforce complete eligible-case intersections.

9. **MAJOR — CORRECT: decision statistics, timing estimands, fronts, and picks remain aligned.**

   **Evidence:** `stats.py:17–22,34–110` implements clipped envelopes, case-level bootstrap, median-of-case-medians timing, strict equivalence bounds, complete six-case timing coverage, and label precedence. `:164–186` uses paired-ratio fronts and includes X-specific fronts in Stage B.

   **Fix:** none in these helpers. Production drivers must supply qualified configurations and complete error cohorts; these helpers cannot establish provenance or eligibility themselves.

10. **MAJOR — CORRECT: the principal smoke controls and timing harness are present and fail closed.**

    **Evidence:** required phase and frozen inventories are checked at `smoke_accept.py:24–45,179–210`. G-fd independently checks stencil rows, Jacobians, weighted residuals, and cancellation-safe Q-DEIM quantities (`smoke.py:313–398`). G-all saves reconstructible evidence (`:402–410`).

    N-under’s amended ladders and own-problem anchors are at `:415–456`; N-grad’s isolated perturbation/oracle/classification machinery is at `:460–534`. Offline predicates reconstruct their outcomes at `smoke_accept.py:340–420`.

    A–B–A burns, synchronization, retained retries, and every-output checks are at `smoke.py:99–119,573–630`; acceptance independently derives drift and checks retry cardinality at `smoke_accept.py:108–141`.

    **Fix:** items 1–2. Actual control firing and completion within the wall limit remain empirical smoke outcomes, not established by this audit.

11. **MINOR — NEEDS-RESTATEMENT: maximal-memory feasibility remains conditional.**

    **Evidence:** `build.py:299–312` simultaneously retains contribution blocks, scaled blocks, and their concatenation. `nnls.py:109,158–160` creates indexed temporaries and potentially large fallback operands. `smoke.py:379–388` uses eager full-field residual autodiff/SVD for the cancellation candidate; `ours.py:39–41` transfers full 2D outputs before restriction.

    I found no further discarded-Jacobian tangent defect. That does **not** establish aggregate GPU peak memory or runtime: shape tracing excludes allocator, executable-cache, and factorization-workspace effects.

    **Fix:** include simultaneous buffers, shape-specific compilations, retries, and transfers in the phase-II resource certificate; use smoke measurements to determine feasibility.

12. **MINOR — CORRECT, with an existing concurrency limitation: cluster staging contains the required safeguards.**

    **Evidence:** `cluster/stage.py:77–119` specifies the GPU partition, paralab output/cache paths, x64/highest precision, preflight exit 42, reference-store environment, manifest validation, and staged-import isolation. `cluster/submit.sh:13–27` checks queues, caps, and unique attempt directories.

    The lock is lane-local (`submit.sh:17`), so another lane can race the account-wide eight-job check. This is an operational limitation, not a computed-number defect.

    **Fix:** use a campaign-wide submission lock if strict atomic account-cap enforcement is required.

13. **MINOR — CORRECT; noted (forgery-only): residual acceptance hardening is nonblocking.**

    **Evidence:** `smoke_accept.py:240–243` partly trusts `kkt['ok']` rather than independently comparing the saved off-support bound. Honest `nnls.py:171–173` prevents a failed KKT check from retaining reason `kkt`.

    **Fix:** optionally compare both saved gradient bounds explicitly. Do not classify this as a launch blocker under COMMON-BRIEF §7.

Phase II still needs the production drivers, generated workload/resource inventory, complete cohort and eligibility enforcement, audited export pins and replay, Q1+Q2 admission for ours, generator/budget ratification, frozen selection, accounting, and checksum-verified collection/cleanup. Their current absence is out of scope.

**Overall verdict: NOT READY.** The blocking registration issue is **the weakened small-mesh G-pod gate**. Resolve it before relying on the regenerated CPU-gate PASS. Also finish invalid-run acceptance handling. **The round-5 autodiff memory blocker is fixed; no new comparable memory blocker was identified.**