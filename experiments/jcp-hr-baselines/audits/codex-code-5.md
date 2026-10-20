**NOT READY for the phase-I smoke.** Most round-4 findings are repaired. A new, reproducible autodifferentiation memory defect threatens the registered real-size G-fd phase.

Read-only review: no files changed, no jobs submitted, no GPU work, and neither smoke nor test suite executed. Small CPU snippets used the prescribed interpreter with bytecode writes disabled. All **17 inherited hashes** and **16 source-commit identity claims** match. The regenerating `checks/test_core.json` is not counted as missing.

Paths below are relative to `experiments/jcp-hr-baselines/`. Items **1–26 track round 4**; resolved items retain their historical severity.

1. **BLOCKER — CORRECT: NNLS’s read-only-array failure remains fixed.**  
   `nnls.py:65–79` copies the gradient before computing support/complement diagnostics.  
   **Fix:** none required.

2. **BLOCKER — CORRECT: underdetermined native GN remains fixed.**  
   `rom.py:117–127` uses QR of \(J^\top\) for the minimum-norm step when there are fewer residual rows than coefficients. Non-finite steps receive reason 5 at `:165–167`.  
   **Fix:** none required.

3. **BLOCKER — CORRECT: the round-4 trajectory-schema hole is closed.**  
   `smoke_accept.py:69–89` requires exact `(steps + 1, r)` trajectories, one-dimensional certificate lists, integral known reasons, and registered step counts. Saved production outputs are checked against scheduled trajectory rows at `:270–273`.  
   **Fix:** none for the reported empty-coefficient-array counterexample. See item 28 for an honest non-finite-output edge case.

4. **MAJOR — CORRECT: Q1 coverage and normalization remain explicit.**  
   `pipeline.py:254–274` requires complete production/tight cohorts, exactly the first-two-case tighter subset, positive finite physical normalizers, pooled stationarity, and every-case distance bounds. Offline reconstruction uses freshly loaded normalizers at `smoke_accept.py:198–205,263–269`.  
   **Fix:** none required.

5. **MAJOR — CORRECT: damping saturation remains rejected.**  
   `rom.py:303–316` and `smoke_accept.py:82–88` reject non-finite damping and damping at or above \(10^{14}\), independently of termination reason.  
   **Fix:** none required.

6. **MAJOR — CORRECT: timing estimand and case coverage remain aligned.**  
   `stats.py:34–46` computes the median of per-case median ratios and bootstraps cases, including the interval for \(\log T\). `:72–86` requires all six registered timing cases.  
   **Fix:** none required.

7. **MAJOR — CORRECT: decision table, envelopes, fronts, and picks remain aligned.**  
   `stats.py:17–22,49–110` implements clipped envelopes, strict worst-error equivalence, standardized-mean equivalence, precedence, and equivalence annotations. `:164–186` implements paired-ratio picks and X-specific Stage-B fronts.  
   **Fix:** none in these helpers. Phase-II callers must enforce eligibility and complete qualified cohorts.

8. **MAJOR — CORRECT: incomplete timing evidence no longer passes the reported examples.**  
   `smoke_accept.py:112–141` checks repeat/retry inventory, discrepancy cardinalities, termination after success, and exhaustion of all four attempts after persistent drift. `:461` selects `(1,2,1)` for N-time.  
   `smoke.py:99–119,578–588` retains attempts, burns before calls, synchronizes outputs, and compares both B outputs separately.  
   **Fix:** none required.

9. **MAJOR — CORRECT: an errored N-under level now breaks acceptance.**  
   `smoke_accept.py:339–371` requires the exact ladder inventory, reconstructs Q1, rescores saved fields, and prevents either family from firing when any level has an evidence-loading error. Explicit construction failures remain distinguishable.  
   **Fix:** none required.

10. **MAJOR — CORRECT: N-grad’s inventory and part-(b) evidence are repaired.**  
    `smoke.py:473–530` records skipped-step identities, perturbation tuples, classification variants, and complete production/anchor evidence for part (b).  
    `smoke_accept.py:375–416` checks the Cartesian inventory, reconstructs oracle counts, and recomputes part-(b) Q1 without demanding that the mutation fail.  
    **Fix:** none for round 4.

11. **MAJOR — CORRECT: N-swap retains the registered fixture and fallback.**  
    `smoke.py:543–563` selects the certified-valid ECSW fixture, then the certified-valid `none` fallback, and exercises the kind controls. `smoke_accept.py:420–453` independently reconstructs selection and swap scores. `score.py:79–90` uses the prescribed four shifted comparisons.  
    **Fix:** none required.

12. **MAJOR — CORRECT: the registered G-fd fixtures are present.**  
    `smoke.py:320–341` implements the perturbed physical state, 4,096 sampled nodes, and 64 complete independent VJP rows. `:347–395` implements directional, autodifferentiation, weighted-row, and cancellation-safe Q-DEIM comparisons.  
    **Fix:** repair their execution-memory defect in item 27.

13. **MAJOR — CORRECT: POD fallback selection and construction gates remain aligned.**  
    `build.py:81–127` implements snapshot POD, re-orthonormalization, and the bounded randomized fallback. `:139–153` accumulates the snapshot norm independently.  
    `pipeline.py:76–98,116–138` compares certificates against the Gram-derived tail and selects the fallback only if it certifies strictly more registered ranks. Construction gates remain at `:142–154,176–185`.  
    **Fix:** none for round 4.

14. **MAJOR — CORRECT: NNLS checkpoints, original-column fallback, and KKT checks remain implemented.**  
    `nnls.py:24–62` implements passive-set stepping/removal; `:119–124` verifies checkpoint residuals directly; `:160` uses original contribution columns for the registered SVD fallback. `:65–79,171–174` checks both support stationarity and all off-support candidates, including banned indices.  
    **Fix:** none required.

15. **MAJOR — CORRECT: checkpointing and maximal-pilot evidence are substantially repaired.**  
    Grid JSON and arrays are saved per completed configuration at `smoke.py:295–302`; N-under’s `none` level is checkpointed before contribution construction at `:431–435`. ECSW pilot diagnostics are preserved at `:225–237`; maximal `none` finiteness evidence is recorded at `:240–245`.

    **Noted (forgery-only):** `smoke_accept.py:239–242` still relies partly on `kkt['ok']` instead of independently comparing the saved off-support gradient with its tolerance. Honest execution cannot produce a `reason='kkt'` record with failed KKT diagnostics because `nnls.py:171–173` renames that outcome.  
    **Fix:** optionally compare both saved gradient bounds directly; no blocker for this residual issue.

16. **MAJOR — CORRECT: G-lmm’s fixture and refinement reporting remain fixed.**  
    `tests/test_core.py:360` selects case 0. `:380–391` implements bounded refinement and records the “pre-asymptotic at 0.01” annotation. G-full covers every registered option at `:276–287`.  
    **Fix:** none required.

17. **MAJOR — CORRECT: metadata pins remain tied to the actual loader.**  
    `score.py:28–36` checks the resolved metadata directory, manifest version, and exact STATUS hash. `cluster/stage.py:94–95` clears `JCPREF_META` and supplies the paralab reference store.  
    **Fix:** none required.

18. **MINOR — NEEDS-RESTATEMENT: resource and campaign-concurrency obligations remain open.**  
    `build.py:299–312` retains contribution blocks while building scaled blocks and their concatenation. `nnls.py:109,158–160` creates potentially large indexed temporaries. `ours.py:39–41` transfers complete 2D fields before restriction. `cluster/submit.sh:17–23` uses a lane-local lock, so independent lanes can race the account-cap check.

    **Fix:** account for simultaneous buffers, transfers, compilation shapes, and retries in the phase-II resource certificate; coordinate a shared submission lock for atomic campaign-cap enforcement. The unwritten production drivers are not themselves phase-I defects.

19. **MAJOR — CORRECT: finite exhausted GN backtracking remains invalid for production.**  
    `rom.py:159–167` returns reason 3 after unsuccessful finite backtracking; `:310` rejects it for production. The amended anchor exception is separate.  
    **Fix:** none required.

20. **MAJOR — CORRECT: G-score no longer silently skips non-finite runs.**  
    `smoke_accept.py:285–306` requires non-finite runs to be classified invalid and have no finite score, and checks decoded-plus-invalid inventory counts.  
    **Fix:** none for the former false-pass path. Item 28 identifies an earlier check that prevents the intended invalid-run handling from working consistently.

21. **MINOR — CORRECT: phase-I adapter execution remains distinguished from G-ours.**  
    `smoke.py:624–626` and `smoke_accept.py:458–468` label this as an execution check. The adapters use the inherited query interfaces at `ours.py:31–37,60–62`.  
    **Fix:** phase II must add audited export pins, Q1+Q2 admission, and timing-case exported-error replay.

22. **BLOCKER — CORRECT: non-finite GN steps cannot qualify as round-off-stalled anchors.**  
    `rom.py:165–167` assigns reason 5 to a non-finite least-squares step. Both certification paths reject it: `rom.py:309–316`, `smoke_accept.py:83–88`.  
    **Fix:** none required.

23. **MAJOR — CORRECT: residuals, Jacobians, and histories match the amended registration.**  
    `hrcore.py:37–44,315–342` implements BE, CN, two-BE-start CNR, and one-BE-start BDF2 consistently with `t2core.SCHEMES`.  
    `rom.py:54–109,212–235,260–266` carries the correct histories through all arms, stepping, and fresh-Jacobian certification. `build.py:238–251,278–311` uses each training state’s startup/main coefficients and histories, the all-node \(J^\top r\) target, and the amended scaling floor. ECSW online rows use square-root weights.  
    **Fix:** none identified in these formulas.

24. **MINOR — CORRECT: Q-DEIM pivot equivalence is now stated narrowly.**  
    A12.3 and `build.py:197–202` explicitly describe modified Gram–Schmidt pivoting and limit LAPACK sequence equality to the registered fixtures. `tests/test_core.py:322–335` checks those fixtures.  
    The untruncated Householder \(R_B\), interpolation backward-error check, and stencil rows remain at `build.py:209–225`.  
    **Fix:** none required.

25. **MAJOR — CORRECT: the reported FD NaN-reduction defect is fixed.**  
    `smoke.py:82–85,338–340,354–362` propagates non-finite discrepancies rather than hiding them with ordinary Python `max`. The corresponding CPU-test reductions at `tests/test_core.py:158,376` prioritize non-finite values.  
    **Fix:** none for this finding.

26. **MAJOR — CORRECT: G-all now has independently reconstructible evidence.**  
    `smoke.py:400–408` saves both trajectories, lattice fields, and certificate lists. `smoke_accept.py:327–335` reconstructs validity and physical-normalized distance.  
    **Fix:** none required.

27. **BLOCKER — WRONG: unjitted G-fd autodifferentiation materializes an enormous discarded Jacobian tangent. New finding.**  
    At `smoke.py:363`, this expression differentiates a function that computes **both** residual and analytic Jacobian:
    ```python
    jax.jacfwd(lambda x: rJt(x, const, nu, co, dt, dat)[0])(c1)
    ```
    The analytic Jacobian construction at `rom.py:64–67` is still executed under eager autodifferentiation before `[0]` discards it. Its tangent has shape `(r, n, r)`.

    **Small CPU reproduction:** with \(n=49,r=4\), the discarded Jacobian’s tangent was an actual synchronized `ArrayImpl` of shape `(4,49,4)`, not merely a symbolic tracing artifact.

    At the registered smoke shapes, one such array occupies:
    - 2D: **137,170,649,088 bytes**.
    - 3D: **131,096,641,536 bytes**.

    These allocations coexist with resident state/advection bases and other intermediates. The 2D allocation plus resident basis arrays already exceeds the declared H200 capacity; multiple tangent intermediates worsen it. This arises during ordinary registered execution, without malformed evidence.

    **Fix:** expose residual-only evaluation for autodifferentiation, or JIT the complete Jacobian-check function with all large arrays passed explicitly so dead-code elimination removes the unused analytic-Jacobian branch. Apply the same pattern to `smoke.py:492`. Verify the absence of `(r,n,r)` intermediates before attempting the real-size smoke.

28. **MINOR — WRONG: NaN-containing invalid runs cannot reach the newly permitted G-score handling cleanly. New finding.**  
    `smoke_accept.py:272` uses `np.array_equal(W, scheduled_C)` without `equal_nan=True`. Matching NaNs therefore produce “W inconsistent with C rows” before `:285–291` can account for a legitimately invalid, unscored run. Such evidence can arise honestly from a non-finite solve; it is not forgery-only.

    **Fix:** compare scheduled outputs with NaN-aware equality only after establishing the run’s invalid classification and absence of a usable score. Retain strict finite equality for valid runs.

The cluster scripts contain GPU preflight exit 42, x64/highest precision, paralab paths, unique attempt directories, manifest verification, staged-import checking, and queue checks (`cluster/stage.py:77–120`; `cluster/submit.sh:13–27`). I found no new large captured-array closure in the principal lane kernels; item 27 is a separate eager-autodifferentiation allocation defect.

Phase II still needs complete cohort/eligibility enforcement, frozen selection, audited export replay, generator/budget ratification, resource and GPU-hour accounting, and checksum-verified collection and cleanup.

**Overall verdict: NOT READY.** The top blocker is **G-fd’s real-size discarded-Jacobian tangent allocation**. Repair that before launching the smoke. Numerical feasibility and control firing remain to be demonstrated by the smoke itself.