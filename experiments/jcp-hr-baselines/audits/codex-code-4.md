**NOT READY for the phase-I smoke.** The non-finite GN anchor bug is fixed, but offline acceptance still accepts incomplete or invalid evidence.

Read-only audit: no files changed, no jobs submitted, no GPU work, and neither smoke nor test suite executed. Small CPU snippets tested acceptance predicates only. All **17 inherited hashes** and **16 source-commit identity claims** match. The regenerating `checks/test_core.json` is not counted as a defect.

Paths below are relative to `experiments/jcp-hr-baselines/`. Items **1–22 correspond to round 3**; resolved items retain their previous severity.

1. **BLOCKER — CORRECT: NNLS read-only-array failure remains fixed.**  
   `nnls.py:65–79` copies the gradient before inspecting support and complement.  
   **Fix:** none required.

2. **BLOCKER — CORRECT: underdetermined native GN remains fixed.**  
   `rom.py:117–127` computes the minimum-norm step through QR of \(J^\top\) for wide Jacobians. Non-finite steps now receive a separate failure code.  
   **Fix:** none required for this finding.

3. **BLOCKER — WRONG: acceptance schema enforcement is improved but incomplete.**  
   Registered step counts, known reason codes, canonical normalizers, and the exact CPU source inventory are now checked (`smoke_accept.py:69–101,159–164,198–199`). However, trajectory validation checks only `C.shape[0]`, not its rank or coefficient dimension (`:75–77`).  
   **CPU reproduction:** an empty coefficient array of shape **`(51, 0)`**, with 50 plausible certificate entries, returns **valid=True and anchor_ok=True**. Saved output coefficients are also not checked against the corresponding trajectory rows.  
   **Fix:** require exact trajectory shapes `(steps + 1, registered_rank)`, one-dimensional certificate arrays, integer reason codes, and consistency between saved `W` and the scheduled rows of `C`.

4. **MAJOR — CORRECT: Q1 cohort coverage remains explicit.**  
   `pipeline.py:254–274` requires complete production/tight cohorts, the prescribed tighter subset, finite positive physical normalizers, pooled stationarity, and every-case distance bounds. Offline distance coverage is now checked at `smoke_accept.py:88–101`.  
   **Fix:** none beyond item 3’s evidence-shape enforcement.

5. **MAJOR — CORRECT: damping saturation remains rejected.**  
   `rom.py:308–316` and `smoke_accept.py:78–84` reject non-finite damping and damping at or above \(10^{14}\).  
   **Fix:** none required.

6. **MAJOR — CORRECT: timing estimand and six-case coverage remain correct.**  
   `stats.py:34–46` bootstraps cases and computes the median of per-case median ratios, including the interval for \(\log T\). `:72–86` prevents callers from shortening the registered cohort.  
   **Fix:** none required.

7. **MAJOR — CORRECT: decision table, envelopes, picks, and fronts remain aligned.**  
   Clipping, strict worst-error equivalence, standardized-mean equivalence, precedence, and annotations match the amendments (`stats.py:17–22,49–110`). Paired-ratio picks and X-specific Stage-B fronts remain present (`:164–186`).  
   **Fix:** none in these helpers. Phase-II callers must enforce complete eligible cohorts and qualification.

8. **MAJOR — WRONG: N-time’s NaN reduction is fixed, but incomplete timing evidence still passes.**  
   Both B-call discrepancies are now saved independently (`smoke.py:554–564`). Acceptance checks their finiteness, but accepts any nonempty discrepancy list and does not require an unsuccessful repeat to exhaust its retries (`smoke_accept.py:108–135`).  
   **CPU reproduction:** acceptance returns True with only **one B discrepancy**, and with one repeat ending after its first failed attempt while other repeats succeed.  
   **Fix:** require discrepancy cardinalities `(1,2,1)` for N-time and `(1,1,1)` for ordinary triples. Require either a successful final attempt or all four failed attempts.

9. **MAJOR — WRONG: N-under rescoring is fixed, but broken ladder evidence can still be ignored.**  
   The exact ladder tags and canonical E0 rescoring are implemented (`smoke_accept.py:305–328`). However, an exception loading or validating one level becomes `levels[tag]['error']`; another firing level can still make the family pass (`:329–335`). This violates A6.7’s missing/errored-item rule.  
   **Fix:** require every ladder entry to have valid evidence or an explicitly permitted construction-failure record before evaluating whether any level fires. Missing array files and malformed run evidence must fail acceptance.

10. **MAJOR — WRONG: N-grad still accepts incomplete and malformed evidence.**  
    `smoke_accept.py:344–358` checks total tuple counts and global uniqueness, but does not enforce each tested step’s exact \(k=0..6\), sign \(=\pm1\) inventory, identify legitimate skipped steps, or require unique classification variants. Part (b) checks only two dictionary keys (`:364–366`); its fields and anchor histories are not persisted (`smoke.py:496–506`).  
    **CPU reproduction:** N-grad(a) passes with perturbations only at step 0, illegal \(k=0..139\), and 14 duplicate classifications. N-grad(b) simultaneously passes with `max_dist=NaN`.  
    **Fix:** validate the exact Cartesian product for every non-skipped step, persist the skipped-step identities and zero-singular-value evidence, require all 14 unique classifications, and save/recompute part (b)’s complete Q1 evidence.

11. **MAJOR — CORRECT: N-swap’s principal acceptance defects are fixed.**  
    Reference and ROM scores are recomputed from pinned references and saved fields; fixture selection is re-derived from certificate evidence, and explicit unavailability is required (`smoke_accept.py:369–403`). Missing grid artifacts separately fail the grid gate.  
    **Fix:** none beyond the shared certificate-validation repairs.

12. **MAJOR — CORRECT: the registered real-size G-fd-a fixture remains implemented.**  
    `smoke.py:305–329` uses the perturbed physical state, 4,096 sampled nodes, and complete Jacobian rows for 64 nodes against independent VJPs. G-fd-b/c retain the registered directional and cancellation-safe comparisons (`:330–383`).  
    **Fix:** address the non-finite reduction issue in item 25.

13. **MAJOR — CORRECT: POD fallback selection remains aligned with A6.**  
    Both state and advection bases use certificate-based fallback selection, choosing the fallback only when it certifies strictly more registered ranks (`pipeline.py:76–98,116–138`). Construction gates remain enforced (`:142–154,176–185`).  
    `build.py:101–127` implements the bounded sketch and re-orthonormalized power iterations; `:139–153` uses the independently accumulated snapshot norm for the tail.  
    **Fix:** none required for this finding.

14. **MAJOR — CORRECT: NNLS checkpoints, fallback, and KKT checks remain implemented.**  
    `nnls.py:24–62` implements passive-set stepping/removal; `:119–124` checks checkpoint residuals directly; `:160` uses the registered original-column SVD fallback. Final KKT checks include banned candidates (`:65–79,171–174`).  
    **Fix:** preserve those diagnostics in the maximal pilot—item 15.

15. **MAJOR — WRONG: checkpointing and maximal-pilot evidence remain partially repaired.**  
    Array chunks now persist per grid configuration and N-under level (`smoke.py:281–286,409–413`). Remaining problems:
    - Grid JSON is installed and saved only **after the entire grid** (`:289–297`). A wall-time kill still loses the certificates and configuration records needed to interpret completed array chunks.
    - The N-under `none` level is not checkpointed before the next contribution/NNLS workload (`:419–428`).
    - The ECSW pilot discards its KKT diagnostics (`:229–231`); acceptance treats the string `reason='kkt'` as sufficient evidence of a certified early stop (`smoke_accept.py:232–236`).
    - The `none` pilot gate checks time, memory, and a declared shape, but no numerical-output finiteness (`:237–239`).

    **Fix:** checkpoint paired JSON/arrays after every completed block; persist and validate raw KKT diagnostics; retain finite-output evidence for the maximal `none` workload. Frozen support and positive-memory checks are now present.

16. **MAJOR — CORRECT: G-lmm’s fixture and reporting are fixed.**  
    The fixture explicitly selects case 0 (`tests/test_core.py:360`). The bounded refinement ladder remains present, and the previously missing “pre-asymptotic at 0.01” annotation is now recorded (`:380–391`).  
    **Fix:** none required.

17. **MAJOR — CORRECT: metadata pins remain tied to the actual loader.**  
    `score.py:28–36` checks the resolved metadata directory, manifest version, and STATUS hash. Staging clears `JCPREF_META` and supplies the paralab store (`cluster/stage.py:94–95`).  
    **Fix:** none required.

18. **MINOR — NEEDS-RESTATEMENT: resource and campaign concurrency obligations remain open.**  
    Contribution construction retains unscaled blocks while building scaled concatenations (`build.py:296–309`); NNLS creates potentially large indexed temporaries (`nnls.py:109,158–160`). The 2D adapter transfers full decoded fields before restriction (`ours.py:39–41`). Submission locking remains lane-local (`cluster/submit.sh:17–23`), so independent lanes can race the account-cap check.  
    **Fix:** include simultaneous buffers, transfers, compiled shapes, retries, and controls in the phase-II resource certificate. Use a shared campaign lock for atomic account-cap enforcement. These unwritten phase-II drivers are not themselves defects.

19. **MAJOR — CORRECT: finite exhausted GN backtracking remains invalid for production.**  
    `rom.py:159–167` returns reason 3 when finite backtracking cannot decrease the residual; production certification rejects it (`:310`). The anchor exception is now separated from non-finite-step failure.  
    **Fix:** none required.

20. **MAJOR — WRONG: G-score’s finite-evidence protection has regressed.**  
    `smoke_accept.py:266–268` silently skips a run when `W` or `F` is non-finite. It does not establish that the run is invalid and carries no finite score. The maxima remain zero, and G-score can pass (`:279–281`).  
    For example, finite `C`, finite matching lattice fields, and a corrupted non-finite `W` can pass Q1 while the coefficient re-decode check is skipped.  
    **Fix:** never skip silently. Require finite coefficients/fields for scored runs; for invalid runs, explicitly validate the invalid classification and absence of a usable score. Track the exact number of completed re-decodes.

21. **MINOR — CORRECT: the adapter check remains correctly distinguished from G-ours.**  
    `smoke.py:600–602` and `smoke_accept.py:408–418` label phase-I adapter execution separately from exported-error replay.  
    **Fix:** none for phase I. Phase II still needs audited export pins, Q1+Q2 admission, and timing-case replay.

22. **BLOCKER — CORRECT: non-finite GN steps no longer qualify as round-off-stalled anchors.**  
    `rom.py:165–167` assigns reason **5** to non-finite least-squares steps. Both production and anchor certification reject it (`:309–316`); offline acceptance does likewise (`smoke_accept.py:79–84`).  
    **Fix:** none required for the round-3 blocker.

23. **MAJOR — CORRECT: the core residuals and histories match the amended registration.**  
    `hrcore.py:37–44,315–342`, `rom.py:54–109,212–235,260–266`, and `build.py:235–248,275–278` consistently implement BE, CN, two-BE-start CNR, and one-BE-start BDF2, including \(c_n,c_{n-1}\) in certification and ECSW training. The certifier uses a fresh Jacobian and no residual exception. A8’s native production tolerance and native anchors are used (`pipeline.py:203–215`; `smoke.py:269–270`).  
    ECSW uses the all-node \(J^\top r\), amended scaling floor, and square-root-weighted online rows.  
    **Fix:** none identified in these formulas.

24. **MINOR — NEEDS-RESTATEMENT: Q-DEIM pivot equivalence should be stated narrowly.**  
    `build.py:178–203` uses successive orthogonal projections with largest-residual-column selection, rather than LAPACK’s Householder `geqp3` implementation. The small random fixtures compare exact pivot sequences (`tests/test_core.py:322–335`), but that does not establish universal finite-precision sequence identity.  
    The untruncated Householder \(R_B\), prescribed interpolation backward-error check, and stencil rows are correctly constructed (`build.py:206–222`).  
    **Fix:** describe the implemented pivot algorithm explicitly and limit equality claims to checked fixtures, or implement the required Householder pivoting algorithm if identical `geqp3` behavior is intended.

25. **MAJOR — WRONG: non-finite FD discrepancies can still be hidden by Python `max`. New finding.**  
    `smoke.py:327–328,342–348` reduces Jacobian-row and directional-FD errors with `max(previous, value)`. As with the repaired N-time defect, a NaN value can leave a finite previous maximum unchanged. The offline gate sees only the reduced scalar (`smoke_accept.py:283–298`). Similar reductions occur in `tests/test_core.py:158,376`.  
    **Fix:** retain component discrepancies and explicitly require every component to be finite before reducing. Propagate any non-finite component as a failed gate.

26. **MAJOR — WRONG: G-all acceptance still trusts in-job validity flags. New finding.**  
    `smoke.py:388–395` saves only the scalar distance and two `valid` booleans. `smoke_accept.py:300–301` accepts those directly. Neither trajectory evidence nor the two lattice outputs is available for independent reconstruction.  
    **Fix:** save both runs’ trajectories, certificate lists, and lattice fields; recompute validity and distance offline using the registered option and canonical \(n_0\).

The inspected main lane JIT kernels pass large arrays as arguments; I found no new large captured-array closure there. Maximal GPU feasibility remains unmeasured. Cluster staging includes GPU preflight exit 42, x64/highest precision, paralab paths, unique attempt directories, manifests, staged-import checks, and queue checks (`cluster/stage.py:77–120`; `cluster/submit.sh:13–27`).

Phase II additionally needs complete eligibility/cohort enforcement, frozen selection, audited export replay, resource and budget accounting, and checksum-verified collection and cleanup.

**Overall verdict: NOT READY.** Top blockers are incomplete acceptance schemas, malformed N-grad evidence passing, incomplete timing and N-under evidence passing, G-score silently skipping invalid coefficients, and checkpoint/pilot evidence that still falls short of A11.