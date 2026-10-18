**Overall: READY for the scoped code audit.** No blocking defect found. Re-submission still requires the in-progress gate suite and job audit required by `DESIGN.md:1799–1808`.

Scoped files match `7f5bb352e` at observed HEAD `5920fe48c`. All probes were CPU-only; no files or scratch created, and `tests/test_core.py` was not run. Paths below are relative to `experiments/jcp-hr-baselines/`.

1. **CORRECT — nonblocking: round-29 optimum-alias defect fixed.**  
   `nnls.py:407–418` synchronizes the gradient before checking the deadline and accepting `kkt`. Controlled-clock expiration at all five checks on the identity fixture—including the second iteration’s post-gradient check—returned `time`; the unreached target remained absent through `ecsw_weights`. Alias creation still requires `kkt` and passing diagnostics (`build.py:354–362`).  
   **Fix:** none required.

2. **CORRECT — nonblocking: entry timing and explicit inner-stop propagation.**  
   The timer precedes initialization (`nnls.py:356–373`). Expiring during the initial gradient returned `time`, elapsed `2`, with no iterations. An inner timeout on the final permitted iteration also returned `time` (`nnls.py:433–448`). A timeout following an actual deletion preserved feasible weights, matching support/factor dimensions, and zero reconstruction error (`nnls.py:307–325`).  
   **Fix:** retain these behaviors.

3. **NEEDS-RESTATEMENT — nonblocking: deadline-reason guarantees remain broader than implementation.**  
   Target completion precedes the new check (`nnls.py:403–406`): expiration during its residual computation returned `taus`, elapsed `2`, against budget `1`. Expiration during the final permitted solve, or detected by its refactor guard, returned `max_iter` (`nnls.py:441–460`). Neither creates an optimum alias. Terminal evidence is also explicitly outside the stopping budget (`DESIGN.md:1816–1819`), qualifying A40.1’s absolute wording.  
   **Fix:** describe cooperative deadline checks and these exceptions, including initialization/residual work in the overshoot inventory; alternatively check before target completion and propagate expiration detected after the final solve/refactor.

4. **NEEDS-RESTATEMENT — nonblocking: G-deadline does not cover “every crossing position.”**  
   The four clock increments at `tests/test_core.py:448–459` exercise the first outer check, first post-gradient check, first inner check, and second outer check. They miss the second post-gradient check—the round-29 regression—and final-iteration/refactor cases. Wrapper assertions explicitly allow `kkt` (`tests/test_core.py:465–466`). Independent probes verified the omitted post-gradient case.  
   **Fix:** add explicit check-index/event-driven cases, require `time` for injected timeout cases, and assert the missing wrapper target directly.

5. **CORRECT — nonblocking: sign-normalized refactor and deletion still match Givens.**  
   `nnls.py:227–235` transforms factors consistently. Independent refactor/delete probes at capacities 13, 96, and 128 gave maximum factor difference `5.33e-15`, reconstruction error `6.22e-15`, and least-squares error `1.39e-15`. G-qrdel retains random factor comparisons, boundary cases, and refactor/delete least-squares coverage (`tests/test_core.py:382–437`).  
   **Fix:** none required; boundary fixtures correctly claim reconstruction/least-squares coverage rather than raw-factor equivalence.

6. **CORRECT — nonblocking: no updating-QR regression found.**  
   Append/reset and refactor handling remain intact (`nnls.py:259–289`). Rejected dependent append restored buffers exactly; a subsequent append succeeded. Independent update/full paths matched checkpoint supports, exercised two deletions, differed in residual by `5.55e-17`, and recorded maximum drift `1.11e-15`. Budget evidence remains persisted (`build.py:375–381`).  
   **Fix:** none identified; retain the required production-interval gate (`tests/test_core.py:360–371`) before submission.