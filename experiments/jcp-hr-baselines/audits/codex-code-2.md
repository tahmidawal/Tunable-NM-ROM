**NOT READY for the phase-I smoke.** The two round-one crashes are fixed, but acceptance still trusts evidence it must independently recompute, and several registered controls remain incomplete.

I made no file changes, submitted no jobs, and ran no GPU work. Read-only CPU checks passed G-fd-a/b/c, N-fd, G-nnls, G-qdeim and N-dom. All **17 inherited hashes** and **16 claimed source-commit identities** match. The absence of `checks/test_core.json` is **not** counted as a defect.

Paths below are relative to `experiments/jcp-hr-baselines/`. Items 1–18 correspond directly to round-one findings; subsequent items are additional findings. Severity on resolved items identifies the original finding.

1. **BLOCKER — CORRECT: round-one NNLS crash fixed.**  
   `nnls.py:65–79` now creates a writable gradient copy and computes support/complement diagnostics without destructive masking. G-nnls passed, including exact tolerance checkpoints.  
   **Fix:** none required for this finding.

2. **BLOCKER — CORRECT: underdetermined native-GN shape failure fixed.**  
   `rom.py:116–126` uses QR of \(J^\top\) for fewer rows than unknowns. A CPU \(2\times3\) example returned the correct minimum-norm step.  
   **Fix:** none required for the dimension mismatch. Qualification of these problems remains wrong; see item 9.

3. **BLOCKER — WRONG: offline acceptance is still not independently recomputed or fully schema-driven.**  
   Improvements include required phase names and coefficient re-decoding, but:
   - `smoke_accept.py:43–52` trusts `cert.valid`, `cert.nonstat`, `cert.steps` and anchor summaries. It ignores the saved production gamma arrays. **CPU reproduction: Q1 returned `True` with production gamma values `NaN` and `2.0`.**
   - `:169` directly trusts N-under’s saved `q1` boolean; its raw trajectories and anchors are not persisted by `smoke.py:376–382`.
   - `:217,227` trusts each timing triple’s `valid` flag rather than recomputing drift and checking finite, positive durations.
   - `:70` trusts `all_pass` plus a gate count; it checks neither the exact 13 CPU-gate names/results nor the supplied source hashes.
   - `:76,98,115–116` derives expected cells, advection ranks and grid size from the submitted output/configuration instead of validating the frozen A5.4 inventory.

   **Fix:** validate the frozen schema; save per-step reasons, damping, gamma and trajectory-finiteness evidence for production and anchors; recompute qualification and drift; verify exact CPU gates and source hashes against the audited/staged code.

4. **MAJOR — WRONG: tighter-anchor coverage is repaired in smoke, but the Q1 interface still cannot enforce cohort coverage.**  
   `smoke.py:254–255` now runs both tighter cases. However, `pipeline.py:247–249` defines the cohort from whatever production records survived and accepts any supplied two-case subset. Dropping the same cases from production and tight records can still qualify a truncated cohort.  
   **Fix:** require explicit expected case identities and the prescribed tighter subset; verify production, tight, tighter and normalization coverage against them.

5. **MAJOR — CORRECT: damping saturation is rejected.**  
   `rom.py:300–311` checks every live damping value for finiteness and the \(10^{14}\) limit, independently of exit reason, and includes that condition in anchor validity.  
   **Fix:** none in the certifier; persist this raw evidence for item 3.

6. **MAJOR — WRONG: timing estimands are fixed, but six-case enforcement remains caller-dependent.**  
   `stats.py:28–45` correctly computes per-case medians, their median \(T\), and case-bootstrap intervals for \(T\) and \(\log T\). However, `:71–83` accepts an arbitrarily short `timing_cases` list. **CPU reproduction: three cases with `timing_cases=[0,1,2]` received “baseline dominates.”**  
   **Fix:** require the six registered identities, independently of caller-supplied coverage. The existing N-dom tests only exercise incomplete ratios against a correctly supplied six-case list.

7. **MAJOR — CORRECT: paired-ratio picks and X-specific fronts are implemented.**  
   `stats.py:161–172` selects roles using paired ratios and constructs the X-specific front. `:175–183` unions those fronts with the absolute-time front and near-front additions.  
   **Fix:** none for these helpers. Phase-II callers must provide qualified configurations and complete same-job, six-case ratios.

8. **MAJOR — WRONG: timing retries are retained, but output agreement is checked only selectively.**  
   `smoke.py:90–108` correctly retains retry attempts and burns before each synchronized call. However, `last[tag]` overwrites previous outputs. N-time compares only the final A output (`:498–500`); the ours comparison checks only the final X output (`:519–521`). B outputs and earlier A outputs are never checked. This violates A6.6’s **every timed output** requirement.  
   **Fix:** compare every call immediately against its independently scored reference and persist a finite discrepancy per call, including both synchronized calls inside N-time’s B.

9. **MAJOR — WRONG: N-under does not enforce A6.3’s underdetermined exclusion.**  
   `smoke.py:387–391` runs the revised support-cap ladder, but every level receives ordinary `q1_config` qualification. Neither that helper nor `smoke_accept.py:168–175` rejects an actual ECSW support smaller than \(r\). Such a level can therefore be credited as qualified degradation, expressly prohibited by A6.3. Raw anchor/distance evidence is also discarded.  
   **Fix:** force support-\(<r\) levels to qualification failure with an explicit reason; persist complete per-level evidence and recompute it offline. Check actual support size, not merely its requested cap.

10. **MAJOR — WRONG: N-grad now exercises shared gamma arithmetic, but still does not test production classification.**  
    `rom.py:239–265` shares tuple-gamma evaluation with the trajectory certifier. `smoke.py:428–443` compares these values with trajectory gamma and an autodifferentiated oracle—substantial improvement. But production nonstationarity labels are generated separately at `rom.py:304–309`, while acceptance independently applies thresholds at `smoke_accept.py:185–190`. A broken production label/count implementation can survive this control. Histories are not saved for replay, and part (b) reports only certification and one distance (`smoke.py:444–447`), not complete Q1.  
    **Fix:** exercise the production classification function on perturbed tuples, save tuple/history evidence, and report complete Q1 for part (b).

11. **MAJOR — CORRECT: N-swap fixture selection, fallback and shift are repaired.**  
    `smoke.py:460–468` tries the registered certified-valid ECSW ID, then certified-valid `none`, and records unavailability. `score.py:79–87` performs exactly the four shifted comparisons. `smoke_accept.py:22–23,210` requires the complete kind-control inventory.  
    **Fix:** none for the original finding. Offline fixture validity should be reconstructed as part of item 3.

12. **MAJOR — WRONG: missing real-size checks were added, but G-fd-a still uses a different fixture.**  
    `smoke.py:335–359` now includes full-field Q-DEIM norm/gradient comparisons and cancellation-safe bounds; the corresponding CPU checks passed. But G-fd-a uses a **projected physical IC** at `:295` and one random Jacobian action at `:300–305`. A1.9 specifies a random state and a Jacobian comparison. One directional action does not establish the registered Jacobian-entry bound.  
    **Fix:** use the frozen random-state fixture and independently check sampled stencil-Jacobian entries, or explicitly amend the test to a specified directional check.

13. **MAJOR — WRONG: construction gates are largely repaired, but fallback selection omits part of the certificate.**  
    Rank enforcement, saturation handling and DEIM backward-error enforcement now exist (`pipeline.py:142–154,176–185,230–240`). However, fallback triggering and candidate counts use projection certificates alone (`:82–90,121–130`). Orthonormality is considered only afterward (`:98,135`). A basis with passing projection errors but failing orthonormality can suppress fallback; a fallback can win the count despite failing orthonormality.  
    **Fix:** include finiteness and orthonormality in each candidate basis’s certification and fallback-selection logic before counting certified ranks.

14. **MAJOR — CORRECT: NNLS support KKT and summary persistence are fixed.**  
    `nnls.py:65–79,171–173` checks nonnegative finite weights, two-sided support stationarity and off-support gradients including banned indices. `build.py:329–330` persists banned counts and KKT diagnostics. Exact residual verification remains at `nnls.py:119–124`.  
    **Fix:** none for this finding. The original-column fallback uses `jnp.linalg.lstsq` (`:160`), rather than the specifically documented device-QR implementation; document that numerical-method deviation or implement the stated fallback.

15. **MAJOR — WRONG: maximal `none` workload is added, but checkpointing still loses completed evidence.**  
    `smoke.py:220–224` adds the maximal-rank query. However, `phase()` saves **before** the caller assigns its returned evidence (`:74–76`, e.g. `:198,218,224`). Arrays are written only after the entire cell returns (`:553–558`). Termination during a later phase loses all previously collected coefficient/field arrays; an uncaught cell exception also leaves the outer `arrays` empty.  
    **Fix:** save after phase results are installed and checkpoint arrays after each completed grid/control block. Acceptance should also validate actual maximal shapes and finite memory measurements, rather than only finite timings/residuals (`smoke_accept.py:101–110`).

16. **MAJOR — WRONG: most CPU-fixture fixes landed, but G-lmm still uses case 1.**  
    G-full now covers all options (`tests/test_core.py:276–287`); bounded order refinement and both N-dom degeneracy regimes are present (`:380–425`). Nevertheless, `:360` still calls `smooth_state(d,L)` with its default `seed=1` (`:49–51`), contradicting the frozen case-0 fixture and the immediately following comment.  
    **Fix:** pass `seed=0` explicitly. Preserve the “pre-asymptotic at 0.01” label when applicable.

17. **MAJOR — CORRECT: metadata pin now checks the loader’s actual files.**  
    `score.py:28–36` verifies the resolved metadata directory and hashes its STATUS file. `cluster/stage.py:93–94` unsets `JCPREF_META` and sets the paralab reference store.  
    **Fix:** none required.

18. **MINOR — NEEDS-RESTATEMENT: phase-II resource/interface obligations remain.**  
    Lattice extraction now selects device rows before host transfer (`pipeline.py:219–223`). Remaining live-buffer costs include unscaled contribution blocks, scaled blocks and concatenation (`build.py:296–309`), and large NNLS advanced-indexing temporaries (`nnls.py:109,158`). The submission lock remains lane-local (`cluster/submit.sh:17–23`); account-cap checks are present but not atomic across lanes.  
    **Fix:** inventory simultaneous buffers, recompilation shapes and all control costs in the future resource certificate; coordinate a shared submission lock if strict account-wide enforcement is required. The unwritten phase-II drivers are not themselves defects.

19. **MAJOR — WRONG: native-GN backtracking failure has an unregistered validity outcome.**  
    A1.15 requires exit reason 3 when no halving decreases the residual. `rom.py:164–166` instead returns reason 2, which `certify()` accepts as valid (`:307`). A CPU example with residual \(r(c)=c^2+1\), its exact Jacobian, and \(c=0.001\) stalled after one iteration and returned **reason 2**.  
    **Fix:** restore reason 3 for exhausted backtracking, or explicitly amend the design and associated validity/fixture-selection semantics. A6.4 does not amend stalled-backtracking behavior.

20. **MAJOR — WRONG: G-score can hide non-finite values.**  
    `smoke_accept.py:246–249` accumulates errors using Python `max` without first checking finiteness. `max(0., NaN)` returns `0.`, so non-finite decoded fields or scores can disappear from the reported worst discrepancy. `score.py:84–87,94–96` likewise lacks explicit finite/shape validation before reducing errors.  
    **Fix:** reject non-finite coefficients, bases, fields, normalizers, scores and discrepancies before any maximum; require exact expected shapes and positive finite \(n_0\).

21. **MINOR — NEEDS-RESTATEMENT: the phase-I ours adapter check is not G-ours.**  
    `smoke.py:504–525` executes the intended inherited adapters and scores case 0, but performs no comparison with exported errors. `smoke_accept.py:228–231` accepts a non-`None` E0 record plus timing agreement. The design’s G-ours requires reproduction of exported errors on the timing cases, while export pins are deferred to phase II.  
    **Fix:** explicitly distinguish the phase-I adapter execution check from G-ours; require pinned exports, all timing-case replay errors, and Q1+Q2 qualification before phase-II comparisons.

The inspected residual/Jacobian formulas and history handling are consistent across BE, CN, two-BE-start CN-R and one-BE-start BDF2: `hrcore.py:37–38,315–342`, `rom.py:53–108,211–231,259–265`, and `build.py:237–248,275–278`. ECSW uses the all-node target, prescribed scaling floor and weighted online rows. Q-DEIM retains the untruncated Householder \(R_B\) and checks the interpolation solve. Its projection-based pivot implementation matched LAPACK on the two tested fixtures; that is not a guarantee of universal finite-precision pivot identity.

The cluster scripts include GPU preflight exit 42, x64/highest precision, paralab paths, unique attempts, manifest checks, queue checks and staged-import isolation. I found no new large captured-array closure in the principal lane kernels. Maximal GPU feasibility remains unmeasured.

**Overall verdict: NOT READY.** Top blockers are **acceptance trusting qualification/timing flags**, **N-under crediting potentially underdetermined ECSW levels**, and **incomplete timed-output verification**. Fix those, the remaining fixture mismatch and GN exit semantics before running the phase-I smoke.