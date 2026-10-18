**NOT READY for the phase-I smoke.** Most round-2 implementation defects are fixed, but acceptance still permits incomplete evidence, and A9 introduces a reproducible false acceptance of a failed anchor solve.

I made no file changes, submitted no jobs, and ran neither the smoke nor the test suite. Small CPU reproductions used the prescribed interpreter with bytecode writes disabled. All **17 inherited hashes** and **16 source-commit identity claims** match. The regenerating CPU gate file is not counted as a defect.

Paths below are relative to `experiments/jcp-hr-baselines/`. Items **1–21 correspond to round 2**; severity on resolved items preserves the original severity.

1. **BLOCKER — CORRECT: NNLS’s read-only-array crash remains fixed.**  
   `nnls.py:65–79` explicitly copies the gradient and evaluates support/complement diagnostics without destructive masking.  
   **Fix:** none required.

2. **BLOCKER — CORRECT: underdetermined GN dimensions remain fixed.**  
   `rom.py:117–127` uses QR of \(J^\top\) for a wide Jacobian and returns the minimum-norm step.  
   **Fix:** none for this issue. Non-finite-step classification now has a separate anchor defect—item 22.

3. **BLOCKER — WRONG: offline acceptance is improved but still not fully independent or schema-driven.**  
   Gamma, damping and reason lists now replace several trusted flags, and drift is recomputed. Remaining holes:
   - `smoke_accept.py:65–73` checks list lengths against the **submitted** `cert['steps']`, rather than the registered scheme’s step count, and still trusts `C_finite`. Full production/anchor coefficient histories are not saved by `smoke.py:278–291`.
   - `smoke_accept.py:176–189` uses saved `n0` without comparing it with the freshly loaded canonical reference normalizer.
   - `:111` accepts any supplied source-hash inventory; an empty dictionary passes `all(...)`.
   - `:295–300` checks surviving timing cases but not the three original repeats, retry numbering or complete attempt inventory.

   A small CPU reproduction accepted one-step certificates for both cases; `run_evidence` also accepted exit reason **99** as valid and anchor-valid. The helper additionally accepts empty distance dictionaries, although current callers populate them.

   **Fix:** derive step counts and shapes from the frozen configuration; require known reason codes, complete histories or independently checkable finiteness evidence, canonical `n0`, an exact source-file inventory, and complete repeat/retry identities.

4. **MAJOR — CORRECT: Q1 cohort coverage is now explicit.**  
   `pipeline.py:254–274` requires exact production/tight coverage, exact tighter coverage, the first-two-case subset, and positive finite normalizers. Smoke callers supply the registered cases at `smoke.py:271,401`.  
   **Fix:** none for the in-job interface. Apply equivalent strictness to offline evidence under item 3.

5. **MAJOR — CORRECT: damping saturation remains rejected.**  
   `rom.py:303–315` rejects non-finite damping and values at or above \(10^{14}\), independently of exit reason. Offline reconstruction repeats that check at `smoke_accept.py:67–73`.  
   **Fix:** none required.

6. **MAJOR — CORRECT: timing estimand and six-case enforcement are fixed.**  
   `stats.py:34–46` computes per-case median ratios, their median \(T\), and case-bootstrap intervals for \(T\) and \(\log T\). `:72–86` rejects a caller-supplied shortened timing cohort.  
   **Fix:** none for this finding.

7. **MAJOR — CORRECT: paired-ratio picks and X-specific fronts remain implemented.**  
   `stats.py:161–186` uses paired ratios for roles and includes X-specific fronts in Stage B. The decision-table precedence, clipped envelopes and strict equivalence rules at `:17–22,49–110` match the governing amendments.  
   **Fix:** none in these helpers; phase-II callers must supply complete, qualified, eligible inputs.

8. **MAJOR — WRONG: every timing call is checked, but N-time can still hide a non-finite second output.**  
   `smoke.py:93–113` now checks each call after synchronization and retains every attempt. However, `:554` combines the two B-call discrepancies using Python `max`. A finite first discrepancy followed by `NaN` returns the finite value: the CPU reproduction `max(np.float64(0), np.float64('nan'))` returned **0.0**. Acceptance sees only the collapsed `dB` (`smoke_accept.py:296–298`).

   **Fix:** persist both B-call discrepancies separately and require each to be finite and within tolerance before reducing them. Also validate repeat/retry coverage as in item 3. The A10.1 switch to the `none` fixture is correctly implemented at `smoke.py:538–556`.

9. **MAJOR — WRONG: N-under qualification is repaired, but its offline degradation test still trusts unrecomputed errors and incomplete ladders.**  
   The amended \(4r,2r,\lceil1.25r\rceil\) ladder, own-problem anchors and actual-support exclusion are present at `smoke.py:395–429`; offline support exclusion is at `smoke_accept.py:237–239`.

   However, acceptance takes E0 errors directly from `v['e']` (`:240–249`) instead of rescoring the saved production fields. G-score covers only the ordinary grid (`:319–336`). Acceptance also iterates whatever ladder entries are present, so missing or malformed levels need not prevent acceptance if another level fires.

   **Fix:** require the exact amended ladder inventory, explicitly represent permitted construction failures, and recompute every level’s E0 errors from its saved fields and pinned references.

10. **MAJOR — WRONG: production classification is now exercised, but N-grad acceptance still lacks complete evidence validation.**  
    `smoke.py:474–490` genuinely calls `RM.certify` on each perturbation variant; part (b) now computes complete Q1 at `:492–500`. These repair the principal round-2 implementation issue.

    Offline acceptance checks only that 14 saved count pairs agree (`smoke_accept.py:260–261`). It neither reconstructs those counts from the tuple gamma values and base trajectory nor requires the expected step/variant inventory. Missing unperturbed tuples pass vacuously at `:258–259`. Part (b) may be absent and is still excluded from required gates at `:267`.

    **Fix:** validate every expected tuple, permitted skipped step and unique variant; reconstruct oracle counts; validate base certification; require part (b)’s complete reported evidence, without requiring its Q1 outcome to fail.

11. **MAJOR — WRONG: N-swap execution is fixed, but offline fixture validation remains incomplete.**  
    Exact fixture selection, certified-valid fallback, four shifted comparisons and kind controls are implemented at `smoke.py:505–530` and `score.py:79–90`.

    Acceptance still trusts saved scoring scalars and treats **any non-dictionary** ROM fixture—including a missing value—as legitimate unavailability (`smoke_accept.py:273–285`). It does not verify the selected ID’s certification or establish that both prescribed fixtures were unavailable.

    **Fix:** recompute reference and ROM swap scores from pinned references/saved fields; validate the selected ID and raw certificate. Accept unavailability only when both registered candidates are demonstrably unavailable.

12. **MAJOR — CORRECT: the real-size G-fd-a fixture is now covered by A7.5.**  
    `smoke.py:301–325` uses a perturbed physical state, sampled residual rows, and complete stencil-Jacobian entries for 64 rows against independent VJP rows. G-fd-b/c and the cancellation-safe Q-DEIM comparisons are at `:326–380`.  
    **Fix:** none for the round-2 fixture mismatch.

13. **MAJOR — CORRECT: basis fallback selection includes orthonormality and finiteness.**  
    `pipeline.py:82–98,121–138,235–251` applies the full certificate before counting passing ranks and selects the fallback only when it certifies strictly more ranks. Construction gates remain enforced at `:142–154,176–185`.  
    **Fix:** none required.

14. **MAJOR — CORRECT: NNLS KKT, checkpoints and fallback registration are aligned.**  
    `nnls.py:65–79,119–124,169–174` checks support stationarity, all off-support candidates including banned indices, and exact checkpoint residuals. `build.py:329–330` persists the diagnostics. The original-column SVD fallback at `nnls.py:160` is now explicitly authorized by A7.2.  
    **Fix:** none for this finding.

15. **MAJOR — WRONG: checkpointing and maximal-pilot validation remain only partially repaired.**  
    `phase(..., store=...)` now installs results before saving (`smoke.py:55–79`), and all maximal workload types exist. But:
    - Grid arrays/records are checkpointed only after the **whole grid** (`:285–293`).
    - N-under saves partial JSON after levels but saves its arrays only after the **whole ladder** (`:421–433`). A later exception or wall-time termination loses completed levels’ array evidence.
    - The frozen acceptance schema omits `pilot_support`; `smoke_accept.py:166–174` does not establish that the NNLS pilot reached the registered support workload or legitimately terminated earlier. Negative sentinel device peaks also pass these two predicates.

    **Fix:** checkpoint paired JSON/array evidence after each configuration/control level. Freeze `pilot_support=1024`, validate support and termination reason, and require positive memory measurements plus finite workload outputs.

16. **MAJOR — CORRECT: the frozen G-lmm case is fixed.**  
    `tests/test_core.py:360` explicitly passes `seed=0`. All registered G-full options, bounded order refinement and both N-dom degeneracy regimes remain present at `:276–287,380–425`.  
    **Fix:** none for the original fixture error. The explicit “pre-asymptotic at 0.01” annotation is still absent; add it to the order-check evidence when applicable.

17. **MAJOR — CORRECT: metadata pins check the loader’s actual metadata.**  
    `score.py:28–36` verifies the resolved metadata directory and exact STATUS hash; `cluster/stage.py:94–95` clears `JCPREF_META` and sets the paralab reference store.  
    **Fix:** none required.

18. **MINOR — NEEDS-RESTATEMENT: phase-II resource and concurrency obligations remain.**  
    `build.py:296–309` retains contribution blocks while forming scaled concatenations; `nnls.py:109,158–160` can create large indexed temporaries. `ours.py:40–42` still transfers the full 2D decoded field before lattice restriction. The submission lock remains lane-local (`cluster/submit.sh:17–23`), so account-cap checks are not atomic across independently submitting lanes.

    **Fix:** include simultaneous buffers, transfers, compilation shapes, retries and controls in the future resource certificate; use a shared campaign submission lock for strict account-wide enforcement. The unwritten phase-II drivers are not themselves defects.

19. **MAJOR — CORRECT: exhausted native-GN backtracking now returns reason 3.**  
    `rom.py:159–167` implements A7.1. Production validity rejects that reason at `:309`.  
    **Fix:** none for production semantics. Distinguish its causes before applying the new anchor exception—item 22.

20. **MAJOR — CORRECT: the original G-score NaN-hiding path is repaired.**  
    `smoke_accept.py:318–337` checks basis, coefficient, field and score finiteness. `score.py:84–87,104–109` rejects invalid field shapes/non-finite inputs and invalid normalizers.  
    **Fix:** none for this specific finding. The analogous timing reduction remains in item 8.

21. **MINOR — CORRECT: the adapter check is now explicitly distinguished from G-ours.**  
    A7.3 authorizes the phase-I execution-only check. `smoke.py:590–592` and `smoke_accept.py:291` label it accordingly.  
    **Fix:** none for phase I. Phase II still needs pinned exports, timing-case error replay, and imported Q1+Q2 qualification.

22. **BLOCKER — WRONG: A9’s round-off exception accepts non-finite GN steps as valid anchors. New defect.**  
    `rom.py:140–142,165–167` maps both a **non-finite least-squares step** and **exhausted finite backtracking** to reason 3. `:313–315` then treats every reason-3 step with sufficiently small gamma as a round-off stall. Offline acceptance repeats this at `smoke_accept.py:71–73`.

    Small CPU reproduction with a \(3\times2\) rank-deficient Jacobian:
    - computed GN step: **`[NaN, -Inf]`**;
    - exit reason: **3**;
    - gamma: approximately **\(5\times10^{-8}\)**;
    - tighter-anchor result: **`anchor_ok=True`, `roundoff_stalls=1`**.

    A9 permits exhausted backtracking at working precision, not a non-finite QR solve.

    **Fix:** persist a distinct failure cause or reason code. Apply A9 only to finite steps with exhausted backtracking; non-finite-step failures must invalidate anchors in both certification paths.

The inspected core formulas and history handling match the registration: `hrcore.py:37–44,315–342`, `rom.py:54–109,212–235,252–266`, and `build.py:237–248,275–278` consistently implement BE, CN, two-BE-start CN-R and one-BE-start BDF2. ECSW uses the all-node target and amended scaling floor. Q-DEIM retains the untruncated Householder \(R_B\), checked interpolation solve and correct stencil rows (`build.py:206–222`). Its pivot implementation is projection-based (`:178–203`); matching the registered small LAPACK fixtures does not establish universal finite-precision pivot identity.

I found no new large captured-array closure in the principal lane kernels. Maximal GPU feasibility remains unmeasured. Cluster staging includes GPU preflight exit 42, x64/highest precision, paralab paths, unique attempts, manifests, queue checks and staged-import checking (`cluster/stage.py:86–120`, `cluster/submit.sh:13–27`).

Phase II will additionally need complete cohort/eligibility enforcement, audited export replay, frozen selection, budget/resource accounting, and checksum-verified collection and cleanup.

**Overall verdict: NOT READY.** Top blockers are **non-finite GN failures accepted as anchors**, **incomplete offline evidence/schema enforcement**, and **negative controls whose acceptance still relies on unrecomputed or incomplete evidence**.