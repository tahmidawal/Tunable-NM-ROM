**NOT READY for the complete phase-I smoke workflow:** the new collection script can delete remote evidence after an unsuccessful transfer. I found no additional blocking numerical defect in the A14/A15 implementation changes.

Read-only review; no files changed, jobs submitted, smoke or test suite run. Small CPU snippets checked NNLS checkpoints and POD saturation. All **17 inherited hashes and 16 source-commit comparisons match**. Paths below are relative to `experiments/jcp-hr-baselines/`.

1. **MAJOR — WRONG: collection can delete evidence after a failed transfer.**

   `cluster/collect.sh:12` suppresses every `rsync` failure with `|| true`; `:13` checks the locally available output manifest; `:17` then permits remote deletion. The manifest generated at `cluster/stage.py:114` covers **output only**, not logs.

   A partial transfer that successfully copies outputs but fails to copy logs can therefore pass verification and delete the only surviving GPU/provenance logs. On repeated collection, an old local manifest introduces another failure path. Neither requires forged evidence.

   **Fix:** require successful transfer before allowing deletion; verify a freshly retrieved manifest covering outputs **and logs**; check that the job is terminal. A failed transfer may leave resumable local files, but must never authorize deletion.

2. **MINOR — NEEDS-RESTATEMENT: the POD memory improvement is real, but maximal-shape feasibility remains unproved.**

   `pipeline.py:84–98,129–143` correctly parks the original candidate on the host before constructing its replacement. `build.py:98` also reduces sign-normalization temporary size.

   However, `build.py:121–130` retains the initial randomized sketch `Y` throughout both power iterations, final basis construction and certification. That unnecessary live array is **17.68 GB in the 2D smoke** and **8.21 GB in the 3D smoke**, before `Qy`, the final basis, QR workspace and resident state operators. Host assembly at `build.py:303–328` removes simultaneous device contribution blocks, but filtering with `Gh[rowsel]` can still temporarily duplicate the host matrix.

   NNLS additionally materializes selected columns for exact residuals and original-column fallback (`nnls.py:123,170`). At the production support of 8,192, the 104,448-row selected matrix alone is approximately **6.85 GB**. The smoke’s support-1,024 pilot does not establish that production branch’s feasibility.

   **Fix:** release `Y` after its first QR and `Qy` after forming the final basis; include gather, factorization and host-filtering copies in phase-II resource accounting. Describe the changes as memory reductions pending GPU measurement, not a demonstrated fit.

3. **MAJOR — CORRECT: ECSW construction and the new support-target path preserve the required mathematics.**

   `build.py:234–255` retains the deduplicated, startup-inclusive training schedule. `:268–280` projects each state and its own history and selects the correct startup coefficients. The target remains the **all-node** \(J^\top r\), while candidate contributions are sampled and block-scaled at `:283–328`.

   `nnls.py:21–22,93,162` implements algebraically equivalent products without an explicit transpose of the full contribution matrix. `:172–174` counts removals and bans after the third drop. Original-column fallback and the all-candidate KKT check remain at `:170,182–184`.

   Small dense and dependent-column CPU examples agreed with SciPy’s optimum to numerical precision; saved checkpoint residuals agreed with direct evaluation.

   **Fix:** none identified in these mathematical changes. GPU lowering and maximal-shape memory still require the pilot.

4. **MINOR — NEEDS-RESTATEMENT: support targets are crossings, not exact support sizes.**

   `nnls.py:125–129` records the first state with **at least** the requested number of positive weights. Batch additions at `:154–156` can overshoot intermediate targets. A CPU example requesting support 8 recorded support 9. `smoke_accept.py:234–235` deliberately accepts this behavior.

   This is a reasonable interpretation of a *target*, but `DESIGN.md:1136–1137` should make it explicit. Moreover, clipping at the maximum requested target means a checkpoint obtained with maximum target \(4r\) need not equal the \(4r\) checkpoint on a path continuing to \(8r\).

   **Fix:** register “first crossing, actual size reported,” including the maximum-path convention. Production extensions should preserve existing configurations rather than silently rebuilding their checkpoints under a different cap. Add checkpoint/drop-count regression coverage to `tests/test_core.py:294`; its current NNLS gate covers optimum and tolerance checkpoints only.

5. **MINOR — NEEDS-RESTATEMENT: the floored-divisor explanation incorrectly promises rejection of arbitrary completion directions.**

   `build.py:88–90` and A14.2 say directions beyond numerical rank are rejected by the certificate. Under A6.1, a saturated prefix can legitimately pass.

   A read-only CPU example with an exactly rank-one snapshot matrix produced an orthonormal four-column basis, and `pipeline.basis_certified` accepted both ranks 1 and 4. That is consistent with the registered saturation rule.

   **Fix:** say that numerical rank is diagnostic and completed prefixes are admitted only through the A6.1 certificate. Do not claim that all beyond-rank columns are rejected.

6. **MAJOR — CORRECT: the A14/A15 smoke changes retain failure-closed acceptance.**

   The maximal Q-DEIM pilot explicitly permits an uncertified `pmax` and records that fact (`smoke.py:237–244`). Actual smoke and N-under advection ranks remain certificate-gated (`smoke_accept.py:221–228`).

   The revised ECSW ladder matches A15.2 (`smoke.py:466–481`; `smoke_accept.py:343–377`). A level fires only with reconstructed Q1 and sufficient rescored degradation; merely failing its solver is insufficient.

   N-swap now chooses a Q1-qualified configuration both in-job and from independently reconstructed Q1 (`smoke.py:576–584`; `smoke_accept.py:433–456`). The unavailable-ROM exception remains the expressly registered reference-fixture fallback.

   Required errored/skipped phases fail acceptance (`smoke_accept.py:207–210`), and failed `ALL` returns nonzero (`:482–486`).

   **Fix:** none identified. Whether the revised ladder actually fires remains an empirical requirement.

7. **MINOR — NEEDS-RESTATEMENT: smk1 supports the main narrative, with several factual corrections.**

   The saved traceback confirms that the 2D failure occurred in `pod_rsvd`; A14’s fallback attribution is supported. The archive also supports the 32/64-point ECSW constructions, the 20,989 banned candidates, failed KKT, and the reported N-under medians.

   Corrections to `DESIGN.md:1126–1151`:

   - The recorded 3D state-POD orthonormality deviation is **1.299e-14**, not bounded by the quoted 7e-15.
   - “Every step non-stationary” applies to the displayed native ECSW runs, not every common-solver run; the log includes 49/50, 48/50 and 46/50.
   - The 3D traceback identifies the failing expression as **`G.T @ b`**, before `_GtG`, rather than `G.T @ G[:, S]` (`runs/smk1/archive/output/smoke.json:18832`). The new code fixes both.
   - Individual FD checks that executed passed, but the complete G-fd inventories did not pass in either cell. `ACCEPTANCE.json:840–858` explicitly lists both failures.

   **Fix:** distinguish passing executed subchecks from complete gates, correct the bounds and failing expression, and retain `ALL=false`. These corrections do not invalidate the central motivation for A14/A15.

8. **MAJOR — CORRECT: residuals, startup history, solvers and Q1 remain consistent with the amended design.**

   `hrcore.py:37–38,315–342`, `rom.py:54–110,240–263`, and `build.py:278–280` consistently implement BE, CN, two-BE-startup CNR and one-BE-startup BDF2.

   `rom.py:280–295` computes fresh Jacobians using each step’s history. `:324–351` counts non-finite gamma as non-stationary without a residual exception and implements the later anchor-stall allowance. `pipeline.py:277–296` requires complete cohorts, physical `n0`, every-case distances and the prescribed tighter subset.

   **Fix:** none identified. Describe anchor validity using A9/A13, rather than the superseded strict A2.7 wording alone.

9. **MAJOR — CORRECT: Q-DEIM, scoring, timing and statistical estimands retain their registered behavior.**

   - `build.py:197–225`: pivot equivalence is correctly limited to the registered fixtures; Householder \(R_B\) is untruncated; interpolation solves have backward-error checks.
   - `score.py:28–36,48–90`: metadata pins, IC/normalizer checks, time-max scoring and S-kind restrictions are present.
   - `smoke.py:120–142`: synchronized A–B–A calls, burns and retained bounded retries; offline timing checks reject non-finite discrepancies.
   - `stats.py:17–22,34–110,164–186`: clipped envelopes, median of case medians, case bootstrap, strict equivalence boundaries, precedence and X-specific picks.
   - `ours.py:31–37,60–62`: inherited query interfaces are used consistently; phase-I adapter execution remains distinct from export replay.

   **Fix:** none identified for phase I. Phase-II callers must enforce eligible-case intersections and qualification before invoking the statistical helpers.

10. **MINOR — CORRECT, noted: forgery-only acceptance hardening remains possible.**

    `smoke_accept.py:240–243` partly trusts the pilot’s KKT `ok` flag; support checkpoints at `:234–235` use saved size/residual summaries rather than persisted weights. Honest construction computes these values and prevents a failed KKT check from retaining an optimum claim.

    **Fix:** optionally persist checkpoint supports/weights and compare both saved KKT gradient bounds directly. Under A12.4 these are **noted, nonblocking forgery-only concerns**, unlike finding 1.

11. **MINOR — CORRECT, with the existing account-cap race: cluster execution safeguards remain present.**

    `cluster/stage.py:77–115` supplies the GPU partition, exit-42 preflight, x64/highest precision, paralab paths, isolated caches, manifest verification and bounded driver execution. `cluster/submit.sh:13–27` checks queues, lane/account caps and unique attempt directories.

    The submission lock at `submit.sh:17` remains lane-local, so independent lanes can race the account-wide cap.

    **Fix:** use a shared campaign submission lock for atomic account-cap enforcement.

Phase II still needs production orchestration, generated workload/live-memory certificates, kappa-aware extension and coverage rules, frozen selection, eligibility enforcement, pinned Q1+Q2 imports and export replay, budget accounting, and verified collection. The regenerated CPU gate artifact must match the staged sources; I did not run or accept its regeneration.

**Overall: NOT READY for the complete phase-I smoke workflow until finding 1 is fixed. Top blocker: remote evidence deletion despite collection failure. No further blocking numerical regression was identified; maximal-shape feasibility and successful control firing remain unproved.**