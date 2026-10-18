**NOT READY for smk2:** the A17 NNLS rewrite passes the targeted numerical checks, but its smoke evidence and offline acceptance still have gaps that should be fixed before spending another smoke run.

This is a targeted Round 15 audit, not a fresh clearance of every unchanged phase-I component. All checks were read-only and CPU-only; no files were written.

1. **MAJOR — WRONG: offline acceptance can accept an explicit KKT violation.**  
   Evidence: `smoke_accept.py:280–284` accepts a pilot ending with `reason == 'kkt'` using `kkt['ok']` and the support-gradient bound. It never checks `off_support_max_gradient`. I reproduced acceptance with support gradient `0`, tolerance `1e-9`, and off-support gradient `1`: the predicate returns **True**.  
   The solver’s own check at `nnls.py:84–98` correctly checks both sides; the defect is the independent acceptance check.  
   **Fix:** independently require finite, nonnegative tolerance, finite gradients, both gradient inequalities, and finite nonnegative weights. Save the pilot’s final support, weights, and gradient evidence rather than relying on `ok`.

2. **MAJOR — NEEDS-RESTATEMENT: checkpoint selection and banning are not independently reconstructed from raw numerical evidence.**  
   Evidence: `smoke.py:252–260` saves checkpoint nodes and weights; `build.py:356–361` retains scalar residuals and ban counts. `smoke_accept.py:241–268` checks weights but trusts those residuals and counts when reconstructing retry and selection. The contribution matrix is discarded at `pipeline.py:182`.  
   Thus, the selection formula is recomputed, but erroneous residual summaries or ban counts can still change the selected path without detection. A16.5’s explicit bookkeeping is implemented; the stronger “recomputed from raw outputs” claim is not established.  
   **Fix before smk2:** preserve compact checkpoint residual vectors, target norms, drop-count arrays, and final KKT evidence; derive the acceptance quantities from them. If independent reconstruction of $G\xi-b$ is required, provide deterministic replay or streamed contribution reconstruction—do not copy the full matrices.

3. **MINOR — WRONG: padding breaks otherwise valid small least-squares problems when the padded width exceeds the row count.**  
   Evidence: `nnls.py:33–38,41–50`. For `G = eye(3)`, `b = [3,2,1]`, and support target `3`, padding produces four columns. Reduced QR returns a `3×4` R, and adding the `4×4` diagonal raises a shape error.  
   This does **not** affect the declared smoke shapes, where rows comfortably exceed padded support.  
   **Fix:** bound padding by the row count and explicitly handle unsupported or rank-deficient passive sets. Add the three-column regression.

4. **MINOR — WRONG: path diagnostics pair the new support size with the previous residual.**  
   Evidence: `nnls.py:140–142,162–169`. On a padded identity fixture, the recorded path is:
   ```
   (1, 1.0), (2, 0.597614...), (3, 0.267261...)
   ```
   Actual residuals at those supports are `0.597614...`, `0.267261...`, and `0`. Checkpoint records themselves are correct.  
   **Fix:** record support and residual from the same iterate; label pre-update logging explicitly if retained.

5. **MINOR — CORRECT: the padded QR solve preserves the first k unknowns at the smoke shapes.**  
   Evidence: `nnls.py:28–51`. With full-rank active columns and padded width no greater than the row count, QR of `[A, 0]` has zero upper-right block. Replacing only padded diagonal entries by one cannot feed padded unknowns into the active solve.  
   CPU comparisons against unpadded `numpy.linalg.lstsq` for seven support sizes spanning padding boundaries gave relative differences below **1.8e-15**.  
   **Fix:** none for the declared shapes; retain explicit padding-boundary coverage.

6. **MINOR — CORRECT: A17 entry, refitting, checkpoint residuals, KKT, and retry configuration are implemented consistently.**  
   Evidence: `nnls.py:54–98,129–175`; `build.py:335–381`; `smoke_accept.py:242–246`. Entry defaults to one index; gradients come from the current residual; the inner loop removes blocking coefficients and refits original columns; final recording covers iteration-bound termination. KKT includes banned indices. Both construction attempts use **40,000** iterations, with the registered retry trigger and tie-breaking order.  
   The existing G-nnls test passed. Another **60** random, duplicated-column, and nearly duplicated-column cases matched SciPy objectives within **5.2e-16** relative difference and passed KKT. None triggered banning, so forced-ban coverage remains unverified.  
   **Fix:** add a deterministic drop-count/retry regression; no numerical correction identified on these cases.

7. **MINOR — NEEDS-RESTATEMENT: memory is plausible for H200, but completion within the smoke deadline remains unproven.**  
   Evidence: `nnls.py:33–38,140–146`; `smoke.py:250,270`; `cluster/stage.py:108`. Power-of-two padding and explicit Q formation increase work beyond an unpadded QR estimate.

   | Shape | G storage | Largest A or Q | Approximate QR + Q work* |
   |---|---:|---:|---:|
   | 2D, m=15,360, support 2,048 | 4.03 GB | 0.252 GB | 2.9×10¹⁴ flops |
   | 3D, m=26,112, support 4,096 | 6.85 GB | 0.856 GB | 3.9×10¹⁵ flops |

   *One refit per support increment; excludes drops, retries, compilation, and gradient scans.* The maximal-rank pilots’ G matrices are approximately **16.1/27.4 GB**. These are component sizes, not peak-memory guarantees. Large arrays are JIT arguments; no new captured-G constant was found.  
   **Fix:** describe runtime as unmeasured. Phase II needs the separately audited updating QR already required by A17, plus measured resource accounting.

8. **MINOR — CORRECT: inherited identity claims checked out.**  
   All **17** listed files match their recorded SHA-256 values. All **16** entries naming source commits also match those Git blobs; the remaining entry is the declared shim.  
   **Fix:** none. The regenerating `checks/test_core.json` was not treated as a defect.

**Overall: NOT READY for smk2.** The top blockers are the incomplete offline KKT predicate and insufficient saved NNLS evidence for independent acceptance. The padded QR mathematics itself is not a blocker.