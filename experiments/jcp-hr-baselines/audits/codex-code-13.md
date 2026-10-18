Targeted review of `c57098e00` against A16.5 and round-12 findings. CPU-only, in-memory probes; no files changed or jobs submitted. Paths below are relative to `experiments/jcp-hr-baselines/`.

1. **MAJOR — CORRECT: both NNLS attempts are now preserved.**  
   `build.py:349–381` retains separate initial/retry diagnostics and checkpoint arrays, with an explicit selection. `pipeline.py:183–188` preserves those arrays; `smoke.py:252–261` writes both attempts’ nodes and weights plus their summaries. Mocked probes verified preservation when either attempt wins.  
   **Fix:** none required for round-12 item 1.

2. **MINOR — CORRECT: construction follows the frozen retry and selection rule.**  
   `build.py:335–336,350,365–376` implements the strict >1% trigger regardless of attainment, equal 20,000-iteration bounds, target-count priority, common-target log-residual comparison, and initial-path tie breaking. Boundary and selection probes passed.  
   **Fix:** retain these probes as regression tests.

3. **MAJOR — WRONG: acceptance still allows a different retry contract.**  
   `smoke_accept.py:239–242` uses the output’s `ban_retry_frac` instead of independently enforcing 0.01. Lines `243–264` never validate `max_iter` or attempt iteration counts. Executing the actual acceptance block accepted:
   - 20% banned candidates, threshold changed to 0.5, and no retry.
   - A reported 40,000-iteration bound and 30,000 completed iterations.

   These violate `DESIGN.md:1199–1204`.  
   **Fix:** enforce the frozen threshold and bound independently; validate candidate counts, banned counts, required attempt inventory, and each attempt’s iteration count. Add rejection tests for these mutations.

4. **MAJOR — WRONG: the replacement gate loses residual finiteness validation and incompletely validates weights.**  
   `smoke_accept.py:248–250` counts positive weights but permits additional negative weights. Lines `259–264` neither require finite checkpoint residuals nor reconcile selected residuals with `ecsw_rs`. The actual block accepted all checkpoint residuals set to NaN; it also accepted a checkpoint containing weights `[1, 1, -1]` with reported support two.  
   **Fix:** require one-dimensional, equal-length node/weight arrays, finite strictly positive saved weights, valid unique integer nodes, and finite nonnegative residuals for every reached checkpoint. Compare selected checkpoint sizes **and residuals** with `ecsw_rs`. Validate both attempts before comparing them.

5. **MINOR — CORRECT: inherited identity claims hold.**  
   All 17 entries in `INHERITED.json` match their hashes; all 16 entries naming committed sources match those sources.  
   **Fix:** none.

**Overall: NOT READY for smoke `smk2`.** Round-12 item 1 is resolved; item 2 remains incomplete. The top blockers are acceptance’s reliance on a mutable retry threshold, missing iteration-bound checks, and acceptance of non-finite residuals or invalid weights.

Phase II must enforce extension-only use of κ=16, incomplete-coverage and banning-quality labels, and construction/validation accounting. This targeted review does not recertify unchanged phase-I internals.