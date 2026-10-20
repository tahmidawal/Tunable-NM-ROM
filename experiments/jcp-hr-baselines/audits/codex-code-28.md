**Overall: NOT READY** — the deadline does not yet justify A38.2’s protection against construction starving Stage A.

Audited scoped files match `4b3edc830`; current HEAD is `cf0a8d1b1`, with only heartbeat changes between them. CPU probes only; no files written, no scratch created, and `tests/test_core.py` was not run. Paths below are relative to `experiments/jcp-hr-baselines/`.

1. **WRONG — blocking: the budget is checked only between outer iterations.**  
   `nnls.py:386` checks elapsed time, but `_lh_refit_upd` can execute many solves and deletions without another check (`nnls.py:302–319`). Refactorisation and terminal evidence generation also run outside deadline checks (`nnls.py:423`, `446–456`). One iteration can therefore overrun substantially; A38.2’s construction-bound claim (`DESIGN.md:1802`) is unsupported.  
   **Fix:** propagate a deadline into the inner loop, stop at a consistent feasible state between expensive operations, and reserve time for evidence finalisation. Document the remaining single-operation overshoot. Persist each attempt’s budget and elapsed time.

2. **CORRECT — nonblocking: window placement and deletion mathematics under the stated non-negative-diagonal invariant.**  
   `nnls.py:177–194` contains the affected Hessenberg block even when `j0 = K-w`; rows in the window have zero entries in columns outside it that would require additional transformation. Clearing slot `k-1` removes the redundant direction. Exhaustive probes for every `j < k ≤ 8`, including individual zero diagonals, differed from Givens by at most `2.11e-15`. Explicit `K=128` probes covered full capacity, right clamping, first/last deletion, and spans exceeding `K/2`; maximum difference was `5.33e-15`.  
   **Fix:** none to the window placement; retain these deterministic cases as regression coverage.

3. **WRONG — nonblocking for numerical validity: production does not maintain the claimed diagonal invariant.**  
   `_qr_active` returns unnormalised Householder factors (`nnls.py:230–233`). Consequently, deletion’s sign normalisation (`nnls.py:183–185`) can flip already-triangular columns, contradicting `nnls.py:171`. A refactor/delete probe started with 52 negative active diagonals and retained 31 afterward. Direct factor comparisons with Givens can differ substantially, although reconstruction remained accurate to `6.22e-15`. Inactive identity entries and zero-diagonal handling passed the probes.  
   **Fix:** normalise active QR factors during refactorisation, transforming `Q`, `R`, and `Qᵀb` consistently; alternatively narrow the factor-equality claim to sign-equivalent factors.

4. **CORRECT — nonblocking: donation is semantically safe; compilation shapes are bounded.**  
   Dynamic slices are consumed before their functional updates (`nnls.py:180–190`); `UpdQR.delete` replaces all donated references (`nnls.py:271`). A compiled CPU probe aliased all three array buffers. This establishes correctness, not GPU temporary-allocation behaviour. For production `K=16384`, `nnls.py:204–206` permits nine window widths, `64…16384`; `j` and `k` remain dynamic.  
   **Fix:** qualify the documentation: row-count-clamped, non-power-of-two capacities can produce a final width `K`, and capacities below 64 use smaller windows.

5. **NEEDS-RESTATEMENT — nonblocking: donation does not eliminate window-product temporaries.**  
   At the requested maximum shape, `G` occupies **27.380 GB**, `Q` **13.690 GB**, and each square window array **2.147 GB**. `Qw @ Qs` (`nnls.py:190`) requires a separate product while its input remains live; sliced inputs, shifted `R`, QR factors, and workspace can add further storage. These sizes do not establish a 120 GB breach, but neither do they certify admission.  
   **Fix:** update A22’s memory discussion for blocked deletion. Preserve the actual process-peak check at `tune.py:318–325`; A28.2 correctly makes that acceptance retrospective.

6. **CORRECT — nonblocking: completed timeout handling preserves evidence and does not manufacture an optimum.**  
   Final residuals and checkpoints are refreshed after exit (`nnls.py:446–463`). A controlled timeout probe returned `reason='time'` with `kkt.ok=True`, retained its reached checkpoint, and preserved the terminal residual. Optimum aliases require specifically `reason=='kkt'` (`build.py:354`). Retry triggering and whole-path selection remain unchanged (`build.py:390–401`). Missing requested configurations become construction failures (`pipeline.py:210–213`, `tune.py:354–359`) and restrict coverage (`analyze.py:163–166`). Terminal arrays are saved at `tune.py:330–338`. Quality labels still describe banning/orthogonality, independently of coverage.  
   **Fix:** add deterministic timeout regression cases; include `'time'` in the solver’s reason documentation.

7. **NEEDS-RESTATEMENT — nonblocking: G-qrdel’s coverage is narrower than the requested guarantee.**  
   `tests/test_core.py:383–390` uses only `K=128`, excludes `k=K`, and explicitly normalises every input. It therefore misses the production refactorisation sign issue, full-capacity deletion, and zero-diagonal fixtures.  
   **Fix:** add deterministic boundary cases, non-power-of-two capacity, zero diagonals, and refactor-then-delete coverage. Compare reconstruction and least-squares results as well as raw factors.