**Overall: NOT READY.**

Scoped files are identical between requested commit `104130435` and observed HEAD `22104f4dd`. CPU-only probes; no files changed, no scratch created, and `tests/test_core.py` was not run. Paths below are relative to `experiments/jcp-hr-baselines/`.

1. **WRONG — blocking: deadline expiration can still produce an optimum alias.**  
   The outer deadline check precedes residual/gradient computation (`nnls.py:394–405`). If time expires during that work, the solver can exit as `kkt` without checking again (`nnls.py:410–413`). `build.py:354–360` then fills unreached targets with optimum aliases, contrary to A39.1.  
   **Reproduction:** controlled clock, `G=I₃`, `b=[1,0,0]`, target 2, budget 1 second; expiration during the second iteration’s gradient returned `reason='kkt'`, `kkt.ok=True`, elapsed 2 seconds, and an unreached target eligible for aliasing.  
   **Fix:** check the deadline after synchronized residual/gradient work and before accepting completion reasons or starting an append. Propagate inner-loop timeout explicitly, including on the final permitted iteration. Add deterministic deadline-crossing regression cases.

2. **NEEDS-RESTATEMENT — nonblocking: A39.1’s overshoot inventory remains incomplete.**  
   `DESIGN.md:1816–1818` omits outer residual/gradient work performed after the last check (`nnls.py:397–405`). Initial gradient computation also precedes the timer (`nnls.py:369,391`), so reported elapsed time excludes that initialization. The `method='full'` inner loop remains unbudgeted (`nnls.py:446`).  
   **Fix:** include residual/gradient and initialization costs in the budget contract, or add checks around them; explicitly limit the inner-loop guarantee to updating QR.

3. **CORRECT — nonblocking: inner stopping state, refactor guard, and evidence persistence.**  
   The inner loop checks before each solve and finishes deletion batches consistently (`nnls.py:307–325`); refactorisation has a deadline guard (`nnls.py:436`). A direct stopped-inner-loop probe preserved feasible weights and matching factors, with reconstruction error `2.22e-16`. Budget/elapsed fields reach attempt summaries (`nnls.py:472`, `build.py:381`). Explicit `time` exits cannot generate optimum aliases; missing checkpoints become construction failures (`build.py:354–362`, `pipeline.py:208–210`).  
   **Fix:** retain this behavior while closing item 1.

4. **CORRECT — nonblocking: sign-normalised refactorisation and subsequent deletion.**  
   `nnls.py:230–235` transforms Q and R consistently and recomputes Qᵀb. Independent refactor-then-delete probes at capacities 13, 96 and 128 matched Givens factor-for-factor within `5.33e-15`; reconstruction error was `5.33e-15`, Qᵀb error `3.55e-15`, and least-squares error `1.08e-15`.  
   **Fix:** none required.

5. **CORRECT — nonblocking: G-qrdel covers A39.3’s stated boundary cases.**  
   `tests/test_core.py:404–420` includes non-power-of-two capacity, full capacity, first/last deletion, large spans and an exactly zero column. Lines `423–437` exercise refactor-then-delete against independent least squares. Boundary fixtures do **not** compare raw factors against Givens; that comparison remains in the random fixtures (`394–399`).  
   **Fix:** optionally retain the independent refactor-then-delete factor comparison as a regression assertion.

6. **CORRECT — nonblocking: no updating-QR regression found in the exercised paths.**  
   Append/reset, drift and capacity handling remain intact (`nnls.py:248–289`). Probes verified exact restoration after rejected dependent appends and successful subsequent appends. An independent update/full comparison produced identical checkpoint supports, four deletions, residual difference `1.11e-16`, and maximum recorded drift `8.88e-16`. A39.4 appropriately acknowledges deletion temporaries (`DESIGN.md:1829–1831`).  
   **Fix:** none identified; the pending production-interval gate remains required by A38.1.

No additional blocking defect was identified in the scoped change.