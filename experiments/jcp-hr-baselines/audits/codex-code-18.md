**NOT READY for phase-II production.** The targeted A22/A23 changes resolve several round-17 defects, but acceptance still lacks sufficient terminal-optimum evidence, and production qualification remains incomplete. This is a targeted round-18 audit, not a fresh certification of the entire lane.

All file references below are relative to `experiments/jcp-hr-baselines/`.

1. **MAJOR — WRONG: A23’s offline optimum check does not establish primal feasibility.**  
   `smoke_accept.py:108–112` checks gradient extrema and tolerance, but not terminal weights, their finiteness/nonnegativity, or terminal support. `missing_ok` then accepts an absent checkpoint using those diagnostics and the reported `final_support` (`287–292`). The saved path arrays contain checkpoint weights, not the terminal optimum (`smoke.py:252–264`).

   A read-only CPU probe confirmed that `opt_ok` returns `True` with `kkt.ok=False` and otherwise passing gradient diagnostics. In contrast, the actual solver’s KKT check includes finite, nonnegative weights (`nnls.py:276–290`). This does **not** demonstrate that smk2’s optimum was invalid; it demonstrates that offline acceptance cannot independently establish the required condition.

   **Fix:** persist terminal support, weights, residual evidence and sufficient KKT evidence; recompute feasibility and terminal support size offline. Bind optimum checkpoints to that terminal solution. Require explicit legacy provenance for the missing-checkpoint exception.

2. **MAJOR — WRONG: orthogonality events are recorded but have no enforced qualification consequence.**  
   `nnls.py:377–380,408–411` records drift events and `drift_ok`; `build.py:371–381` preserves them. However, `pipeline.py:175–201` admits the resulting checkpoints without a quality designation, and `smoke_accept.py:252–294` never examines drift evidence. A path with excessive drift can therefore pass the ECSW acceptance gate without the limitation required by A22.3 (`DESIGN.md:1362–1367`).

   Additionally, `NaN > DRIFT_TOL` is false, so nonfinite drift does not increment `drift_events`.

   **Fix:** classify nonfinite or excessive drift as an event; propagate an explicit `quality limited (orthogonality)` status and enforce its reporting restrictions. Offline acceptance must recompute that status from persisted drift values. The future production verdict driver must consume it.

3. **MAJOR — NEEDS-RESTATEMENT: A22’s resource estimate remains internally inconsistent.**  
   Active-bucket QR is implemented correctly in `nnls.py:227–240`, but `DESIGN.md:1351` gives approximately \(2.2\times10^{17}\) append flops. Its own formula \(8mKs\), using the stated dimensions, gives approximately **\(2.24\times10^{14}\)**. The stated memory traffic is consistent with the formula. Also, the refactor estimate must account for explicit Q formation (`nnls.py:183–186`), not only factorization.

   Refactor buffer replacement remains outside the donated kernels (`238–240`), so donation of append/delete/reset does not establish the claimed refactor peak.

   **Fix:** correct the arithmetic and certify complete refactor costs and transient memory. Enforce A22.2’s measured **120 GB** admission threshold before the maximal production path. This is an outstanding qualification requirement, not evidence of an OOM.

4. **MINOR — CORRECT: the principal round-17 implementation repairs are present.**  
   Append, reset and delete declare donation (`nnls.py:122,131,163`); rejected appends restore their slot (`214–218`). Refactorization and drift use active-width buckets (`227–240`). Capacity is clamped to the row count (`199,360`). Per-attempt diagnostics survive construction (`build.py:371–381,398–401`) and pipeline storage (`pipeline.py:185–188`).

   **Fix:** no further structural correction identified here. Retain executable aliasing and peak-memory checks; donation declarations alone are not a maximal-shape memory measurement.

5. **MINOR — CORRECT: G-nnls now encodes the requested production-interval stability gate.**  
   `tests/test_core.py:360–373` compares intervals 1024 and 256, requires identical checkpoint supports and relative residual agreement, at least two production-interval refactors, bounded drift, and actual deletions. The rectangular regression is present at `352–359`.

   **Fix:** retain the tests and obtain their completed, source-pinned result before admission. I did **not** run the test suite or treat the currently running test as passed.

6. **MINOR — CORRECT: A23’s construction-side optimum substitution is implemented.**  
   `build.py:352–360` requires terminal `reason == 'kkt'` and the solver’s complete KKT result before supplying missing targets from the terminal solution. It marks them `optimum`, preserves actual size/residual, and computes their residual vectors (`365–368`). Attempt summaries retain the flag (`379–381`).

   **Fix:** update the stale “unreached target is a construction failure” docstring (`344`) and complete the independent acceptance evidence described in finding 1.

All 17 `INHERITED.json` entries match their declared hashes; all 16 entries naming a source commit also match that source byte-for-byte. The remaining entry is the declared shim.

**Overall: NOT READY for phase-II production.** Top blockers are incomplete offline optimum certification, unenforced orthogonality limitations, and outstanding stability/resource qualification. A diagnostic smoke can gather evidence, but should not establish production readiness until these conditions are satisfied.

No files were changed, jobs submitted, GPU work performed, or tests run.