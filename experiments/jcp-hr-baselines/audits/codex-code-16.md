**READY for smk2 on this targeted Round 16 review.** The round-15 blocking acceptance defects are repaired. This is clearance to run the smoke, not evidence that its controls pass or that production is ready.

Reviewed commit `8b2beabed`. All checks were read-only and CPU-only; no files changed. Paths below are relative to `experiments/jcp-hr-baselines/`.

1. **MINOR — CORRECT: round-15 finding 1, incomplete KKT acceptance, is fixed.**  
   `smoke_accept.py:296–301` now requires finite gradient summaries and tolerance, nonnegative tolerance, and **both** support and off-support inequalities. `smoke_accept.py:288–293` independently checks saved weights and residual magnitude. The former off-support-gradient counterexample now fails.  
   **Fix:** none required for that defect. These remain checks of saved gradient summaries, not independent reconstruction of the gradient.

2. **MINOR — CORRECT: round-15 finding 2 is resolved for support-checkpoint selection under A18’s evidence contract.**  
   `nnls.py:194–203` produces checkpoint residual vectors, target norm and drop counters. `build.py:356–368` retains these for support checkpoints; `smoke.py:252–265` persists both attempts. `smoke_accept.py:248–278` recomputes ban counts, validates residual norms, counts positive weights, and reconstructs whole-path selection. A mismatched count or residual fails acceptance.  
   **Fix:** none for support-path selection. Describe this accurately as reconstruction from saved residual vectors and counters; it does not independently reconstruct \(G\xi-b\) or replay the drop history.

3. **MINOR — WRONG: “each checkpoint” persistence overstates the implementation.**  
   A18.1 (`DESIGN.md:1227–1229`) says each checkpoint’s residual vector is saved. Although `nnls.py:195–198` supplies vectors for tolerance checkpoints too, `build.py:364–365` reduces those checkpoints to scalar size/residual summaries. Only support checkpoints reach `smoke.py:257–263`.  
   **Fix:** persist tolerance-checkpoint supports, weights and residual vectors too, or explicitly limit A18.1’s persistence claim to support checkpoints. This does not affect the registered support-based selection.

4. **MINOR — CORRECT: round-15 finding 3 and the rank-deficiency policy are repaired.**  
   `nnls.py:45–66` bounds QR padding by the row count, routes wide passive sets directly to SVD, and implements A18.2’s diagonal-ratio fallback with `rcond=1e-13`.  
   CPU checks: the three-column identity regression succeeds; dependent and wide passive-set solutions agree with NumPy within **9.1e-16** in coefficient norm.  
   **Fix:** add these cases to `tests/test_core.py`; its existing NNLS tests (`:293–339`) do not explicitly cover these new branches.

5. **MINOR — CORRECT: round-15 finding 4’s returned path pairing is fixed.**  
   `nnls.py:159–160,186,189–192` pairs each attained support with its own residual, including iteration-limit termination. The identity fixture now records:
   ```
   (1, 0.597614...), (2, 0.267261...), (3, 0)
   ```
   **Fix:** the progress log at `nnls.py:187–188` still combines the new support with the preceding residual; label that residual “pre-update” or print after refresh. Returned path data are correct.

6. **MINOR — NEEDS-RESTATEMENT: the archived dry-run evidence remains incomplete.**  
   A18.3 corrects the iteration arithmetic and distinguishes the endpoints (`DESIGN.md:1233–1240`). However, `runs/nnls_fixture/README.md:3–6` names `G.sha256`, `t4.log` and `t7.log`, which were absent during this audit. Moreover, `t_old.py:2–8` imports the current implementation and defaults, so running it now does not reproduce the archived pre-A17 algorithm.  
   **Fix:** supply the missing records and pin the old script to its historical implementation and parameters. Until then, qualify the exact repaired-path/reference comparison as incompletely archived. This is not a smoke execution blocker.

7. **MINOR — CORRECT: targeted numerical and inherited-identity checks pass.**  
   The existing G-nnls gate passed. Thirty additional random/duplicated/nearly duplicated-column cases passed KKT and matched SciPy residual norms within **2.1e-15**. All **17** `INHERITED.json` entries match their hashes; all **16** source-backed entries match their recorded Git blobs.  
   **Fix:** retain a deterministic forced-ban/retry regression; these numerical checks do not establish that branch’s execution.

8. **MINOR — NEEDS-RESTATEMENT: smoke readiness does not establish production feasibility.**  
   A18.5 correctly leaves updating QR, checkpoint comparisons and revised resource accounting outstanding (`DESIGN.md:1242–1246`). Repeated full QR remains in `nnls.py:33–39`. Missing or failed required smoke phases fail acceptance (`smoke_accept.py:207–210,534–540`); the cluster deadline is enforced at `cluster/stage.py:106–113`.  
   **Fix:** phase II must freeze and audit the final construction algorithm, measure both attempts through \(16r\), and implement the production orchestration/resource contracts. Do not extrapolate smoke-rank runtime to production.

**Overall: READY for smk2. No remaining smoke-blocking defect was identified in the targeted A18 changes.** The principal outstanding production blockers are construction feasibility/resource certification and the unwritten phase-II drivers. The regenerating `checks/test_core.json` was not treated as a defect.