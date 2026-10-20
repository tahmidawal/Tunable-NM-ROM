**Do not launch J1 unchanged.** The final-panel closure bug can crash after the per-setting work finishes. Timing failures also do not prevent timing-based deployment decisions.

Read-only audit: no files modified, no jobs submitted, no GPU runs. I checked the source, ran isolated NumPy/Python probes, verified both configs against their generator, and checked all 76 reference arrays. Other workspace edits appeared during the audit; the findings below concern the inspected 2D implementation.

1. **CORRECT — metric normalization and shared-node strides.**

   [w2d.py:222](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-wide-bank/experiments/jcp-wide-bank/w2d.py:222), lines 328–332 and 402–416.

   With `L=1024`, `::4` maps the padded `1025×1025` fields to exactly `257×257`. Refined errors use the restricted initial-field norm; distances from `gref` use the full-mesh initial-field norm. Evolved maxima exclude index zero.

   The reference generator uses the same initial condition and aligned nodes. Across all 76 ST/S arrays, the largest relative initial-field discrepancy from local regeneration was `3.93e-17`; all boundaries were zero. Thus using the regenerated initial norm instead of `norm(ref[0])` is equivalent here to roundoff.

   **Fix:** none required. An explicit initial-field agreement assertion would protect this assumption against future reference changes.

2. **CORRECT mathematically — M-prefix slicing; bitwise equivalence is not established for every operation.**

   [w2d.py:308](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-wide-bank/experiments/jcp-wide-bank/w2d.py:308), lines 343–369; vendored `hops.py:35–47`, `qcore.py:185–227`.

   I checked every registered M at `L=1024`: `(kx, ky, lam)` is **bitwise identical** to the corresponding prefix of the M=2048 table.

   The construction is separable in test columns:

   - `Gq` and `Gs` do not depend on M.
   - `Psi[:,j] = L*w*psi_j` depends only on that test.
   - Each row of A projects the bank against one test.
   - Neither construction performs an M-dependent normalization or fit.

   Consequently, slicing produces the same mathematical operator as building directly at `(R′,M)`.

   **Qualification:** differently shaped GEMMs may change floating-point reduction choices. Also, adding the 256 boundary to the nested bank changes decode summation grouping relative to `qstudy.py`. Do not claim bitwise-identical complete rollouts without testing that stronger claim.

   **Fix:** none to the slicing formula; use a numerical tolerance for direct-build parity.

3. **CORRECT — eligibility, m*, and named diagnostics.**

   [w2d.py:133](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-wide-bank/experiments/jcp-wide-bank/w2d.py:133), lines 490–527; [make_configs.py:22](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-wide-bank/experiments/jcp-wide-bank/make_configs.py:22).

   Eligibility requires finite output fields, no reason-3 exits, and budget exits ≤1% of the steps. With 50 steps, **one budget exit correctly makes the rollout ineligible**. Reason 3 also covers nonfinite residual exits in the vendored LM.

   Candidates require every case eligible; selection scans non-control family members in increasing m. Primary and secondary tolerances are correctly `2.5e-4` and `1e-3`. Both convergence gates disable primary selection; `m_d` and `m_rho` remain separately named diagnostics.

   Both supplied JSON configs matched the 2D generator.

   **Fix:** none to this selection logic.

4. **NEEDS-FIX — control discrimination differs from A0-2’s literal two-metric rule.**

   [w2d.py:503](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-wide-bank/experiments/jcp-wide-bank/w2d.py:503).

   The code correctly reports distance and rho outcomes separately. However, `would_be_selected` additionally requires `all_eligible`.

   Counterexample: a control passes both metric thresholds but has one damping-limit exit. The code declares the setting discriminating. A0-2 explicitly says a control passing **both metric criteria** invalidates m*.

   There is a wording tension with A0-3’s general candidate eligibility; the implementation silently resolves it in favor of eligibility.

   **Fix:** record `passes_both_metrics` separately and use it for A0-2 discrimination. Keep eligibility as its own diagnostic. If eligibility is intentionally part of control discrimination, amend the governing specification explicitly.

5. **NEEDS-FIX — K-conv/K-target are correctly scoped, but reference-file integrity is incomplete.**

   [w2d.py:271](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-wide-bank/experiments/jcp-wide-bank/w2d.py:271), lines 465–497.

   **Correct:** `gref_check` runs on every selected case; both rollout eligibilities enter K-conv; the bar is `2.5e-5`. K-target checks refinement and flux agreement on all `gref` reached states, with the correct `1e-5` bar. Mesh, timestep, tolerances, acceptance, case uniqueness, cohort hashes, shape, finiteness, and field-content hashes are checked.

   **Gap:** line 294 hashes the extracted `f257` array, not the NPZ file. This matches the historical reference driver but does not satisfy K-ref’s literal “SHA256 of every file” contract. The external `refs2d` directory is also outside the job’s staged manifest.

   **Fix:** stage and verify the reference bundle’s file checksums and pin its manifest; retain the array hashes as semantic checks.

   All 76 local reference arrays passed their stored array hashes. Local cohort descriptor hashes differed from the cluster hashes, consistent with the documented cross-platform issue; I did not treat that as evidence of a cluster failure.

6. **WRONG — final timing calls capture the wrong solver and cold start. Timing acceptance is also incomplete.**

   [w2d.py:371](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-wide-bank/experiments/jcp-wide-bank/w2d.py:371), lines 553–560 and 594–616.

   The retained callable binds only `data`:

   ```python
   lambda u, nu, data=data: fq(u, nu, data, cold)
   ```

   `fq` and `cold` are shared loop cells. At final-panel execution, both refer to the last setting/rank. A pure-Python reproduction produced:

   ```text
   intended R512: solver_R128, data_R512, cold_R128
   intended R128: solver_R128, data_R128, cold_R128
   ```

   If an earlier rank has a deployed arm, its final warmup encounters incompatible coefficient dimensions. This happens **after the expensive main sweep**. The historical driver binds `fq` and isolates `cold` inside `run_setting()`.

   **Fix:**

   ```python
   lambda u, nu, data=data, fq=fq, cold=cold: fq(u, nu, data, cold)
   ```

   Additional timing findings at [w2d.py:160](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-wide-bank/experiments/jcp-wide-bank/w2d.py:160):

   - **Correct:** A1→B→A2, seeded randomization, whole-output synchronization, per-subject drift, phase-1 hashes, and raw invocation records. Family choice is saved before the final panel.
   - **Wrong acceptance:** drift/determinism failures are recorded but ignored when choosing `deployed`; there is no job-wide timing-withdrawal state. **Fix:** propagate K-time failure into timing validity and prevent those medians from supporting deployment/cost claims.
   - **Incomplete pairing:** the raw records support A2-8, but the required `(case, phase)` paired ratios are never calculated. **Fix:** compute those ratios and their median, alongside the per-setting median ratio; require both H3 bars.
   - **Burn defect:** `clear_caches()` invalidates `_BURN`; subsequent burn duration includes compilation. **Fix:** compile and synchronize `_BURN` before starting its duration clock.

7. **NEEDS-FIX — projection-floor mathematics is correct; the consistency check is not enforced.**

   [w2d.py:570](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-wide-bank/experiments/jcp-wide-bank/w2d.py:570).

   The direct shared-node bank, thin SVD, `1e-12*sigma_max` rank cutoff, orthogonal projection, ST/S normalization, and `lstsq(rcond=1e-12)` match A2-6/A3-6. Omitting the boundary rows is correct for these zero-wall references.

   But `resid_check` is only recorded. A discrepancy above the required `1e-10` still produces apparently usable floors and `complete=True`.

   **Fix:** record an explicit floor-check verdict and reject/mark unavailable floors failing the relative—or prescribed near-zero absolute—comparison. Guard rank zero explicitly.

8. **CORRECT — reached-state residual Jacobian.**

   [w2d.py:430](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-wide-bank/experiments/jcp-wide-bank/w2d.py:430); vendored `qcore.py:303–310`.

   The driver computes

   \[
   J=S\left(A+\Delta t\left(\partial N/\partial c+\nu\Lambda A\right)\right).
   \]

   Since \(S(I+\Delta t\,\nu\Lambda)=I\), this equals the vendor’s

   \[
   J=A+\Delta t\,S\,\partial N/\partial c.
   \]

   The first case’s viscosity is correct, and `pop[0][k-1]` indexes reached states 1, 25, and 50 correctly.

   **Fix:** none to the formula. For full compliance with “records the singular values,” persist the spectrum: currently only extrema, rank, and condition are stored.

9. **NEEDS-FIX — retained references defeat the claimed block lifetime; memory sampling races.**

   [w2d.py:432](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-wide-bank/experiments/jcp-wide-bank/w2d.py:432), lines 560–590.

   `del arms` and `del blocks` do not release all their arrays:

   - `gd` retains the previous `gref` data.
   - Loop variables such as `arm`, `d`, and `data` retain rule data.
   - `final` intentionally retains every deployed arm’s data.
   - Smaller-M JAX slices must be budgeted as allocations, not assumed NumPy-style views.

   Old reference blocks can therefore remain resident while the next rank’s blocks are built. `jax.clear_caches()` does not free live arrays or repair closure bindings.

   **Fix:** put each setting/rank in a function scope; return an explicit minimal final-panel record and release temporary reference-rule data before the next build. Include retained final-panel blocks and slicing copies in memory estimates.

   At lines 123–129, the sampling thread and `interval()` update/reset maxima without a lock. **Fix:** synchronize both operations and take boundary samples. As required by A4, label these sampled peaks as potentially missing sub-0.2-second spikes.

10. **NEEDS-FIX — inherited large JIT captures, recompilation, and failure handling.**

   [qcore.py:133](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-wide-bank/experiments/jcp-wide-bank/vendor/quad2d/qcore.py:133), lines 135–153; [w2d.py:321](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-wide-bank/experiments/jcp-wide-bank/w2d.py:321).

   **Good:** query/rho kernels receive the large mesh and quadrature blocks as explicit arguments. The ordinary Python arm lambda does not itself make those arrays JIT constants.

   **Remaining policy violation:** `_grad` closes over model parameters, including an 8 MiB bank weight matrix; the head JITs also capture parameter arrays. This is inherited, not introduced by the new query wrapper, but it remains inconsistent with the repository’s large-array capture rule.

   **Fix:** use a documented lane-local adapter with parameters passed explicitly to gradient/head JITs; preserve the vendored files unchanged.

   Recreating JIT functions per setting and clearing caches forces recompilation, including final-panel warmups. That is not automatically a numerical error, but must enter the calibrated budget.

   Finally, reached-state SVD runs before the convergence gate. Nonfinite failed trajectories can make `np.linalg.svd` raise instead of marking the setting unavailable. **Fix:** check trajectory/Jacobian finiteness first, persist the failure, and skip dependent diagnostics.

11. **NEEDS-FIX — imports and config paths are correct; the submission cap is not atomic.**

   [stage.py:18](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-wide-bank/experiments/jcp-wide-bank/cluster/stage.py:18), lines 31, 56–104.

   All repository imports reachable from `w2d` are staged: `qcore`, `engines`, `sep_common`, `arms`, `hops`, `hfast`, `bkfast`, and `hari_quadrature`. Remaining imports are standard-library or environment packages. **`iterative_paths.py` is not needed** by this driver.

   The generated command correctly changes into the lane and uses `configs/j1_2d.json`. Checkpoint location, rotation path, GPU preflight, precision flags, manifests, and the 2D test64 guard are correct.

   At [submit.sh:10](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-wide-bank/experiments/jcp-wide-bank/cluster/submit.sh:10), two simultaneous submissions can both observe an empty queue and submit different attempts. Counting only `RUNNING/PENDING` also omits other nonterminal states.

   **Fix:** serialize the final queue check and `sbatch` using a shared-filesystem lane lock; reject any nonterminal lane job. Verify the returned job ID and directory after submission.

   A4’s calibrated `BUDGET.json` is neither required nor validated. **Fix:** require it for production submissions and check requested walltime against the registered bound.

12. **NEEDS-FIX — collection can delete an active job; independent acceptance is missing.**

   [collect.py:22](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-wide-bank/experiments/jcp-wide-bank/cluster/collect.py:22).

   `--partial --cleanup` can checksum a snapshot and then delete the directory of a still-running job. No scheduler-state or completion check prevents it. Transfer hashes prove the snapshot transferred correctly, not that the producer finished.

   **Fix:** forbid cleanup for partial collection; require terminal job state and the intended completion evidence before deletion. Keep partial manifests separate from the final `OUTPUTS.sha256`.

   Also, A0-2 requires `audit_w.py` to independently recompute errors/selection and detect swapped-reference and 1% field-perturbation faults. That auditor is absent. `complete=True` at [w2d.py:623](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-wide-bank/experiments/jcp-wide-bank/w2d.py:623) means execution finished, not that results passed acceptance.

   **Fix:** implement the specified independent acceptance path and distinguish `complete` from `accepted`, with failed-gate reasons.

| Item | Verdict | Required action |
|---|---|---|
| 1. Normalizations/strides | CORRECT | Optional initial-field agreement assertion |
| 2. M-prefix slicing | CORRECT mathematically | Do not claim untested bitwise rollout parity |
| 3. Eligibility/m*/diagnostics | CORRECT | None |
| 4. Controls | NEEDS-FIX | Resolve A0-2’s metric-only discrimination rule |
| 5. Convergence/target/reference gates | NEEDS-FIX | Add reference-file checksum provenance |
| 6. A–B–A/final panel | **WRONG — blocker** | Bind `fq`/`cold`; enforce K-time; compute paired ratios; precompile burn |
| 7. Projection floor | NEEDS-FIX | Enforce SVD/lstsq agreement |
| 8. Reached-state Jacobian | CORRECT | Persist full spectra if required |
| 9. Memory/residency | NEEDS-FIX | Release dangling references; synchronize sampler |
| 10. JIT/failure handling | NEEDS-FIX | Explicit parameter arguments; guard failed-state SVD |
| 11. Staging/submission | NEEDS-FIX | Atomic cap guard and calibrated-budget enforcement |
| 12. Collection/acceptance | NEEDS-FIX | Protect active jobs; implement independent auditor |