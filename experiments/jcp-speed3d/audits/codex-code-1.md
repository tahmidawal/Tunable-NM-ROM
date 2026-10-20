**NOT READY FOR CLUSTER SMOKE.** The core numerical algebra passed the small checks, but the replay, predictor implementation, configuration round-trip, and acceptance logic have material defects.

This was read-only: no files or git state changed, no jobs submitted, and no GPU work. CPU checks ran sequentially with bytecode and persistent compilation caching disabled. All files listed in `INHERITED.json` matched their recorded hashes.

Paths below are relative to `experiments/jcp-speed3d`.

1. **QN evolution — CORRECT algebra; NEEDS-FIX records and replay interface.**

   Numerical checks on a random \(M=7,\ R'=3,\ m=11\) problem found:
   
   - `adv(c)` versus `0.5*contract(c)@c`: maximum absolute difference \(1.11\times10^{-15}\).
   - Analytic gradient versus autodiff: relative differences \(1.23\times10^{-17}\), \(1.29\times10^{-17}\), and \(1.67\times10^{-16}\) for BE, CN, and BDF2.
   - IMEX normal-equation residuals: below \(3.9\times10^{-15}\).

   The implementation at [qn3.py:91](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/qn3.py:91) correctly cancels the scaled linear Jacobian to \(A\). Candidate activation, finite-residual masking, lowest-index ties, BE startup, and history initialization agree with A2.

   Refresh application versus execution accounting, emergency refresh on the final iteration, recomputation of `gam`, damping refactor, backtracking, trust clipping, and reason precedence are consistent with the amended algorithm (`qn3.py:144–228`). Injecting NaN advection produced reasons `[3,3,3]`, frozen states, and refresh counts `[1,0,0]`.

   Two shortcomings:
   
   - `mu` is carried but never returned or saved (`qn3.py:226`, `s3common.py:238`). Add per-step damping to the record.
   - `n_exec` is a **batch-total counter broadcast to every member**. Summing its saved array over members overcounts by batch size. Store it once per step or explicitly provide an aggregation convention.

   A QN reason of 4 can disagree with fresh certification because its denominator uses cached `Jn`; that is anticipated by the design. Certification and E-c protect against this in grid analysis. The downstream acceptance defects in item 9 weaken those protections.

2. **Certification — CORRECT within the specified numerical tolerance.**

   [qn3.py:254](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/qn3.py:254) initializes \(J(c_0)\), carries the preceding Jacobian into \(F_n\), initializes the previous state to \(c_0\), and applies BE startup coefficients for BDF2.

   On the tiny BE/CN/BDF2 trajectories, certified versus recorded FS ratios had **zero difference on fixed-sweep steps**. Recontracting a Jacobian versus extrapolating its history need not be bitwise identical on every problem; G2’s tolerance is the appropriate check.

   **Fix the test:** `test_s3.py:95–97` reduces with `max(worst, NaN)`, which can retain zero and report success. Explicitly reject nonfinite certification values and differences before reduction.

3. **FSP — CORRECT with `extra=False`; WRONG with `extra=True` relative to A1.3.**

   Tiny tests found bitwise equality of **all returned arrays** against FS for BE, CN, and BDF2 with extras disabled.

   With extras enabled, [fs3p.py:134](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/fs3p.py:134) evaluates P3 through the cheap residual and contracts a fresh Jacobian whenever **either P3 or P4** wins (`:147`). A1.3 requires:
   
   - P3 residual and Jacobian from cubic extrapolation of carried Jacobians;
   - an extra contraction only when P4 wins.

   The carry adds `wp3`, but no `Jp3` (`:169–193`). This changes both floating-point predictor decisions and measured cost.

   **Fix:** carry and shift `Jp3`; form cubic `J3`; evaluate P3 through `rj(J3,p3)`; contract only for P4; record the additional contraction. The coefficient-history shift itself is correct.

4. **Shared query — CORRECT projection/decode; WRONG timestep serialization.**

   Mesh projection, the single-member no-vmap path, mesh-free Cholesky projection, mesh-equivalent adaptive scale, `1e-12` FS tolerance scaling, decode orientation, and lattice distance are consistent with the design (`s3common.py:128–205,247–254`). ADT receives the scale and applies its scheduled tolerance internally.

   **Reproduced blocker:** [s3common.py:59](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/s3common.py:59) formats timesteps with `.8g`; `parse_key` treats that display value as executable input. Consequently:

   ```text
   qn|CN|0.016666667|k4
   → nsteps(0.016666667)
   → AssertionError
   ```

   Any selected \(0.05/3\) configuration fails when reconstructed for later modes.

   **Fix:** use a lossless structured spec, a canonical timestep identifier, or normalize parsed values to the registered timestep set. Keep display formatting separate.

   `run_valid` implements the intended grid E-a checks, but mesh-free condition-number invalidation is absent; see item 6.

5. **Grid/context — NEEDS-FIX coverage, replay, and timing acceptance.**

   The setting definitions and FOM ladder generator match the amended tables. However, [s3run.py:37](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/s3run.py:37) restricts FS-K2 to CN/BDF2 on both deployed and reduced rules. BE FS-K2 is missing from the solver/decomposition population.

   **Fix:** include BE FS-K2, including reduced-rule BE configurations eligible for \(B^*\).

   Tight/tighter comparison and lattice distances are correctly computed. Replay step selection, adaptive-start exclusion, baseline reproduction tolerance, and label precedence agree with A2/A3 (`replay.py:23–46,128–170`).

   **Replay intervention iv is wrong:** [replay.py:153](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/replay.py:153) supplies history and `k0`, but QN then chooses a new best-of-five predictor. It never receives the saved FS predictor index. This is not the prescribed intervention from the same candidate.

   **Fix:** add an explicit initial-candidate override for replay, preserve the original step history, refresh at that candidate, and report fresh terminal certification and success.

   Timing warm-up, burn-in, randomized A–B–A order, and missing-hash rejection are implemented (`s3ctx.py:163–192`). T1 does dispatch and synchronize twice. But:
   
   - T1’s two SHA computations occur **inside** the measured callable (`:213–216`), adding host hashing to its nominal doubled query cost.
   - No code enforces T1’s `[1.8,2.2]` acceptance band.
   - The mesh fallback to T1 (`s3modes.py:408–409`) does not ensure C_old accuracy hashes exist, so that fallback panel fails.
   - The saved local grid panel has **102/102 hash failures and zero C_old/gl32 accuracy rows**. Current `s3run.py:133–134` adds that rollout; the existing smoke therefore does not validate the current fix.

   **Fix:** move T1 validation outside its timer, enforce its gate, and require expected hashes before entering any panel.

6. **FOM/profile/mesh/replication modes — NEEDS-FIX; several required decisions are unimplemented.**

   **FOM calibration:** level-0 qualification is mandatory and looser levels use the required maximum lattice difference (`s3modes.py:83–110`). For FD, its unchanged initial lattice field gives the specified \(n_0\). The convergence-predicate limitation is in item 8.

   **E0:** in-job loading correctly checks the store-version and STATUS pins, uses `field='finest'`, and checks normalizer/t0 (`s3ctx.py:220–254`). It returns reference eligibility, but does not enforce it. Persist the checked reference hashes in the result as well.

   **FOM selection cannot yet be performed validly.** `mode_fom` records first-call timings but no warm repeated comparison panel (`s3modes.py:145–172`). `seconds_first` includes compilation for early entries and is not a valid “cheapest calibrated FOM” measure. No lane-level matcher or K10 implementation exists.

   **Fix:** add warm repeated FOM screening in one job, retain timing arrays, implement matching with convergence/reference-eligibility checks, and run K10.

   **Mesh mode:** construction at \(n\), mesh-free initialization, ADT at the same mesh, and same-mesh panels are wired correctly. Defects:
   
   - G5 only covers requested ROMs; neither current smoke nor `test_fom.py` guarantees required W256 coverage.
   - K7 is recorded but not included in the mesh gate (`bigtables.py:94–101`, `s3modes.py:335`).
   - Tight/tighter validity is not propagated alongside agreement (`s3modes.py:363–368`).
   - A matched FD configuration is only timed when its **own** L equals the current ROM L; PS is only timed at the first L (`:389–392`). Thus cross-resolution matches are missing for other ROM meshes.

   **Fix:** time every chosen cross-resolution match against each relevant ROM mesh, independently of the FOM’s input resolution; enforce G5/K7 and both ADT validity checks.

   **Profile/mesh-free qualification:** the code collects useful trajectories and condition numbers, and initializes its ADT comparator with the same projection. But it does not implement H3.2’s combined eligibility/error verdict, reject the deployed mesh-free path when condition exceeds \(10^{10}\), or determine whether K5 actually fails the required accuracy criteria (`s3modes.py:190–204`).

   The profiling kernels also use surrogate algebra: fixed `.01`, no actual scheme scaling/history, identity damping instead of \(\mu\operatorname{diag}H\), and an unused viscosity argument (`:229–237`). These are not the specified QN components. G3 and the post-start split are not evaluated.

   **Fix:** profile extracted kernels using actual stage inputs; calculate G3; implement H3.2/K5 qualification and make it a prerequisite for mesh claims.

   Batching’s G4 and whole-query permutation checks are structurally sound. K6 uses `min(distance)>bar`, requiring **every** member to differ; rejecting equality of the batch requires `max(distance)>bar`. Timed batch repetitions also lack output-hash checks (`:272–288`).

   Converged-rule sensitivity and reduced-rule ADT runs exist, but completeness and acceptance are not enforced (`:302–316`).

   **Replication freeze is ineffective.** [s3ctx.py:45](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/s3ctx.py:45) accepts any nonempty `frozen_sha256`; it never reads or hashes a freeze file or checks configuration membership. Replication also lacks ADT comparisons for recomputing E-c.

   **Fix:** verify the immutable freeze and all selected decisions before constructing the replication cohort; stage the freeze and validate its hash; run required replication eligibility comparators.

7. **Streaming tables — CORRECT mathematics; NEEDS-FIX gate coverage.**

   [bigtables.py:21](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/bigtables.py:21) uses the correct normalization, denominator, one-based modes, zero-based gather, and x-slowest/z-fastest slab ordering.

   A random small MLP using `common.features` compared against direct `common.phiT` gave:
   
   | Mesh n | M | Relative Frobenius error |
   |---:|---:|---:|
   | 9 | 17 | \(2.73\times10^{-16}\) |
   | 17 | 17 | \(6.79\times10^{-16}\) |

   Required fix: run and enforce the complete G5/K7 bank population described in item 6. These tiny checks do not establish production-size feasibility.

8. **PS/FOM wrapper — CORRECT basic port; NEEDS-FIX convergence acceptance.**

   PS reuses the nonlinear operator and ETDRK4 coefficients and includes synthesis in its jitted query. Two tiny N=8 comparisons against `pscore` gave relative differences \(9.4\times10^{-18}\) and \(1.5\times10^{-17}\). The required N=32 G7 remains a separate test.

   FD lattice indexing is correct: `k*(L/64)-1` (`fomrun.py:34–38`). Tolerance levels are correct, and recorded residuals are relative residuals.

   [fomrun.py:73](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/fomrun.py:73) nevertheless:
   
   - does not test the recorded Newton limit/exit condition;
   - truncates to available residual records rather than rejecting incomplete evidence;
   - returns unconditional success for PS.

   A fabricated record with `nwt=20` and a small residual was accepted. `fom_accuracy` can likewise record nonfinite PS output as `converged=True`, merely omitting its score.

   **Fix:** record and enforce termination reasons, require complete stage evidence, and require finite PS fields. Distinguish convergence on the last permitted iteration explicitly if that is intended.

9. **Scoring/analysis — WRONG acceptance logic despite correct metric formulas.**

   The error normalization, evolved-time maximum, uncertainty endpoint propagation, pooled E-b, E-c, and H2 3% tie rule are correct.

   Material failures:
   
   - **Reference eligibility is ignored.** `analyze.py:84–105` reads scores/errors but never uses each score’s `eligible`. I changed every reference eligibility flag to false in memory: configuration results were unchanged, including 35 eligible configurations.
   - K1/K2/K3 are reported but never block protected claims (`analyze.py:230–245`).
   - H1.2/H1.3 ignore the ADT resolution gate. H1.4 chooses from `eligible` configurations without requiring valid panels or ADT gates (`:201–217`). The invalid local grid panel still produces an H1.4 “fail (screening)” verdict.
   - There is no final confirmed H1.4/H2 verdict implementation; `analyze.py` ends with screening selections.
   - Completion, cohort alignment, score provenance, and K4/distance-check failures are recorded without acceptance enforcement.

   **Fix:** centralize claim-specific acceptance, including reference eligibility, cohort/provenance completeness, ADT resolution, applicable controls, and timing validity. Produce final speed verdicts only from valid direct confirmation panels.

   Two controls differ from their specification:
   
   - [score3.py:124](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/score3.py:124) scores the second field against the first reference; A1.10 specifies **case 0’s field against case 1’s reference**. Select those case IDs explicitly and use that direction.
   - K9 correctly constructs seeded perturbations, but assumes row zero is case 0 and duplicates the threshold logic instead of invoking a shared production E-c classifier (`analyze.py:147–161`). Resolve case IDs and share the classifier.

   Offline scoring pins store version but not STATUS (`score3.py:33–39`). Add the same STATUS check and record both hashes.

10. **Cluster machinery — CORRECT basic staging/submission; NEEDS-FIX readiness and resource evidence.**

   The current active import graph is covered by `stage.py`’s file list. In particular, `qpanel` is staged; t2core’s unstaged 2D imports are lazy and are not reached by these drivers. `test_fom.py` needs `--skip-e0` on the cluster because its comparison source uses a local absolute path.

   Private attempt directories, locked cap-three submission, MANIFEST verification, GPU preflight, precision environment, reference environment, output checksums, and collection-before-deletion are present.

   Remaining issues:
   
   - Only local configs currently exist; generated cluster smoke/production configs are absent. `stage.py:51` requires a committed config blob.
   - The resource certificate check verifies a few fields but does not establish smoke provenance, inventory completeness, or mesh feasibility (`stage.py:53–56`).
   - Grid inventory omits warm-ups, replay interventions, certification accounting, and some baseline work; other modes mostly list configurations (`s3run.py:57–72`, `s3modes.py:29–39`).
   - H200 is the default, but other GPU types remain accepted.
   - Neither staging nor execution verifies the replication freeze.

   **Fix:** generate and commit reviewable configs after repairs; derive complete resource estimates from pinned smoke evidence; enforce H200 and freeze requirements.

11. **Production answerability — NEEDS-FIX end-to-end gates.**

   The lane currently lacks the complete selection/freeze/matcher workflow promised by `make_configs.py:1–2`; the referenced lane-level `select.py` does not exist. H3 qualification, final confirmation verdicts, FOM matching/K10, and G8’s stored-Step-2 Gram-versus-lattice check are also absent.

   Smoke scripts save booleans but do not fail the process when mandatory gates fail (`test_s3.py:100–102,150–151`; `test_fom.py:90`). A completed job therefore does not imply a successful smoke.

   **Fix:** implement a machine-readable acceptance report with a nonzero exit on mandatory failures, preserving diagnostic output. Wire every later selection and freeze to that report.

**Required fixes, blocking first:**

1. **Before smoke:** repair lossless timestep reconstruction, FSP’s P3 Jacobian path, and QN replay’s fixed candidate.
2. **Before accepting smoke:** enforce finite gate results, mandatory test failures, T1, complete G5/K7 coverage, and expected timing hashes; rerun the stale local-grid hash validation.
3. **Before production selection:** enforce reference eligibility, ADT gates, K1–K4, cohort/provenance completeness, and valid timing panels.
4. **Before production:** restore BE FS-K2 coverage and implement final confirmation-based H1/H2 verdicts.
5. **Before FOM matching:** implement warm ladder timing, strict finite/convergence checks, the matcher, K10, and cross-resolution panels.
6. **Before mesh/profile claims:** implement H3.2/K5 qualification, condition gating, valid ADT comparisons, faithful profiling kernels, G3, and G8.
7. **Before replication:** implement and verify the immutable freeze and frozen configuration membership.
8. **Before cluster staging:** generate the configs and complete resource/feasibility evidence.
9. **Non-blocking diagnostics:** persist QN damping, clarify `n_exec` aggregation, fix K4/K9 construction details, adjust K6’s failure predicate, hash timed batches, and record ROM input preparation and reference provenance.

**Overall verdict: NOT READY.**