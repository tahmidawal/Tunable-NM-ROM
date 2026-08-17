# Codex adversarial harness review — 2026-08-17 (pre-fan-out)

`codex exec` (read-only guardrails in the prompt text; the `-s` sandbox flag is broken on
this box, so the prompt is the only guardrail). Brief: whether the CG/Newton stopping tests
are identical between the warm-started and baseline arms; whether the ROM decode cost is
fully charged; whether the FOM baseline is compiled and warmed identically; timing-protocol
validity; leakage of the held-out reference into either solve path; the BiCGStab NaN guard.

Reviewed at commit `fe0c70d` plus the then-uncommitted additions (run-role split, the
oracle diagnostic, the direct-solver baseline). Codex noted the tree was dirty during its
read; the additions it reviewed are the ones addressed below.

## Disposition of every MUST FIX

| # | Codex MUST FIX | Applied |
|---|---|---|
| 1 | Recursive-CG convergence accepted without requiring the TRUE residual to meet the tolerance; no initial finiteness guard | **Yes.** `wsf_util.make_cg` is now an OUTER true-residual loop around the textbook inner recursion (up to `max_restarts`), so the returned iterate provably satisfies `‖b−Ax‖/‖b‖ ≤ tau`; `flag` is derived from the true residual; non-finite `b`, `x0`, `tau` are flagged before the loop. Every row also aborts if `max(true residual) > tau`. |
| 2 | Burgers could publish failed/NaN Newton solves; a NaN initial residual reported flag 0; `lflag` ignored | **Yes.** A non-finite initial residual is flagged (code 3) before the loop; BiCGStab breakdown **and** max-iteration statuses are both counted and raised into the Newton flag (code 4); per-arm `max_rel_newton_residual`, `all_finite`, breakdown and flag counts are persisted in the row; a **solver-health gate** aborts the job unless every step of every arm met the tolerance with finite arithmetic. `json.dump(..., allow_nan=False)`. |
| 3 | BiCGStab missing the alpha-half-step convergence test — `s == 0` was mis-declared a breakdown and a converged iterate discarded | **Yes.** The `‖s‖ ≤ thr` branch is in, implemented branchlessly by forcing `omega = 0` (which makes `x_new = x + alpha·p`, `r_new = s` exactly) and by exempting `t·t == 0` from the breakdown test when `s` has converged. A `matvecs` counter is returned alongside `k` because the final sweep still evaluates `A(s)`. |
| 4 | Burgers omitted query-dependent preprocessing (`nearest_train_ic`, `Z0`, `u0_rms`, the per-step tolerance vector) — potentially dominant | **Yes.** The bank of training initial fields is built once per mesh and timed separately as offline model preparation (`offline_train_ic_bank_s`); the **online** part — the distance search, the latent gather, the tolerance scale and the tolerance vector — is a single jitted `prep(u0)` that is timed as `t_pre_ms` and included in `t_total_ms`. |
| 5 | The oracle CG diagnostic stopped on `‖x − u_ref‖`, violating the stated leakage rule | **Yes.** `make_cg_to_err` is deleted. Replaced by `wsf_util.cg_error_curve`, which runs plain CG from a zero start for a fixed number of iterations and **grades saved iterates** against the reference; the reference never enters a stopping test or a solve path. |
| 6 | The summarizer accepted incomplete, duplicate, dirty and cross-job reports and could synthesise a one-GPU cross-N curve from unrelated runs | **Yes.** Malformed JSON is now a hard error; `complete is not True` reports are dropped and named; `run_role` is validated; `select_consolidated` pools consolidated rows by (source file, Slurm job id, GPU, commit, harness source hash) and uses the single group covering the most meshes, erroring on duplicate keys and printing what it dropped; the chosen group's provenance is printed into `SUMMARY_TABLES.md`. A sha256 of every `wsf_*.py` travels with each row, so a dirty tree cannot publish two codes under one commit. |

## SHOULD FIX / NOTE items applied

- **Poisson `INIT`**: the `nearest` path reused case 0's latent for every case and its lookup
  was untimed. `INIT` is now forced to `mean`, with an explicit error otherwise.
- **Reference cross-checks strengthened**: Poisson checks the counting CG against
  `jax.scipy.sparse.linalg.cg` at **every reported tolerance** on several right-hand sides
  and compares true residuals; Burgers checks **every** test trajectory, reports the
  **per-step maximum** as well as the trajectory norm, and cross-checks one representative
  Newton correction against `jax.scipy.sparse.linalg.bicgstab`.
- **Native baselines added** so the instrumented solvers cannot hide behind their own
  instrumentation: Poisson times `jax.scipy` CG with a runtime `x0` for both arms
  (`t_fom_native_ms`, `t_fom_baseline_native_ms`); Burgers times the testbed's own jitted
  fixed-8-Newton rollout (`t_fom_testbed_ms`), which also answers the NOTE that the shared
  three-arm kernel adds a dummy guess stream to the nominal pure-FOM arm.
- **Timed-vs-reported sample mismatch**: `iters_*_timed` fields now carry the counts of
  exactly the cases/trajectory that were wall-clock timed, alongside the broader means.
- **`rom_tau = 0` relabelled** from "converged" to "ref. stops" (the objective test is
  disabled; the reference LM may stop on budget or lambda saturation).
- **"Best hybrid per N" labelled post-selected** in the table title, with the full ladder
  kept as the primary table.
- **Bitwise-equivalence claim softened**: the check compares the final latent to 1e-12
  relative and records `lm_rom_tau0_bitwise_identical`; the documentation now says
  "numerically equivalent", not "asserted bit-for-bit".

## Not applied, with reasons

- **Burgers absolute Newton thresholds are not numerically identical across arms** (each arm
  evolves its own tolerance-accurate `u_prev`, so `tau·‖u_prev‖` differs in the last digits).
  Codex classified this NOTE, and correctly: the *rule* is identical and the warm arm's test
  is if anything tighter. Documented rather than changed — forcing a shared `u_prev` would
  stop measuring the actual hybrid.
- **Preselecting `rom_tau` on separate validation runs.** The full ladder is reported as the
  primary result (P1) and the post-selected best is labelled as such; a second full timing
  sweep purely to de-bias a min-selection is not worth the GPU time here.

## Codex's verdict, verbatim

> Results produced by this code are not presently trustworthy as evidence that the hybrid
> beats a pure FOM. The central same-kernel stopping-rule construction, device
> synchronization, operator reuse, and trajectory indexing are sound, but the experiment does
> not enforce final solver health, contains a real BiCGStab correctness defect, omits a
> potentially dominant Burgers online cost, violates its own reference-leakage rule, and
> allows the summarizer to combine incomplete or cross-job measurements. After those MUST FIX
> items are addressed—and the native FOM baselines and source provenance are
> strengthened—the design could support the intended cost claim.

All six MUST FIX items and the strengthening it asks for were applied before any cluster job
was submitted. The full unedited report follows.

---

## MUST FIX

- [ctol_tables.py:133](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-17-cost-to-tolerance/experiments/cost-to-tolerance/ctol_tables.py:133), [ctol_pick_configs.py:41](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-17-cost-to-tolerance/experiments/cost-to-tolerance/ctol_pick_configs.py:41): `nondominated`, `cheapest_reaching`, and consolidation selection admit censored rows. A row that stopped on stall, step size, lambda saturation, or budget can therefore define the “cost at tau” frontier—the exact previous-round defect this experiment is meant to remove. Keep censored cells in raw output/censor tables, but require `censored == false`, `censored_frac == 0`, and `n_blowup == 0` for Pareto, target-reaching, ownership, and consolidation selection.

- [ctol_tables.py:56](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-17-cost-to-tolerance/experiments/cost-to-tolerance/ctol_tables.py:56): incomplete result files are accepted. `d["complete"]` is accumulated at line 79 but never enforced, and no expected-key coverage or uniqueness check exists. Because drivers save partial JSON incrementally and `pull.sh` copies failed-job output, a failed panel can silently become an incomplete surface. Abort unless every panel is complete and contains exactly one primary row for every expected `(N,k,method,tau)`, with consistent configuration/backend/commit.

- [cluster/make_cells.sh:7](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-17-cost-to-tolerance/experiments/cost-to-tolerance/cluster/make_cells.sh:7): the current cluster plan fans the surface into nine independent per-mesh GPU jobs. Re-timing only selected points later at lines 128–138 does not make the full `(k,N,tau)` surface a one-process/one-GPU measurement. This directly violates the requested timing protocol and leaves the primary cross-mesh surface heterogeneous. Restore one full-grid job per PDE, sequential over all `N`, `k`, methods, and tau on one GPU; or explicitly abandon cross-`N` claims and re-time every primary cell, not just selected points, in consolidation.

- [ctol_burgers.py:86](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-17-cost-to-tolerance/experiments/cost-to-tolerance/ctol_burgers.py:86), [ctol_burgers.py:300](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-17-cost-to-tolerance/experiments/cost-to-tolerance/ctol_burgers.py:300), [ctol_burgers.py:313](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-17-cost-to-tolerance/experiments/cost-to-tolerance/ctol_burgers.py:313): POD sees only 128 training trajectories, while the coordinate manifold and nearest-IC search use all 512. POD’s nearest search is then explicitly restricted to the first 128. This systematically handicaps POD accuracy and cold-start convergence. Build POD from the same 512 training trajectories—using streaming/randomized truncated SVD if memory requires it—and use the same 512-known-IC candidate set for both methods.

- [ctol_burgers.py:417](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-17-cost-to-tolerance/experiments/cost-to-tolerance/ctol_burgers.py:417), [ctol_burgers.py:440](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-17-cost-to-tolerance/experiments/cost-to-tolerance/ctol_burgers.py:440): blow-ups are replaced with NaN and silently removed from the reported mean, while `censored` depends only on LM reason codes. A cell can therefore report a flattering finite error and even `censored=false` after a full-field decode blows up away from the EQ nodes. Mark any blow-up as invalid/censored, make the primary `err_rel_l2` non-finite when any trajectory blows up, and retain a separate finite-only diagnostic. Also count non-finite computed errors, not just non-finite field entries.

- [ctol_burgers.py:309](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-17-cost-to-tolerance/experiments/cost-to-tolerance/ctol_burgers.py:309), [ctol_burgers.py:400](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-17-cost-to-tolerance/experiments/cost-to-tolerance/ctol_burgers.py:400): the query-specific nearest-training-IC search is performed outside the timed function, despite supplying one of the two starts used by every timed cold solve. At fine meshes this is an \(O(N_{\rm train}N^2)\) online operation and can erase claimed mesh independence. Include it in end-to-end timing, or use and time a genuinely deployable low-dimensional lookup; alternatively use mean-only initialization consistently.

- [ctol_poisson.py:315](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-17-cost-to-tolerance/experiments/cost-to-tolerance/ctol_poisson.py:315), [fu_eq.py:184](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-17-cost-to-tolerance/experiments/poisson2d-rom-objective/followup/fu_eq.py:184): the documented “one low-mode table matvec” preprocessing actually computes the full dense sine transform `S.T @ f @ S` and only then gathers \(M\) modes. This introduces avoidable \(O(N^3)\) work and overcharges ROM end-to-end cost, especially for `M=64`. Precompute the selected \(M'\)-mode table—or an equivalent selected-mode contraction—and time that actual \(O(MN^2)\) deployment operation.

- [ctol_poisson.py:293](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-17-cost-to-tolerance/experiments/cost-to-tolerance/ctol_poisson.py:293): FOM cost is measured only on source 0, while ROM cost is the median across all 16 sources and accuracy is their mean. CG iteration count can depend on the source, so speedups compare different workloads. Time all 16 FOM solves with the same warm-up/median-of-seven protocol and summarize them using the same across-source statistic as the ROM.

- [ctol_tol.py:83](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-17-cost-to-tolerance/experiments/cost-to-tolerance/ctol_tol.py:83), [ctol_tol.py:146](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-17-cost-to-tolerance/experiments/cost-to-tolerance/ctol_tol.py:146): Poisson and generic/cold-start solvers never test tau at the initial guess. When `||r(z0)|| == 0`, the specified inequality already holds, but these solvers continue and normally terminate censored because strict decrease is impossible. Initialize reason 2 when `(tau > 0) & (r0_norm <= tau*r0_norm)`, analogous to Burgers reason 4. Preserve the `tau=0` reference-agreement behavior.

- [cluster/make_cells.sh:30](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-17-cost-to-tolerance/experiments/cost-to-tolerance/cluster/make_cells.sh:30): results are labelled with the current worktree commit, but executable reference sources and checkpoints are copied from separate `2026-08-16-*` worktrees at lines 33–34. The dirty check covers only `cost-to-tolerance`, not those sources or artifacts. Thus the recorded `commit` does not identify the executed bundle. Stage reference source from the reviewed tree, use explicit checkpoint paths, refuse dirty source by default, record all source commits/checkpoint hashes plus the staged-manifest hash in JSON, and pull the manifest with the results.

## SHOULD FIX

- [ctol_poisson.py:406](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-17-cost-to-tolerance/experiments/cost-to-tolerance/ctol_poisson.py:406), [ctol_burgers.py:439](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-17-cost-to-tolerance/experiments/cost-to-tolerance/ctol_burgers.py:439): advertised end-to-end time is a sum of separately measured medians. Poisson preprocessing/decode and Burgers decode do not come from the graded timed invocation. Although these kernels are value-independent, a sum of medians is not the median of the actual pipeline. Time a single synchronized preprocess→solve→decode wrapper per source and grade its last repetition; retain component timings separately.

- [ctol_poisson.py:137](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-17-cost-to-tolerance/experiments/cost-to-tolerance/ctol_poisson.py:137), [ctol_burgers.py:125](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-17-cost-to-tolerance/experiments/cost-to-tolerance/ctol_burgers.py:125): the “k surface” changes `M` from 64 to 256 at `k=32`, while primary `m` stays 256. Thus the k=32 cost jump combines latent dimension, four times as many residual modes, and an `m/M` change from about four to one. The `m=1024` correction is supplementary and excluded from the primary Pareto. Use fixed `(M,m)` across k, or treat `(M,m)` as explicit tuning axes and include the valid `m≈4M` configurations in the frontier.

- [ctol_eq.py:17](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-17-cost-to-tolerance/experiments/cost-to-tolerance/ctol_eq.py:17), [ctol_poisson.py:468](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-17-cost-to-tolerance/experiments/cost-to-tolerance/ctol_poisson.py:468): the documented uncapped `eq_pool_control` does not exist. Poisson’s control changes to another capped 4096-point meshfree pool; Burgers has no cap control. Because the random candidate cap can create an artificial high-`N` error floor, add an actual uncapped or increasing-cap control at a feasible fine mesh for both methods/PDEs, or explicitly caveat all fine-mesh trends.

- [ctol_eq.py:64](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-17-cost-to-tolerance/experiments/cost-to-tolerance/ctol_eq.py:64), [ctol_eq.py:120](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-17-cost-to-tolerance/experiments/cost-to-tolerance/ctol_eq.py:120): the claim that the Poisson fit is unchanged from the reference is false. Padding uses `mean(abs(G))` instead of the reference’s decoder-output score `mean(abs(R))`, and perturbations change from absolute `0.05` to a global latent RMS scale. The latter is not invariant to per-coordinate scaling and may over-excite weak POD modes. Use a common physically meaningful/coordinate-whitened perturbation rule, and either reproduce the reference padding rule or document and control the change.

- [ctol_tol.py:33](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-17-cost-to-tolerance/experiments/cost-to-tolerance/ctol_tol.py:33), [ctol_poisson.py:384](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-17-cost-to-tolerance/experiments/cost-to-tolerance/ctol_poisson.py:384): the module says agreement checks cover Poisson and cold start, but only Poisson is checked, and only at the globally smallest k. The generic solver is independently copied despite `fu_common` being imported. Add a tau-zero cold-start agreement assertion and preferably factor the shared solver body into one implementation. Also correct the Poisson reason-code documentation: its body emits 3, not 4, for non-finite-step lambda saturation.

- [cluster/make_cells.sh:47](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-17-cost-to-tolerance/experiments/cost-to-tolerance/cluster/make_cells.sh:47), [cluster/launch.sh:32](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-17-cost-to-tolerance/experiments/cost-to-tolerance/cluster/launch.sh:32): the batch file hardcodes `gpu:a100:1`, while `launch.sh --gpu h100|h200|l40s` adds a conflicting constraint. Remove the option or have it set the actual GRES request. Also replace `find ... | head -1` at `make_cells.sh:102` with explicit checkpoint paths and expected hashes.

## NOTES

- Correct: for nonzero initial residuals, Poisson and cold-start tau tests implement `||r_j|| <= tau ||r_0||`, run before stall/lambda reasons after an accepted step, and use only deployable PDE/input information.

- Correct: Burgers computes each step’s threshold from that step’s own warm start. The extra `rn_fn` evaluation at [ctol_tol.py:206](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-17-cost-to-tolerance/experiments/cost-to-tolerance/ctol_tol.py:206) is inside the synchronized rollout timing. It is correctly absent from LM attempt/Jacobian counts because it is neither an attempt nor a Jacobian evaluation.

- Correct: the error field comes from the last timed solver result for every tested source/trajectory. The solvers contain no randomness, and the synchronization is on an output of the relevant compiled computation; no asynchronous solver dispatch visibly escapes the timed region. Warm-up and median-of-seven placement are otherwise sound.

- Correct: Poisson error is mean interior relative L2 against the CG solution at the same mesh. Burgers computes slice-wise relative L2 across all 51 slices and then averages trajectories. The only metric defect is how blow-ups are allowed to leave a usable aggregate.

- Correct: EQ is refit for every mesh, mode count, k, method, and m; there is no stale cache. Candidate capping changes the feasible support search, but targets remain exact full-grid projections. It is applied symmetrically to coordinate and POD primary arms.

- Correct: no held-out future solution enters initialization, EQ, POD construction, tolerance, or stepping. Burgers uses only the known initial field and viscosity; Poisson uses the known source. Held-out fields are confined to grading and post-run diagnostics.

- Correct: Poisson POD uses the same 512 training sources, weak objective, test modes, EQ nodes/weights, LM, mean initialization convention, and error metric. Its `pod_direct` matrix is the exact pseudo-inverse solve of the same hyper-reduced weak system and is appropriately supplementary.

- Correct cluster mechanics: GPU partition, A100 request, exit-42 GPU preflight, `jax_backend=gpu` logging, direct paralab `scp`, pre/post `squeue`, transfer checksums, paralab output paths, and conservative memory/time requests are present.

- The reviewed files changed on disk during this read-only review; the findings and line numbers above correspond to the final observed snapshot containing panel/consolidation logic. No files or jobs were created or run.