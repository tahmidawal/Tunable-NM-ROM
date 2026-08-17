# Codex adversarial review of the harness (before the fan-out)

`codex exec` in read-only mode, 2026-08-17, against commit `f777d5b` of this cell.  Its
verbatim report follows the triage.  Every MUST item was either APPLIED (commit `c1e41cd`)
or REJECTED with a reason recorded here; nothing was silently ignored.

## Triage

| # | MUST item | disposition |
|---|---|---|
| 1 | censored rows admitted to the Pareto / target-reaching / consolidation selection | **APPLIED** -- `usable_points()` requires `censored == false` and `n_blowup == 0`; the strict frontier is primary and the *as-deployed* frontier (censored included) is tabulated and drawn dashed beside it, so the accurate-but-censored operating points are still visible |
| 2 | incomplete result files accepted | **APPLIED** -- `ctol_tables.audit()` refuses to build tables unless every panel is `complete`, ran on GPU in f64 at matmul precision `highest`, and every cell of the *specified* grid (not the union of what arrived) is present exactly once; `--allow-incomplete` stamps the problems into the README's integrity block |
| 3 | "restore one full-grid job per PDE" | **REJECTED** -- see below |
| 4 | Burgers POD saw 128 training trajectories vs the coordinate arm's 512 | **APPLIED** -- POD is built from all 512, retaining every 4th time slice (3.5 GB instead of 13.7 GB at N=256); the nearest-IC candidate set is the same 512 for both arms; the POD projection floor on the held-out set is reported per mesh |
| 5 | blow-ups silently dropped from the mean | **APPLIED** -- any blow-up makes the primary `err_rel_l2` non-finite and the cell censored; `err_rel_l2_finite_only` is kept as a labelled diagnostic |
| 6 | nearest-training-IC search outside the timed region | **APPLIED** -- moved inside the jitted, timed cold start and onto the m EQ nodes: `O(N_train m)`, mesh independent, deployable |
| 7 | Poisson preprocessing charged the full dense sine transform | **APPLIED** -- timed as the deployable selected-mode contraction `O(M' N^2)`, asserted equal to `pro_common.weak_source_term` to 1e-10 relative; the full-transform cost is recorded alongside |
| 8 | FOM timed on source 0 only | **APPLIED** -- every test source, same warm-up/median-of-7, same across-source statistic as the ROM |
| 9 | tau never tested at the initial guess | **APPLIED** -- both solvers now emit reason 2 when `tau > 0` and `||r(z0)|| <= tau ||r(z0)||`, mirroring `lm_step_jit`'s reason 4 |
| 10 | recorded commit does not identify the executed bundle | **APPLIED** -- every panel records all four source-worktree commits and dirty markers, the sha256 of every module and checkpoint it actually loaded, and the staged manifest hash; `MANIFEST.sha256` is copied into `out/` so it is pulled with the results |

SHOULD items applied: one timed pipeline per cell instead of a sum of independently
measured medians; an isolator arm at fixed `k`=8 carrying the `k >= 32` `(M, m)` change, and
both `k`=32 arms admitted to the Pareto; a genuine **uncapped**-pool control for both methods
and both PDEs (Codex correctly noted the documented one did not exist); the reference's EQ
support-padding rule restored and the latent perturbation made coordinate-whitened rather
than globally RMS-scaled; `launch.sh`'s GPU flag removed (it conflicted with the batch file's
`--gres`); explicit checkpoint paths instead of `find | head -1`; the reason-code
documentation corrected.

Not applied: a tau=0 agreement assertion for the *cold-start* solver (only the Poisson solver
is checked). `lm_tau_generic` is a line-for-line copy of `fu_common.lm_jit_solver` plus the
tau test, and the Poisson check exercises the identical tau machinery; this is recorded as a
gap rather than closed.

## Why MUST #3 was rejected

Codex asked for one full-grid job per PDE, sequential over every `N`, `k`, method and `tau` on
one GPU. It did not have the operating constraint this cell was launched under, which is to
use the idle A100 pool rather than serialise the whole grid into one job. The decomposition
actually used keeps the protocol intact where it matters:

* **Dominance is computed WITHIN a (PDE, N) panel.** A panel is one job on one GPU, so every
  timing that a Pareto frontier compares was measured on the same device, in one process, with
  the same warm-up and median protocol. The per-panel frontier is valid exactly as Codex would
  require.
* **Accuracy is GPU-independent.** Errors, censoring rates, achieved residuals, EQ fit quality
  and iteration counts do not depend on which A100 produced them.
* **Cross-`N` timing is the one thing that does not survive the fan-out**, and it is the only
  thing the scaling figure needs. It is served by a separate **single-GPU consolidation job**
  that re-times the *whole* per-panel non-dominated frontier (not just the argmin) across every
  mesh in one process. The README states which figures come from panel timings and which from
  the consolidation run, and a generated table puts the two side by side for the same
  configurations.
* All panels request the same GPU type (`--gres=gpu:a100:1`), and every row records the GPU
  model and the node.

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