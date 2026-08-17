# Adversarial Codex review of the Heat-2D latent-stepping harness (2026-08-16)

Run before the cluster fan-out with `codex exec` (read-only brief; the `-s` sandbox flag is
broken on this box, so the guardrails were carried in the prompt). Brief: the residual vs the
FOM's actual discrete operator and `dt`, held-out leakage into the ROM path, timing-protocol
validity, POD comparability, weak-form units and weights.

## Triage — what was applied

| item | verdict | action (commit `5eedcf0`, `90171ca`) |
|---|---|---|
| MUST-1 Stage-1 boundary row is `u_{n+1}`, not `u_{n+1} - u_n` | **valid bug** | fixed; residual identity now compared row-by-row in the `n^2` layout AND on a random field with non-zero walls (the FOM trajectory has zero walls and could never expose it). Verified 8.9e-16. |
| MUST-2 timing pipeline inconsistencies (7 sub-items) | **valid** | timed rollout now uses `ops['tol_scale']` (was `sqrt(m)` = 2.7–7.9x stricter than the evaluated run); each of the jitted-IC and python-IC end-to-end numbers is now one realizable pipeline (nearest-IC search + cold start + rollout + decode of all 51 slices, blocked); the python IC LM no longer rebuilds its jits per call; Galerkin is no longer charged twice for decoding and is labelled `python_loop_incl_decode`; POD is timed with projection + rollout + reconstruction; POD timing is skipped for non-LSPG variants instead of silently timing the LSPG scan. |
| MUST-3 cluster TOCTOU / unvalidated cell name | **valid, partially applied** | cell-name regex, refusal to reuse a cell dir that already has output, FATAL (not warning) on a non-unit post-submit queue count, staged manifest hash + dirty-worktree count in the job log. A true distributed lock was judged unnecessary: `hlat/` has exactly one submitter (this agent) and every cell has its own directory. |
| SHOULD-1 checks did not assert; LSPG normal equations unchecked | **valid** | checks now abort; `weakall(alpha=0)` vs strong-form full grid compared on `\|\|r\|\|`, `J^T r`, `J^T J`, `JD^T r`, plus decoder wall values `== 0` exactly. |
| SHOULD-2 tolerance absolute in `u0`, not the decaying state; not comparable across M | **valid** | stopping rule changed to `\|\|r\|\| <= max(GN_RTOL*\|\|r(z_n)\|\|, GN_TOL*rms(u0)*tol_scale)`; both `\|\|r\|\|` and `\|\|r\|\|/\|\|r(z_n)\|\|` reported. |
| SHOULD-3 Galerkin root tolerance representation-dependent | **valid** | relative criterion `\|\|g\|\| <= GN_RTOL*\|\|g_0\|\|`. |
| SHOULD-4 training generates TEST; `TEST_SEED` unguarded | **valid (hygiene)** | `build_data(with_test=False)` in the trainer; `TEST_SEED != SEED` asserted; TEST parameter rows asserted disjoint from TRAIN/VAL. |
| SHOULD-5 NaN steps can count as completed | **valid** | `nan_step` distinguished from `lambda_max`; NaN terminations truncate the rollout; non-finite NNLS weights and checkpoint parameters abort. |
| SHOULD-6 POD is a representation control, not a production POD | **valid** | added the **direct reduced POD-Galerkin ROM** (`V^T A_kappa V`, a k x k solve per step) as the honest linear speed competitor, alongside the same-solver POD control. The IC asymmetry is kept and documented: `V^T u0` IS the exact minimiser of the POD misfit, so running LM there would only handicap POD. |
| SHOULD-7 checkpoint/manifest validation | **valid** | `k_lat`, `Z_train`/`V` shapes and parameter finiteness checked; the Stage-1 sweep manifest is mandatory and the trunk input width is checked. |
| SHOULD-8 EQ diagnostics are in-sample and row-scaled | **valid** | EQ weights now validated OUT OF FIT on a disjoint set of training latents (relative mode-projection error mean/max, weight sum/min/max, finiteness); the in-sample number is renamed `rel_fit_insample_scaled`. |
| NIT-1 what alpha=1 LSPG actually minimises | **accepted, no code change** | folded into the README: with all modes and exact sums `W = A^-1`, so alpha=1 LSPG minimises `\|\|D(z) - A^-1 D(z_n)\|\|` — the Euclidean distance to the exact backward-Euler step, not the residual norm. With M modes it constrains only the low-mode part of that state error. |
| NIT-3 `weakall` accepted an alpha suffix | **valid** | rejected now. |
| NIT-4 summarizer hard-coded `n_test=16` | **valid** | read from the config. |
| NIT-5 JSON `NaN` tokens | **not applied** | only this repo's summarizer consumes these files. |
| NIT-6 stage freshness vs source edits | **partially applied** | manifest hash + dirty count are recorded in the log; cells are always rebuilt immediately before launch. |
| SHOULD-4 (2nd half) "choosing the best TEST variant would turn TEST into selection data" | **accepted as a caveat** | stated explicitly in the README verdict: the variant grid is reported in full and no variant is selected on TEST. |

## Codex's "verified correct" list (independently confirmed by the run-time checks)

`dx = 1/(n-1)`, Laplacian sign and `u - u_prev - dt*kappa*lap(u)` agree with the FOM
(`local_vs_fom_maxabs` 7.8e-16); the FOM mask makes wall values exactly zero and the
polynomial hard-BC factor is exactly zero there (`decoder_wall_maxabs` 0.0); the sine-mode
eigenvalue formula, grid `i/(n-1)`, unit normalisation and interior flattening order are
consistent, so `Phi^T L u = -diag(lam) Phi^T u` is exact (`weakall_vs_fd_*` 2.3e-16 …
7.3e-16); no held-out leakage in the Stage-2 ROM path; the POD basis comes from exactly the
same TRAIN snapshots; no silent f32 in the Stage-2 path; the GPU preflight makes CPU
fallback non-silent; Slurm logs go to the project path, not the quota'd home.

---

## Full report as returned by Codex

# MUST-FIX

1. **Stage-1 does not use the FOM’s boundary-row residual.**  
   File: [hlat_stage1.py:83](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-16-heat2d-rom-latent-stepping/experiments/heat2d-rom-latent-stepping/hlat_stage1.py:83)

   `r_res` appends `Uz[n + 1][bnd]`, but the actual linear-system residual is
   `Uz[n + 1][bnd] - Uz[n][bnd]` because `implicit_op` is the identity on boundary rows. These differ for the sweep decoder, which has no hard boundary condition. Thus the `resid` and `both` Stage-1 arms optimize a different boundary objective than claimed.

   The check at [hlat_stage1.py:128](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-16-heat2d-rom-latent-stepping/experiments/heat2d-rom-latent-stepping/hlat_stage1.py:128) cannot expose this: the true FOM trajectory has zero walls, and it sorts absolute residual entries before comparison, discarding signs and row locations.

   **Minimal fix:** use `Uz[n + 1][bnd] - Uz[n][bnd]`. For the check, scatter interior and boundary residuals back into an `n²` vector and compare directly, without sorting or taking absolute values first. Alternatively, describe the implemented term honestly as an independent Dirichlet penalty rather than the FOM residual.

2. **The reported timing numbers are assembled from inconsistent pipelines and tolerances.**  
   Files: [hlat_rom.py:304](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-16-heat2d-rom-latent-stepping/experiments/heat2d-rom-latent-stepping/hlat_rom.py:304), [hlat_rom.py:371](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-16-heat2d-rom-latent-stepping/experiments/heat2d-rom-latent-stepping/hlat_rom.py:371)

   Several independent problems make speedups non-apples-to-apples:

   - Timed weak/EQ rollouts use `sqrt(ops["m"])` at line 310, while accuracy rollouts use `ops["tol_scale"] = sqrt((N-2)²)`. At `N=64`, `eq512` is timed with a tolerance 2.74× stricter than the evaluated algorithm; `eq256` is 3.88× stricter. At `N=128`, `eq256` is 7.88× stricter.
   - `speedup_vs_fom_end_to_end_jit_ic` adds the cost of the jitted IC solver to a rollout initialized by the **Python** IC result. The `ics_jit` latents computed at lines 230–240 are never selected and fed into that timed rollout. It therefore is not the timing of a realizable pipeline.
   - The nearest-training-IC search is outside both IC timers, although it is required online.
   - Galerkin timing calls `bc.rollout`, which already decodes all 51 full fields, and then the reported end-to-end number adds `decode_all_slices_s` again. Galerkin is double-charged for decoding.
   - Coordinate LSPG “rollout-only” returns latents, whereas the FOM rollout returns all 51 fields. The separate decode timing partly addresses this, but the prominently reported rollout speedup compares different outputs.
   - POD timing omits both the initial projection and reconstruction of the 51 fields, yet is compared with a FOM that produces those fields. This inflates the POD speedup.
   - If a user makes a Galerkin variant the first `POD_VARIANTS` entry, lines 371–378 silently call `ops["rollout_jit"]`, which is always the LM/LSPG scan, while labeling it as Galerkin.
   - `ic_once` calls `fit_ic`, which constructs fresh `jax.jit` wrappers on every invocation. Its warm-up does not reliably exclude those recompilations, contradicting the compile-exclusion claim. This biases the Python-IC number conservatively.

   **Minimal fix:** time one actual pipeline per implementation: select the nearest training IC, run the chosen IC solver, select its best latent by IC misfit, feed that latent into the rollout using `ops.get("tol_scale", sqrt(m))`, decode the actual returned latents once, and block on the fields. Do the equivalent projection/rollout/reconstruction for POD. Keep Galerkin separate until it has a device implementation, and do not add decoding if `bc.rollout` already performed it.

3. **Cluster launching does not enforce the claimed one-job-per-directory invariant.**  
   Files: [make_cell.sh:6](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-16-heat2d-rom-latent-stepping/experiments/heat2d-rom-latent-stepping/cluster/make_cell.sh:6), [launch.sh:11](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-16-heat2d-rom-latent-stepping/experiments/heat2d-rom-latent-stepping/cluster/launch.sh:11)

   The `squeue` check and `sbatch` submission are a TOCTOU race: two concurrent launchers can both observe zero jobs, copy into the same directory, and submit. Both jobs then write the same checkpoint/JSON paths. The post-submit count only warns after corruption has become possible.

   In addition, `cell` is not validated before being embedded in `STAGE`, `REMOTE`, `rm -rf`, and remote shell commands. An empty/path-traversal/metacharacter value can delete unintended staging directories or alter remote commands.

   **Minimal fix:** require a nonempty conservative cell-name regex, use an atomic remote lock or a unique immutable run directory, and make a non-unit post-submit job count fatal. Record the run directory/manifest hash in the result JSON.

# SHOULD-FIX

1. **The residual and `weakall` “assertions” do not assert anything, and the LSPG check is incomplete.**  
   File: [hlat_rom.py:142](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-16-heat2d-rom-latent-stepping/experiments/heat2d-rom-latent-stepping/hlat_rom.py:142)

   Deviations are only written to JSON; a failed identity still permits the run to finish. The `weakall` check compares residual norms and the Galerkin quantity `JD.T @ r`, but does not check the claimed LSPG normal equations `J.T @ r` and `J.T @ J`. Consequently, the statement “asserted in every run” in [hlat_common.py:163](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-16-heat2d-rom-latent-stepping/experiments/heat2d-rom-latent-stepping/hlat_common.py:163) is false.

   **Minimal fix:** abort on nonfinite/excessive deviations and compare all of:

   - residual norms;
   - `J_fd.T @ r_fd` versus `J_weak.T @ r_weak`;
   - `J_fd.T @ J_fd` versus `J_weak.T @ J_weak`;
   - `JD_fd.T @ r_fd` versus `JD_weak.T @ r_weak`;
   - decoder wall values versus zero.

2. **The stopping tolerance is absolute relative to `u0`, not relative to the decaying state.**  
   Files: [hlat_common.py:294](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-16-heat2d-rom-latent-stepping/experiments/heat2d-rom-latent-stepping/hlat_common.py:294), [blat_common.py:837](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-16-burgers2d-rom-latent-stepping/experiments/burgers2d-rom-latent-stepping/blat_common.py:837)

   If `d = ||u_n||/||u_0||`, the allowed residual relative to the current state grows to `10^-9/d`. Thus:

   - `d=10^-3` permits about `10^-6` relative to the current state;
   - `d=10^-4` permits `10^-5`;
   - `d=10^-6` permits `10^-3`.

   At `N=64`, the lowest-mode eigenvalue is about `19.735`; at `kappa=0.5`, its 50-step backward-Euler decay is about `0.0900`. Localized high-frequency content can decay much further. A `10^-3` total decay does not automatically make the warm start pass—the alpha-1 lowest-mode warm residual is still roughly `0.047 u_n`, so `tol_at_init` would require decay near `2×10^-8`—but the late-time criterion is nonetheless 1000× looser in relative terms.

   Across `M`, the same threshold is also not a per-equation-equivalent tolerance. For an isotropically distributed residual at `N=64`, a 64-mode projection contains roughly `sqrt(64/3844)=0.129` of the full norm, making the effective test about 7.75× easier than `weakall`; for `M=16`, about 15.5×.

   **Minimal fix:** use a relative-plus-absolute criterion based on the available previous decoded state or its modal coefficients, e.g. `rtol * current_scale + atol * initial_scale`, and report both residual-to-initial and residual-to-current ratios. Document whether invariance across `M` or RMS-per-equation comparability is intended.

3. **The Galerkin root tolerance is dimensionally and representation dependent.**  
   File: [blat_common.py:743](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-16-burgers2d-rom-latent-stepping/experiments/burgers2d-rom-latent-stepping/blat_common.py:743)

   The solver compares `||JD.T r||` with `tol_abs * 1e-2`, even though `JD.T r` has additional decoder-tangent/latent-scaling units. POD has an orthonormal basis; the nonlinear decoder’s latent scaling is arbitrary. The same numerical threshold therefore is not the same stopping rule across POD and auto-decoder Galerkin arms.

   **Minimal fix:** normalize the root by a documented Jacobian scale or use relative reduction from the initial root, while retaining a separately scaled residual test.

4. **Strict held-out hygiene is weaker than the comments claim.**  
   Files: [hlat_train_ad.py:58](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-16-heat2d-rom-latent-stepping/experiments/heat2d-rom-latent-stepping/hlat_train_ad.py:58), [hlat_common.py:140](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-16-heat2d-rom-latent-stepping/experiments/heat2d-rom-latent-stepping/hlat_common.py:140)

   `hlat_train_ad` says TEST is untouched, but `build_data()` always generates TEST and checks its residual during training. It does not feed the fields into training, so this is not parameter leakage, but it is not literally untouched. Also, `TEST_SEED` is environment-overridable with no guard against `TEST_SEED == SEED`.

   Stage 2 evaluates many solver/M/m/alpha variants on TEST. The code reports them all and does not select a winner, which is legitimate. Any later claim based on choosing the best of these TEST results would, however, turn TEST into model-selection data.

   **Minimal fix:** let training request train/VAL data without generating TEST; assert the test seed/parameter rows do not overlap training; select ROM hyperparameters on VAL and lock them before the final TEST evaluation.

5. **Numerical solver failures can still count as completed trajectories.**  
   Files: [blat_common.py:783](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-16-burgers2d-rom-latent-stepping/experiments/burgers2d-rom-latent-stepping/blat_common.py:783), [blat_common.py:871](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-16-burgers2d-rom-latent-stepping/experiments/burgers2d-rom-latent-stepping/blat_common.py:871)

   A Galerkin NaN step can return the unchanged, finite latent with reason `nan_step`; rollout only truncates for a nonfinite decoded field or `nan_at_init`. Jitted LSPG similarly conflates NaN steps and ordinary `lambda_max` in reason code 3. Such trajectories can be included in `n_completed` and accuracy averages.

   **Minimal fix:** distinguish NaN-step from lambda exhaustion and mark any numerical-NaN termination as failed/truncated. Report “completed horizon” separately from “all steps solved successfully.” Add finite checks for checkpoint parameters, EQ weights, and cross-check values.

6. **The POD arm is fair as a representation-isolation control, but not as a competitive optimized POD baseline.**  
   Files: [hlat_rom.py:347](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-16-heat2d-rom-latent-stepping/experiments/heat2d-rom-latent-stepping/hlat_rom.py:347), [blat_common.py:816](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-16-burgers2d-rom-latent-stepping/experiments/burgers2d-rom-latent-stepping/blat_common.py:816)

   POD uses the same nonlinear LM/Galerkin stepping machinery, which is useful for isolating representation quality. But this linear heat/POD system admits a direct reduced linear solve; iterative damped LM handicaps a production-quality POD method. Conversely, POD receives an exact orthogonal projection for its IC while the auto-decoder receives a finite-budget nonlinear fit.

   **Minimal fix:** retain this arm as “same-solver POD,” but add or clearly distinguish a direct reduced backward-Euler POD baseline. If strict end-to-end solver parity is claimed, run the same finite-budget IC algorithm for both.

7. **Checkpoint and Stage-1 manifest validation can silently mislabel or misinterpret runs.**  
   Files: [hlat_rom.py:107](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-16-heat2d-rom-latent-stepping/experiments/heat2d-rom-latent-stepping/hlat_rom.py:107), [hlat_stage1.py:51](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-16-heat2d-rom-latent-stepping/experiments/heat2d-rom-latent-stepping/hlat_stage1.py:51)

   Stage 2 does not check `ck["k_lat"]` against the checkpoint config or current `K_LAT`; the outer report can therefore claim one latent size while executing another. Stage 1 proceeds when its results manifest is absent. Different `(N_FREQ,T_FREQ)` combinations can have the same input width, so a missing manifest is not guaranteed to fail by shape.

   **Minimal fix:** validate latent dimensions and all architecture fields, including stored array shapes. Make the Stage-1 manifest mandatory or embed architecture metadata inside the checkpoint.

8. **EQ diagnostics do not establish out-of-fit quadrature reliability.**  
   File: [hlat_common.py:186](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-16-heat2d-rom-latent-stepping/experiments/heat2d-rom-latent-stepping/hlat_common.py:186)

   EQ is fitted to only 64 selected training latents. `rel_fit` is computed after independently row-scaling `G` and `b`, so it is a standardized in-sample fit, not a relative error in the physical projection norm. No validation-latent projection error, weight sum/range, or conditioning statistic is reported.

   **Minimal fix:** validate weights on disjoint TRAIN/VAL latents, report unscaled modal projection errors and weight diagnostics, and fail on nonfinite or grossly inaccurate quadrature.

# NITS/OBSERVATIONS

1. **What alpha-1 LSPG minimizes is mathematically coherent, but it is not ordinary residual LSPG.**  
   File: [hlat_common.py:277](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-16-heat2d-rom-latent-stepping/experiments/heat2d-rom-latent-stepping/hlat_common.py:277)

   Let `A = I + dt*kappa*(-L)`. With all modes and exact sums, `W=A^-1`, so

   `||W(A D(z)-D(z_n))|| = ||D(z)-A^-1 D(z_n)||`.

   Thus alpha 1 chooses the manifold state closest in Euclidean L2 to the exact backward-Euler step from the previous decoded state. That is a defensible state-error/preconditioned-residual metric for linear heat. With only `M` modes it minimizes only the low-mode part of that state error and leaves high-frequency decoder artifacts unconstrained. Alpha 0 is the ordinary discrete residual metric. These should not be described as the same LSPG objective with harmless numerical scaling.

2. **Weak-form units are internally consistent.**

   `Phi` has unit discrete 2-norm, and all targets are discrete sums, so omitting `dx²` is correct for the stated discrete Euclidean inner product. EQ weights approximate those dimensionless grid sums, not continuum area integrals. Full-grid sums and EQ therefore have the same intended units, although only approximately the same scale.

3. **`weakall` is not strictly “alpha=0 by definition.”**  
   File: [hlat_common.py:351](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-16-heat2d-rom-latent-stepping/experiments/heat2d-rom-latent-stepping/hlat_common.py:351)

   `weakall` defaults to zero, but `weakalla1` is accepted. Either force zero/reject a suffix or soften the docstring.

4. **Summary metadata can be stale or false.**  
   File: [hlat_summarize.py:7](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-16-heat2d-rom-latent-stepping/experiments/heat2d-rom-latent-stepping/hlat_summarize.py:7)

   The heading hard-codes 16 test trajectories, and the first `*.out` log is used even if a directory contains logs from multiple runs. Select the completed log corresponding to the JSON and read `n_test` from its config.

5. **JSON may contain non-standard `NaN` tokens.**  
   File: [hlat_rom.py:405](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-16-heat2d-rom-latent-stepping/experiments/heat2d-rom-latent-stepping/hlat_rom.py:405)

   Python accepts these, but strict JSON consumers do not. Serialize missing statistics as `null` plus an explicit status.

6. **Staging protects transfer integrity, not source freshness.**

   The local/remote aggregate checksum catches stale extra remote files, and rerunning into a dirty remote directory normally fails loudly. It does not detect that a stage was built before later source edits, and the logged git commit omits dirty-worktree content. Record the manifest hash and dirty/source hashes, or rebuild immediately before launch.

# Verified correct

- [heat2d_film.py:88](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-13-heat2d-coord-decoder/experiments/heat2d-coord-decoder/heat2d_film.py:88) and [hlat_common.py:163](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-16-heat2d-rom-latent-stepping/experiments/heat2d-rom-latent-stepping/hlat_common.py:163) agree on `dx=1/(n-1)`, Laplacian sign, and `u-u_prev-dt*kappa*lap(u)`.

- The FOM mask after each CG step makes wall values exactly zero; the polynomial hard-BC factor `16x(1-x)y(1-y)` is exactly zero at grid endpoints. POD wall rows are zero because the training snapshots are zero there.

- The sine-mode eigenvalue formula, grid `i/(n-1)`, unit normalization, and interior flattening order agree with the full-grid `i*n+j` extraction. Therefore the discrete identity `Phi.T L u = -diag(lambda) Phi.T u` is exact for hard-zero walls.

- For `weakall`, alpha 0, and exact grid sums, both LSPG normal equations and the Galerkin root are mathematically identical to full-grid strong form. The implemented `JD.T @ r` cross-check is the correct Galerkin identity; it simply needs enforcement and the missing LSPG checks.

- No actual held-out trajectory leakage was found in the Stage-2 ROM evolution. EQ uses training latents/fields only; cold starts use known `u0`; tolerance uses known `u0`; `kappa_test` is permitted; and `U_true` is consumed only after stepping to compute errors. Oracle per-snapshot inference is clearly separated and not reused.

- Stage 1 selects its two starts by the optimization objective, not held-out trajectory error. `z_test` and future `U_test` are used only for oracle/diagnostic/error reporting after optimization.

- Under the default `TEST_SEED=SEED+1`, TEST is a separate independent parameter draw from the seed used for TRAIN/VAL.

- The POD basis is built in host float64 from exactly the same TRAIN snapshots used by the auto-decoder. POD test projection floors use the same held-out test snapshots and the matching time-stepping variants use the same residual operators and budgets.

- The main LSPG and FOM rollout timers both use batch size one, jitted device execution, warm-ups, median repetitions, and a device synchronization. Their compile exclusion is sound for persistent closures.

- No silent float32 downcast was found in the Stage-2 auto-decoder/POD path: x64 is enabled before imports, the auto-decoder parameters and latent optimization are float64, and POD is explicitly float64. Stage 1 evaluates an originally float32 sweep checkpoint after promoting its stored weights; inherited time-feature construction still has float32 quantization, consistent with that checkpoint’s original model.

- The import ordering currently prevents the advertised `HIDDEN` collision in fresh script processes, and assertions catch mismatched heat/Burgers `N`, split sizes, seed, timestep, and sweep width.

- Cluster submission performs an explicit GPU preflight, so CPU fallback is not silent. Slurm logs are directed to the project path, not a quota-limited home directory.