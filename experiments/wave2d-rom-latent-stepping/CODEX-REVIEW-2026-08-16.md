## MUST FIX

- **A failed LSPG rollout can be reported as complete.** `n_done` correctly detects non-finite residuals/reason 5, but `complete` depends only on finite snapshot latents—not `n_done`, decoded fields, or metrics (`wlat_common.py:728-755`). A `nan_at_init` step may return the previous finite latent, continue the scan, and enter the headline averages as “complete.” Energy NaNs are then hidden by `nanmean`/`nanmax`, while per-time curves are survivor-only (`wlat_rom.py:62-90`). Fix: require `n_done == n_steps`, finite decoded fields/metrics, and finite requested energy diagnostics; invalidate every state after the first bad step. Report success-conditioned errors only alongside completion rate, and never silently drop invalid energy values.

- **The recursive-velocity energy is not, by itself, a reliable stability metric.** The current claim calls it “the stability diagnostic” and computes it from every decoded substep (`wlat_common.py:48-51`, `wlat_common.py:756-777`). For displacement error \(e_k\),
  \[
  \delta v_k=2(e_k-e_{k-1})/dt-\delta v_{k-1}.
  \]
  A time-constant bias cancels, but an isolated state error creates a persistent \(O(e/dt)\) alternating velocity, random state errors accumulate roughly as \(\sqrt{k}/dt\), and \(e_k=(-1)^k e\) causes linearly growing resonance. Thus energy drift can be dominated by decoder noise rather than ROM dynamics. Fix: retain this as a clearly labelled “kinematic-recursion energy,” but also reconstruct
  \(v_k^{dyn}=v_{k-1}+\frac{dt\,c^2}{2}(Lu_{k-1}+Lu_k)\), report its energy, and report the resulting kinematic defect. Calibrate both estimators on the same-dt FOM.

- **Weak-mode counts silently clip while the report preserves the requested, false value.** `test_modes` slices the available \((n-2)^2\) modes without validating `M` (`deps/burgers2d-rom-latent-stepping/blat_common.py:360-375`), while `make_weak_ops` records the requested `M` (`wlat_common.py:514-560`). At the default \(N=16\), only 196 modes exist, yet default `weak256` variants are labelled `M=256` (`wlat_rom.py:46-54`). Fix: assert `1 <= M <= (n-2)^2`, or explicitly replace and report `M_actual`.

- **The foundational CN-equivalence verification can be skipped while `all_ok` remains true.** V1 is only set when 80 appears in `RS_LIST` (`wlat_verify.py:60-87`), but `all_ok` defaults a missing V1 to success (`wlat_verify.py:162`). Fix: always run RS=80, require finite per-trajectory errors, gate on the maximum rather than only the mean, and make a missing V1 a failure.

- **The claimed training-data energy guard is not implemented.** `build_data` says it aborts for excessive FOM energy drift “anywhere,” but only checks the fresh test trajectories and merely checks that training displacements are finite (`wlat_common.py:182-202`). The dependency calculates and prints training drift but never returns or thresholds it (`deps/wave2d-coord-rom/wave2d_film.py:200-228`). Fix: return the training maximum drift and abort on non-finite or \(>10^{-9}\), or enforce that threshold inside `build_trajectories`.

- **Galerkin timing is understated after an early failure.** A truncated Galerkin rollout records the shortened wall time, pads the arrays, and still divides by the full requested `n_steps` (`wlat_common.py:717-733`). That incorrect value enters the reported median (`wlat_rom.py:87-88`). Fix: preserve the number of attempted steps before padding and divide by that number; exclude incomplete runs from throughput comparisons or label them separately.

- **“One job per directory” is not enforced.** The launcher checks only currently queued jobs, then reuses the remote directory with `mkdir -p` and overlays files (`cluster/launch.sh:10-17`). Once an earlier job finishes, the same cell can be submitted again, leaving old `out/` and `logs/` to be mixed with or overwritten by the new run. Fix: create the remote cell directory atomically and fail if it already exists, or require a unique immutable run ID plus a submission sentinel.

## SHOULD FIX

- **Default weak variants are weighted/preconditioned LSPG, not ordinary weak LSPG.** The factor \((1+a\lambda)^{-\text{WEAK\_ALPHA}}\), with default alpha 1, changes the nonlinear least-squares minimizer; `weakl` adds another change (`wlat_common.py:266-271`, `wlat_common.py:543-554`). The configuration records it, but variant names do not (`wlat_rom.py:46-54`). Label tables as weighted LSPG and include `WEAK_ALPHA=0` as the standard comparison.

- **The “end-to-end” speedup needs a narrower label or a true whole-pipeline timing.** It correctly adds latent rollout, jitted IC fit, and 51 full-grid decodes, but sums separately measured medians and excludes the all-substep decode/velocity/energy work used for the reported energy diagnostic (`wlat_rom.py:255-281`, `wlat_rom.py:315-321`; `wlat_common.py:756-777`). Meanwhile the FOM executable computes stored energies (`deps/wave2d-coord-rom/wave2d_film.py:162-175`). Rename it “online field-prediction speedup, diagnostics excluded,” or time one complete callable per replicate with matching requested outputs.

- **Solver diagnostics under-report rejected work and conflate numerical failures.** `_lm_while` distinguishes only the final generic `lambda_max`; repeated non-finite trial steps are not separately identified (`wlat_common.py:383-406`). Although attempts and accepts are computed, rollout discards them and reports `nJs` as “iters” (`wlat_common.py:424-437`, `wlat_common.py:731-737`). Preserve attempts/rejections and a `nan_step_lambda_max` reason; label the existing count `n_jac_evals`.

- **The data fingerprint is too weak for provenance.** Shape is stored, but Stage 2 compares only two global moments with a \(10^{-6}\) tolerance (`wlat_common.py:207-208`, `wlat_rom.py:107-112`). Small/local corruption or some configuration changes can pass. Check shape and all FOM/data configuration fields, and store a cryptographic or suitably quantized per-trajectory hash.

- **Stage 1’s residual-only arm does see the test initial condition through initialization.** The prose says it “never sees u0,” but nearest-training-IC selection uses `U_te[i,0]` (`wlat_stage1.py:7-17`, `wlat_stage1.py:139-150`). This is allowed information, not future leakage, but the ablation is “no IC term in the objective,” not “no u0.” Relabel it or use u0-independent initializations.

- **V2b is not actually the FOM’s own trajectory.** It generates states using the same Newmark implementation being checked, despite the comment claiming an FOM trajectory test (`wlat_verify.py:96-105`). Relabel it as a Newmark self-consistency check; V1 is the independent comparison against `make_rollout`.

- **The staged manifest is not enforced.** `make_cell.sh` writes `MANIFEST.sha256`, but launch compares current local files to current remote files and excludes the manifest (`cluster/make_cell.sh:47`, `cluster/launch.sh:14-16`). A locally altered stage therefore passes transport verification. Run `sha256sum -c MANIFEST.sha256` locally and remotely. Also make `pull.sh` require completed-job/`ALL-DONE` status before accepting a snapshot (`cluster/pull.sh:9-15`).

## CONSIDER

- **POD’s exact IC projection is a legitimate advantage, but not the same cold-start problem.** POD receives the analytic global least-squares projection, whereas the coordinate decoder receives a budgeted, local best-of-two LM solve (`wlat_common.py:658-679`). IC misfits are recorded (`wlat_rom.py:87-90`), which helps. Recommendation: disclose the difference explicitly and compare downstream error conditional on IC misfit rather than deliberately degrading POD.

- **EQ arms do not literally use the same collocation rule across decoders.** Cache keys include decoder kind and rank, so coordinate and POD controls learn different supports and weights from their respective reconstructed training snapshots (`wlat_rom.py:202-222`). This is defensible decoder-specific hyper-reduction, but add a matched-support arm if claiming “same collocation.”

- **Guard weak systems against rank deficiency.** Nothing asserts \(M\ge k\), and even \(M\ge k\) does not ensure `Phi_q.T @ J_D` has rank \(k\) (`wlat_common.py:509-563`). Defaults are dimensionally safe, but custom variants can be underdetermined or badly conditioned. Report the smallest singular value/condition number and reject clearly rank-deficient configurations.

## VERIFIED CORRECT

- **The u-only residual is exactly the FOM’s CN scheme.** From the FOM solve,
  \[
  u_{n+1}-u_n=dt\,v_n+a(Lu_n+Lu_{n+1}),
  \]
  and its velocity update gives
  \[
  v_{n+1}-v_n=\tfrac{dt\,c^2}{2}(Lu_n+Lu_{n+1}),
  \]
  so \(u_{n+1}-u_n=\frac{dt}{2}(v_{n+1}+v_n)\) exactly (`deps/wave2d-coord-rom/wave2d_film.py:127-142`). Combining adjacent steps yields the stated three-level residual; with \(v_0=0\), the stated first-step residual follows. The signs and \(a=(cdt/2)^2\) are correct in strong, weak, same-dt FOM, and Stage 1 (`wlat_common.py:232-256`, `wlat_common.py:467-476`, `wlat_common.py:543-554`, `wlat_stage1.py:89-98`).

- **Boundary treatment is equivalent.** The FOM uses identity boundary rows and masks both \(u\) and \(v\) (`deps/wave2d-coord-rom/wave2d_film.py:115-142`); `make_newmark_fom` does the same (`wlat_common.py:219-228`). Coordinate and POD decoders satisfy zero walls by construction (`deps/burgers2d-rom-latent-stepping/blat_common.py:196-230`), while Stage 1 explicitly adds boundary values to its residual (`wlat_stage1.py:90-98`). The off-grid strong arm intentionally uses the continuum Laplacian and is correctly labelled a control (`wlat_common.py:444-501`).

- **Velocity reconstruction is exact for an exact scheme trajectory.** Starting from \(v_0=0\), `v = 2*(u-up)/DT - v` is precisely the CN kinematic relation (`wlat_common.py:769-773`). The energy formula matches the FOM’s forward-difference invariant, including the \(dx^2\) measure (`wlat_common.py:153-160`; `deps/wave2d-coord-rom/wave2d_film.py:146-153`). The concern above applies only when decoded states do not satisfy full-grid CN exactly.

- **ROM and same-dt FOM indexing are correct.** ROM uses `DT_SNAP/RS` and exactly `NUM_STEPS*RS` substeps (`wlat_common.py:103-108`, `wlat_common.py:687-700`). `make_newmark_fom(N, RS)` uses the same dt; after constructing \(u_1\), the first interval performs `rs-1` additional steps and every later interval performs `rs`, yielding exactly 51 snapshots through \(u_{50RS}\) (`wlat_common.py:220-259`). Stage 2 constructs that reference with `wc.RS` and compares matching snapshot arrays to both references (`wlat_rom.py:134-147`, `wlat_rom.py:237-244`).

- **No held-out future state enters the Stage 2 solve path.** `U_te[:,n>=1]` is used for the same-dt/FOM reference comparison, oracle floors, and post-rollout metrics (`wlat_rom.py:134-184`, `wlat_rom.py:237-244`). Cold starts, biased collocation, and tolerance scaling use only allowed \(u_0\) information (`wlat_rom.py:117`, `wlat_rom.py:186-198`, `wlat_rom.py:224-225`). EQ snapshots, POD, and latent initializers come from training trajectories (`wlat_rom.py:113-115`, `wlat_rom.py:159-164`, `wlat_rom.py:200-221`).

- **The auto-decoder and POD basis do not train on TEST or VAL.** Training uses only `U[:N_TRAIN]`; VAL is used solely for reported projection floors, and the fresh TEST data returned by `build_data` is not consumed by training (`wlat_train_ad.py:60-83`). POD and latent initialization are fitted from the same training snapshots (`wlat_train_ad.py:67-93`).

- **Core timing mechanics are otherwise sound.** The FOM and LSPG rollout are jitted, warmed twice, measured by median, and explicitly blocked (`wlat_rom.py:255-268`, `wlat_rom.py:299-314`; `deps/burgers2d-rom-latent-stepping/blat_common.py:892-901`). `state_of` remains inside the compiled latent scan (`wlat_common.py:419-438`). The coordinate IC solver is genuinely returned as a jitted function and is directly timed, not estimated (`wlat_common.py:658-679`, `wlat_rom.py:270-276`). The FOM baseline is batch one on the same reported JAX backend.

- **POD full-form comparability is sound.** Its basis is fitted only on training snapshots (`wlat_train_ad.py:67-74`), matching variants call the same operators/solver/budget, and both arms use the same true-\(u_0\) tolerance scale (`wlat_rom.py:227-245`, `wlat_rom.py:287-297`). Strong `tol_scale` is identical for equal collocation size, and weak `tol_scale` is identical for equal grids (`wlat_common.py:506`, `wlat_common.py:559-560`).

- **Weak-form normalization and quadrature units are correct.** The product sine norm is \((n-1)/2\), exactly the `modes_at` normalization (`deps/burgers2d-rom-latent-stepping/blat_common.py:360-389`). Exact targets and NNLS candidates both represent unscaled discrete sums, so the EQ weights absorb node-count measure consistently (`wlat_common.py:281-311`). Current and carried projections both use the same `Phi_q` and quadrature rule (`wlat_common.py:520-554`).

- **V3 is set up correctly.** It sets `WEAK_ALPHA=0` before tracing/evaluating the all-mode weak operator, uses every interior mode, and compares the weak and strong 2-norms on a non-solution (`wlat_verify.py:107-132`). With orthonormal \(\Phi\), equality of norms and LSPG minimizers holds only in this unweighted all-mode case, exactly as asserted.

- **Other requested mechanics are correct.** `traj_metrics` implements the stated mean-time norm divided by trajectory RMS normalization (`wlat_common.py:163-170`); the strong-form Galerkin tangent reshape selects stencil/point column 0, which is the center value \(u\) (`wlat_common.py:604-609`); full-grid energy decoding includes every latent substep and snapshot selection includes both endpoints (`wlat_common.py:738-773`); x64 is enabled before numerical construction (`wlat_common.py:66-101`). GPU preflight exits 42, Slurm logs target the project filesystem, transport/pull checksums are compared, and no logs are directed to cluster home (`cluster/make_cell.sh:19-41`, `cluster/launch.sh:14-16`, `cluster/pull.sh:10-14`).