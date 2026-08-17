# Heat-2D INR-decoder latent-stepping ROM (2026-08-16)

Port of the Burgers-2D latent-stepping ROM (`exp/2026-08-16-burgers2d-rom-latent-stepping`)
to the **linear** Heat-2D testbed (`exp/2026-08-13-heat2d-coord-decoder`), using the recipe
established by the Poisson objective study (`exp/2026-08-16-poisson2d-rom-objective`):
hard-BC FiLM coordinate auto-decoder + warm-started LM latent stepping on a **weak-form
Galerkin residual** against low sine test modes, hyper-reduced by capped Lawson–Hanson
NNLS quadrature ("EQ") fitted on decoder-output snapshots.

Heat is the cleanest case in the series: the equation is **linear**, so the weak form is
*exact* — projecting the backward-Euler residual on the discrete sine eigenvectors turns the
Laplacian into a multiplication by `-lam_i` and **no stencil, derivative or nonlinear term
has to be evaluated at all**. Only the decoder *output* at the quadrature nodes is needed.

Everything numeric is f64. The ROM knows `u0`, the diffusivity `kappa` (a legitimate family
parameter), the PDE and the training set — never the held-out trajectory.

## Design

| piece | what |
|---|---|
| FOM | `heat2d_film.make_rollout` imported verbatim: `u_t = kappa lap u` on `[0,1]^2`, `u=0` walls, backward Euler `dt=0.005 x 50`, CG per implicit step (`tol 1e-10`). Its implicit operator `A_kappa u = u - dt kappa L u` (interior rows; identity on the walls) defines the residual the ROM steps: `R_n(u) = A_kappa u - u_n`. |
| family | `z = (cx, cy, w, a, log kappa)`, `kappa in [0.01, 0.5]`. TRAIN = first 512 trajectories of the seed-0 draw (identical to the sweep). **TEST = 16 fresh trajectories from `TEST_SEED=1`** — the sweep checkpoints were model-selected on the VAL split, so VAL is not a test set. Data are regenerated on the cluster from the seed; `build_data` aborts if any FOM trajectory has rel. residual > 1e-8 through the FOM's own implicit operator (observed ~1e-10, set by the FOM's own CG tolerance of 1e-10 — this testbed does not reach the 1e-13 of the Newton-based Burgers FOM). |
| decoder (Stage 2) | `u(x;z) = eps * b(x) * FiLM(x;z)`, hard Dirichlet factor `b = 16x(1-x)y(1-y)`, `ms_parametric` FiLM net (5x256, `n_freq` Nyquist-capped), trained as an **auto-decoder**: one latent per (trajectory, time) snapshot, latents initialised from the top-K POD coefficients (no true `z` anywhere), inverse-energy row weights, lazy per-row Adam on the latents, time-smoothness penalty. `hlat_train_ad.py`. |
| POD control | same solver on `u = V c`, `V` from the same TRAIN snapshots (spatial Gram, host f64), `k in {6,8,16,32,64}`. |
| residual (strong form, control) | point-local BE residual at m interior nodes from the decoder at the m 5-point stencils; `be_residual_from_stencil == ` the FOM interior residual (asserted every run). |
| residual (weak form, the recipe) | test modes = M lowest **discrete** sine modes `phi_i`, exact eigenvectors of the FOM's ghost-zero 5-point Laplacian (`-L phi_i = lam_i phi_i`, `lam = 4/dx^2 (sin^2(pi kx/(2(n-1))) + sin^2(pi ky/(2(n-1))))`). Because heat is linear: `R_i(z) = w_i [ phi_i^T (u - u_n) + dt kappa lam_i phi_i^T u ]`, `w_i = (1 + dt kappa lam_i)^-alpha`, `alpha = 1` — **exact**, needs only `phi_i^T u`. `weakc<M>` uses the continuum eigenvalues `pi^2(kx^2+ky^2)` instead (O(h^2) sensitivity arm). |
| hyper-reduction | `eq<m>` / `eqoff<m>`: capped Lawson–Hanson NNLS quadrature weights on interior grid nodes / a 4096-point meshfree random pool, fitted to reproduce the mode projections `phi_i^T u` of DECODER-OUTPUT snapshots at 64 training latents. Because the heat weak form needs no stencil, the meshfree pool is available for the **exact** discrete form too (unlike Burgers, where the upwind operator forced `eqoff` onto the continuum form). |
| solvers | `lspg`: LM on `||R||` (on-device `lax.while_loop`; whole rollout as one `lax.scan` for timing). `galerkin`: damped Newton on `J_D^T R = 0` (weak form: test functions = mode-projected tangents `Phi^T dD/dz`), backtracking on `||g||`. Warm start `z_n`, budget 30 attempts/step. Cold start `z_0` = best-of LM data-misfit to the **known u0**; both a Python LM and a **jitted on-device LM** (`make_ic_solver_jit`) are run and timed. |
| stopping rule | `||r|| <= max(GN_RTOL·||r(z_n)||, GN_TOL·rms(u0)·tol_scale)`, `GN_RTOL=1e-6`, `GN_TOL=1e-9`. The Burgers reference used the absolute term alone. Two reasons to change it, one measured and one structural: (i) heat decays over the horizon — computed exactly in the discrete sine basis over the 16 test trajectories, `||u_50||/||u_0||` is 5.7e-2 (min) / 2.8e-1 (median) / 7.7e-1 (max) — so a `u0`-pinned absolute tolerance is up to ~17x looser in relative terms at `t=T` (not the >1e3 the review feared, but not nothing); (ii) more importantly it is **not comparable across `M`**: an `M`-mode projection of an isotropic residual carries only `sqrt(M/(N-2)^2)` of its norm, making the same absolute threshold ~7.8x easier at `M=64` than at `weakall`. The Galerkin root uses the matching relative criterion `||g|| <= GN_RTOL·||g_0||`. Both `||r||` and `||r||/||r(z_n)||` are reported. |
| POD, second control | **direct reduced POD-Galerkin**: for a linear PDE a real POD ROM never runs LM — `V^T A_kappa V` is `k x k`, so the whole rollout is 50 `k x k` solves. This arm is the honest linear competitor on speed; the same-solver POD arm above is the *representation* control. |
| what alpha=1 actually minimises | with `A = I + dt·kappa·(-L)` and all modes with exact sums, `W = A^-1`, so `||W(A·D(z) - D(z_n))|| = ||D(z) - A^-1 D(z_n)||`: alpha=1 LSPG picks the manifold state **closest in Euclidean L2 to the exact backward-Euler step** from the previous decoded state — a state-error metric, not the residual norm. With only `M` modes it constrains the low-mode part of that state error and leaves high-frequency decoder artefacts free. `alpha=0` (`weak64a0`) is the ordinary discrete-residual metric and is reported as its own arm. |
| exactness cross-check | `weakall` = ALL `(N-2)^2` modes with `alpha=0`: `Phi` is then square-orthogonal, so `r_weak = Phi^T r_fd` and both the LM normal equations (`J^T J`, `J^T r`) and the Galerkin gradient (`J_D^T Phi Phi^T r_fd`) are identical to the strong-form full-grid ones. Verified numerically in `report["checks"]` and again as a full rollout variant. |
| Stage 1 pilot | `hlat_stage1.py`: the heat sweep's `(z,t)`-conditioned decoder (5-dim true-parameter latent, N=64 checkpoint); solve for `z` by LM on (a) IC misfit only, (b) the space-time BE residual over all 50 steps (interior + FOM boundary rows; a PDE-consistency ablation that never sees `u0`), (c) both (`IC_W` = sqrt(50) RMS-balanced, and 1). |
| metrics | trajectory rel-L2 = mean over the 51 slices of `||u_ROM - u_FOM||/||u_FOM||`, mean/median/max over 16 test trajectories over **completed** rollouts (blow-ups counted separately, never averaged in); per-time curves; Jacobian evaluations and LM attempts, cold step 0 vs warm; ROM-vs-FOM timing: warm-up 2, median of 7, `block_until_ready`, same device/process, FOM = the same jitted implicit CG solver at batch 1; rollout-only plus IC fit (Python **and** jitted) and 51-slice decode timed separately and composed into an end-to-end speedup. |
| oracle floors (labelled) | POD projection of the test trajectories; per-snapshot LM-inferred latents on the held-out field (the auto-decoder's held-out representation error = the latent-stepping floor). |

Files: `hlat_common.py` (heat operators, weak form, NNLS-EQ, objective parser, jitted IC
solve; all PDE-agnostic machinery imported from `blat_common`), `hlat_train_ad.py`,
`hlat_rom.py`, `hlat_stage1.py`, `hlat_summarize.py`,
`cluster/{make_cell.sh,launch.sh,hlat_cells.sh}`, `CODEX-REVIEW-2026-08-16.md` (adversarial
review run before the fan-out; every MUST and almost every SHOULD applied — see its triage
table).

## Published comparison

The public `main` heat ROM rollout is **frozen after step 1** (a real bug: it never warm-starts
the next step). The fixed version on branch `fix/heat-rollout-warm-start` gives
**rel_l2 = 2.83e-2 at 0.17x speedup** for `heat2d_n64` — that is the number this experiment
has to beat. Only the FOM is reused from the public package here; the public ROM solver is not.

<!-- RESULTS -->
