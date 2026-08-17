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

## Cluster runs (Tufts, 2026-08-17, commit `90171ca`, `dirty=0`)

| cell | job | GPU | `jax_backend` | what |
|---|---|---|---|---|
| `s1_n64` | 2482867 | A100-PCIE-40GB | gpu | Stage 1, N=64, 16 test, budget 100, `IC_W` sqrt(50) and 1 |
| `ad_n64_k4` | 2482870 | A100-PCIE-40GB | gpu | auto-decoder K=4 (60k steps, batch 128, `P_SUB` 2048) + the full 19-variant ROM study |
| `ad_n64_k8` | 2482872 | **A100 80GB PCIe (pax007)** | gpu | K=8 |
| `ad_n64_k16` | 2482874 | A100-PCIE-40GB | gpu | K=16 |
| `ad_n128_k8` | 2482880 | **A100 80GB PCIe (pax007)** | gpu | K=8 at N=128 (N-flatness at fixed K, M, m) |

All five: `JAX_DEFAULT_MATMUL_PRECISION=highest`, f64, data regenerated from the seed on the
cluster (max FOM rel. residual 1.0e-10 = the FOM's own CG tolerance), one isolated job
directory each, checksum-verified pull, cluster directories deleted. Raw JSONs, checkpoints and
logs under `runs/`; every table below is regenerated by `hlat_summarize.py` into
`SUMMARY_TABLES.md`. **`ad_n64_k8` and `ad_n128_k8` landed on the same physical node
(`pax007`, A100 80GB PCIe)**, so the N=64 -> N=128 comparison is like-for-like; the K=4 and
K=16 cells ran on 40 GB A100s, so absolute times across those cells are not comparable (ratios
within a cell are).

### Identity checks (asserted, every cell)

| check | K=4 | K=8 | K=16 | N=128 K=8 |
|---|---|---|---|---|
| point-local BE residual vs the FOM's own operator, max abs | 1.9e-16 | 1.9e-16 | 1.9e-16 | 2.2e-16 |
| `weakall`(alpha=0) vs strong-form full grid, `\|\|r\|\|` rel dev | 6.2e-16 | 4.9e-15 | 5.0e-15 | 2.1e-14 |
| ... `J^T r` | 3.2e-14 | 1.3e-14 | 1.2e-14 | 4.6e-14 |
| ... `J^T J` | 6.4e-15 | 5.6e-15 | 1.9e-14 | 1.7e-14 |
| ... Galerkin `J_D^T r` | 2.0e-15 | 2.7e-15 | 1.4e-15 | 2.7e-15 |
| decoder value on the wall nodes, max abs | 0.0 | 0.0 | 0.0 | 0.0 |
| FOM trajectory residual through the FOM's own residual | 7.6e-11 | 7.6e-11 | 7.6e-11 | 9.7e-11 |

The `weakall` identity also holds *end to end*: `lspg:full:weakall` reproduces `lspg:full:fd`
and `galerkin:full:weakall` reproduces `galerkin:full:fd` to **all printed digits of the
trajectory error and of the iteration counts** in every cell (e.g. K=8: 3.348e-02 / 22.1 cold /
8.48 warm for both). The FOM residual floor here is 1e-10, not the 1e-13 of the Newton-based
Burgers FOM — it is set by this testbed's CG tolerance (`CG_TOL=1e-10`).

**Zero blow-ups anywhere**: 0/16 in every one of the 19 coordinate variants x 3 K values, the
11 variants at N=128, and all 20 POD variants per cell. Every step terminates on the stall
criterion (795/800 steps at K=8; 5 on budget), i.e. at the decoder's own residual floor
(mean `||r||/||r(z_n)||` = 4.5e-2) — the tolerance never binds.

### Stage 1 — space-time LSPG on the existing (z,t) sweep decoder (N=64, 16 fresh test trajectories)

| arm | `IC_W` | traj rel-L2 mean | median | max | mean \|z − z*\| |
|---|---|---|---|---|---|
| oracle (true z) | — | **9.85e-3** | | 7.26e-2 | 0 |
| `ic` (IC misfit only) | — | 1.14 | 5.62e-1 | 4.88 | 0.63 |
| `resid` (space-time BE residual only; never sees u0) | — | 8.19e-1 | 8.70e-1 | 1.11 | 1.33 |
| `both` | **sqrt(50)** | **3.31e-2** | **2.59e-2** | 1.37e-1 | **3.68e-2** |
| `both` | 1 | 3.91e-2 | 3.28e-2 | 1.51e-1 | 4.33e-2 |

Same qualitative result as Burgers and Poisson: the space-time residual **plus** the known IC
identifies the 5 true parameters (to 3.7e-2) and lands within 3.4x of the oracle, while IC
alone is badly under-determined (kappa is invisible at t=0) and residual alone collapses onto
the low-amplitude branch. Unlike Burgers, the RMS-balanced `IC_W = sqrt(50)` beats `IC_W = 1`
here. The gap to the oracle is explained by the sweep decoder's own PDE inconsistency: its
residual at the *true* z is already 2.4e-2 relative.

### Stage 2 — auto-decoder latent-stepping ROM, N=64 (16 fresh test trajectories, 50 steps)

Trajectory rel-L2 = mean over the 51 slices, mean over the 16 test trajectories.

| | K=4 | K=8 | K=16 |
|---|---|---|---|
| auto-decoder TRAIN recon at learned latents | 7.35e-3 | 6.20e-3 | 5.90e-3 |
| **ORACLE held-out inferred latents** (per snapshot, budget 40) | 2.04e-2 | **1.16e-2** | **6.71e-3** |
| IC fit (cold start from the known u0; = the ROM at t=0) | 7.06e-2 | 4.14e-2 | 2.32e-2 |
| `lspg:full:fd` (strong FD residual, full grid) | 5.97e-2 | 3.35e-2 | 1.58e-2 |
| `galerkin:full:fd` | 5.09e-2 | 2.50e-2 | 1.11e-2 |
| `lspg:full:weak16` | **3.30e-2** | 2.50e-2 | 9.04e-2 ⚠ |
| `lspg:full:weak32` | 3.67e-2 | 2.05e-2 | 2.07e-2 |
| **`lspg:full:weak64`** | 3.96e-2 | **1.87e-2** | 1.07e-2 |
| `lspg:full:weak128` | 4.19e-2 | 1.90e-2 | 9.86e-3 |
| `lspg:full:weak256` | 4.31e-2 | 2.15e-2 | **9.73e-3** |
| `lspg:full:weak64a0` (alpha = 0) | 4.24e-2 | 1.95e-2 | 1.10e-2 |
| `galerkin:full:weak64` | 3.96e-2 | 1.87e-2 | 1.07e-2 |
| `lspg:eq256:weak64` (NNLS-EQ, 256 grid nodes) | 3.93e-2 | 1.96e-2 | 1.14e-2 |
| `lspg:eq512:weak64` | 3.96e-2 | 1.92e-2 | 1.05e-2 |
| `lspg:eqoff256:weak64` (**meshfree** pool, 256 pts) | 3.94e-2 | 1.97e-2 | 1.10e-2 |
| `lspg:eqoff512:weak64` | 3.95e-2 | 1.88e-2 | 1.07e-2 |
| `lspg:eq512:weak256` / `eq1024:weak256` | 4.29e-2 / 4.30e-2 | 2.19e-2 / 1.97e-2 | 9.79e-3 / 9.75e-3 |
| `lspg:full:weakc64` (continuum eigenvalues) | 3.97e-2 | 1.87e-2 | 1.06e-2 |
| `lspg:full:weakall` = `lspg:full:fd` (exactness check) | 5.97e-2 | 3.35e-2 | 1.58e-2 |
| POD same-solver, `lspg:full:weak64`, k = 6/8/16/32/64 | 1.81e-1 / 1.29e-1 / 6.17e-2 / 2.32e-2 / 1.42e-1 ⚠ | (same) | (same) |
| POD same-solver, `lspg:full:fd`, k = 6/8/16/32/64 | 1.88e-1 / 1.37e-1 / 6.79e-2 / 2.58e-2 / 7.30e-3 | (same) | (same) |
| POD projection floor (test), k = 6/8/16/32/64 | 1.81e-1 / 1.29e-1 / 6.13e-2 / 2.26e-2 / 6.33e-3 | | |

Five things this says:

1. **The weak form beats the exact discrete residual.** At K=8, `weak64` (1.87e-2) is **1.8x
   better** than `weakall` = the exact full-grid strong-form residual (3.35e-2) — same manifold,
   same solver, same stopping rule, and (by the asserted identity) literally the same residual
   vector up to an orthogonal rotation and the `alpha` weighting. Against the *best* strong-form
   arm, `galerkin:full:fd`, the margin is smaller: 1.34x at K=8 (1.87e-2 vs 2.50e-2), 1.54x at
   K=4, and only **1.14x at K=16** (9.73e-3 vs 1.10e-2). The effect is consistent in mean,
   median and max at every K, and it cannot be a discretisation artefact, because on this linear
   PDE the weak form is *exact*.
   **What we did not show:** the causal story usually offered — "the low-mode projection discards
   the high-frequency residual the K-dimensional tangent space cannot fix" — is *consistent with*
   these numbers but is **not demonstrated** by them. Two other explanations survive: the
   `alpha`-weighting changes the metric (though `weak64a0` at alpha=0 keeps most of the gain, so
   this is at most a small part), and the projected Jacobian `Phi^T J` is far better conditioned
   at M=64 than the full `J`, which changes the LM trajectory and where it stalls (warm iteration
   counts drop from 8.5 to 6.3 at K=8). Separating those would need a residual-spectrum
   diagnostic that this study does not have.
2. **M must comfortably exceed K, and the best M grows with K.** The argmin moves M=16 (K=4) ->
   M=64 (K=8) -> M=128–256 (K=16), with the error monotone in M on either side of the argmin at
   every K. **Three K values are not a scaling law** — read this as a trend and a sizing rule of
   thumb (M of order 10K), not as `M ~ 16K`. ⚠ The two entries marked above are the failure mode:
   `weak16` at K=16 (M = K, 9.0e-2) and POD `weak64` at k=64 (M = k, 1.42e-1) — when the
   number of test modes equals the number of unknowns the Petrov–Galerkin system becomes
   square, the least-squares regularisation disappears, and the step can chase an exact root of
   a rank-deficient projection. **Use M >= 4K.**
3. **Hyper-reduction is essentially free.** `eq256` (256 of 3844 interior nodes, 15x fewer)
   costs 5% of accuracy at K=8 and is 5x cheaper per step (3.2 vs 16.0 ms). The **meshfree**
   pool matches the grid pool to within 1% (1.97e-2 vs 1.96e-2 at m=256) — and unlike Burgers,
   where the upwind operator forced meshfree quadrature onto the *continuum* residual, heat's
   weak form needs no stencil, so the meshfree rule quadratures the **exact FOM operator**.
   Out-of-fit EQ diagnostics (disjoint training latents): relative mode-projection error
   9.4e-4 to 3.4e-3, all weights positive and finite.
4. **Galerkin and LSPG coincide once the residual is projected.** In the strong form Galerkin
   is clearly better (K=8: 2.50e-2 vs 3.35e-2; K=16: 1.11e-2 vs 1.58e-2; N=128: 2.01e-2 vs
   3.78e-2), but with `weak64` the two agree to 3 digits at every K and at N=128. The low-mode
   projection already performs the filtering that the Galerkin test space performs.
5. **The nonlinear manifold buys ~7x in latent dimension.** At equal latent size k=8 the
   coordinate ROM is 1.87e-2 against POD's 1.29e-1 — **6.9x better**. POD needs k ≈ 40 to
   reach the coordinate ROM's K=8 accuracy. Note the honest other side: POD at k=64 reaches
   6.8e-3, better than the coordinate ROM at K=16, and same-solver POD essentially attains its
   own projection floor at every k (so the POD arm is not being handicapped by the solver).
   `alpha = 1` (the `A^-1`-weighted / state-error metric) beats `alpha = 0` consistently but
   only by 3–7%.

Per-time trajectory error (t-index 0/10/20/30/40/50), K=8:

| arm | t=0 | 10 | 20 | 30 | 40 | 50 |
|---|---|---|---|---|---|---|
| oracle inferred latents | 4.14e-2 | 1.68e-2 | 1.04e-2 | 7.16e-3 | 5.72e-3 | 4.96e-3 |
| `lspg:full:weak64` | 4.14e-2 | 2.37e-2 | 1.77e-2 | 1.50e-2 | 1.38e-2 | 1.29e-2 |
| `lspg:eq256:weak64` | 4.14e-2 | 2.48e-2 | 1.88e-2 | 1.59e-2 | 1.46e-2 | 1.37e-2 |
| `galerkin:full:fd` | 4.14e-2 | 2.82e-2 | 2.50e-2 | 2.30e-2 | 2.18e-2 | 2.09e-2 |
| `lspg:full:fd` | 4.14e-2 | 3.66e-2 | 3.42e-2 | 3.22e-2 | 3.09e-2 | 2.99e-2 |

Every arm starts at the same value (the cold-start IC fit, 4.14e-2, which is *worse* than the
oracle at t>0), then **improves** with time: the heat solution smooths, so the manifold
represents it better as the rollout proceeds. No arm diverges. The ROM tracks 2.6x above the
oracle floor at t=T and the gap is flat in time.

### N-flatness: N=64 -> N=128 at fixed K=8, M=64, m=256 (same GPU, same node)

| | N=64 | N=128 | ratio |
|---|---|---|---|
| FOM dof | 4 096 | 16 384 | 4x |
| FOM rollout (jitted CG, batch 1, 50 steps) | 49 ms | 121 ms | **2.5x** |
| `lspg:full:weak64` (all 3844 / 15876 interior nodes) step | 16.0 ms | 62.6 ms | 3.9x |
| **`lspg:eq256:weak64` step** | 3.2 ms | 3.5 ms | **1.09x** |
| **`lspg:eq256:weak64` rollout** | 119 ms | 120 ms | **1.008x** |
| `lspg:eq256:weak64` traj rel-L2 | 1.96e-2 | 1.83e-2 | 0.93x |
| oracle inferred-latent floor | 1.16e-2 | 1.20e-2 | 1.03x |
| rollout-only speedup vs FOM | 0.41x | **1.01x** | |

This is the cleanest result in the study: with the weak form hyper-reduced to m=256 quadrature
points the online cost is **independent of the FOM dimension** (1.008x for a 4x refinement)
while the accuracy is unchanged, and the FOM cost grows — so the rollout crosses break-even at
N=128. The non-hyper-reduced weak form scales with n, as expected (3.9x).

### Timing and the honest speed verdict (N=64, K=8, A100 80GB; median of 7 after 2 warm-ups, `block_until_ready`)

| stage | time | note |
|---|---|---|
| FOM: same jitted implicit CG solver, batch 1, 50 steps, 51 output slices | **49 ms** | ~1 ms/step on 4096 dof |
| ROM cold start, jitted on-device LM (nearest-train-IC search + 2 LM solves, best-of) | 156 ms | **dominates** |
| ROM cold start, Python-loop LM (same rule, prebuilt jits) | 236 ms | only 1.5x slower — the cost is real device work, not Python overhead |
| ROM rollout, `eq256:weak64`, device `lax.scan` | 119 ms | 3.2 ms/step |
| ROM decode of all 51 slices | 10 ms | |
| **ROM end to end** (cold start + rollout + decode, one realizable pipeline) | **287 ms** | **0.17x vs FOM** |
| direct reduced POD-Galerkin, k=8 (50 k x k solves) | **3.5 ms** | **14.3x vs FOM**, at 1.31e-1 |
| direct reduced POD-Galerkin, k=64 | 9.0 ms | **5.5x vs FOM**, at 6.8e-3 |

At N=128 the same ladder is FOM 121 ms, ROM end-to-end 0.15x, POD-direct k=8 **35x** / k=64
**13x**.

**Verdict.** The recipe *transfers* on every axis except speed:

- **Accuracy**: yes. The weak-form Galerkin objective with M ~ 16K low sine modes beats the
  exact discrete residual by 1.8x, hyper-reduces to m ~ 4M points for a 5% accuracy cost, works
  meshfree, is insensitive to discrete-vs-continuum eigenvalues, never blows up, and lands
  1.5x (K=8) to 1.6x (K=16) above the held-out oracle floor of its own manifold.
- **vs the published heat ROM** (`fix/heat-rollout-warm-start`, rel_l2 **2.83e-2 at 0.17x**):
  beaten on accuracy at *exactly* the same speedup — `lspg:eq256:weak64` at K=8 gives
  **1.96e-2 at 0.17x** (1.4x better), and `lspg:full:weak64` at K=8 gives **1.87e-2** (1.5x
  better), `weak256` at K=16 gives **9.73e-3** (2.9x better). Note `main`'s public heat ROM
  rollout is frozen after step 1 (a real bug), so only the FOM is reused from the public
  package here.
- **Cost scaling**: yes, and this is the strongest single number — hyper-reduced online cost is
  flat in n (1.008x for 4x the dof), *for the rollout*. Two mesh sizes on one GPU with one
  variant is evidence of the mechanism (nothing in the hyper-reduced step touches n: m=256
  decoder evaluations and an M x K least-squares solve), not a measured scaling curve; and the
  **end-to-end** pipeline is not flat, because the cold start still scales with n (156 ms ->
  640 ms). The flatness claim applies to the rollout only, and the prose above says so.
- **Wall-clock speedup vs this FOM**: **no, and it should not be expected to be.** Heat 2D is
  linear, so the correct ROM is the `k x k` reduced operator `V^T A_kappa V`, which is 5.5–38x
  faster than the FOM and at k=64 is also *more accurate* than the nonlinear ROM at K=16. A
  neural decoder cannot compete with that: one decoder evaluation at 256 points already costs
  more than a 64x64 reduced solve. The nonlinear manifold's value on this problem is purely
  **accuracy per latent dimension** (6.9x at equal k), which matters when the latent has to be
  small (control, inverse problems, downstream conditioning) — not throughput.

### Statistical honesty (n_test = 16)

The per-trajectory distribution is heavy-tailed (K=8, `weak64`: mean 1.867e-2, median 1.461e-2,
max 4.96e-2), so with 16 trajectories only differences of roughly 20% or more that move mean,
median **and** max in the same direction should be read as real. On that standard:

- **Real**: weak-M vs `weakall`/`fd` (1.8x, all three statistics); the `M = K` collapse
  (9.0e-2 with a 6.4e-1 worst case at K=16); Galerkin vs LSPG in the **strong** form; the
  coordinate ROM vs POD at equal k (6.9x); the K-ladder (2.04e-2 -> 1.16e-2 -> 6.71e-3 floors
  and 3.30e-2 -> 1.87e-2 -> 9.73e-3 ROM).
- **Ties, and described as such**: `weak64` vs `weak128` at K=8 (1.867e-2 vs 1.902e-2, 2%);
  `weak128` vs `weak256` at K=16; LSPG vs Galerkin under `weak64` (0.3%); `eq` vs `eqoff` at
  equal m (<1%); discrete vs continuum eigenvalues (<0.2%); alpha=1 vs alpha=0 (3–7% — a
  consistent sign at all three K, but each individually within noise).
- The `eq256` penalty over the full grid (5%) is at the edge; it is consistent in sign at all
  three K and at N=128, which is why it is stated as "about 5%" rather than as a tie.
- All tables use the same statistic throughout (mean over 51 slices, then mean over 16
  trajectories) with the median shown alongside; no arm is quoted on a median where another is
  quoted on a mean.
- **Limitation**: `hlat_rom.py` stores only the aggregates per variant, not the 16
  per-trajectory values, so *paired* significance tests cannot be recomputed from the archived
  JSONs. The judgements above are from the mean/median/max triple.

### Caveats

- **The end-to-end speedup is cold-start-bound, not rollout-bound.** The IC latent solve fits
  `u0` on the *full* `n^2` grid with a budget-100 LM; it is 156 ms against a 119 ms rollout at
  N=64 and 640 ms at N=128 (where it is 5x the rollout). It is the one part of the online path
  that was never hyper-reduced. Projecting the IC misfit onto the same M modes / m quadrature
  points is the obvious next step and would move the N=128 end-to-end number from 0.15x to
  roughly rollout-bound (≈1x). This is a real limitation of the numbers reported here, not a
  claim about the method's ceiling.
- **The FOM baseline is strong.** It is a fully jitted, batched CG on a 5-point Laplacian with
  a warm start from the previous step (`x0=u_prev`) and a 1e-10 tolerance — about the fastest
  reasonable heat solver on a GPU. A general sparse direct solver would make every ROM look
  much better; we deliberately did not use one.
- **Absolute times are comparable only within a cell.** `ad_n64_k8` and `ad_n128_k8` shared one
  node (pax007, A100 80GB PCIe), so the N-flatness table is like-for-like; the K=4 and K=16
  cells ran on 40 GB A100s.
- **No variant was selected on TEST — but the headline numbers quote the best variant per K,
  which is the same thing done informally.** The full 19-variant x 3-K grid is reported so the
  reader can see the whole surface, and the *qualitative* conclusions (weak > strong, M >> K,
  hyper-reduction cheap, meshfree == grid, Galerkin == LSPG under projection) hold across the
  whole grid rather than at one point. But "best ROM 1.87e-2 at K=8" is a maximum over 19
  correlated arms evaluated on the same 16 trajectories and is therefore optimistically biased.
  A publishable headline needs the configuration locked on VAL and re-measured once on TEST;
  this study did not do that.
- **The comparison to the published `2.83e-2 @ 0.17x` is cross-code, not a re-run.** That
  number comes from `fix/heat-rollout-warm-start` — different code, its own `heat2d_n64`
  configuration and test split, and its own timing protocol on unknown hardware. It is quoted
  as the target because it is the project's best published heat ROM, but "beaten at exactly the
  same speedup" should be read as "in the same ballpark on both axes, better on accuracy", not
  as a controlled head-to-head. Re-running that branch's ROM inside this harness would be
  required for a real comparison.
- **`M >= 4K` is a rule of thumb inferred from two failure points** (`weak16` at K=16, POD
  `weak64` at k=64), both of which have M exactly equal to the number of unknowns. The
  *mechanism* (a square Petrov–Galerkin system loses the least-squares regularisation) is clear;
  the specific factor 4 is not measured — we have no data between M=K and M=2K.
- Stage 1's gap to the oracle is limited by the sweep decoder's own PDE inconsistency
  (residual 2.4e-2 at the true z), not by the solver.
- The FOM residual floor is 1e-10 (this testbed's CG tolerance), so "converged data" here means
  1e-10, not the 1e-13 of the Burgers testbed.

