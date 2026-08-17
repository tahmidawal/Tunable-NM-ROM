# Wave-2D INR-decoder latent-stepping ROM (2026-08-16/17)

A real reduced-order model on the FiLM **coordinate** decoder for the Wave-2D
testbed (`exp/2026-08-14-wave2d-coord-rom`), ported from the Burgers-2D round
(`exp/2026-08-16-burgers2d-rom-latent-stepping`) and the Poisson objective study
(`exp/2026-08-16-poisson2d-rom-objective`).  The manifold is the nonlinear
coordinate net; the online solve is latent time stepping by warm-started
Gauss–Newton / Levenberg–Marquardt on **the discrete Crank–Nicolson residual of
the FOM**; hyper-reduction is the weak-form Galerkin + NNLS-EQ recipe.

Wave is the case the recipe had not been tried on: **hyperbolic and
dissipation-free**.  Nothing damps the modes the decoder gets wrong, so
projection error does not decay — energy drift and long-horizon error growth are
first-class results here, not footnotes.

Everything numeric is f64.  The ROM knows the initial condition `u0`, that
`u_t(·,0)=0`, the wave speed `c`, the PDE, and the training set — never the
held-out trajectory.

## Design

| piece | what |
|---|---|
| FOM | `wave2d_film.make_rollout` imported verbatim: `u_tt = c^2 lap u`, `u=0` walls, first-order `(u,v)` system, **Crank–Nicolson**, 5-point Laplacian, CG (tol 1e-10) per implicit step, `SUBSTEPS=80` sub-steps per stored snapshot (`dt_FOM = 2.5e-4`), 51 stored snapshots over `T=1`. Discrete energy `E = dx^2[0.5||v||^2 + 0.5 c^2 u^T(-L)u]` conserved to ~1e-12. |
| ROM step | **the decoder represents `u` only.** `v` is eliminated exactly by the trapezoidal relation `v_{n+1} = 2(u_{n+1}-u_n)/dt - v_n`, turning CN into the three-level Newmark (average-acceleration) residual in `u`: `R(u) = (u - 2u_n + u_{n-1}) - a L(u + 2u_n + u_{n-1})`, `a = (c dt/2)^2`; first step (`v_0 = 0`): `R(u) = (u - u_0) - a L(u + u_0)`. Carried state = the previous **two** latents' summaries. Verified equivalent to the FOM's CN operator both algebraically (Codex) and numerically (V1/V2 below). |
| ROM time step | `dt = DT_SNAP / RS`. `RS = 80` **is** the FOM's operator; smaller `RS` adds a CN time-discretisation error, measured separately by a u-only Newmark FOM at the same `dt` (`make_newmark_fom`). Every ROM error is reported against **both** the 80-substep FOM and that same-`dt` FOM. Primary cells use `RS = 20` (1000 latent steps), where the same-`dt` FOM is 1.16e-3 from the true FOM at N=64 — roughly 10x below the manifold floor. `RS in {8,20,40}` is swept at K=8. |
| family | `z = (cx, cy, w, a, log c)`, `cx,cy~U(0.15,0.85)`, `w~U(0.05,0.20)`, `a~U(1,10)`, `c~logU(0.5,2)`. TRAIN = the first 512 trajectories of the seed-0 draw (identical to the sweep). **TEST = 16 fresh trajectories from `TEST_SEED=1`** — the sweep checkpoints were model-selected on VAL, so VAL is never used as a test set. Data are regenerated on the cluster from the seed and `build_data` aborts if the FOM's relative energy drift exceeds 1e-9 on **either** split (observed ~3e-11). |
| decoder (Stage 2) | `u(x;z) = eps * b(x) * FiLM(x;z)`, hard Dirichlet factor `b = 16x(1-x)y(1-y)`, `ms_parametric` FiLM net (5x256, `n_freq` Nyquist-capped, ~440k params), trained as an **auto-decoder**: one latent per (trajectory, time) snapshot (26 112 rows at N=64), latents initialised from the top-K POD coefficients (no true `z` anywhere), **per-trajectory** inverse mean-square weights (wave snapshot norms pass near zero during kinetic/potential exchange, so per-snapshot weights would spike), lazy per-row Adam on the latents, time-smoothness penalty. `wlat_train_ad.py`. |
| POD control | the same solver on `u = V c`, `V` from the TRAIN snapshots (spatial Gram, host f64), `k in {6,8,16,32,64}`. |
| residual: strong (control) | **point-local**: the Newmark residual at `m` interior nodes from the decoder at the `m` 5-point stencils — the exact FOM interior operator. Collocation `full | rand<m> | biased<m> | offgrid<m>` (`offgrid` = continuum strong form with autodiff Laplacians, a *continuum* objective, labelled as such). |
| residual: weak (the recipe) | test modes = the `M` lowest discrete sine modes `phi_i`, the **exact** eigenvectors of the FOM's ghost-zero Laplacian (`-L phi_i = lam_i phi_i`, exact on the square). With `p = Phi^T u` carried as `M` numbers: `R_i = w_i[(1 + a lam_i)(p_i + p_{n-1,i}) - 2(1 - a lam_i) p_{n,i}]`, first step `R_i = w_i[(1 + a lam_i)p_i - (1 - a lam_i)p_{0,i}]`, `w_i = (1 + a lam_i)^-alpha`. **The wave operator is linear, so no derivative of the decoder is ever needed** and grid and meshfree quadrature share the same discrete `lam_i`. `M in {64,144,256}`. Variants: `weak<M>` (alpha=1, a *weighted* least-squares problem — disclosed), `weaku<M>` (alpha=0, unweighted, the standard comparison), `weakl<M>` (extra `lam^-1/2` energy-type weighting). |
| hyper-reduction | `eq<m>` (grid nodes) / `eqoff<m>` (meshfree pool of 4096 interior points) — capped Lawson–Hanson NNLS quadrature weights fitted to reproduce the exact grid projections `Phi^T u` of **DECODER-OUTPUT** snapshots at 64 training latents. `m ~ 4M`. |
| solvers | `lspg`: LM on `||R||` entirely on device (`lax.while_loop` step, the whole rollout one `lax.scan`); `galerkin`: damped Newton on `J_D^T W R = 0` (test functions = mode-projected tangents `Phi^T dD/dz`), backtracking on `||g||`, Python loop. Warm start `z_n`; budget 30 attempts/step. Cold start `z_0` = a **jitted** best-of LM fit of the decoder to the KNOWN `u0` from {mean training latent at t=0, latent of the training trajectory with the nearest IC}. |
| Stage 1 pilot | `wlat_stage1.py`: the sweep's (z,t)-conditioned decoder (5-dim true-parameter latent, N=64 checkpoint); solve for `z` by LM on (a) IC misfit only, (b) the space-time Newmark residual over the horizon (an objective ablation — no IC term; `u0` still initialises it), (c) both, IC block weight `IC_W in {1, sqrt(50)}`. |
| energy | `v` reconstructed on the decoded fields two ways: **kinematic** (the trapezoidal recursion) and **dynamic** (`v_k = v_{k-1} + (dt c^2/2)(L u_{k-1} + L u_k)`, the FOM's own update). The kinematic recursion has a `(-1)^k` mode that amplifies decoder noise like `sqrt(k)/dt`, so the dynamic estimate is the fair long-horizon diagnostic and the kinematic–dynamic velocity defect is reported alongside. Both are calibrated on an exact Newmark trajectory (V7). |
| metrics | **traj-RMS (PRIMARY, the wave testbed's metric)**: `mean_t ||u_ROM - u||_t / sqrt(mean_t ||u(t)||^2)`; per-snapshot `snap` metric alongside. Mean/median/max over 16 test trajectories **over completed rollouts only**; a rollout counts as complete only if every step finished and every latent, field, error and energy value is finite. |

Files: `wlat_common.py` (all machinery), `wlat_verify.py`, `wlat_train_ad.py`,
`wlat_rom.py`, `wlat_stage1.py`, `wlat_summarize.py`, `cluster/{make_cell,launch,pull,cells}.sh`,
`CODEX-REVIEW-2026-08-16.md`, `deps/` (frozen copies of the imported modules, see
`deps/PROVENANCE.md`).

## Verification (`wlat_verify.py`, run at the start of every cell)

| check | what it asserts |
|---|---|
| V1 | the u-only Newmark rollout at `RS=80` reproduces the `(u,v)` CN FOM — i.e. the v-elimination is exact |
| V2 | the strong (full + subset) and weak residual operators vanish on exact Newmark states, and a **non-solution does not** |
| V2b | Newmark self-consistency of the Stage-1 residual formula |
| V3 | the weak form with **all** `(n-2)^2` modes and `alpha=0` has exactly the same 2-norm as the strong full-grid residual (`Phi` orthonormal) — "weak with all modes == full Galerkin" |
| V5 | the ROM's time-discretisation floor as a function of `RS` |
| V7 | both energy estimators calibrated on an exact Newmark trajectory |
| V8 | `train_trajectories` reproduces the frozen `wf.build_trajectories` bit-for-bit |

