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

## Cluster runs

| cell | job | GPU | what |
|---|---|---|---|
| `wverify_n64` | 2478633 | A100 80GB | verification + the `RS` time-step table at N=64 |
| `ws1_n64_icw1` | 2480278 | A100 80GB | Stage 1, N=64, `IC_W=1` |
| `ws1_n64_icwsqrt50` | 2480282 | A100 80GB | Stage 1, `IC_W=sqrt(50)` |
| `ws1_n64_diag` | 2481931 | A100 80GB | Stage 1 oracle-init / multistart diagnostic |
| `wad_n64_k4` | 2481531 | A100 80GB | auto-decoder K=4 (80k steps, batch 128, P_SUB 2048) + the full ROM study |
| `wad_n64_k8` | 2481533 | A100 80GB | K=8, plus the `RS in {8,20,40}` ROM-time-step arm |
| `wad_n64_k16` | 2481537 | A100 80GB | K=16 |
| `wad_n128_k8` | 2481538 | A100 80GB | K=8 at N=128 (flat-in-N check at fixed k, M, m) |

Every cell: `jax_backend=gpu`, `JAX_DEFAULT_MATMUL_PRECISION=highest`, f64, data regenerated
from seed 0 on the cluster (train and test FOM energy drift 3.2e-11 / 1.2e-11, both
thresholded at 1e-9), isolated job directories created atomically, sha256-verified staging
and pull, cluster directories deleted afterwards.  The per-interval step diagnostic
(`wlat_stepdiag.py`) was run on the local GB10 against the pulled K=8 checkpoint (the data it
regenerates agree with the cluster's to 5.9e-16 in the global moments; the sha256 differs
because CG iterates differ in the last bits across GPU/JAX builds).

## Verification results (N=64, job 2478633)

- FOM energy drift **1.6e-12**; the u-only Newmark rollout's **2.3e-9**.
- **V1** u-only Newmark at RS=80 vs the (u,v) CN FOM: **2.8e-8** (max 8.1e-8) — CG-tolerance
  limited (the FOM uses tol 1e-10, ours 1e-12, over 4000 steps). **The v-elimination is exact.**
- **V2** residual operators on exact Newmark states: strong-full 2.1e-16, strong-subset
  1.7e-16, weak 2.1e-16; a NON-solution gives 2.6e-4.
- **V2b** Newmark self-consistency of the Stage-1 residual formula: 2.9e-17.
- **V3** weak form with all 3844 modes, `alpha=0`, vs the strong full-grid residual:
  3.095926e-01 vs 3.095926e-01, **relative difference 3.6e-16**.
- **V7** both energy estimators on an exact Newmark trajectory: kinematic drift 1.5e-12,
  dynamic drift 5.4e-12, kinematic-vs-dynamic velocity defect 9.1e-12.
- **V8** `train_trajectories` reproduces the frozen `wf.build_trajectories`: max |diff| **0.0**.

### The ROM time-step floor (`RS`), N=64

The u-only Newmark FOM at the ROM's own `dt`, measured against the 80-substep FOM
(traj-RMS, mean over 6 trajectories):

| `RS` | `dt` | latent steps | traj-RMS vs the FOM | max |
|---|---|---|---|---|
| 1 | 2.0e-2 | 50 | 1.33e-1 | 2.39e-1 |
| 2 | 1.0e-2 | 100 | 5.40e-2 | 1.26e-1 |
| 4 | 5.0e-3 | 200 | 2.17e-2 | 7.44e-2 |
| 8 | 2.5e-3 | 400 | 7.22e-3 | 2.84e-2 |
| 16 | 1.25e-3 | 800 | 1.85e-3 | 7.28e-3 |
| **20** | **1.0e-3** | **1000** | **1.16e-3** | 4.57e-3 |
| 40 | 5.0e-4 | 2000 | 2.32e-4 | 9.17e-4 |
| 80 | 2.5e-4 | 4000 | 2.78e-8 | 8.08e-8 |

Second order asymptotically (ratio 2.5 -> 3.0 -> 3.9 -> 4.0 as `dt` halves).  `RS=20` was
chosen for the primary cells: 1.16e-3 is roughly 10x below the manifold floor, so the ROM's
own time discretisation cannot explain any of the errors below.  Every ROM number is
nevertheless reported against **both** the 80-substep FOM and the same-`dt` Newmark FOM
(they agree to 4 digits everywhere, confirming this).

## Stage 1 — space-time LSPG on the (z,t) sweep decoder (N=64, 16 fresh test trajectories)

| arm | traj-RMS mean | median | max | mean \|z − z*\| | obj(found)/obj(z*) |
|---|---|---|---|---|---|
| oracle (true z) | **3.18e-2** | — | 8.37e-2 | 0 | 1 |
| `ic` (IC misfit only) | 8.83e-1 | 9.40e-1 | 1.36 | 0.428 | 0.728 (16/16 below) |
| `resid` (no IC term in the objective) | 1.04 | 9.56e-1 | 1.70 | 1.59 | 0.171 (16/16 below) |
| `both`, `IC_W=1` | 6.34e-1 | 7.31e-1 | 1.35 | 0.269 | 0.808 (16/16 below) |
| `both`, `IC_W=sqrt(50)` | 6.07e-1 | **2.11e-1** | 1.35 | 0.301 | — |
| `both_oracleinit` (LM started AT the true z — ORACLE diagnostic) | 5.93e-1 | 6.14e-1 | 1.35 | 0.256 | 0.817 (16/16 below) |

**Stage 1 does not transfer from Burgers, and the reason is not optimisation.**  On Burgers
`both` reached the oracle (7.3e-3 vs 3.8e-3).  Here it lands 19x above it, and the
oracle-initialised arm settles the question: started exactly at the true parameter vector,
LM *walks away* from it to `|z − z*| = 0.26` and finds an objective **18% lower** than the
objective at the truth, on **16 of 16** trajectories.  The residual + IC-misfit objective's
minimiser genuinely is not the true parameter.

Two things cause this, and both are wave-specific:

1. The sweep decoder's own PDE inconsistency at the true `z` is 5.6e-3 relative — comparable
   to the objective's variation across the latent space, so the objective can be lowered by
   moving to latents whose (wrong) fields happen to be more Newmark-consistent.  `resid`
   alone drives the objective to 0.171 of its value at the truth while `|z − z*|` reaches
   1.59: with no dissipation and no data term, a smoother, lower-amplitude field is always
   more PDE-consistent.
2. Even a *good* parameter estimate does not buy a good trajectory. `both` fits `u0`
   essentially perfectly (per-time error 1.1e-2 at t=0 — the oracle's own level) and then
   grows monotonically to ~0.94 by T=1.  Nothing damps a phase error: `|δz| ≈ 0.27` out of a
   ~[-1,1]^5 range already dephases the wave by O(1) over one crossing time.

