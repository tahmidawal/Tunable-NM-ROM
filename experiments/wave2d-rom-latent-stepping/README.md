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

Second order asymptotically (ratio 2.5 -> 3.0 -> 3.9 -> 4.0 as `dt` halves).  This table is a
6-trajectory sample and the spread is wide (max 4x mean); the Stage-2 cells recompute the
same quantity on all **16 test trajectories** and report `RS=20` -> **2.39e-3**, `RS=8` ->
1.39e-2, `RS=40` -> 4.80e-4, which are the numbers used in the comparisons.  `RS=20` was
chosen for the primary cells: 2.4e-3 is ~70x below the ROM errors and ~30x below the
manifold floor, so the ROM's own time discretisation cannot explain any of the errors below.  Every ROM number is
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

## Stage 2 — auto-decoder latent-stepping ROM

All numbers are traj-RMS against the 80-substep FOM, mean over 16 held-out trajectories,
`RS=20` unless stated.  **Zero blow-ups anywhere** (0/16 in every one of the 3 x 16
coordinate variants and the 3 x 15 POD variants at N=64, and at N=128); warm Gauss-Newton
converges in ~5 Jacobian evaluations per step, cold starts in ~4.

### Floors and headline numbers (N=64)

| | K=4 | K=8 | K=16 |
|---|---|---|---|
| auto-decoder TRAIN recon at learned latents | 1.31e-1 | 7.03e-2 | 5.99e-2 |
| **ORACLE held-out inferred latents** (per snapshot, budget 60) | 3.11e-1 | 1.72e-1 | 1.11e-1 |
| IC fit (cold start from the known `u0`) | 1.86e-1 | 1.13e-1 | 7.52e-2 |
| ROM `lspg:eq256:weak64` (the recipe) | 1.02e0 | 8.78e-1 | 6.80e-1 |
| best coordinate variant at that K | 9.45e-1 (`full:weakl64`) | 8.41e-1 (`full:weakl64`) | 5.92e-1 (`full:weakl64`) |
| **ROM / its own oracle floor** | **3.3x** | **5.1x** | **6.2x** |
| kinematic energy `E_T/E_0` (`eq256:weak64`) | 0.27 | 0.27 | 0.40 |
| POD-LSPG k=6 / 8 / 16 / 32 / 64 (same solver) | 4.29e-1 / 3.42e-1 / 2.19e-1 / 1.41e-1 / **8.38e-2** | same | same |
| POD projection floors k=6 / 8 / 16 / 32 / 64 (test) | 4.28e-1 / 3.42e-1 / 2.17e-1 / 1.38e-1 / 8.20e-2 | | |
| **POD-LSPG / its own projection floor** | **1.002 / 1.002 / 1.006 / 1.020 / 1.022** | | |
| POD kinematic energy `E_T/E_0` | **1.0000** at every rank | | |

### The two findings

**1. The linear ROM inherits the FOM's conservation; the nonlinear one does not.**
The Newmark residual is *linear* in `u`, so projecting it onto a fixed linear subspace `V`
returns a Newmark scheme on the reduced operator `Vᵀ L V` — still symmetric negative
definite, still exactly energy-conserving, still unconditionally stable.  That is precisely
what the POD arms show: **POD-LSPG sits at its projection floor at every rank (ratio
1.00–1.02) and conserves the discrete energy to four digits.**  A nonlinear manifold has no
such structure — the tangent space rotates with `z`, so the per-step Gauss-Newton update is
not energy-consistent.  The coordinate ROM dissipates 61–84% of the energy and lands 3.3x
(K=4) / 5.1x (K=8) above **its own** oracle floor.  The manifold's better approximation power
at matched dimension (K=8 oracle floor 1.72e-1 vs POD-8 3.42e-1, 2.0x better) is entirely
destroyed by the stepping, and on this problem the *linear* ROM wins outright.

**The gap widens as the manifold improves.**  Tripling the latent dimension from K=4 to K=16
improves the oracle floor 2.8x (3.11e-1 -> 1.11e-1) but the ROM only 1.5x (1.02 -> 6.80e-1),
so the ROM/floor ratio *grows* 3.3x -> 5.1x -> 6.2x.  The ROM error is not manifold-limited;
it is limited by the amplification mechanism below, and a better manifold does not help much.
(One suggestive exception: the `lam^-1/2` energy-weighted weak form `weakl64` is the best
coordinate arm at every K -- 9.45e-1 / 8.41e-1 / 5.92e-1 -- and it is the only arm whose
energy *grows* rather than decays (`E_T/E_0` = 0.58 / 1.17 / 6.26).  That the best arm is the
one that fights the numerical dissipation is a hint that an explicitly energy- or
symplectic-constrained latent step is the thing to try next; it is not tested here.)

This is the mirror image of the Burgers result, where the same solver, same residual and same
collocation gave the coordinate manifold a 12.6x advantage over POD at matched dimension.
Taken together the two rounds say: **the nonlinear manifold buys accuracy when the PDE damps
the error it injects, and loses when the PDE conserves it.**

**2. Every objective/hyper-reduction knob that mattered on Poisson and Burgers is irrelevant
here — including the ones that were supposed to matter on wave.**

| knob | result at K=8, N=64 |
|---|---|
| Galerkin vs LSPG | **identical to 4 significant figures** in all 16 variants (8.783e-1 both, 8.952e-1 both, 8.762e-1 vs 8.767e-1) |
| test modes `M` = 64 / 144 / 256 | 8.78e-1 / 8.71e-1 / 8.85e-1 — within 1.5% |
| weighted (`alpha=1`) vs unweighted (`weaku`) weak form | 8.952e-1 vs 8.931e-1 |
| `lam^-1/2` energy weighting (`weakl`) | 8.408e-1 — the only knob worth anything (4% better), and it *inflates* the energy (`E_T/E_0` 1.17) |
| weak vs strong FD residual | 8.95e-1 vs 8.77e-1 |
| meshfree (`eqoff256`) vs grid (`eq256`) NNLS quadrature | 8.791e-1 vs 8.783e-1 |
| **NNLS-EQ `m=256` (= 4M nodes) vs the full 3844-node grid** | **8.783e-1 vs 8.952e-1 — the 15x-cheaper quadrature is if anything slightly better** |
| random 512-node strong collocation | 8.939e-1 |
| off-grid continuum strong form | 8.735e-1 |

So the **hyper-reduction half of the recipe is validated on wave** (NNLS-EQ at `m ≈ 4M`
reproduces the full-grid answer at 1/5 of the per-step cost, from a meshfree pool as
happily as from grid nodes, with NNLS relative fit residuals of 1.0e-2 at m=256 and 7e-5 at
m=1024) even though the ROM it accelerates does not work.  The failure is upstream of every
objective choice.

### Where the error comes from: injection vs amplification

`wlat_stepdiag.py`, K=8, started from **oracle latents fitted to the exact Newmark sub-step
fields** (so the two-level warm start is exact), 8 trajectories x 4 start snapshots, then run
for `H` snapshot intervals:

| `H` (snapshot intervals, = 20H latent steps) | 1 | 2 | 5 | 10 |
|---|---|---|---|---|
| ROM `lspg:eq256:weak64` | 2.046e-1 | 2.249e-1 | 3.749e-1 | 6.578e-1 |
| oracle latent fit of the FOM state at the same time | 1.959e-1 | 1.720e-1 | 1.730e-1 | 1.949e-1 |
| **excess (the ROM's own contribution)** | **+8.7e-3** | +5.3e-2 | +2.0e-1 | **+4.6e-1** |
| `hold` control (freeze the latent) | +9.6e-2 | +2.7e-1 | +7.4e-1 | +1.16e0 |

**One interval of latent stepping is accurate**: the excess is 4% of the oracle floor, and
11x smaller than doing nothing.  The stepping is not broken.  The excess then grows roughly
like `H^1.7` — the injected error is *amplified*, not damped, which is what "dissipation-free"
means for a ROM.

The ROM time-step arm closes the argument.  Refining `dt` makes the answer **worse**:

| `RS` | latent steps | same-`dt` FOM error (the time-discretisation floor) | ROM `lspg:eq256:weak64` |
|---|---|---|---|
| 8 | 400 | 1.39e-2 | **8.428e-1** |
| 20 | 1000 | 2.39e-3 | 8.783e-1 |
| 40 | 2000 | 4.80e-4 | 9.063e-1 |

The time-discretisation error falls 29x while the ROM error *rises* 8%.  The per-step
injection is set by the manifold's representation error, which does not shrink with `dt`, so
more steps simply means more injections.  A latent-stepping ROM on a conservative PDE is not
a convergent scheme in `dt`.

### Timing (N=64, one A100 per cell, median of 7 after 2 warm-ups, `block_until_ready`)

The IC latent solve is a **jitted** LM (78–97 ms), so — unlike the Burgers round — it does not
dominate; the full 51-slice decode is 10 ms.

| | rollout | ms / latent step | speedup vs the 80-substep FOM (384 ms) |
|---|---|---|---|
| FOM (CN/CG, 4000 substeps, batch 1) | 384 ms | 0.096 | 1x |
| same-`dt` u-only Newmark FOM (`RS=20`) | 161 ms | 0.16 | 2.4x |
| coord `lspg:eq256:weak64`, K=8, `RS=20` | 2443 ms | 2.44 | **0.16x** |
| coord `lspg:eq256:weak64`, K=8, `RS=8` | 1095 ms | 2.74 | 0.35x |
| coord `lspg:eqoff256:weak64` (meshfree) | 2420 ms | 2.42 | 0.16x |
| coord `lspg:full:weak64` | 13.5 s | 13.5 | 0.03x |
| coord `lspg:full:fd` | 62.1 s | 62.1 | 0.006x |
| POD k=8, `RS=20` | 453 ms | 0.45 | 0.85x |
| POD k=8, `RS=8` | 184 ms | 0.46 | **2.09x** |
| POD k=64, `RS=20` | 954 ms | 0.95 | 0.40x |

**No coordinate ROM pays at N=64.**  The wave FOM is an SPD CG solve on 4096 unknowns at
0.096 ms per sub-step; a single f64 FiLM decoder evaluation at 256 quadrature nodes with its
Gauss-Newton Jacobian costs 25x that.  This reproduces, inside a real ROM, the wave testbed's
own timing conclusion ("the wave FOM is CHEAP, so full-field surrogate decoding does not pay
at fine resolution").  The n-free per-step cost is real — `eq256` costs the same 2.4 ms per
step at N=64 and N=128 — but the FOM's cost grows too slowly for it to matter at these sizes.

### Flat in N (K=8, `RS=20`, fixed `M`, `m`)

| | N=64 | N=128 |
|---|---|---|
| ROM `lspg:eq256:weak64` | 8.783e-1 | 8.783e-1 |
| `galerkin:eq256:weak64` | 8.783e-1 | 8.783e-1 |
| `lspg:eq1024:weak256` | 8.846e-1 | 8.680e-1 |
| `lspg:eqoff256:weak64` | 8.791e-1 | 8.779e-1 |
| `lspg:full:weak64` | 8.952e-1 | 9.008e-1 |
| `lspg:full:fd` | 8.767e-1 | 8.637e-1 |
| ms per latent step, `eq256:weak64` | 2.44 | 2.48 |

The `eq256:weak64` number is **identical to four significant figures at N=64 and N=128**, and
the per-step cost is flat (2.44 -> 2.48 ms) while the number of grid points quadruples — the
n-free-cost property of the hyper-reduced coordinate ROM is confirmed on wave.  It just does
not buy anything here, because the failure mode is resolution-independent too.

## The published wave-ROM comparison, and why it is not a like-for-like number

The paper rebuttal's wave results were a **frozen POD basis with a damped LSPG rollout**:
4.71e-3 at 0.7x speedup (N=64), 4.94e-3 at 1.6x (N=128), 1.63e-2 at 3.3x (N=256).

**These cannot be reproduced or contradicted by this experiment, and the README does not claim
either.**  The problem here is a 5-parameter family (`cx, cy, w, a, log c`, `c` spanning a
factor of 4) with a *global* basis fitted to 512 training trajectories and evaluated on 16
trajectories from a fresh seed.  On that problem the **POD projection floor itself** — the
best any frozen linear basis can do, before any ROM — is 4.28e-1 at k=6, 8.20e-2 at k=64, and
POD-LSPG achieves 8.38e-2 at k=64, i.e. it is *at* its floor.  A frozen-basis wave ROM
reaching 4.7e-3 is therefore reporting on a substantially narrower problem (a much smaller
parameter range, a per-trajectory or per-regime basis, a shorter horizon, or a higher rank);
the rebuttal number and the 8.38e-2 here are measurements of different things.

What this experiment *does* say about that claim is structural and transfers: on a
dissipation-free PDE a frozen-basis LSPG ROM sits essentially exactly at its own projection
floor and conserves energy to four digits (ratio 1.00–1.02, `E_T/E_0` = 1.0000 at every rank
from 6 to 64), so a good frozen-basis wave ROM number is a statement about the basis, not
about the rollout.  The speedup side is consistent with the rebuttal's: 0.4–0.9x at N=64
here for POD at `RS=20` (2.1x at `RS=8`), against a 0.7x reported there.

## Reading the results

1. **The recipe ports, the machinery is right, and the ROM does not work on wave.**  Every
   verification passes at machine precision (the v-elimination is exact, the residual
   operators vanish on exact Newmark states, the weak form with all modes reproduces the
   strong one to 4e-16).  Every one of 3 x 16 coordinate variants and 3 x 15 POD variants
   completes 16/16 rollouts with no blow-up.  And the coordinate ROM lands at 0.59–1.02
   traj-RMS against a FOM it should be tracking — 3.3–6.2x above its own held-out floor.
2. **The mechanism is amplification, not injection.**  One snapshot interval of latent
   stepping from an oracle start costs +8.7e-3 (4% of the floor, 11x better than freezing the
   latent); the excess then grows like `H^1.7`, and refining the ROM time step by 5x makes the
   final error 8% *worse*.  There is nothing to damp the error the manifold injects.
3. **On this PDE the linear ROM is the better ROM, for a structural reason.**  A linear
   subspace turns the linear Newmark residual back into a Newmark scheme on `Vᵀ L V` —
   conservative and unconditionally stable by construction — and POD-LSPG duly sits at its
   projection floor with `E_T/E_0` = 1.0000.  A nonlinear manifold has no such structure.
   This is the exact converse of the Burgers finding, and the pair of results is the honest
   statement: **the nonlinear-manifold advantage is contingent on the PDE damping the
   error the ROM injects.**
4. **The hyper-reduction half of the recipe is nevertheless validated on wave.**  NNLS-EQ
   quadrature with `m = 4M = 256` nodes reproduces the full 3844-node grid answer to within
   2% at 1/5 of the per-step cost, from a meshfree pool as well as from grid nodes, at both
   N=64 and N=128, with an n-free per-step cost (2.44 -> 2.48 ms as `n` quadruples).  That
   part is problem-independent and carries forward.
5. **Stage 1 is a separate negative with a separate cause.**  Solving for a global parameter
   vector by minimising the space-time residual plus the IC misfit fails here because the
   objective's minimiser is not the true parameter — proven by starting LM at the truth and
   watching it walk away to an 18% lower objective on 16/16 trajectories.  The precondition
   the Burgers round satisfied silently (the decoder's own PDE residual at the truth being
   small compared to the objective's variation over the latent space) does not hold for an
   undamped, broadband solution.
6. **Cost.**  No coordinate ROM pays at these resolutions: the wave FOM is 4000 SPD CG solves
   at 0.096 ms each, and one f64 FiLM evaluation with its Gauss-Newton Jacobian at 256
   quadrature nodes costs 25x that.  The jitted IC solve (78–97 ms) is no longer the
   bottleneck it was in the Burgers round.

## What would have to change

Not tested here, listed because the diagnosis points at them: (i) an energy- or
symplectic-constrained latent step (the best coordinate arm at every K is the one whose
weighting fights the numerical dissipation); (ii) a manifold trained with a *rollout* loss
rather than a per-snapshot reconstruction loss, so the injected error is penalised; (iii) a
decoder accurate enough that the amplified error stays small over the horizon — but the
ROM/floor ratio *grows* with K here, so this looks like the weakest of the three.

## Caveats

- **Single training seed**, 16 test trajectories per cell, one training budget per K (80k
  steps, batch 128, 2048 points/row/step).  The auto-decoder floors here (7.0e-2 at K=8) are
  20x above the Burgers round's (3.5e-3) — some of that gap is the problem (undamped,
  broadband) and some is very likely under-training; **the ROM/oracle ratio, not the absolute
  error, is the quantity this experiment measures reliably.**
- The 5x256 FiLM architecture and `n_freq` cap were inherited from the Burgers/heat rounds
  without a wave-specific capacity search.  A larger or differently-parameterised decoder
  would lower the floors; whether it would change the *amplification* result is exactly the
  open question, and the step diagnostic is the instrument for answering it.
- `RS=20` is the primary configuration.  The `RS` arm covers only K=8.
- Test split = 16 trajectories from `TEST_SEED=1`; the Stage-1 sweep checkpoints were
  model-selected on VAL, so VAL is not used as a test set anywhere.
- The step diagnostic ran on the local GB10 rather than the cluster (queue quota); its data
  agree with the cluster's to 5.9e-16 in the global moments but are not bit-identical, so its
  absolute numbers should be read against its own oracle column, which is what the "excess"
  quantity does.
- POD's cold start is an exact least-squares projection while the coordinate decoder's is a
  budgeted best-of-two LM fit.  That favours POD at t=0; both IC misfits are reported so the
  reader can separate it from the rollout.
- The EQ supports/weights are fitted per decoder (coordinate and each POD rank get their own),
  so "the same collocation" means the same *rule and budget*, not literally the same nodes.
- Both energy estimators are reported.  On a trajectory this far from the FOM neither is a
  clean invariant, and their disagreement (the kinematic-vs-dynamic velocity defect) is itself
  part of the diagnosis; both are calibrated to ~1e-12 on an exact trajectory (V7).
- Two-reviewer gate: the Codex adversarial review (`CODEX-REVIEW-2026-08-16.md`, all MUST
  items applied) plus self-review; a second independent model reviewer could not be spawned.
