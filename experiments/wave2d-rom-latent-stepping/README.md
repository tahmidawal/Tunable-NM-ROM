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
| ROM time step | `dt = DT_SNAP / RS`. `RS = 80` **is** the FOM's operator; smaller `RS` adds a CN time-discretisation error, measured separately by a u-only Newmark FOM at the same `dt` (`make_newmark_fom`). Every ROM error is reported against **both** the 80-substep FOM and that same-`dt` FOM. Primary cells use `RS = 20` (1000 latent steps), where the same-`dt` FOM is **2.39e-3** from the true FOM over the 16 test trajectories at N=64 — 72x below the K=8 manifold floor and 368x below the ROM errors. `RS in {8,20,40}` is swept at K=8. |
| family | `z = (cx, cy, w, a, log c)`, `cx,cy~U(0.15,0.85)`, `w~U(0.05,0.20)`, `a~U(1,10)`, `c~logU(0.5,2)`. TRAIN = the first 512 trajectories of the seed-0 draw (identical to the sweep). **TEST = 16 fresh trajectories from `TEST_SEED=1`** — the sweep checkpoints were model-selected on VAL, so VAL is never used as a test set. Data are regenerated on the cluster from the seed and `build_data` aborts if the FOM's relative energy drift exceeds 1e-9 on **either** split (observed ~3e-11). |
| decoder (Stage 2) | `u(x;z) = eps * b(x) * FiLM(x;z)`, hard Dirichlet factor `b = 16x(1-x)y(1-y)`, `ms_parametric` FiLM net (5x256, `n_freq` Nyquist-capped at 31 for N=64, 463k params), trained as an **auto-decoder**: one latent per (trajectory, time) snapshot (26 112 rows at N=64), latents initialised from the top-K POD coefficients (no true `z` anywhere), **per-trajectory** inverse mean-square weights (wave snapshot norms pass near zero during kinetic/potential exchange, so per-snapshot weights would spike), lazy per-row Adam on the latents, time-smoothness penalty. `wlat_train_ad.py`. |
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
| `ws1_n64_diag` | 2481931 | A100 80GB | Stage 1 + the oracle-init / multistart diagnostic arms |
| `wad_n64_k4` | 2481531 | A100 80GB | auto-decoder K=4 (80k steps, batch 128, `P_SUB` 2048) + the full ROM study |
| `wad_n64_k8` | 2481533 | A100 80GB | K=8, plus the `RS in {8,20,40}` ROM-time-step arm |
| `wad_n64_k16` | 2481537 | A100 80GB | K=16 |
| `wad_n128_k8` | 2481538 | A100 80GB | K=8 at N=128 (flat-in-N check at fixed k, M, m) |

Every cell: `jax_backend=gpu`, `JAX_DEFAULT_MATMUL_PRECISION=highest`, f64, `git_commit` echoed
into the log, data regenerated from seed 0 on the cluster (train / test FOM relative energy
drift 3.2e-11 / 1.2e-11 at N=64 and 3.4e-11 / 1.4e-12 at N=128, both thresholded at 1e-9 with a
hard abort), isolated job directories created atomically, `MANIFEST.sha256` verified locally and
after transport, checksum-verified pull, cluster directories deleted afterwards.

The per-interval step diagnostic (`wlat_stepdiag.py`) ran on the **local GB10** against the
pulled K=8 checkpoint because the cluster GPU quota was full.  Its regenerated data agree with
the cluster's to 5.9e-16 in the global moments; the sha256 differs because CG iterates differ in
the last bits across GPU/JAX builds, so its absolute errors should be read against **its own**
oracle column (which is what the reported "excess" does).

`wad_n128_k8` has since **completed** (Slurm 2481538, `COMPLETED`, exit 0:0, 13:47:03 elapsed;
pulled 2026-08-17 with matching sha256 manifests).  `"finished": true`, all 16 coordinate
variants, all 15 POD control arms and the full timing block are in
`runs/wad_n128_k8/out/wlat_rom_N128_K8.json`, and every table below is regenerated from it.

## Verification results

All from `wlat_verify.py`, which runs at the start of every cell.  N=64 values from job 2478633
(`wverify_n64`); V7/V8 were added after that cell and come from the three `wad_n64_*` cells,
which re-run the whole verification.

- FOM relative energy drift **1.6e-12**; the u-only Newmark rollout at RS=80 **2.3e-9**.
- **V1** u-only Newmark at RS=80 vs the (u,v) CN FOM: **2.8e-8** (max 8.1e-8).  This is *not* a
  machine-precision agreement and is not expected to be: the two rollouts are algebraically
  identical but use independent CG solves (the FOM at tol 1e-10, ours at 1e-12) over 4000 steps,
  so the difference is a random walk of the CG residuals.  What it establishes is that **the
  trapezoidal v-elimination is exact for this discretisation** (the algebraic derivation is in
  `CODEX-REVIEW-2026-08-16.md`, section "VERIFIED CORRECT"); the gate is 1e-6 on the max.
- **V2** residual operators fed exact Newmark states, at machine precision: strong-full
  2.1e-16, strong-subset 1.7e-16, weak 2.1e-16.  A **non**-solution gives 2.6e-4, so the
  operators are not trivially zero.
- **V2b** Newmark self-consistency of the Stage-1 residual formula: 2.9e-17.  (This uses the
  same u-only implementation it checks; V1 is the independent comparison.)
- **V3** weak form with all 3844 interior modes and `alpha=0` vs the strong full-grid residual:
  3.095926e-01 vs 3.095926e-01, **relative difference 3.6e-16** — "weak with all modes == full
  Galerkin/LSPG", at machine precision.
- **V7** both energy estimators on an exact Newmark trajectory (N=64): kinematic drift 2.2e-12,
  dynamic drift 7.1e-12, kinematic-vs-dynamic velocity defect 1.4e-11.  At N=128: 4.4e-10,
  7.1e-10, 1.4e-9.  Both estimators are therefore calibrated; the large values reported for the
  coordinate ROM below are not artefacts of the estimators themselves.
- **V8** `train_trajectories` reproduces the frozen `wf.build_trajectories`: max |diff| **0.0**
  in all four cells.

### The ROM time-step floor (`RS`), N=64

The u-only Newmark FOM at the ROM's own `dt`, against the 80-substep FOM (traj-RMS, mean over
the **6** trajectories of the verification sample; `N_VERIFY=6`):

| `RS` | `dt` | latent steps | traj-RMS vs the FOM | max |
|---|---|---|---|---|
| 1 | 2.0e-2 | 50 | 1.33e-1 | 2.39e-1 |
| 2 | 1.0e-2 | 100 | 5.40e-2 | 1.26e-1 |
| 4 | 5.0e-3 | 200 | 2.17e-2 | 7.44e-2 |
| 8 | 2.5e-3 | 400 | 7.22e-3 | 2.84e-2 |
| 16 | 1.25e-3 | 800 | 1.84e-3 | 7.28e-3 |
| **20** | **1.0e-3** | **1000** | **1.16e-3** | 4.57e-3 |
| 40 | 5.0e-4 | 2000 | 2.32e-4 | 9.16e-4 |
| 80 | 2.5e-4 | 4000 | 2.78e-8 | 8.08e-8 |

Successive halving ratios are 2.46, 2.49, 3.01, 3.91 (RS 1->2->4->8->16), approaching but not
reaching the second-order value 4; RS 20->40 gives 4.98.  This is a 6-trajectory sample with a
wide spread (max ~4x mean).  **The Stage-2 cells recompute the same quantity on all 16 test
trajectories**, and those are the numbers used in every comparison below: `RS=8` -> 1.388e-2,
`RS=20` -> **2.386e-3**, `RS=40` -> 4.802e-4 (N=64); `RS=20` -> 1.081e-2 (N=128).  At `RS=20`
and N=64 the ROM's own time-discretisation error is 72x below the K=8 manifold floor (1.719e-1)
and 368x below the ROM error (8.783e-1), so it cannot explain any of the errors below.  Directly:
every ROM error measured against the same-`dt` FOM instead of the true FOM changes by at most
0.021% relative.

## Stage 1 — space-time LSPG on the (z,t) sweep decoder (N=64, 16 fresh test trajectories)

| arm | traj-RMS mean | median | max | mean \|z − z*\| | obj(found)/obj(z*) |
|---|---|---|---|---|---|
| oracle (true z) | **3.184e-2** | — | 8.365e-2 | 0 | 1 |
| `ic` (IC misfit only) | 8.830e-1 | 9.397e-1 | 1.361 | 0.428 | 0.728 (16/16 below) |
| `resid` (no IC term in the objective) | 1.041 | 9.561e-1 | 1.697 | 1.592 | 0.171 (16/16 below) |
| `both`, `IC_W=1` | 6.342e-1 | 7.307e-1 | 1.353 | 0.269 | 0.808 (16/16 below) |
| `both`, `IC_W=sqrt(50)` | 6.068e-1 | **2.114e-1** | 1.349 | 0.301 | — |
| `both_multistart` (6 nearest-IC starts) | 6.610e-1 | 7.303e-1 | 1.364 | 0.284 | 0.805 (16/16 below) |
| `both_oracleinit` (LM started **AT** the true z — ORACLE diagnostic, not a ROM) | 5.928e-1 | 6.142e-1 | 1.351 | 0.256 | 0.817 (16/16 below) |

**Stage 1 does not transfer from Burgers, and the failure is not an optimisation failure.**  On
Burgers `both` reached the oracle (7.3e-3 vs 3.8e-3).  Here it lands 19x above it; six
restarts do not help (6.610e-1); and the oracle-initialised arm settles the question — started
exactly at the true parameter vector, LM *walks away* to `|z − z*| = 0.256` and finds an
objective **18% lower** than the objective at the truth, on **16 of 16** trajectories.  The true
parameter is therefore **not a minimiser of the implemented objective**.  (What LM converges to
need not be the global minimiser either; the rows terminate on budget or `lambda_max`.)

Two candidate explanations, **neither directly measured here**:

1. The sweep decoder's own PDE inconsistency at the true `z` is 5.58e-3 relative — not small
   compared to the objective's variation over the latent space, so the objective can be lowered
   by moving to latents whose (wrong) fields happen to be more Newmark-consistent.  `resid`
   alone drives the objective to 0.171 of its value at the truth while `|z − z*|` reaches 1.59,
   which is the behaviour one expects if smoother, lower-amplitude fields are more
   PDE-consistent — but this experiment does not measure the amplitude or smoothness of the
   fields it lands on.
2. Even a decent parameter estimate does not buy a good trajectory.  `both` fits `u0` essentially
   perfectly (per-time error 1.1e-2 at t=0, the oracle's own level) and then rises to ~0.94 by
   T=1 (the rise is not strictly monotone).  `|δz| ≈ 0.27` out of a ~[-1,1]^5 range already
   dephases the wave by O(1) over one crossing time.

## Stage 2 — auto-decoder latent-stepping ROM

traj-RMS against the 80-substep FOM, mean over 16 held-out trajectories, `RS=20` unless stated.
**Zero blow-ups anywhere**: all 93 main N=64 variants (48 coordinate, 45 POD) plus the 8 RS-arm
variants report `n_blowup=0`, `n_completed=16`, `n_total=16`; so do all 31 N=128 variants
(16 coordinate, 15 POD) and all 32 step-diagnostic cases.  Warm Gauss-Newton takes 4.94 / 5.19 / 5.38 Jacobian
evaluations per step at K=4/8/16 and cold starts 3.56 / 4.13 / 4.00.

### Floors and headline numbers (N=64)

| | K=4 | K=8 | K=16 |
|---|---|---|---|
| auto-decoder TRAIN recon at learned latents | 1.312e-1 | 7.03e-2 | 5.99e-2 |
| **ORACLE held-out inferred latents** (per-snapshot LM, budget 60) | 3.114e-1 | 1.719e-1 | 1.106e-1 |
| IC fit (cold start from the known `u0`) | 1.860e-1 | 1.125e-1 | 7.52e-2 |
| ROM `lspg:eq256:weak64` (the recipe) | 1.019 | 8.783e-1 | 6.801e-1 |
| best coordinate variant at that K (`full:weakl64`) | 9.453e-1 | 8.408e-1 | 5.916e-1 |
| **ROM / its own oracle floor** | **3.27x** | **5.11x** | **6.15x** |
| POD-LSPG k=6 / 8 / 16 / 32 / 64 (`lspg:full:fd`, same solver) | 4.290e-1 / 3.424e-1 / 2.188e-1 / 1.408e-1 / **8.378e-2** | same | same |
| POD projection floors, **TEST** split, k=6…64 | 4.282e-1 / 3.417e-1 / 2.174e-1 / 1.381e-1 / 8.201e-2 | | |
| **POD-LSPG / its own projection floor** | **1.002 / 1.002 / 1.006 / 1.019 / 1.022** | | |

### The two findings

**1. The linear ROM stays at its projection floor and (nearly) conserves energy; the nonlinear
one does neither.**

| | POD-LSPG, k=6…64 | coordinate ROM `eq256:weak64`, K=4/8/16 |
|---|---|---|
| ROM / its own floor | 1.002 … 1.022 | 3.27 / 5.11 / 6.15 |
| kinematic `E_T/E_0` | 1.000000 … 1.000003 | 0.265 / 0.272 / 0.395 |
| kinematic max drift | 2.8e-7 … 2.9e-6 | 0.75 / 0.75 / 0.61 |
| dynamic `E_T/E_0` | 1.0046 … 1.0091 | 50.7 / 14.5 / 28.6 |
| dynamic max drift | 0.014 … 0.018 | 0.75 … 6.4 |
| kinematic-vs-dynamic velocity defect | 0.14 … 0.18 | 1.02 … 1.06 |

The two velocity reconstructions disagree, so "conserves energy" has to be stated per estimator.
Under **either** one the contrast is unambiguous: POD-LSPG is within 1% of its initial energy at
every rank, while the coordinate ROM is off by a factor of 3–4 (kinematic: it loses 61–74% of
the energy) or 15–50 (dynamic: it gains), with a velocity defect of order one.  The robust
statement is **severe energy inconsistency in the coordinate ROM and near-conservation in the
POD ROM**, not specifically dissipation.

**Why (derivation, not a measurement).**  The Newmark residual is *linear* in `u`.  Write
`A = I − aL` and `B = I + aL`, both symmetric.  On a **fixed** subspace `u = V c`,
`R = A V c_{n+1} − 2 B V c_n + A V c_{n−1}`.  Galerkin gives
`(VᵀAV)(c_{n+1} + c_{n−1}) = 2 (VᵀBV) c_n`; LSPG gives
`(VᵀAᵀAV)(c_{n+1} + c_{n−1}) = 2 (VᵀAᵀBV) c_n`, and `AᵀB = I − a²L²` is symmetric, so **both**
reduce to a three-term recurrence `S(c_{n+1} + c_{n−1}) = 2 T c_n` with `S` symmetric positive
definite and `T` symmetric — the *same* matrix multiplying `c_{n+1}` and `c_{n−1}`, i.e. a
time-reversible recurrence, which is the structural property that makes the FOM conservative.
The weak form inserts a weight `W` that is diagonal in the sine-mode basis, in which `A` and `B`
are also diagonal, so `T` stays symmetric there too (and indeed `pod_k*:full:weak256` and
`pod_k*:eq1024:weak256` give the same numbers as `full:fd` to 4 digits).  On a **nonlinear**
manifold `S` and `T` depend on `z`, the time-reversal symmetry is lost, and nothing bounds the
per-step energy change.  Note this derivation explains the *observation*; the observation is the
evidence, and it does not claim the reduced LSPG operator equals `VᵀLV` (it does not — that is
the Galerkin operator).

**The gap widens as the manifold improves.**  Tripling K from 4 to 16 improves the oracle floor
2.8x (3.114e-1 -> 1.106e-1) but the ROM only 1.5x (1.019 -> 6.801e-1), so the ROM/floor ratio
*grows* 3.27 -> 5.11 -> 6.15.  (K=4/8/16 are three separately trained models, so this is not a
clean single-variable sweep.)  At matched dimension the manifold is the better approximator
(K=8 oracle floor 1.719e-1 vs POD-8 3.417e-1, 2.0x better) — and the stepping gives all of it
back: on this problem the *linear* ROM is the better ROM at every rank tested.

**2. Every objective / hyper-reduction knob that mattered on Poisson and Burgers is nearly
irrelevant here** (K=8, N=64):

| knob | result |
|---|---|
| Galerkin vs LSPG (4 matched pairs run) | 8.783078e-1 vs 8.783100e-1; 8.846253e-1 vs 8.846264e-1; 8.952084e-1 vs 8.952089e-1; and `full:fd` 8.766662e-1 vs 8.762220e-1 — the first three agree to 7 figures, the last to 3 |
| test modes `M` = 64 / 144 / 256 (EQ) | 8.783e-1 / 8.714e-1 / 8.846e-1 — within 1.5% |
| weighted (`alpha=1`) vs unweighted (`weaku`) weak form | 8.952e-1 vs 8.931e-1 |
| `lam^-1/2` energy weighting (`weakl64`) | 8.408e-1 — the only knob worth anything (4% better) |
| weak vs strong FD residual (full grid) | 8.952e-1 vs 8.767e-1 |
| meshfree (`eqoff256`) vs grid (`eq256`) NNLS quadrature | 8.791e-1 vs 8.783e-1 (0.09%) |
| **NNLS-EQ `m=256` (= 4M nodes) vs the full 3844-node grid** | **8.783e-1 vs 8.952e-1 — 15x fewer nodes, 5.5x less per-step time, 1.9% *better*** |
| random 512-node strong collocation | 8.939e-1 |
| off-grid continuum strong form | 8.735e-1 |

So at K=8, N=64 the **hyper-reduction half of the recipe reproduces the full-grid answer**:
`eq256` differs from `full:weak64` by 1.9%, meshfree from grid by 0.09%, at 2.44 vs 13.49 ms per
step.  NNLS relative fit residuals for the coordinate decoder are 1.007e-2 (m=256, M=64),
3.90e-3 (m=576, M=144), 1.92e-3 (m=1024, M=256) and 1.046e-2 for the meshfree m=256 pool; the
POD arms fit far better (7.25e-5 for k=64 at m=1024) because their snapshots are exactly in the
span.  This is a statement about the tested cells, not a general one — but the same holds at
N=128 (below), which is the part of the recipe that survives here.

### Where the error comes from

`wlat_stepdiag.py`, K=8, `RS=20`, started from **oracle latents fitted to the exact Newmark
sub-step fields** (so the two-level warm start is exact), 8 trajectories x 4 start snapshots
(= 32 cases, correlated within a trajectory), then run for `H` snapshot intervals:

| `H` (snapshot intervals, = 20H latent steps) | 1 | 2 | 5 | 10 |
|---|---|---|---|---|
| ROM `lspg:eq256:weak64` | 2.046e-1 | 2.249e-1 | 3.749e-1 | 6.578e-1 |
| oracle latent fit of the FOM state at the same time | 1.959e-1 | 1.720e-1 | 1.730e-1 | 1.949e-1 |
| **excess = ROM − oracle** | **+8.7e-3** | +5.3e-2 | +2.0e-1 | **+4.6e-1** |
| `hold` control (freeze the latent, do nothing) | +9.6e-2 | +2.7e-1 | +7.4e-1 | +1.16 |

**One snapshot interval of latent stepping is accurate**: the excess is 4.5% of the oracle floor
and 11.0x smaller than freezing the latent.  The stepping is not broken.  The excess then grows
as `H^1.69` (least-squares fit of `log(excess)` on `log H`, H=1..10).

**This is consistent with error accumulation *or* with amplification, and the diagnostic does
not separate them.**  The oracle endpoint is refit at every `H` and the "excess" is a difference
of relative-error norms, not a decomposition; no controlled one-time perturbation is propagated
and no per-step forcing is switched off.  What is established is that a single interval costs
little and ten intervals cost 53x more than one, in a PDE with nothing to damp the difference.

The ROM time-step arm is the sharpest available handle (K=8, `lspg:eq256:weak64`, 3 points):

| `RS` | latent steps | same-`dt` FOM error (the time-discretisation floor) | ROM |
|---|---|---|---|
| 8 | 400 | 1.388e-2 | **8.428e-1** |
| 20 | 1000 | 2.386e-3 | 8.783e-1 |
| 40 | 2000 | 4.802e-4 | 9.063e-1 |

Refining `dt` by 5x drops the time-discretisation floor 28.9x and **raises** the final ROM error
by 7.5%.  For this configuration, refinement does not improve the answer.  Three non-asymptotic
points at one latent dimension, one variant, one seed do not prove that latent stepping is
non-convergent in `dt` in general — but they do rule out "just take smaller steps" as a fix here,
and they are what one expects if the per-step injection is set by the manifold's representation
error (which does not shrink with `dt`) rather than by the truncation error.

### Timing (N=64, one A100 per cell, median of 7 after 2 warm-ups, `block_until_ready`)

| | rollout | ms / latent step | speedup vs the 80-substep FOM (385 ms) |
|---|---|---|---|
| FOM (CN/CG, 4000 substeps, batch 1) | 385 ms | 0.096 | 1x |
| same-`dt` u-only Newmark FOM (`RS=20`, 1000 CG solves) | 161 ms | 0.16 | 2.4x |
| coord `lspg:eq256:weak64`, K=8, `RS=20` | 2443 ms | 2.44 | **0.158x** |
| coord `lspg:eq256:weak64`, K=8, `RS=8` | 1095 ms | 2.74 | 0.351x |
| coord `lspg:eqoff256:weak64` (meshfree) | 2420 ms | 2.42 | 0.159x |
| coord `lspg:full:weak64` | 13.5 s | 13.49 | 0.029x |
| coord `lspg:full:fd` | 62.1 s | 62.09 | 0.006x |
| POD k=8, `RS=20` | 453 ms | 0.45 | 0.849x |
| POD k=8, `RS=8` | 184 ms | 0.46 | **2.09x** |
| POD k=64, `RS=20` | 954 ms | 0.95 | 0.404x |

Separately timed: the **jitted** IC latent solve is 78 ms (K=4), 95 ms (K=8) and **831–841 ms
(K=16** — it runs much closer to the `IC_BUDGET=100` LM cap at 16 latent dimensions, and at K=16
it *does* dominate the `eq256` rollout's own 2.98 s only mildly but is worth 28% of it; this is
disclosed rather than hidden in an end-to-end number).  The full 51-slice decode is 10.0–10.7 ms.
Speedups above are **rollout-only**; the composed online-field-prediction figures (rollout + IC
solve + full decode) are in the JSONs under `speedup_vs_fom_online_field_prediction`.

**No coordinate ROM pays at N=64.**  One K=8 `eq256` latent step costs 2.443 ms = 25.4x one FOM
sub-step (0.096 ms), and it averages 5.19 Gauss-Newton iterations, so ~0.47 ms — ~4.9 FOM
sub-steps — per iteration.  This reproduces, inside a real ROM, the wave testbed's own timing
conclusion: the wave FOM is a cheap SPD CG solve, so surrogate decoding does not pay at these
resolutions.  Cross-cell absolute times are not comparable (different A100s); within-cell ratios
are.

### Flat in N (K=8, `RS=20`, fixed `M`, `m`)

| | N=64 | N=128 |
|---|---|---|
| auto-decoder TRAIN recon | 7.03e-2 | 7.44e-2 |
| ORACLE held-out inferred latents | 1.719e-1 | 1.749e-1 |
| IC fit | 1.125e-1 | 1.068e-1 |
| same-`dt` Newmark FOM (the `RS=20` floor) | 2.386e-3 | 1.081e-2 |
| `lspg:eq256:weak64` | 8.7831e-1 | 8.7831e-1 |
| `galerkin:eq256:weak64` | 8.7831e-1 | 8.7831e-1 |
| `lspg:eq1024:weak256` | 8.846e-1 | 8.680e-1 |
| `lspg:eqoff256:weak64` (meshfree) | 8.791e-1 | 8.779e-1 |
| `lspg:full:weak64` | 8.952e-1 | 9.008e-1 |
| `lspg:full:weakl64` | 8.408e-1 | 8.406e-1 |
| `lspg:full:fd` | 8.767e-1 | 8.637e-1 |
| **ROM / oracle floor** | **5.11x** | **5.02x** |
| ms per latent step, `eq256:weak64` | 2.443 | 2.451 |
| POD projection floors (test) k=6/8/64 | 4.282e-1 / 3.417e-1 / 8.201e-2 | 4.304e-1 / 3.446e-1 / 9.009e-2 |
| POD **ROM** `lspg:full:fd` k=6 | 4.2899e-1 | 4.3115e-1 |
| POD **ROM** `lspg:full:fd` k=8 | 3.4238e-1 | 3.4530e-1 |
| POD **ROM** `lspg:full:fd` k=16 | 2.1875e-1 | 2.2331e-1 |
| POD **ROM** `lspg:full:fd` k=32 | 1.4080e-1 | 1.4684e-1 |
| POD **ROM** `lspg:full:fd` k=64 | 8.3778e-2 | 9.5010e-2 |
| ms per step, POD k=8 / k=64 `full:fd` | 0.453 / 0.954 | 0.528 / 1.475 |
| FOM Newmark `RS=20` rollout (s) | 0.1607 | 0.1945 |

`eq256:weak64` is **identical to five significant figures at N=64 and N=128**, the ROM/floor
ratio moves 5.11 -> 5.02, and the per-step cost is flat (2.443 -> 2.451 ms, +0.3%) while the
number of grid points quadruples — the n-free per-step cost of the hyper-reduced coordinate ROM
is confirmed on wave.  It buys nothing here, because the failure mode is resolution-independent
too: every POD ROM arm is likewise flat in N (k=8: 3.4238e-1 -> 3.4530e-1, +0.9%) and stays
2.5x more accurate than the coordinate ROM at the same k.

The completed cell adds one comparison the N=64 cell alone could not make: **the coordinate
ROM's cost scales better in N than POD's.**  The N=64 and N=128 cells ran on different A100s, so
raw millisecond ratios across them are not admissible; what *is* admissible is a **within-cell
ratio**, where the card's speed cancels, compared between cells.  Those ratios move the right
way for the hyper-reduced arm:

| within-cell ratio | N=64 | N=128 |
|---|---|---|
| coordinate `eq256:weak64` step / POD k=8 `full:fd` step | 5.39x | 4.64x |
| coordinate `eq256:weak64` step / POD k=64 `full:fd` step | 2.56x | 1.66x |
| coordinate `eq256:weak64` speedup vs the FOM rollout | 0.158x | 0.204x |
| POD k=6 `full:fd` speedup vs the FOM rollout | 0.871x | 1.004x |
| POD k=64 `full:fd` speedup vs the FOM rollout | 0.404x | 0.339x |

POD LSPG runs on the full grid, so its work grows with n; the hyper-reduced coordinate step does
not.  Accordingly the coordinate arm closes ground on POD (5.39x -> 4.64x at k=8, 2.56x -> 1.66x
at k=64) and improves against the FOM (0.158x -> 0.204x) over the same 4x in grid points that
*degrades* POD k=64 (0.404x -> 0.339x).  So hyper-reduction delivers its mesh-independence
promise here — the half of the recipe that works — while the accuracy failure is untouched by
resolution, and the coordinate arm is still 4.9x below break-even at N=128.

## The published wave-ROM comparison, and why it is not a like-for-like number

The paper rebuttal's wave results were a **frozen POD basis with a damped LSPG rollout**:
4.71e-3 at 0.7x speedup (N=64), 4.94e-3 at 1.6x (N=128), 1.63e-2 at 3.3x (N=256).

**This experiment can neither reproduce nor contradict those numbers, and does not try.**  The
problem here is a 5-parameter family (`cx, cy, w, a, log c`, with `c` spanning a factor of 4),
a *global* basis fitted to 512 training trajectories, and evaluation on 16 trajectories from a
fresh seed over one crossing time.  On that problem the **POD projection floor itself** — the
best any frozen linear basis can do, before any ROM — is 4.282e-1 at k=6 and 8.201e-2 at k=64,
and POD-LSPG achieves 8.378e-2 at k=64, i.e. it is *at* its floor.  The two reported errors are
therefore measurements of different problems; **which of the many possible differences (parameter
range, basis construction, rank, metric normalisation, horizon, damping) accounts for the gap
cannot be determined from anything in this experiment.**  The one point of contact is the speedup
side, which is consistent: 0.40–0.85x here for POD at `RS=20` and 2.09x at `RS=8`, against 0.7x
reported there.

## Reading the results

1. **The harness checks out, every tested rollout completes, and the coordinate ROM is far from
   the FOM.**  The operator checks pass at machine precision (2e-16), the weak-all-modes identity
   at 3.6e-16, V1 at 2.8e-8 (CG-tolerance limited, as expected), the data reproduction exactly.
   All 101 N=64 variants x 16 trajectories complete with no blow-up.  And the coordinate ROM
   lands at 0.59–1.02 traj-RMS, 3.3–6.1x above its own held-out floor, while POD-LSPG lands at
   1.00–1.02x above its own.
2. **A single latent step is accurate; ten snapshot intervals are 53x worse.**  The excess over
   the oracle is +8.7e-3 after one interval (4.5% of the floor, 11x better than doing nothing)
   and +4.6e-1 after ten, growing as `H^1.69`.  Refining the ROM time step 5x makes the final
   error 7.5% worse rather than better.  Whether this is best described as accumulation or as
   amplification is not settled by this diagnostic.
3. **On this PDE the linear ROM is the better ROM, and there is a structural reason available.**
   For a fixed subspace, both Galerkin and LSPG reduce the linear Newmark residual to a
   *time-reversible* three-term recurrence `S(c_{n+1}+c_{n−1}) = 2T c_n` with `S` SPD and `T`
   symmetric; POD-LSPG duly sits at its projection floor with kinematic energy drift 3e-7…3e-6
   and dynamic drift 1.4–1.8%.  A nonlinear manifold makes `S` and `T` state-dependent and loses
   that symmetry, and the coordinate ROM's energy is off by 3–50x with an order-one velocity
   defect.  This contrasts with the Burgers round, where the same solver gave the coordinate
   manifold a 12.6x advantage over POD at matched dimension, and **motivates** testing whether
   the PDE's dissipation is what decides it — it does not establish that, since dissipation is
   confounded here with the PDE, the data family, the decoder, the training budget and the
   integrator.
4. **The hyper-reduction half of the recipe works on wave.**  At K=8, NNLS-EQ with `m = 4M = 256`
   nodes reproduces the full 3844-node grid answer within 1.9% at 5.5x less per-step time, from a
   meshfree pool as well as from grid nodes, at both N=64 and N=128, with per-step cost flat in
   `n` (2.44 -> 2.48 ms as `n` quadruples).  Demonstrated on these cells, not proven in general.
5. **Stage 1 is a separate negative with a proven mechanism.**  Starting LM at the true parameter
   and watching it move to an 18% lower objective on 16/16 trajectories proves the truth is not a
   minimiser of the implemented objective; six restarts do not help.  The *physical* reason (the
   decoder's 5.6e-3 residual at the truth, and lower-amplitude fields being more PDE-consistent
   without a damping term) is a hypothesis this experiment does not measure.
6. **Cost.**  No coordinate ROM pays at N=64 or N=128: the wave FOM is 4000 SPD CG solves at
   0.096 ms each; one hyper-reduced K=8 latent step is 25.4x that.  The jitted IC solve
   (78–95 ms at K=4/8) is no longer the bottleneck it was in the Burgers round, but at K=16 it
   costs 831–841 ms and must be quoted.

## Candidates to test next

Suggested by the diagnosis, **none tested here and not ranked**: (i) an energy-aware or
symplectic latent step — the best coordinate arm at every K is `weakl64`, whose `lam^-1/2`
weighting changes the energy trend, but its dynamic energy grows 58–845x, so this is a
correlation and not evidence that an energy constraint would help; (ii) a manifold trained with a
*rollout* loss rather than a per-snapshot reconstruction loss, so the injected error is
penalised; (iii) a more accurate decoder — but the ROM/floor ratio grows with K across the three
models trained here, so this is not obviously the lever.

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
- **No confidence intervals or seed-to-seed variability anywhere.**  The 16 test trajectories and
  the 32 (correlated: 8 trajectories x 4 start times) step-diagnostic cases are reported as point
  estimates only; medians and maxima are in the tables and JSONs, spreads are wide (`traj_rel_max`
  is typically 1.3–1.5x the mean).
- The "oracle floor" is a **budgeted** nonlinear latent inference (LM, budget 60 in the ROM cells,
  80 in the step diagnostic, best of two initialisations), not a certified projection minimum.
  Optimisation error inflates it, which *understates* the ROM/floor ratios.
- K=4, K=8 and K=16 are three **separately trained** models, so the trend in the ROM/floor ratio
  cannot be attributed to latent dimension alone.
- The `dt` sweep is one latent dimension, one variant, one decoder, three non-asymptotic points.
- The N=128 cell is complete (`"finished": true`); its POD control arms and timing block are now
  reported.  Cross-cell **absolute** times remain incomparable (different A100s); the flat-in-N
  claims above are within-cell ratios plus the N=64/N=128 comparison of *relative* growth, which
  is the weaker but valid reading.
- Strong LSPG and the weighted weak form are least-squares (normal-equation) schemes; the
  structural argument in "Finding 1" is a derivation about a *fixed* subspace, and the reduced
  LSPG operator is **not** `VᵀLV` (that is the Galerkin operator).  The evidence for the
  conclusion is the measured ratio and energy drift, not the derivation.
- The step diagnostic's "excess" is a difference of relative-error norms against a
  **refit** oracle; it does not isolate injection from amplification.
- Timing speedups are rollout-only and hardware-specific; the K=16 IC-solve cost (831–841 ms) is
  ~10x the K=4/K=8 cost and is disclosed separately rather than folded into a speedup.
- Two-reviewer gate: two Codex adversarial passes — `CODEX-REVIEW-2026-08-16.md` (harness review
  before the fan-out; all MUST items applied) and `CODEX-REVIEW-2-2026-08-17.md` (this document's
  numbers audited against the JSONs and its claims audited for overreach; every mismatch it found
  is corrected above and its overreach findings are reflected in the wording) — plus self-review.
  A second independent *model family* reviewer could not be spawned from this fork.
