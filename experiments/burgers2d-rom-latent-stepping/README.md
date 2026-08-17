# Burgers-2D INR-decoder latent-stepping ROM (2026-08-16)

A real reduced-order model on the FiLM **coordinate** decoder for the Burgers-2D
testbed (`exp/2026-08-14-burgers2d-coord-rom`): the manifold is the nonlinear
coordinate net, the online solve is latent time stepping by warm-started
Gauss–Newton / Levenberg–Marquardt on the **discrete backward-Euler residual of
the FOM**, and hyper-reduction is the weak-form Galerkin recipe found by the
Poisson objective study (`exp/2026-08-16-poisson2d-rom-objective`).

Everything numeric is f64.  The ROM knows the initial condition `u0`, the
viscosity `nu`, the PDE and the training set — never the held-out trajectory.

## Design

| piece | what |
|---|---|
| FOM | `burgers2d_film.make_rollout` imported verbatim: `u_t + u(u_x+u_y) = nu lap u`, u=0 walls, BE dt=0.005 x 50, first-order sign-upwind advection, centered diffusion, Newton/BiCGStab (converged-residual guard). Its `residual(u,u_prev,nu)` **is** the operator the ROM steps. |
| family | `z=(cx,cy,w,a,log nu)`; TRAIN = first 512 trajectories of the seed-0 draw (identical to the sweep); **TEST = 16 fresh trajectories from `TEST_SEED=1`** (the sweep checkpoints were model-selected on the VAL split, so VAL is not a test set — Codex MUST-fix). Data are regenerated on the cluster from seed; `build_data` aborts if any FOM trajectory has rel. residual > 1e-8 (observed ~1e-12). |
| decoder (Stage 2) | `u(x;z) = eps * b(x) * FiLM(x;z)`, hard Dirichlet factor `b = 16x(1-x)y(1-y)` (`BC_MODE=poly`), `ms_parametric` FiLM net (5x256, n_freq Nyquist-capped 31 at N=64, ~440k params), trained as an **auto-decoder**: one latent per (trajectory, time) snapshot (26 112 rows at N=64), latents initialised from the top-K POD coefficients (no true z anywhere), inverse-energy row weights (relative-error training), lazy per-row Adam on the latents, time-smoothness penalty `T_SMOOTH * ||z_{i,n+1}-z_{i,n}||^2` on paired rows. `blat_train_ad.py`. |
| POD control | same solver on `u = V c`, `V` from the TRAIN snapshots (spatial Gram, host f64), k in {8,16,32,64}. |
| residual (strong form, control) | **point-local**: BE residual at m interior nodes from the decoder at the m 5-point stencils (`be_residual_from_stencil` == FOM interior residual to 0.0, asserted in every run). Collocation `full | rand<m> | biased<m>` (biased by the known u0) `| offgrid<m>` (continuous strong form via autodiff derivatives — a *continuum* objective, labelled as such). |
| residual (weak form, the recipe) | test modes = M lowest discrete sine modes `phi_i` (exact eigenvectors of the FOM's ghost-zero Laplacian, `-L phi_i = lam_i phi_i`); `R_i(z) = w_i [phi_i^T(u-u_n) + dt(phi_i^T N(u) + nu lam_i phi_i^T u)]`, `w_i=(1+dt nu lam_i)^-alpha`, alpha=1. `weak<M>`: `N` = the FOM's upwind advection at the quadrature nodes (needs the 5-point stencil, exact FOM operator, hyper-reduces on grid nodes); `weakc<M>`: continuum advection integrated by parts `-1/2 sum_q om_q u_q^2 (dphi_i/dx+dphi_i/dy)(x_q)`, `lam_i^c = pi^2(kx^2+ky^2)` — no decoder derivatives, **meshfree** (nodes anywhere), but it targets the continuum PDE (O(h) from the upwind FOM). Hyper-reduction: `eq<m>` (grid nodes) / `eqoff<m>` (meshfree pool of 4096 points) — capped Lawson–Hanson NNLS quadrature weights fitted to reproduce the mode projections of DECODER-OUTPUT snapshots (`u` and `N(u)` / `u^2` at 64 training latents). |
| solvers | `lspg`: LM on `||R||` (on-device `lax.while_loop`, same acceptance rule as the Python reference; whole rollout as one `lax.scan` for timing); `galerkin`: damped Newton on `J_D^T W R = 0` (weak form: test functions = mode-projected tangents `Phi^T dD/dz`), backtracking on `||g||`. Warm start `z_n`; budget 30 attempts/step; stop on RMS residual <= 1e-9 x RMS(u0), stall, or lambda blow-up. Cold start `z_0` = best-of LM data-misfit to the **known u0** from {mean training latent at t=0, latent of the training trajectory with the nearest IC}. |
| Stage 1 pilot | `blat_stage1.py`: the sweep's (z,t)-conditioned decoder (5-dim true-parameter latent, N=64 checkpoint); solve for z by LM on (a) IC misfit only, (b) the space-time BE residual (interior + FOM boundary rows; a PDE-consistency ablation, never sees u0), (c) both (IC block weight `IC_W`, default sqrt(50) = RMS-balanced; IC_W=1 sensitivity arm). Inits: family mean / nearest-IC training z, viscosity coordinate set from the known nu. |
| metrics | trajectory rel-L2 = mean over the 51 slices of `||u_ROM - u_FOM||/||u_FOM||`, mean/median/max over 16 test trajectories **over completed rollouts** (blow-ups counted separately, never averaged in); per-time curves with survivor counts; iterations (Jacobian evals) and LM attempts, cold step 0 vs warm; ROM-vs-FOM timing: warm-up 2, median of 7, `block_until_ready`, same device, FOM = the same jitted implicit solver at batch 1; ROM rollout-only (device scan) plus IC-fit and 51-slice full decode timed separately, end-to-end speedup composed. |
| oracle floors (labelled) | POD projection of the test trajectories; per-snapshot LM-inferred latents on the held-out field (mean / nearest-train inits, budget 40). |

Files: `blat_common.py` (all machinery), `blat_train_ad.py`, `blat_rom.py`,
`blat_stage1.py`, `cluster/make_cell.sh` + `cluster/launch.sh` (one job dir per
cell under `paralab/tawal01/blat/<cell>`, sha256-verified staging, exit-42 GPU
preflight), `CODEX-REVIEW-2026-08-16.md` (adversarial review; all MUST items
applied — see the commit message of `3efd6de`).

## Smoke (local GB10, N=16, N_TRAIN=24, 300 training steps — interfaces only)

`smoke/`: FOM residual of the FOM trajectory 2.3e-13; point-local residual ==
FOM residual to 0.0; every variant (fd / weak / weakc x full / eq / eqoff /
galerkin, POD control) runs end-to-end; Stage 1 `both` recovers the true z to
5.7e-2 (traj rel 1.1e-2 vs oracle 3.0e-3) while `ic` alone (2.2e-1) and `resid`
alone (9.7e-1) do not.

## Cluster runs (Tufts, submitted 2026-08-16, commit 3efd6de)

| cell | job | GPU | what |
|---|---|---|---|
| `s1_n64` | 2468405 | A100 80GB | Stage 1, N=64, 16 test, budget 100, IC_W sqrt(50) and 1 |
| `ad_n64_k4` | 2468407 | A100 80GB | auto-decoder K=4 (60k steps, batch 128, P_SUB 2048) + full ROM study |
| `ad_n64_k8` | 2468411 | H100 | K=8 |
| `ad_n64_k16` | 2468412 | A100 40GB | K=16 |
| `ad_n128_k8` | 2468414 | A100 40GB | K=8 at N=128 (flat-in-N check at fixed k, M, m) |

All five cells: `jax_backend=gpu`, `JAX_DEFAULT_MATMUL_PRECISION=highest`, f64, data
regenerated from seed (max FOM rel. residual 1e-12), isolated dirs, checksummed pull,
cluster dirs deleted.  Raw JSONs/logs/checkpoints under `runs/`; all tables regenerated by
`blat_summarize.py` -> `SUMMARY_TABLES.md`.  (`ad_n128_k8` was resubmitted as job 2470764
after the first attempt OOM'd a 40 GB A100 in the post-training eval — fixed by chunking.)

## Results

### Stage 1 — space-time LSPG on the (z,t) sweep decoder (N=64, 16 fresh test trajectories)

| arm | IC_W | traj rel-L2 mean | median | max | mean \|z − z*\| |
|---|---|---|---|---|---|
| oracle (true z) | — | **3.83e-3** | | 1.65e-2 | 0 |
| `ic` (IC misfit only) | — | 2.26e-1 | 1.71e-1 | 8.66e-1 | 0.67 |
| `resid` (space-time BE residual only; PDE-consistency ablation, never sees u0) | — | 9.52e-1 | 9.43e-1 | 1.16 | 1.71 |
| `both` | sqrt(50) | 2.22e-2 | 1.15e-2 | 1.80e-1 | 5.8e-2 |
| `both` | **1** | **7.29e-3** | **3.47e-3** | 5.20e-2 | 1.2e-2 |

The space-time residual plus the known IC identifies the 5 true parameters to 1e-2 and puts
the trajectory error at the oracle (median 3.5e-3 vs 3.8e-3; the mean is one outlier); IC
alone is under-determined (nu, and the width/amplitude trade-off) and residual alone falls
into the low-amplitude branch.  Sanity: FOM trajectory residual through the FOM's own
residual 2.8e-13; our point-local + boundary-row residual matches it to 0.0.

### Stage 2 — auto-decoder latent-stepping ROM, N=64 (16 fresh test trajectories, 50 steps)

Floors and headline numbers (trajectory rel-L2 = mean over 51 slices; mean over 16
trajectories; **zero blow-ups anywhere**: 0/16 in every one of the 3 x 14 coord variants and
the 3 x 16 POD variants):

| | K=4 | K=8 | K=16 |
|---|---|---|---|
| auto-decoder TRAIN recon (learned latents) | 6.37e-3 | 3.52e-3 | 3.95e-3 |
| ORACLE held-out inferred latents (per snapshot, budget 40) | 5.24e-2 | 1.15e-2 | 7.40e-3 |
| IC fit (cold start from the known u0; = oracle at t=0) | 8.09e-2 | 2.31e-2 | 1.75e-2 |
| ROM `lspg:full:fd` (strong FD residual, full grid) | 8.90e-2 | 2.01e-2 | 1.11e-2 |
| ROM `galerkin:full:fd` | 8.51e-2 | 1.80e-2 | 9.51e-3 |
| ROM `lspg:rand512:fd` (random 512 nodes, strong form) | 9.39e-2 | 2.30e-2 | 1.60e-2 |
| ROM `lspg:offgrid512:fd` (continuum strong form, autodiff) | 2.67e-1 | 1.75e-1 | 9.07e-2 |
| ROM **`lspg:full:weak64`** (weak Galerkin, 64 sine test modes) | 7.74e-2 | **1.65e-2** | 9.62e-3 |
| ROM **`lspg:eq256:weak64`** (NNLS-EQ, m=256 nodes) | 7.74e-2 | 1.74e-2 | 1.10e-2 |
| ROM `lspg:eq512:weak64` | 7.75e-2 | 1.68e-2 | 9.91e-3 |
| ROM `lspg:full:weak256` / `eq1024:weak256` | 8.08e-2 / 8.05e-2 | 1.67e-2 / 1.68e-2 | **8.97e-3** / 9.15e-3 |
| ROM `lspg:full:weakc64` (continuum weak form; = `eq512` / `eqoff512` within 1%) | 8.93e-2 | 4.56e-2 | 4.31e-2 |
| POD-LSPG k=8 / 16 / 32 / 64 (`lspg:full:fd`) | 2.09e-1 / 9.73e-2 / 4.32e-2 / 1.40e-2 | same | same |
| POD projection floors k=8 / 16 / 32 / 64 (test) | 1.96e-1 / 8.90e-2 / 3.79e-2 / 1.22e-2 | | |

Per-time (t-index 0/10/20/30/40/50), K=8: oracle 2.32e-2 / 1.44e-2 / 1.10e-2 / 9.2e-3 /
8.4e-3 / 8.1e-3; `full:weak64` 2.31e-2 / 1.91e-2 / 1.65e-2 / 1.47e-2 / 1.40e-2 / 1.37e-2;
`eq256:weak64` 2.31e-2 / 2.05e-2 / 1.75e-2 / 1.55e-2 / 1.47e-2 / 1.44e-2; `full:fd`
2.31e-2 / 2.21e-2 / 2.00e-2 / 1.89e-2 / 1.86e-2 / 1.83e-2; `full:weakc64` 2.31e-2 →
6.20e-2 (grows monotonically: the continuum operator drifts away from the upwind FOM).

Solver behaviour (K=8): warm steps take 5.7 (weak) / 6.7 (fd) Jacobian evaluations, cold
step 0 takes 8–13; every step ends on the stall criterion (the residual floor is the decoder
error, tolerance 1e-9 never binds); LM lambda never saturates.

Timing (per test trajectory 0, median of 7 after 2 warm-ups, block_until_ready, one device
per cell; K=8 cell ran on an H100, K=4 on an A100-80GB, K=16 on an A100-40GB — ratios
within a cell are like-for-like, absolute times across cells are not):

| K=8 (H100) | rollout | speedup vs FOM 321 ms |
|---|---|---|
| FOM (same jitted implicit Newton/BiCGStab solver, batch 1) | 321 ms | 1x |
| coord `eq256:weak64` (device scan; 4.7 ms/step) | 196 ms | **1.64x** |
| coord `eq512:weak64` | 341 ms | 0.94x |
| coord `eqoff512:weakc64` (meshfree, single decoder eval per node) | 114 ms | 2.8x (but 4.5e-2 vs the FOM) |
| coord `full:weak64` / `full:fd` | 2.0 s / 2.4 s | 0.16x / 0.13x |
| POD k=8 / 64 (`full:fd`) | 33 ms / 56 ms | 9.6x / 5.7x |
| + IC solve (python LM, 100-attempt budget, 2 starts) | 610 ms | end-to-end `eq256` 0.40x |
| + full decode of all 51 slices | 5.4 ms | |

K=16 (A100-40GB): FOM 424 ms, `eq256:weak64` 466 ms (0.91x), `eq512` 823 ms, `eqoff512:weakc64`
235 ms (1.8x); K=4 (A100-80GB): FOM 420 ms, `eq256` 185 ms (2.3x).

### N=128, K=8 — flat in N at fixed (k=8, M, m) (job 2470764, A100-40GB)

| | N=64 (H100) | N=128 (A100-40GB) |
|---|---|---|
| auto-decoder TRAIN recon | 3.52e-3 | 3.77e-3 |
| ORACLE held-out inferred latents | 1.15e-2 | 1.28e-2 |
| IC fit | 2.31e-2 | 2.41e-2 |
| `lspg:full:fd` / `galerkin:full:fd` | 2.01e-2 / 1.80e-2 | 2.24e-2 / 1.84e-2 |
| `lspg:rand512:fd` | 2.30e-2 | 2.76e-2 |
| `lspg:full:weak64` / `eq256` / `eq512` | 1.65e-2 / 1.74e-2 / 1.68e-2 | 1.90e-2 / 1.89e-2 / 1.94e-2 |
| `lspg:full:weak256` / `eq512` / `eq1024` | 1.67e-2 / 1.70e-2 / 1.68e-2 | 1.66e-2 / 1.72e-2 / 1.67e-2 |
| `lspg:*:weakc64` (continuum weak form) | 4.5e-2 | 3.0e-2 |
| POD-LSPG k=8 / 16 / 32 / 64 | 2.09e-1 / 9.7e-2 / 4.3e-2 / 1.40e-2 | 2.16e-1 / 1.05e-1 / 5.0e-2 / 1.78e-2 |
| ROM (`eq256:weak64`) / oracle | 1.51x | 1.48x |
| FOM rollout | 321 ms | 1160 ms |
| coord `eq256:weak64` rollout (speedup) | 196 ms (1.64x) | 269 ms (**4.3x**) |
| coord `eq512:weak64` | 341 ms (0.94x) | 477 ms (2.4x) |
| coord `eqoff512:weakc64` | 114 ms (2.8x) | 161 ms (7.2x) |
| POD k=8 / k=64 | 33 / 56 ms (9.6x / 5.7x) | 50 / 101 ms (23x / 11x) |
| ms per ROM step, `eq256:weak64` | 4.7 | 6.5 |

At fixed (k, M, m) the ROM error is flat in N (the ROM/oracle ratio 1.51 -> 1.48; the
`weak256` numbers are identical to 3 digits), the per-step ROM cost is flat up to the GPU
change (H100 -> A100-40GB) while the FOM's grows 3.6x, so the speedup grows from 1.6x to
4.3x (m=256) — the n-free-cost claim, live in an end-to-end nonlinear ROM.  The
continuum-weak-form gap to the upwind FOM shrinks 4.5e-2 -> 3.0e-2 as h halves, consistent
with it targeting the continuum PDE (the FOM's own O(h) error shrinks 3.8e-2 -> 1.8e-2 vs the
512^2 reference).  POD k=64 with `weak64` (square Petrov-Galerkin) again unstable, as at N=64.


## Reading the results

1. **The nonlinear coordinate manifold ROM works on advection-dominated Burgers, and it is
   the manifold that does the work.**  With the *same* solver, same residual, same
   collocation and same test modes, POD-LSPG at k=8 sits at 2.1e-1 (12.6x worse than the
   coord ROM at K=8) and needs k=64 (1.4e-2) to match the K=8 coordinate decoder (1.65e-2);
   at K=16 the coord ROM (9.0e-3) beats POD-64.  Everything is stable (no blow-up in 16 x
   14 x 3 rollouts), warm-started GN converges in ~6 Jacobian evaluations per step, cold
   starts in ~10.  This is the answer to reviewer 5mgh's "the stability comes from the solver
   formulation, not the nonlinear manifold": here the solver is identical across arms and the
   accuracy gap is the manifold's.
2. **The ROM tracks the decoder's own held-out floor.**  Coord ROM / oracle-inferred-latent
   floor = 1.48x (K=4), 1.44x (K=8), 1.21x (K=16, `weak256`).  The residual is dominated by
   the sharpest early states: at t=0 every variant equals the IC-fit misfit (2.3e-2 at K=8),
   which is exactly the oracle's t=0 number — the decoder cannot represent the initial bump
   better than that; by t=T the ROM is 1.7x above the oracle (1.37e-2 vs 8.1e-3).  K below
   the intrinsic dimension (K=4 < 6 = 5 params + time) is representation-limited (oracle
   5.2e-2 despite a 6.4e-3 train recon).
3. **The residual objective matters less than on Poisson, hyper-reduction matters more.**
   The strong FD residual is only 1.2x worse than the weak Galerkin form (2.0e-2 vs 1.65e-2
   at K=8) — the backward-Euler operator `I + dt(...)` amplifies grid-scale decoder error by
   O(1 + dt nu/dx^2) ~ 2–3, not by (N-1)^2 as the Poisson Laplacian does — and even random
   collocation of the strong residual works (2.3e-2 at m=512, unlike Poisson).  What the weak
   form buys is quadrature: **NNLS-EQ with m=256 = 4M nodes reproduces the full-grid weak
   number (1.74e-2 vs 1.65e-2) at 1/10 of the per-step time**, m=512 within 2%; the NNLS fit
   residual (0.6% at m=256, 0.1% at m=512) tracks it.  Off-grid strong-form collocation
   (autodiff derivatives) fails (0.09–0.27), as on Poisson.
4. **The continuum weak form (`weakc`, meshfree, no decoder derivatives) is a different
   model.**  It lands at 4.3–4.6e-2 *versus the upwind FOM* at every K, growing in time — the
   same size as the FOM's own discretization error at N=64 against the 512^2 reference
   (3.8e-2, `burgers2d-coord-rom` README) — i.e. it is consistent with the continuum PDE the
   FOM only approximates to first order.  Judged against the FOM it is 2.7x worse than
   `weak`; judged against the continuum it may well be better (not measured here; the 512^2
   reference exists for that comparison).  It is also the cheapest coordinate ROM (one
   decoder evaluation per node, meshfree pool as good as grid nodes: 114 ms vs 321 ms FOM).
5. **Cost.**  Per-step work is n-free by construction (m decoder evaluations x 5 stencil
   points x (1 + K) for the Jacobian, an M x m matvec, a K x K solve); at N=64 the coord ROM
   is at parity with the FOM (0.9–2.3x, device dependent) while POD is 6–12x faster: the
   5x256 f64 FiLM decoder is the cost.  Two honest caveats: (i) the online IC solve is a
   Python-loop LM (0.6–1.2 s) and dominates end-to-end; it is an implementation choice (a
   jitted LM would be ~10x cheaper) — reported separately so the reader can compose; (ii)
   everything is f64 on GPUs whose f64 rate is 1/2 (A100/H100) — f32 inference of the decoder
   would be a further 2x.  Different cells ran on different GPUs; only within-cell ratios are
   comparable.
6. **Galerkin vs LSPG.**  Identical to three digits on the weak form; on the strong form
   Galerkin is 10–15% better; no stability difference.  Petrov-Galerkin with M test modes
   needs M > k: POD k=64 with `weak64` (square system) is unstable (1.66) while `weak64` with
   k <= 32 and `fd` with any k are fine — documented as a caveat, not used for any claim.

## Follow-up (2026-08-17) — k ladder, multi-seed, EQ knobs, honest cost

`followup/` extends this study without touching anything above: new code in `followup/`,
new results in `runs/followup/<cell>/`, generated tables in `followup/FOLLOWUP_TABLES.md`,
figures in `followup/figs/` (also copied to `Plots/`).  Every table above is unchanged;
the K=4/8/16 rows of the frozen round are reused verbatim as three points of the new k
ladder.

Everything below: N=64 unless stated, 16 fresh test trajectories from `TEST_SEED=1`, f64,
`jax_backend=gpu`, `JAX_DEFAULT_MATMUL_PRECISION=highest`, one isolated cluster job
directory per cell, data regenerated on the cluster from the seed, checksummed pull,
cluster directories deleted.  Adversarial review before the fan-out:
`CODEX-REVIEW-followup.md` (all MUST items applied; disposition table in that file).

### New code

| path | what |
|---|---|
| `blat_train_ad.py` | + `TRAIN_SEED`: the FiLM network initialisation and the minibatch / collocation-point order.  The per-snapshot latents are initialised **deterministically** from the top-K POD coefficients, and the data draw, the split and the `TEST_SEED` test set never see it — so a multi-seed sweep varies training randomness only.  Default = the data seed, reproducing the frozen weights and latents; non-default seeds get their own file name (`_S<seed>`), as do their ROM reports. |
| `followup/fu_common.py` | an **exact** `lax.while_loop` port of `ms_autodecoder.lm_solve` (the Python-loop LM that `blat_common.fit_ic` uses for the online cold start, which dominated end-to-end time at 0.6–1.3 s): same damping, acceptance, both stopping tests applied only after an accepted step, same accounting.  Plus the nearest-training-IC rule by **field** distance at the tested mesh, matching `blat_rom.py`. |
| `followup/fu_timing.py` | one-GPU, one-process cost measurement.  `MODE=n`: N ladder at fixed (k, M, m) — the coordinate decoder is meshfree so the same N=64 checkpoint runs at every N, with the EQ weights refit per N and the FOM's own Newton residual asserted < 1e-8 before anything is timed.  `MODE=k`: K ladder + the POD control, and (with `POD_KS=` empty) the m/M cost ladder.  2 warm-ups, median of 7, `block_until_ready`, FOM = the testbed's own jitted implicit rollout at batch 1. |
| `followup/fu_summarize.py`, `fu_style.py` | tables + figures straight from the JSONs (verified to reproduce the frozen K=4/8/16 rows exactly). |
| `followup/cluster/{make_cell.sh,launch.sh,fu_cells.sh,pull.sh}` | `blat2/` namespace: one job per cell dir, sha256-verified staging and pull, `squeue` checked before and after every submit, `git_commit` + `git_dirty` recorded in every batch script, cluster dirs deleted after a verified pull. |

`blat_common.fit_eq_weights` additionally reports per-row NNLS fit diagnostics (median /
p95 / max of |Gw−b|/|b| over the fitted rows) so a single badly-fitted test mode cannot
hide behind a small global norm; the max is dominated by rows whose target projection is
near zero and should be read next to the median.

### Cells

| cell | job | what |
|---|---|---|
| `bk_K{2,6,12,24,32}` | 2481734–2481748 | k ladder: retrain the auto-decoder at each K (60k steps, the frozen budget), then the ROM sweep.  K=4/8/16 are the frozen `runs/ad_n64_k{4,8,16}` cells, reused verbatim. |
| `bs_S{1,2}` | 2481749/2481753 | multi-seed: K=8, `TRAIN_SEED` 1 and 2 (seed 0 = the frozen `runs/ad_n64_k8`) |
| `bm_m` | 2481756 | m ladder at (K=8, M=64): m ∈ {64…1024} for the exact-FOM weak form on grid nodes, and for the continuum weak form on grid nodes *and* a meshfree pool, + the full grid |
| `bm_M` | 2481759 | M ladder at m ≈ 4M |
| `bt_m` | 2481763 | per-rollout / per-iteration cost of every m and M ladder point, one GPU, one process |
| `bt_n` | 2488105 | cost vs N on one GPU, N ∈ {32,64,128,256} sequential in one process |
| `bt_k` | 2488123 | cost vs K on one GPU + the POD control |

### k ladder — the manifold is worth ~8x in k, and the knee is the intrinsic dimension

Trajectory rel-L2 (mean over the 51 slices, then over 16 held-out trajectories); **zero
blow-ups in every cell**.  Figure: `followup/figs/burgers_k_ladder.png`.

| K | 2 | 4 | 6 | 8 | 12 | 16 | 24 | 32 | (POD only) 64 |
|---|---|---|---|---|---|---|---|---|---|
| coordinate ROM `full:weak64` | 4.47e-1 | 7.74e-2 | 1.73e-2 | **1.65e-2** | 1.13e-2 | 9.62e-3 | 1.02e-2 | 1.47e-2 | — |
| coordinate ROM `eq256:weak64` | 4.48e-1 | 7.74e-2 | 1.81e-2 | 1.74e-2 | 1.22e-2 | 1.10e-2 | 1.21e-2 | 1.85e-2 | — |
| inferred-latent floor (oracle) | 1.97e-1 | 5.24e-2 | 1.41e-2 | 1.15e-2 | 8.61e-3 | 7.40e-3 | 6.35e-3 | 5.79e-3 | — |
| POD, **same weak objective** `full:weak64` | 6.06e-1 | 3.93e-1 | 2.70e-1 | 2.09e-1 | 1.36e-1 | 9.50e-2 | 5.74e-2 | 4.06e-2 | 1.66 (square) |
| POD-LSPG `full:fd` (frozen round's control) | 6.06e-1 | 3.92e-1 | 2.70e-1 | 2.09e-1 | 1.38e-1 | 9.73e-2 | 6.03e-2 | 4.32e-2 | 1.40e-2 |
| POD projection floor | 6.02e-1 | 3.75e-1 | 2.57e-1 | 1.96e-1 | 1.27e-1 | 8.90e-2 | 5.39e-2 | 3.79e-2 | 1.22e-2 |

1. **The knee is at the intrinsic dimension 6** (5 family parameters + time): the ROM falls
   5.8x from K=2 to K=4 and another 4.5x from K=4 to K=6, then flattens (1.73e-2 → 9.6e-3
   over K=6…16).  K=4 < 6 is representation-limited — the *oracle* is already 5.2e-2 there.
2. **POD needs ~7x the latent dimension for the same accuracy.**  With the same solver, the
   same residual and the same test modes, POD at k=8 is 2.09e-1 — **12.7x** the coordinate
   ROM at K=8.  Interpolating the POD `full:fd` curve *inside* the measured range (k=32
   4.32e-2 → k=64 1.40e-2, slope k^−1.63), POD needs **k ≈ 60** to reach the coordinate
   ROM's K=8 value of 1.65e-2 — a **7.4x** reduction in latent dimension for the same
   accuracy.  Read the other way round, POD-64 (1.40e-2) is 1.18x *better* than the
   coordinate ROM at K=8 and 1.24x better than at K=12 (1.13e-2 — comparable), and the
   coordinate manifold overtakes it at K=16 (9.62e-3, 1.46x better than POD-64).  So the
   crossover in *accuracy at equal k* is everywhere in favour of the manifold; the crossover
   in *best achievable accuracy* is at coordinate K ≈ 12–16 against POD k=64.  POD beyond
   k=64 was not measured (the stored basis has rank 64).
3. **The ROM tracks its own floor up to K=16.**  ROM/oracle = 1.23 (K=6), 1.44 (K=8), 1.30
   (K=16) — then it detaches: 1.61 at K=24 and 2.53 at K=32, which is the same phenomenon as
   item 4.
4. **Beyond K=16 the ROM stops improving even though the floor does** (K=24 1.02e-2,
   K=32 1.47e-2 against oracle floors 6.35e-3 and 5.79e-3), and the solver visibly works
   harder: warm iterations per step rise from 5.5 (K=6) to 9.5 (K=24) to 15.7 (K=32).  With
   a fixed 60k-step budget the extra latent directions are increasingly poorly conditioned
   for the online solve — the useful range here is K = 6…16.
5. **POD sits just above its projection floor** — `full:fd`/floor ranges 1.006–1.15 over
   k=2…64 — so the linear control is solver-limited by at most ~15% and the gap to the
   coordinate ROM is the manifold's, not the solver's.  The same-weak-objective POD row is
   the like-for-like comparison and is essentially identical to `full:fd` up to k=32
   (within 6%); at k=64 = M it is a square weak Petrov-Galerkin system and is unstable
   (1.66), as in the frozen round — it is excluded from the figure by that condition and
   kept in the table.

### Multi-seed (K=8, three training seeds)

`TRAIN_SEED` changes the FiLM network initialisation and the minibatch / collocation-point
order.  The per-snapshot latents are initialised deterministically from the top-K POD
coefficients, and the data draw, the split and the `TEST_SEED` test set are untouched — so
the spread below is training randomness and nothing else.

| quantity | seed 0 | seed 1 | seed 2 | mean ± std |
|---|---|---|---|---|
| train reconstruction (learned latents) | 3.52e-3 | 3.55e-3 | 3.48e-3 | 3.52e-3 ± 3.3e-5 |
| ORACLE inferred-latent floor (held out) | 1.15e-2 | 1.24e-2 | 1.11e-2 | 1.17e-2 ± 6.3e-4 |
| IC fit at t=0 | 2.31e-2 | 2.15e-2 | 2.09e-2 | 2.18e-2 ± 1.1e-3 |
| ROM `full:weak64` | 1.65e-2 | 1.57e-2 | 1.46e-2 | **1.56e-2 ± 9.6e-4** |
| ROM `eq256:weak64` | 1.74e-2 | 1.61e-2 | 1.52e-2 | **1.62e-2 ± 1.1e-3** |
| ROM `full:fd` (strong-form control) | 2.01e-2 | 1.91e-2 | 1.93e-2 | 1.95e-2 ± 4.9e-4 |
| POD-LSPG k=8 control | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 (std 0 by construction) |

The headline is reproducible to a **6% coefficient of variation over three training seeds on
one fixed test set** (not a confidence interval), and the 12.7x gap to POD is two orders of
magnitude outside that spread.  The weak form's 1.2x edge over the strong FD residual
(1.56e-2 vs 1.95e-2) is ~4 sample standard deviations, so it survives seeding as well.  The
POD basis is a deterministic function of the training snapshots, so its control row carries
no training-seed variance; only `k8:lspg:full:fd` was actually run in all three cells, and
`FOLLOWUP_TABLES.md` marks the other POD variants as seed-0 only rather than reporting a
spurious zero spread for them.

### The hyper-reduction knobs — m and M ladders (K=8)

Per-step times are the median-of-7 device-synced rollout of the `bt_m` cost cell divided by
the 50 steps (one GPU, one process).  Figure: `followup/figs/burgers_eq_knobs.png`.

| m | 64 | 128 | 256 | 512 | 1024 | full grid (3844) |
|---|---|---|---|---|---|---|
| `weak64` (exact FOM operator), grid EQ | 6.54e-2 | 1.95e-2 | 1.74e-2 | 1.68e-2 | 1.67e-2 | 1.65e-2 |
| ↳ ms per ROM step | 2.37 | 3.56 | 5.50 | 9.03 | 17.55 | 61.52 |
| ↳ NNLS relative fit | 2.1e-1 | 4.9e-2 | 6.2e-3 | 1.0e-3 | 1.5e-4 | 0 |
| `weakc64` (continuum), grid EQ | 1.11e-1 | 5.53e-2 | 4.47e-2 | 4.51e-2 | 4.55e-2 | 4.56e-2 |
| `weakc64` (continuum), **meshfree** pool | 1.20e-1 | 5.41e-2 | 4.49e-2 | 4.53e-2 | 4.55e-2 | 4.56e-2 |
| ↳ ms per ROM step (meshfree) | 1.50 | 1.82 | 2.33 | 3.09 | 4.71 | 12.87 |

| M | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|
| full grid | 1.88e-2 | 1.68e-2 | 1.65e-2 | 1.66e-2 | 1.67e-2 |
| NNLS-EQ, m = 4M | 2.86e-2 | 1.83e-2 | 1.74e-2 | 1.67e-2 | 1.68e-2 |
| ms per ROM step, full / EQ | 59.4 / 2.40 | 61.6 / 3.72 | 61.5 / 5.50 | 62.0 / 9.09 | 62.3 / 17.7 |

- **m ≈ 4M is the knee, and it is 11x cheaper than the full grid**: m=256 gives 1.74e-2
  against 1.65e-2 full-grid (5% worse) at 5.50 ms/step against 61.5 ms/step.  m=512 closes
  it to 2% at 9.0 ms/step; m=128 (2M) is already 18% worse and m=64 (M) fails outright.
- **The NNLS fit residual predicts the ROM error**: it falls 2.1e-1 → 1.5e-4 across the
  ladder and the error saturates exactly where it drops below ~1e-3.  It is computed
  offline, before the ROM is ever run.
- **The meshfree candidate pool matches grid nodes** for the continuum weak form at every m
  (4.49e-2 vs 4.47e-2 at m=256) and is the cheapest *coordinate* arm at m=256 (2.33 ms/step,
  one decoder evaluation per node, no stencil; its own m=64/128 points are cheaper still at
  1.50/1.82 ms/step, and every POD arm is cheaper again at ~0.7 ms/step) — but `weakc` is a
  *different model*: it targets
  the continuum PDE, so against the upwind FOM it sits at 4.5e-2 at every m, exactly as in
  the frozen round.  The meshfree pool is not available for the exact-FOM `weak` form,
  which needs grid neighbours for the upwind stencil.
- **M is saturated at 64**: M=16 is genuinely under-resolved (1.88e-2 even on the full
  grid), M=32 is within 2%, and M ≥ 64 changes nothing while costing linearly in m = 4M.

### Online cost vs N — the whole online path can be made mesh-size independent

One GPU, all four meshes measured **sequentially in one process**, 2 warm-ups then the
median of 7 `block_until_ready` repetitions.  The FOM is the testbed's own jitted implicit
rollout at batch 1 — the function that produced the truth — and its Newton residual is
asserted below 1e-8 before anything is timed.  The coordinate decoder is meshfree, so the
*same* N=64 checkpoint runs at every N and only the EQ quadrature is refit.  Figure:
`followup/figs/burgers_cost_vs_N.png`.

(All numbers are test trajectory 0 — the timed trajectory.  The end-to-end columns are
**measured**: the rollout is re-timed *and re-graded* starting from the hyper-reduced cold
start's own latent, so no number here is a composition of pieces measured from different
starting points.)

| N | FOM rollout | ROM `eq256:weak64` | speedup | ms / step | ms / Jacobian | cold start, full grid (python / jitted) | cold start, EQ m=256 | rollout FROM the EQ start (its traj. error) | decode 51 slices | end-to-end (EQ start, with / without decode) |
|---|---|---|---|---|---|---|---|---|---|---|
| 32 | 197 ms | 261 ms | 0.76x | 5.21 | 1.00 | 778 / 31.6 ms | 12.9 ms | 259 ms (1.01e-2) | 2.4 ms | 0.72x / 0.72x |
| 64 | 442 ms | 260 ms | **1.70x** | 5.20 | 1.00 | 863 / 94.9 ms | 16.0 ms | 266 ms (1.09e-2) | 9.8 ms | 1.52x / 1.57x |
| 128 | 1215 ms | 264 ms | **4.61x** | 5.28 | 1.03 | 1278 / 369.7 ms | 18.1 ms | 255 ms (8.90e-3) | 40.8 ms | 3.88x / 4.46x |
| 256 | 2148 ms | 258 ms | **8.32x** | 5.16 | 1.02 | 2818 / 1468.2 ms | 19.2 ms | 250 ms (9.32e-3) | 164.4 ms | **4.95x / 7.96x** |

1. **The rollout is flat: 258–264 ms across a 64x range of degrees of freedom** (N=32→256,
   1024→65536 DOF, ±1.2%), with ~255 Jacobian evaluations and 1.00–1.03 ms per Jacobian
   evaluation at every N, while the FOM grows 10.9x.  The speedup therefore *grows* with the
   mesh: 0.76x → **8.32x**.  The ROM error is flat too (8.90e-3–1.09e-2 on the timed
   trajectory).
2. **Two arms show the same thing from the other side.**  `eqoff512:weakc64` (a genuinely
   meshfree quadrature pool, one decoder evaluation per node) is flat at ~150 ms and reaches
   **14x** at N=256; the *non*-hyper-reduced `full:weak64` grows ~840 ms → ~56 s,
   a factor **67** — steeper even than the FOM's 10.8x, because its per-step work is O(n)
   with a much larger constant (a FiLM decoder evaluation and a Jacobian column per node).
   That arm is the control showing the flatness comes from the hyper-reduction, not from the
   manifold.  Note that the primary `eq256:weak64` arm is mesh-size *independent* but not
   mesh-*free*: its EQ nodes are grid nodes and its residual uses the FOM's upwind stencil.
   Only the `weakc`/`eqoff` arms are meshfree.
3. **The cold start was the last O(n) piece, and it did not have to be.**  Fitting the
   latent to the known u₀ on the full grid costs 34.5 ms at N=32 but 1672 ms at N=256 and
   capped the end-to-end speedup near 1.0x.  Fitting it instead on the *same* m=256 EQ nodes
   with the same weights makes it mesh-size independent — 17–32 ms at every N.  The
   end-to-end numbers in the last column are **measured, not composed**: the rollout is
   re-timed *and re-graded* starting from that hyper-reduced latent (a first version of this
   cell composed the EQ fit time with a rollout timed from the full-grid latent, which the
   verification pass correctly rejected — see `CODEX-REVIEW-followup-2.md`).  End-to-end goes
   from 0.72x at N=32 to **7.96x at N=256** (4.95x if the full field is also decoded at all
   51 slices).  And the price turns out to be nil where it matters: the *t=0* misfit is
   indeed worse (2.11e-2 → 2.51e-2 at N=32, 3.47e-2 → 4.31e-2 at N=256, latent within 2–5%),
   but the *trajectory* error from the hyper-reduced start is the same or slightly better
   (1.01e-2 vs 1.15e-2 at N=32, 9.32e-3 vs 1.07e-2 at N=256) — the warm-started latent solve
   forgets a marginally worse initial guess within a few steps.
4. **The 51-slice full-field decode (2.8 → 181 ms) is genuinely O(n) and is the output, not
   the solve.**  It is reported separately so the reader can compose whichever end-to-end
   number matches their use case; a ROM asked for functionals rather than fields never pays
   it.
5. **The jitted cold start is the same algorithm as the Python one it replaces**: an exact
   `lax.while_loop` port of `ms_autodecoder.lm_solve`, landing on the same latent to 1e-16
   relative in every row, at 2–25x the speed (the ratio shrinks with N because the Python
   loop's per-iteration overhead becomes negligible against O(n) work).

### Per-iteration cost and iterations vs k (one GPU, N=64, FOM 422 ms)

| K | 2 | 4 | 6 | 8 | 12 | 16 | 24 | 32 |
|---|---|---|---|---|---|---|---|---|
| coordinate ROM rollout `eq256:weak64` | 184 ms | 187 ms | 222 ms | 277 ms | 360 ms | 448 ms | 793 ms | 2104 ms |
| speedup vs FOM | 2.30x | 2.25x | 1.90x | 1.52x | 1.17x | 0.94x | 0.53x | 0.20x |
| Jacobian evaluations (50 steps) | 314 | 258 | 246 | 262 | 261 | 274 | 360 | 764 |
| ms per Jacobian evaluation | 0.59 | 0.73 | 0.90 | 1.06 | 1.38 | 1.63 | 2.20 | 2.75 |
| POD-LSPG rollout at the same k | 38 ms | 35 ms | 39 ms | 35 ms | 44 ms | 44 ms | 41 ms | 46 ms |
| POD ms per Jacobian evaluation | 0.22 | 0.19 | 0.20 | 0.18 | 0.23 | 0.23 | 0.21 | 0.24 |

Per-iteration cost grows **4.7x** from K=2 to K=32 (0.59 → 2.75 ms) — the `jacfwd` over the
FiLM decoder is linear in K — and the iteration count is flat at ~260 up to K=16 and then
blows up (360 at K=24, 764 at K=32).  Combined, the rollout is 11x more expensive at K=32
than at K=2 while the *accuracy* is no better than K=12's, which is the cost-side statement
of the K ≤ 16 useful range found above.  **The accuracy/cost sweet spot is K = 6–8**: 1.7e-2
at 1.9–1.5x the FOM, and 4.2x/8.1x once the mesh is 128/256.  The POD control is
6–12x faster than the FOM at every k (its "decoder" is one 4096×k matvec) and its
per-iteration cost barely moves with k — the coordinate manifold buys accuracy per latent
dimension and pays for it in cost per iteration.  This is the honest trade: at N=64 POD-64
(1.40e-2, 5.8x faster than the FOM) and coordinate-K=8 (1.65e-2, 1.5x faster) are
comparable in accuracy, and POD wins on time; by N=256 the coordinate ROM's cost is
unchanged while both the FOM and any full-grid arm have grown ~10x.

### Caveats

- 16 held-out trajectories per cell; means carry tails.  Medians are printed beside the
  means in `followup/FOLLOWUP_TABLES.md`.  Zero blow-ups anywhere in the follow-up: 0/16 in
  every one of the 8 x 4 k-ladder variants, the 3 x 4 seed variants, the 17 m-ladder
  variants and the 10 M-ladder variants.
- Zero blow-ups is not the same as convergence at every step.  Time steps that end on the
  per-step LM budget rather than the residual/stall criterion are counted per cell in
  `FOLLOWUP_TABLES.md` and are **included** in every mean; they are rare up to K=16 and
  common at K=32 (233 of 800 steps for `eq256:weak64`).
- The k ladder trains one decoder per K at one budget (60k steps).  K=24 and K=32 are
  visibly under-trained *for the online solve* (their held-out floors keep improving while
  the ROM does not), so "the ROM stops improving past K=16" is a statement about this
  budget, not about the manifold's capacity.
- Timing tables are one GPU, one process, median of 7 — but `bt_n`, `bt_m` and `bt_k` are
  three different jobs on three different (possibly different-model) GPUs, so numbers may be
  compared **within** each table and not across them.  The k ladder's *errors* are
  hardware-independent (f64) and are compared freely.  Accuracy statistics are always
  16-trajectory means; timing is always test trajectory 0.  The N=128/256 speedups are
  measured at K=8 only, so the "K = 6–8 sweet spot" combines a 16-trajectory accuracy
  statement with a single-trajectory K=8 timing statement.
- "Per-iteration cost is linear in K" is the expected `jacfwd` scaling, offered as an
  explanation; the *measured* change is 4.7x over a 16x increase in K, so other terms
  (the fixed m x 5 stencil evaluation, the M x m matvec) clearly dominate at small K.
- The "knee at the intrinsic dimension" and the "under-trained / poorly conditioned past
  K=16" readings are descriptions consistent with one ladder at one budget; no
  longer-training or conditioning control was run.
- The multi-seed cells were not pinned to one GPU model; only errors are taken from them.
- `weakc` (the continuum weak form, the only variant that admits a meshfree quadrature pool)
  is stated against the upwind FOM only.  It is O(h) away from that FOM by construction; its
  4.5e-2 is the same size as the FOM's own discretization error at N=64 against the 512²
  reference (3.8e-2, `burgers2d-coord-rom` README).  Against the continuum it may well be
  better; not measured.
- The hyper-reduced cold start is only available for grid-node EQ sets: u₀ is known on the
  mesh, so an off-grid node set would need an interpolation step that was not implemented.
  The `weakc`/meshfree arms therefore still report the full-grid cold start.
- `blat_common.test_modes` takes exactly M modes from a stable sort, which splits a
  degenerate `(kx,ky)/(ky,kx)` eigenshell at M = 16, 128 and 256, so the retained set is not
  exactly x/y symmetric at those M.  This is the frozen round's convention and was kept for
  comparability (Codex SHOULD-fix, deliberately not applied — see
  `CODEX-REVIEW-followup.md`); the effect is a fraction of one mode out of M.
- Two-reviewer gate, both archived: `CODEX-REVIEW-followup.md` (adversarial harness review
  before the fan-out; all MUST items applied) and `CODEX-REVIEW-followup-2.md` (a second
  read-only pass that re-derived 1 311 table cells, 143 timing medians and all 8 figures
  from the raw JSONs).  The second pass found no error in any generated table and 19 errors
  in hand-written prose or figure titles, all fixed here — including one that required
  rerunning a cell (the EQ-cold-start end-to-end numbers were compositions, not
  measurements).  Both files carry a per-item disposition table.

## Caveats

- Single training seed; 16 test trajectories per cell (means carry tails: K=8 `eq512:weak64`
  max 3.8e-2, median 1.54e-2).
- The auto-decoder was trained once per K with one budget (60k steps, batch 128, 2048
  points/row/step); K=16's held-out floor (7.4e-3) is likely under-trained relative to K=8.
- Test split = 16 trajectories from `TEST_SEED=1`; the sweep checkpoints (Stage 1) were
  model-selected on the VAL split, so VAL is not used as test anywhere here.
- Cross-cell timing on different GPUs; the FOM baseline is the testbed's fixed-8-iteration
  Newton scan with a converged-residual guard (a harness FOM, not an optimised solver).
- `weakc` accuracy is stated against the upwind FOM only; against the 512^2 reference it is
  a separate measurement (not run).
- The two-reviewer gate: Codex adversarial review (`CODEX-REVIEW-2026-08-16.md`, all MUST items
  applied) + self-review; a second independent model reviewer could not be spawned from this
  fork.  Numerical cross-checks in every run: point-local residual == FOM residual (0.0), FOM
  trajectory residual ~1e-13, POD ortho dev ~1e-15, EQ fit residuals reported.

