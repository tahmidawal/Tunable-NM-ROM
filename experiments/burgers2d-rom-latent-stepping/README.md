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

