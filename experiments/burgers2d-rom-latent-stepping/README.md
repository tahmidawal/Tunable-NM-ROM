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

RESULTS: pending (this section is filled in when the jobs land).
