# Burgers-2D coordinate-decoder testbed (2026-08-14)

Does the heat-2D coord-decoder result (exp/2026-08-13-heat2d-coord-decoder)
survive a **nonlinear, advection-dominated** PDE? Scalar viscous Burgers on
the unit square with homogeneous Dirichlet walls:

    u_t + u * (u_x + u_y) = nu * lap(u),      u = 0 on the boundary

Advection is the classic linear-ROM killer (slowly decaying Kolmogorov
width); this testbed measures how the POD floor, the grid-tied CP-style
control, and the FiLM coordinate decoder `u(x, y, t; z)` respond.

## FOM discretization

- **Time**: backward Euler, `dt = 0.005`, 50 steps (T = 0.25). dt is FIXED
  across every mesh resolution so cross-N comparisons isolate spatial error
  (time-stepping error is common and cancels), exactly as in heat-2D.
- **Nonlinear solve**: full Newton per implicit step (default `NEWTON_ITERS=8`,
  warm-started from the previous step). The Jacobian is nonsymmetric
  (advection), so inner solves are matrix-free **BiCGStab** with `J*v`
  computed by `jax.jvp` of the residual; `LIN_TOL=1e-10`, `LIN_MAXITER=2000`.
  Every rollout returns per-step relative residuals `||R(u_new)||/||u_prev||`
  and the scripts assert/warn above 1e-8 — Newton non-convergence cannot pass
  silently.
- **Space**: second-order centered diffusion; **first-order upwind** advection
  (upwind direction by sign(u), which stays >= 0 for this family). Upwind is
  monotone and robust at cell Peclet >> 1 on the coarse grids; the price is
  spatial order ~1 in advection-dominated regions (order ~2 where diffusion
  dominates). The self-convergence check below confirms the order observed.
- f64 throughout data generation and reference solves.

## Parameter family and range reasoning

    z = (cx, cy, w, a, log nu), normalized to ~[-1,1] as in heat-2D
    cx, cy ~ U(0.15, 0.85)   blob center
    w      ~ U(0.05, 0.20)   blob width
    a      ~ U(0.5, 2.0)     blob amplitude (= advection speed scale)
    nu     ~ logU(0.01, 0.1) viscosity

- **Front resolvability at the 512 reference**: the viscous front width scale
  is ~4*nu/a >= 4*0.01/2 = 0.02, i.e. ~10 cells at h_512 = 1/511. The
  reference resolves every front in the box.
- **Advection dominance on coarse grids**: cell Peclet a*h/nu on the N=16
  grid reaches 2*(1/15)/0.01 ~ 13 >> 1; global Re = a*L/nu spans 5-200. The
  low-nu/high-a corner is genuinely advection-dominated; the high-nu corner
  is diffusive — the family spans both regimes.
- **Front motion**: the leading front of a positive pulse moves at ~a/2 per
  axis, so displacement over T is up to 0.25 — clearly visible dynamics, and
  extreme near-wall corners (cx=cy=0.85, a=2) reach the wall late in the
  window, forming a thin but resolvable boundary layer (~nu/a >= 0.005 = 2.5
  cells at 512).

## Scripts

All share the seed-0 parameter draw (val = last `N_VAL` of the draw), the
heat-2D env-var interface (`N`, `STEPS`, `GT_STEPS`, `N_TRAIN`, `N_VAL`,
`SEED`, `N_FREQ`, `T_FREQ`, `P_POINTS`, ...), plus `NEWTON_ITERS`, `LIN_TOL`,
`LIN_MAXITER`, `POD_TIME_STRIDE`.

| script | what |
|---|---|
| `burgers2d_validate.py` | pre-training FOM validation: self-convergence at N=64..512 (fixed dt) + Newton iteration counts at parameter-box corners. Run FIRST. |
| `burgers2d_film.py` | one training cell: FOM data gen at resolution `N`, POD floors (f64 Gram, **all** time slices: `POD_TIME_STRIDE=1`, the heat-round adversarial-review fix — POD and neural arms use identical slice sets), grid-tied control, FiLM coord-net. Writes `burgers2d_results_N{N}.json` + checkpoints. |
| `burgers2d_refgen.py` | N=512 f64 reference trajectories for the val params at the `EVAL_TIMES` slices -> `ref512_val.npz`. |
| `burgers2d_convergence_eval.py` | trained nets evaluated natively on the 512 grid vs the reference + coarse-FD data floors. |

Differences from heat-2D beyond the PDE: rollouts return `(snaps, rel_res)`
(Newton audit); POD default is all slices, so the N=256 cell's Gram is
26112^2 f64 (~5.5 GB matrix, ~20+ GB with eigh workspace — fits an 80 GB
A100/H100; request `--mem 96G`); biased training-point sampling follows the
advected front (center shifted by ~a*t/4, width ~2*sqrt(w^2+2*nu*t) +
a*t/4).

## Cluster sweep (parent submits; one isolated job dir per cell)

Per cell N in {16, 32, 64, 128, 256}:

    N=$N STEPS=120000 GT_STEPS=40000 python burgers2d_film.py outdir

plus one refgen job and one convergence-eval job (needs all five checkpoints
+ `ref512_val.npz`, `N_TRAIN`/`N_VAL` in env matching training).

## Validation results (local GB10, f64, seed-independent params)

From the committed `burgers2d_validation.json` (jax_backend=gpu, f64):

- **Self-convergence** (rel L2 vs the N=512 solution, aggregated over the
  t>0 EVAL_TIMES slices, fixed dt):

  | case | N=64 | N=128 | N=256 | order 64->128 | order 128->256 |
  |---|---|---|---|---|---|
  | mid (a=1.2, nu=0.03) | 4.10e-2 | 1.81e-2 | 6.17e-3 | 1.18 | 1.56 |
  | sharp corner (a=2, nu=0.01) | 1.62e-1 | 8.19e-2 | 3.02e-2 | 0.98 | 1.44 |

  Orders sit between 1 (upwind advection) and 2 (centered diffusion), as
  expected; the sharp advective corner is order ~1 at coarse N and improves
  as the front resolves. Errors are ~5x heat's at matched N — Burgers is a
  genuinely harder spatial problem, which is the point of the testbed.
- **Newton behavior at box corners** (iterations to relative residual
  <= 1e-10, max/median over all 50 steps, N=256 and N=512): **max 3
  iterations everywhere** (2 for the mild case), zero unconverged steps,
  final residuals ~1e-14..1e-16. `NEWTON_ITERS=8` has a >2x safety margin.

Pipeline smoke (N=16, N_TRAIN=24, STEPS=400 — interface validation only, not
converged accuracy): all three arms + refgen (8 trajs at 512^2, max Newton
res 5.2e-14) + convergence eval ran end-to-end; smoke JSONs committed under
`smoke/`.

## Local smoke

    source /etc/profile.d/jax-mem.sh
    N=16 N_TRAIN=24 N_VAL=8 STEPS=400 GT_STEPS=200 EVAL_EVERY=100 \
      JAX_DEFAULT_MATMUL_PRECISION=highest \
      jaxrun /home/tahmid/Dev/.venv/bin/python burgers2d_film.py smoke

Smoke validates interfaces and end-to-end execution, not converged accuracy.
