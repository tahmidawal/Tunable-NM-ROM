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

## Results (2026-08-14, Tufts A100 sweep, seed 0, jobs 2357967-69 / 2358516 / 2360592-93)

### In-resolution head-to-head (mean rel-L2, 120k steps)

| N   | POD-6 (equal-dim linear) | POD-24  | grid-tied (rank 24) | FiLM coord-net |
|-----|--------------------------|---------|---------------------|----------------|
| 16  | 2.638e-1                 | 5.165e-2 | 7.873e-2           | 2.985e-3       |
| 32  | 2.751e-1                 | 5.857e-2 | 7.988e-2           | 2.528e-3       |
| 64  | 2.833e-1                 | 6.376e-2 | 8.610e-2           | 3.192e-3       |
| 128 | 2.884e-1                 | 6.717e-2 | 8.499e-2           | 2.833e-3       |
| 256 | 2.912e-1                 | 6.917e-2 | 8.641e-2           | 3.568e-3       |

- The coord-net beats equal-dimension POD-6 by **~90-110x** — the widest gap of
  the four PDEs studied (heat ~30x, wave ~13-16x, Poisson ~7.5x): nonlinear
  advection steepens fronts that linear bases cannot track, while the FiLM
  decoder barely notices them.
- The grid-tied control is stuck ~1.3x ABOVE its own POD-24 ceiling at every N
  (7.9-8.6e-2 vs 5.2-6.9e-2) — the disease signature, now on a nonlinear PDE.

### Convergence + mesh transfer vs the N=512 reference (native 512^2 eval)

| train N | data-floor (discretization bound) | FiLM net vs reference |
|---------|-----------------------------------|-----------------------|
| 16      | 1.469e-1                          | 1.413e-1              |
| 32      | 7.588e-2                          | 7.427e-2              |
| 64      | 3.809e-2                          | 3.986e-2              |
| 128     | 1.761e-2                          | 1.924e-2              |
| 256     | 6.487e-3                          | 8.713e-3              |

- **Strictly monotone, 16x across the sweep** — the largest and cleanest
  convergence range of the program. The net tracks its data floor within a few
  percent through N=128 (slightly BELOW it at N=16/32: the net's smoothness
  partially cancels first-order upwind error in the coarse training data) and
  opens to only 1.34x at N=256 — Burgers error is data-limited everywhere in
  this sweep; no capacity plateau reached yet.
- Caveats: single seed; Nyquist cap varies feature width across N cells
  (n_freq 7/15/31/32/32); min(8192, N^2) points/step; POD rows are
  train-fitted SVD bases, not certified floors for this val metric; the
  data floor reflects the first-order upwind scheme (order ~1-1.6 observed).

### Solver-robustness postmortem (this round's bug find)

The original run corrupted the N=128 cell and the first 512^2 reference with
non-deterministic NaNs: the fixed-length Newton scan kept calling BiCGStab
after convergence, and on a ~machine-eps residual its rho/omega inner products
can underflow to a NaN step (environment/rounding dependent — the identical
replay came back clean). Masked by a `max(x, NaN) == x` audit bug. Both fixed
(`burgers2d: guard Newton scan...`, `burgers2d: NaN-propagating residual
audit...`); all shipped data was regenerated or proven clean (finite POD Gram
== finite training data). N in {16,32,64,256} cells ran pre-guard but their
data is verified clean; N=128 and the reference are post-guard reruns.

Provenance: A100 80GB, jax_backend=gpu in every log;
JAX_DEFAULT_MATMUL_PRECISION=highest; data regenerated from seed 0 on-cluster
per cell; isolated job dirs; cluster tree deleted after checksummed pull.
Local artifacts: `sweep/` (results JSONs, film+grid-tied checkpoints, logs,
`burgers2d_convergence.json`), `ref512_val.npz` (untracked, 0.81G —
regenerate via burgers2d_refgen.py).

## ViT-CP arm (pending)

`burgers2d_vitcp.py` trains the repo's PUBLISHED CP decoders (imported
verbatim from heat/src CPDecoder and poisson/src LinearCPDecoder, flax) as
(z,t)-conditioned arms on the same data/metrics — the literal published
baseline next to the simplified grid-tied control. Published rank=256/
hidden=256/per-N optimizer settings; latent input = concat(z, 2*tau-1)
(latent_dim 6, no encoder — testbed protocol). Cluster cells not yet run.
`N=$N STEPS=120000 python burgers2d_vitcp.py <outdir>`

## Online timing vs the FOM (2026-08-15, one A100, batch 1, median of 5; job 2379562)

| N | FOM rollout /traj | FiLM full-field /traj (native) | speedup |
|---|---|---|---|
| 16 | 93 ms | 13.0 ms | 7.2x |
| 32 | 128 ms | 13.1 ms | 9.8x |
| 64 | 197 ms | 14.9 ms | 13.3x |
| 128 | 574 ms | 52.5 ms | 10.9x |
| 256 | 1035 ms | 195 ms | 5.3x |
| 512 | 2993 ms | 696-786 ms (net on 512^2) | 3.9-4.3x |

Protocol: FOM = full implicit Newton/BiCGStab rollout, f64, batch 1; net =
51-slice full-field reconstruction, f32, per-slice jitted calls
(`pde_timing.py`, timing/timing.json). Interpretation caveats: (i) this is
SURROGATE-INFERENCE speedup — the decoder conditioned on the true z decodes
EVERY grid point at EVERY slice, the most pessimistic online mode; the ROM
deployment decodes only m EQ points per GN iteration (n-free — see the
cost-scaling branches), which is Phase 3's measurement. (ii) batch-1 latency
favors neither side; net throughput batches trivially. (iii) the 13-15 ms
floor at N<=64 is kernel-launch latency (51 sequential dispatches).
