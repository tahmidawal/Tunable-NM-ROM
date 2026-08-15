# Wave-2D coordinate-decoder testbed (2026-08-14)

Does the heat-2D coord-decoder result (exp/2026-08-13-heat2d-coord-decoder)
survive the jump to a **hyperbolic, dissipation-free** PDE? Wave is the purest
transport stress case for linear ROMs: no diffusion ever damps high modes, the
solution is a traveling/reflecting wave packet, and the Kolmogorov width decays
slowly — equal-dimension POD should be far worse here than for heat, while the
FiLM coordinate decoder `u(x, y, t; z)` sees only a 6-input smooth map.

## PDE and FOM discretization

    u_tt = c^2 * lap(u)   on [0,1]^2,   u = 0 on the boundary,
    u(x,0) = masked Gaussian blob,   u_t(x,0) = 0

- First-order system (u, v = u_t), **Crank–Nicolson** (trapezoidal): for this
  linear system CN coincides with implicit midpoint, is unconditionally stable
  and conserves the discrete energy `E = 0.5||v||^2 + 0.5 c^2 u^T(-L)u`
  **exactly** (up to CG solve tolerance). Measured relative energy drift over
  the full horizon: **<= 1e-11** (typically 1e-13..1e-15) at every resolution —
  the solver-correctness invariant, printed by every script.
- Each implicit step solves the SPD system `(I - (c dt/2)^2 L) u^{n+1} = rhs`
  with CG (tol 1e-10); 5-point Laplacian, same boundary-mask pattern as heat.
- **dt is FIXED across all mesh resolutions** (unconditional stability makes
  this free), so the N=512 reference comparison isolates spatial error, same
  design as heat. `T_FINAL=1.0`, 51 stored snapshots (`NUM_STEPS=50`),
  `SUBSTEPS=80` CN steps per snapshot interval → `dt = 2.5e-4`.
- SUBSTEPS was set by the explicit dispersion check in `wave2d_selfconv.py`
  (dt-halving diff at N=512): at SUBSTEPS=40 the temporal error was comparable
  to the N=512 spatial error; at 80 it is 0.171x (representative) / 0.042x
  (stress) of even the **N=256** spatial error. Residual temporal error is
  common-mode across resolutions (same dt everywhere) and cancels in the
  net-vs-reference comparison.
- f64 everywhere in data generation; data stored f64.

## Parameter family (mirrors heat: 5 params + t)

    z = (cx, cy, w, a, log c), normalized to ~[-1,1]
    cx, cy ~ U(0.15, 0.85),  w ~ U(0.05, 0.20),  a ~ U(1, 10),
    c ~ logU(0.5, 2)

- `c ~ logU(0.5, 2)`: at T=1 the fastest waves cross the unit domain ~2x
  (reflections off the Dirichlet walls are well inside the horizon), the
  slowest cross ~half — the family spans pre-reflection and post-reflection
  regimes without every trajectory becoming reverberation soup.
- Amplitude `a` enters linearly (the PDE is linear); kept for symmetry with the
  heat family and for z-conditioning. The nonlinearity the decoders must
  capture is entirely the (cx, cy, w, c, t) dependence — which is exactly the
  transport structure that kills linear bases.
- IC is the blob times the boundary mask (heat convention). For wide blobs
  this leaves a ~0.4%-of-peak kink at the walls that propagates undamped — see
  the self-convergence caveat below.

## Self-convergence of the FOM (`selfconv.json`, fixed dt, cubic interp to 512)

| case | N=64 | N=128 | N=256 | observed order |
|---|---|---|---|---|
| representative (w=0.125, c=1) | 7.386e-3 | 2.294e-3 | 1.008e-3 | 1.69, 1.19 |
| stress (w=0.05, c=2) | 1.661e-1 | 4.012e-2 | 8.077e-3 | 2.05, 2.31 |

- The **stress case is clean 2nd order**: sharpest blob at the fastest speed,
  errors dominated by resolved-scheme dispersion.
- The representative case degrades toward ~1.2 by N=256: its absolute error
  (~1e-3) approaches the floor set by the masked-IC boundary kink (a small
  non-smooth component that no dissipation removes) plus residual CN
  dispersion. Kept as-is deliberately — the IC construction matches heat, and
  the floor (~1e-3) sits well below the data floors that matter at testbed
  scale. Honest caveat, not a bug.
- CN dt-halving diff at N=512: 1.7e-4 / 3.4e-4 (repr/stress) — the temporal
  error bound quoted above.

## Metrics (wave-specific — READ before comparing to heat)

Wave snapshot norms oscillate (kinetic <-> potential energy exchange) and
`||u(t)||` can pass near zero, making the heat-style per-snapshot relative L2
spiky. Every result therefore carries TWO metrics:

- **traj (PRIMARY)**: `mean_t ||pred - u||_t / sqrt(mean_t ||u(t)||^2)` —
  normalized by the trajectory-RMS norm, bounded away from zero by energy
  conservation. JSON keys without suffix (`film_coord`, `grid_tied`, `pod`).
- **snap**: heat-style `mean_t ||pred - u||_t / ||u(t)||` — recorded alongside
  under `*_snap` keys for cross-PDE comparability.

Training losses (both neural arms) are normalized by the per-trajectory mean
square for the same reason.

## Three arms (per training resolution N)

- **POD floors**: f64 Gram of the space-time training snapshot matrix, fitted
  AND evaluated on **ALL 51 time slices** (the heat adversarial-review fix —
  no POD_TIME_STRIDE mismatch; fit stride is env-overridable but defaults 1).
  r_max additionally capped by numerical Gram rank.
- **grid-tied control**: MLP(z,t) -> rank-24 coefficients over learned rank-1
  spatial factors (CP structure) — the trainability-disease control.
- **FiLM coord-net**: identical architecture to heat (5x256 trunk, ~450-470k
  params), Fourier features in x, y, t. Spatial `n_freq <= (N-1)//2` (Nyquist
  cap). **Time bandwidth defaults to the full snapshot Nyquist
  `T_FREQ = NUM_STEPS//2 = 25`** (heat used 8): wave solutions oscillate at
  temporal frequency `c*k/2pi` — up to ~12 cycles over T for in-family content
  — so under-capping the time features would alias the physics, not just slow
  training. Per-step training points: min(8192, N^2), half uniform / half
  concentrated on the expanding wavefront annulus (radius ~ c*t, width ~ 2w).

## Files

- `wave2d_film.py` — FOM + all three arms for one N.
  `N=64 [STEPS=120000] [GT_STEPS=40000] [SEED=0] python wave2d_film.py <outdir>`
- `wave2d_refgen.py` — N=512 f64 reference for the 64 val trajectories at
  EVAL_TIMES {0,10,20,30,40,50}; stores per-trajectory energy drift.
- `wave2d_selfconv.py` — solver validation (energy, spatial order, dt check).
- `wave2d_convergence_eval.py` — trained checkpoints evaluated natively on the
  512 grid vs the reference + coarse-FD data floors (both metrics).
- `smoke/` — N=8 end-to-end smoke artifacts (interface validation only).

## Running the sweep (cluster)

One isolated job dir per cell (N in {16,32,64,128,256} x wave2d_film.py, plus
one for wave2d_refgen.py; wave2d_convergence_eval.py afterwards on the pulled
checkpoints). Every job: venv, `JAX_DEFAULT_MATMUL_PRECISION=highest`,
`jax_backend=gpu` preflight. Memory note: the stride-1 POD Gram at N=256 is
26112^2 f64 (~5.5 GB) plus a ~14 GB snapshot copy and ~15 GB f64 data — fits
an 80 GB A100 but not much smaller; if an OOM appears, set POD_TIME_STRIDE=2
(fit-side only; eval stays all-slice) before shrinking anything else.

## Smoke status (local GB10, N=8, 400/200 steps — interface only)

jax_backend=gpu; energy drift 4.1e-15; POD ortho dev 1.4e-12 (rank-capped);
all three arms train with finite losses; results JSON + checkpoints written.
Numbers at this scale are meaningless by design.

## Results (2026-08-14, Tufts A100 sweep, seed 0, jobs 2357973-75 / 2358517-18 / 2357978)

### In-resolution head-to-head (traj-RMS metric, 120k steps)

| N   | POD-6 (equal-dim linear) | POD-24  | grid-tied (rank 24) | FiLM coord-net |
|-----|--------------------------|---------|---------------------|----------------|
| 16  | 4.407e-1                 | 1.591e-1 | 3.579e-1           | 2.801e-2       |
| 32  | 4.466e-1                 | 1.773e-1 | 3.739e-1           | 3.059e-2       |
| 64  | 4.497e-1                 | 1.847e-1 | 3.726e-1           | 2.850e-2       |
| 128 | 4.513e-1                 | 1.885e-1 | 3.730e-1           | 3.052e-2       |
| 256 | 4.520e-1                 | 1.904e-1 | 3.757e-1           | 3.507e-2       |

- The coord-net beats equal-dimension POD-6 ~13-16x and POD-24 ~5.4-6.5x.
  POD floors are ~4x worse than heat's at the same ranks (POD-24 1.6-1.9e-1
  vs heat 4.2-4.5e-2): the dissipation-free transport problem has the slow
  Kolmogorov decay the coordinate-decoder program predicts.
- The grid-tied control is stuck ~2x ABOVE its own POD-24 ceiling at every N —
  the same disease signature as Poisson/heat/Burgers, now on a hyperbolic PDE.
- Per-snapshot-normalized variants (`*_snap` keys in the JSONs) tell the same
  story ~10% higher (snapshot norms pass near zero during energy exchange).

### Convergence + mesh transfer vs the N=512 reference (native 512^2 eval)

| train N | data-floor (discretization bound) | FiLM net vs reference |
|---------|-----------------------------------|-----------------------|
| 16      | 2.134e-1                          | 2.332e-1              |
| 32      | 1.045e-1                          | 1.173e-1              |
| 64      | 5.631e-2                          | 6.785e-2              |
| 128     | 3.708e-2                          | 4.984e-2              |
| 256     | 2.578e-2                          | 4.368e-2              |

- **Strictly monotone, 5.3x across the sweep — no non-monotone step** (cleaner
  than heat's aggregate or Poisson's wiggle). The net tracks its data floor
  within 9-20% through N=64 and opens to ~1.7x at N=256 (capacity beginning to
  bind); the in-res uptick at N=256 (3.5e-2) is metric wiggle, not regression —
  against the fixed reference N=256 is the best cell.
- Wave data floors are ~20x heat's at matched N (2.6e-2 vs 1.4e-3 at N=256):
  without dissipation, sharp features never smooth out, so wave error is
  data-limited nearly everywhere in this sweep.
- Caveats (inherited from the heat round's adversarial review, unresolved
  here too): single seed; Nyquist cap varies feature width across N cells
  (n_freq 7/15/31/32/32); per-step sampled points min(8192, N^2); POD rows are
  train-fitted SVD bases, not certified floors for this val metric.

Provenance: commits `de11ee4`..`509df59` + CPU-POD/audit fixes; A100 80GB,
jax_backend=gpu in every log; JAX_DEFAULT_MATMUL_PRECISION=highest; data
regenerated from seed 0 on-cluster in every cell; isolated job dirs; cluster
dirs deleted after checksummed pull. Local artifacts: `sweep/` (results JSONs,
film+grid-tied checkpoints, logs, `wave2d_convergence.json`), `ref512_val.npz`
(untracked, 769M — regenerate via wave2d_refgen.py).

## ViT-CP arm (pending)

`wave2d_vitcp.py` trains the repo's PUBLISHED CP decoders (imported verbatim
from heat/src CPDecoder and poisson/src LinearCPDecoder, flax) as
(z,t)-conditioned arms on the same data, reporting both testbed metrics —
the literal published baseline next to the simplified grid-tied control.
Note: the published per-sample relative loss up-weights near-silent wave
snapshots (kept as published; documented in the script docstring). Cluster
cells not yet run. `N=$N STEPS=120000 python wave2d_vitcp.py <outdir>`
