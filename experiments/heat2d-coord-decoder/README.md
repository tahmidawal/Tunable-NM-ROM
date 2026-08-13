# Heat-2D coordinate-decoder testbed (2026-08-13)

Does the Poisson coord-decoder result (exp/2026-08-12-coord-decoder) survive the
jump to a **time-dependent** PDE? This testbed mirrors the repo's heat FOM —
`du/dt = kappa * lap(u)` on the unit square, u=0 walls, backward Euler dt=0.005
x 50 steps, CG per implicit step — with a fixed 1-blob family so the true
parameter vector has fixed length:

    z = (cx, cy, width, amplitude, log kappa),  cx,cy ~ U(0.15,0.85),
    w ~ U(0.05,0.20), a ~ U(1,10), kappa ~ logU(0.01,0.5)

The architectural extension over Poisson: **time is a coordinate.** The decoder
is `u(x, y, t; z)` — Fourier features in x, y AND t, a 5x256 trunk
FiLM-modulated by (z, t), ~450-470k params. No grid-tied parameters in space or
time. Spatial Fourier bandwidth is Nyquist-capped in-code per training grid
(n_freq <= (N-1)//2 — the aliasing landmine from the Poisson round), time
bandwidth capped at NUM_STEPS//2.

Three arms per training resolution N, all in one cluster job
(`heat2d_film.py`): f64-Gram POD floors of the space-time snapshot matrix
(linear yardstick), a grid-tied CP-style decoder with MLP(z,t) coefficients
over learned rank-24 spatial factors (disease control), and the FiLM coord-net.
Errors are mean relative L2 over 64 held-out val trajectories and all 51 time
slices; f32 network inference with norms/references in f64; seed 0, single seed.

## Headline result 1 — in-resolution head-to-head (Tufts A100s, 120k steps)

| N   | POD-6 (equal-dim linear) | POD-24  | grid-tied (rank 24) | FiLM coord-net |
|-----|--------------------------|---------|---------------------|----------------|
| 16  | 1.99e-1                  | 4.20e-2 | 1.14e-1             | 7.59e-3        |
| 32  | 1.98e-1                  | 4.36e-2 | 1.06e-1             | **6.21e-3**    |
| 64  | 1.98e-1                  | 4.45e-2 | 9.99e-2             | 6.94e-3        |
| 128 | 1.98e-1                  | 4.50e-2 | 9.93e-2             | **6.29e-3**    |
| 256 | 1.98e-1                  | 4.53e-2 | 8.96e-2             | 7.13e-3        |

- The solution manifold over (z, t) is 6-dimensional; POD-6 is the
  train-fitted SVD basis of that dimension — a strong linear baseline, though
  not a certified floor for this metric (SVD optimality is train-set absolute
  error, the table reports val-set mean relative L2; see the metric caveat in
  the Poisson testbed README). The coord-net beats it **~30x**, beats POD-24
  ~6.5x, and roughly matches POD-64 (~7-8e-3 in-res) from 6 inputs. One known
  mismatch (adversarial review): POD rows are fitted AND evaluated on every
  4th time slice (POD_TIME_STRIDE=4) while the neural columns average all 51
  slices — the POD numbers should be recomputed on all slices for an exact
  apples-to-apples table.
- The grid-tied control repeats the disease **more cleanly than in Poisson**:
  stuck at 0.09-0.11, 2-2.5x ABOVE the POD-24 floor its own rank permits, flat
  in N while its parameter count grows with the grid.
- Time behaves: per-time error is balanced (worst at t=0 where the blob is
  sharpest, ~2-7e-2 -> settling to ~5e-3-1e-2 for t>0; see
  `film_per_time` in the per-N JSONs). No drift or time-axis pathology.

## Headline result 2 — convergence + mesh transfer vs the N=512 reference

Each net evaluated NATIVELY on the 512x512 grid (1,024x more points than the
N=16 training grid; zero retraining, zero interpolation) at time slices
t-index {0,10,20,30,40,50}, against f64 reference trajectories with the same
dt (so the comparison isolates spatial error). `sweep/heat2d_convergence.json`:

| train N | data-floor (discretization bound) | FiLM net vs reference |
|---------|-----------------------------------|-----------------------|
| 16      | 2.82e-2                           | 4.58e-2               |
| 32      | 1.05e-2                           | 2.17e-2               |
| 64      | 5.07e-3                           | 1.50e-2               |
| 128     | 2.79e-3                           | 9.55e-3               |
| 256     | 1.42e-3                           | **9.53e-3**           |

**The six-time aggregate error falls monotonically with training resolution —
4.8x across the sweep — then flattens at the capacity floor (~9.5e-3).**
Honest caveats (from adversarial review): the final N=128->256 step improves
only 0.2%, and per-time errors at t-index 20/30/40/50 individually WORSEN
7-16% on that step while t=0/10 improve — only the aggregate is monotone.
The Nyquist cap also means N=16/32 cells have narrower feature widths (442k
vs 468k params) and per-step sampled points are min(8192, N^2), so
architecture/budget are not exactly fixed across resolutions; a
fixed-bandwidth control and multi-seed replication are needed before calling
this a clean convergence law. Same overall shape as the Poisson curve: data-limited at coarse N (the 16-grid can't resolve w=0.05
blobs), capacity-limited from N~128. A net trained on a 32x32 grid, evaluated
on 512x512, lands at 2.17e-2 — 2x its own training data's discretization bound.

## Reading the two results together

The repo's wall — error refusing to improve (or rising) with resolution — is
reproduced by the grid-tied arm and absent in the coordinate decoder, now in a
time-dependent setting: the coord-net's in-resolution error is flat ~6-7e-3
across a 256x range of grid sizes with near-constant parameter count
(442k-468k, varying only via the Nyquist-capped feature width), and its
error against the continuum *descends* as the training data improves. Adding
time as a network input (rather than a grid axis) worked on the first try;
nothing about the time dimension broke the architecture.

Caveats: single seed; decoder conditioned on true params (isolates the decoder
from encoder/ROM machinery); 1-blob family is simpler than the repo's 1-3-blob
heat benchmark; t=0 (sharp IC) remains the hardest slice at every N.

## Files

| path | what |
|---|---|
| `heat2d_film.py` | data gen (backward-Euler CG rollouts) + POD floors + grid-tied + FiLM coord-net, one job per N |
| `heat2d_refgen.py` | N=512 reference trajectories for the val params (seed 0) |
| `heat2d_convergence_eval.py` | nets + data-floors vs the 512 reference (per-N module reload keeps Nyquist caps consistent) |
| `sweep/` | per-N results JSONs, FiLM + grid-tied checkpoints, slurm logs, `heat2d_convergence.json` |
| `smoke/` | local N=16 tiny-budget smoke |
| `ref512_val.npz` | 805M reference (gitignored; regenerate via `heat2d_refgen.py`) |

## Provenance

- Branch `exp/2026-08-13-heat2d-coord-decoder`; scripts committed before launch
  (commit 1c15acc), results added after.
- Tufts jobs 2341579-2341583 (N=16/32/64/128/256) + 2341584 (reference), gpu
  partition, one A100 each, one isolated job dir per job
  (`paralab/tawal01/heat2d-coord/N*`, deleted after pulling). Every log
  contains `jax_backend=gpu` twice (preflight + script) and zero
  error/OOM/truncation lines. `JAX_DEFAULT_MATMUL_PRECISION=highest`
  throughout; data regenerated from seed 0 inside each job.
- Convergence eval ran locally on the GB10 (`jax_backend=gpu`).
- Config: 120k FiLM steps / 40k grid-tied steps, n_train=512 trajectories,
  n_val=64, batch 32, min(8192, N^2) sampled points per step (half importance-sampled
  around the diffusion-widened blob, std 2*sqrt(w^2+2*kappa*t)), POD Gram in
  f64 (ortho dev ~1e-11), CG tol 1e-10.

## Next steps

1. Multi-seed replication (here and in the Poisson testbed).
2. The integration experiment: this decoder behind the GN/EQ solver in
   `heat/`/`poisson/` — latent time-stepping through z instead of
   conditioning on true params — measuring actual ROM accuracy/speedup.
3. Optional: the repo's full 1-3-blob family (padded/masked conditioning).
