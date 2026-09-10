# Why the Heat NM-ROM under-performs — diagnosis and fix

Date: 2026-05-21. Working copy of `heat/`.

## Premise

The Poisson NM-ROM converges from a *cold start* (z = 0) to good accuracy.
The Heat NM-ROM *warm-starts* every step from the previous latent code — a
strictly easier setting — yet is less accurate. The physics is not the
problem. The problem is that the Heat ROM minimizes a residual that is **not**
the Heat FOM equation.

## The FOM equation (what the data satisfies)

Backward Euler, one step:

    A(u_n) = u_{n-1},   A(u) = u - dt*kappa*Lap(u)

with `Lap(u) = (sum_neighbours - 2d*u_c)/dx^2` (positive discrete
Laplacian), Dirichlet `u = 0` on the boundary. Boundary rows are identity,
so boundary neighbours simply contribute their true value (0) to an
interior node's stencil. There is **no masking of interior centres** —
ever.

The per-step FOM residual at an interior node `c` is therefore

    r_c(u_n) = u_n,c - dt*kappa*Lap(u_n)_c - u_{n-1},c                 (1)

## What the ROM solver actually minimized (the bug)

`solver/nm_rom.py :: f_norm_eq` (original):

    u_st = (h @ V_eq + bias).reshape(n_eq, stencil_w) * mask_st        # <-- BUG
    lap  = (2d*u_st[:,0] - sum(u_st[:,1:])) / dx^2
    f    = u_st[:,0] + dt*kappa*lap

Two defects relative to (1):

1. **`mask_st` zeroes the centre column.** `mask_st` is 0 for any stencil
   node — *including the centre* — whose flat index decodes to a boundary
   coordinate. For an EQ centre one node inside the boundary, the centre
   itself is interior (mask should be 1) but a neighbour is on the
   boundary. The original code multiplies the *whole* `(n_eq, stencil_w)`
   block by `mask_st`, so whenever a neighbour is masked it does not just
   zero that neighbour — the layout means centres get corrupted too, and
   the masked Laplacian no longer equals the FOM Laplacian. The FOM never
   masks; it lets the true boundary value (0) flow into the stencil.

2. **EQ design matrix uses the wrong operator.** `eq/nnls.py ::
   compute_eq_weights` builds the NNLS design matrix `G` from
   `|u_i - decode(encode(u_i))|` — the **autoencoder reconstruction
   error**. The solver, however, minimizes the **time-step residual** (1).
   Poisson builds `G` from `|K u_i|` — the *same* operator the solver
   uses. Heat's hyper-reduction was selecting nodes/weights optimal for a
   functional the solver never sees.

Net effect: even a perfectly converged Gauss-Newton solve lands on the
wrong field, and the error compounds over 50 steps. This is exactly the
"data / residual must match the FOM operator" failure mode — here it is
the *solver* residual, not the training data, that is inconsistent.

## The fix

Mirror Poisson's design:

* **Interior-only EQ.** Restrict EQ centres to `[2, N-3]^d` so the full
  `(2d+1)` stencil is strictly interior. The `(2d+1)`-point stencil then
  never touches the boundary and `mask_st` is unnecessary. Drop `mask_st`
  entirely.

* **Correct residual.** `f_norm_eq` becomes the exact left side of (1)
  with no masking:

      lap = (sum(u_st[:,1:]) - 2d*u_st[:,0]) / dx^2      # positive Lap
      f   = u_st[:,0] - dt*kappa*lap                     # A(u) at centre

  residual = scale * f_norm(z) - u_prev_eq.

  (The original wrote `2d*u_c - sum(neighbours)` = `-Lap`, then
  `u_c + dt*kappa*(-Lap)` = `u_c - dt*kappa*Lap`. Sign was accidentally
  right; the masking was the defect. The rewrite makes the sign explicit
  and matches `fom/heat.py` term for term.)

* **EQ design matrix from the step operator.** Build `G` rows from
  `|A(u_i) - u_{i-1}|` over *consecutive snapshot pairs* of each
  trajectory, i.e. the actual FOM step residual the solver drives to
  zero — not reconstruction error.

### Defect 3 — the `u_prev` representation seam

In `rollout`, step 0 sets `u_prev_eq = u0_flat[eq_indices]` — the true
*field* `u_0` at EQ centres. But every later step set
`u_prev_eq = scale * f_norm(z_new)` = `scale * A(u_hat)` — the *operator
applied* to the decoded field, not the field. The residual
`R = scale*A(u_hat(z)) - u_prev_eq` only equals the backward-Euler step
residual `A(u_n) - u_{n-1}` if `u_prev_eq` is the previous **field**
`u_{n-1}`. Feeding `A(u_{n-1})` instead silently changes the recurrence
the ROM marches, and the error compounds over 50 steps.

Fix: a new `u_center_eq(z)` returns the decoded field value at EQ
centres (`u_st[:,0]`); `rollout` now carries `scale * u_center_eq(z_new)`
as `u_prev_eq`, consistent with the step-0 initialization.

## Files changed

* `fom/heat.py` — added `HeatFOM.step_residual` = `|A(u_curr) - u_prev|`,
  the FOM operator EQ must be built from.
* `eq/nnls.py` — `compute_eq_weights` now takes a `step_residual_fn`
  built from the FOM operator and consecutive snapshot pairs; EQ centres
  restricted to `[2, N-3]^d` via `_eq_candidate_indices(margin=2)`.
* `solver/nm_rom.py` — `f_norm_eq` rewritten to (1) with no `mask_st`
  (`mask_st` construction removed); added `_u_stencil` and `u_center_eq`;
  `rollout` carries the decoded field (not `A(field)`) as `u_prev`.
* `scripts/run_rom.py` — reconstructs per-trajectory snapshot blocks and
  passes the FOM step-residual closure to `compute_eq_weights`.
* `tests/test_smoke.py` — updated to the new `compute_eq_weights` signature.

The AE architecture and training are unchanged — the defect was entirely
in the ROM/EQ layer. There is no pre-existing checkpoint, so one AE
training pass is still required; had a checkpoint existed, only the
cheap ROM stage would need re-running.

## Empirical result (2026-05-21, job 879296, heat2d_n64)

After the three fixes:

| metric                       | README claim | this run  |
|------------------------------|--------------|-----------|
| 2D N=64 ROM rel-L2 (mean)    | 5.21e-3      | 4.87e-2   |
| 2D N=64 speedup (median)     | 39.6x        | 0.14x     |
| AE val loss (best)           | —            | 8.40e-3   |
| **AE reconstruction rel-L2** | —            | **4.72e-2** (mean), 6.6e-1 (max) |
| GN iters per ROM step        | —            | 3.4 (cap 12), converging |

The decisive number is the **AE reconstruction rel-L2 = 4.72e-2**, which
the ROM rel-L2 (4.87e-2) matches almost exactly. The ROM solve is
healthy — GN converges in ~3 iterations, EQ selects 100 sensible nodes.
The operator-consistency fixes did their job: the ROM now faithfully
tracks whatever the autoencoder manifold can represent.

**The remaining limiter is the autoencoder itself, not the ROM.** The
trained AE cannot reconstruct the validation fields below ~4.7e-2, so
the ROM cannot beat that floor. The README's 5.21e-3 is a paper-target
number this code's AE — as configured — does not reach.

Root cause is architectural, as flagged in the initial review:
* CP decoder is a low-rank *separable* factorization; sums of off-axis
  Gaussian blobs are not separable and need many CP terms.
* mean-pool ViT encoder discards blob localization.

Next lever for accuracy is the AE (decoder rank, encoder pooling,
training length) — not the ROM/EQ layer, which is now correct.

The README's 39.6x speedup is also not reproduced (0.14x here): on this
L40S the 50-step CG FOM is fast and the per-step latent GN solve does
not amortise. Speedup is a separate axis from the accuracy fix.

## Defect 4 — too few EQ training snapshots (3D, job 879323)

heat3d_n32 first ran with the config's `n_eq_samples: 16`. The ROM
**diverged**: mean rel-L2 13.0, individual trajectories at 26-58, the
decoded field blowing up to max|u|~3.9 against a true max|u|~0.08.

It is NOT the GN iteration cap (sweeping gn_max_iters 3->50 changed
nothing) and NOT the EQ node count (64 vs 6000 identical). The cause is
the number of *training snapshots* fed to NNLS:

| n_eq_samples | ROM mean rel-L2 | max   |
|--------------|-----------------|-------|
| 16 (config)  | 1.31e+1         | 5.9e1 |
| 32           | 5.66e+0         | 4.0e1 |
| 64           | 3.87e-1         | 1.4e0 |
| 128          | 1.58e-1         | 4.9e-1|
| 256          | 1.01e-1         | 2.1e-1|

With only 16 snapshots the NNLS design matrix cannot characterise the
step-residual operator across the parameter range, so the EQ weights
are badly under-determined and the latent GN minimises a misweighted
residual -> divergence. 256 snapshots -> rel-L2 0.10, bounded, and that
0.10 again matches the 3D AE reconstruction floor (1.26e-1) -- so once
EQ is well-conditioned the ROM is AE-limited, the same pattern as 2D.

Fix: heat3d_n32.yaml `n_eq_samples 16 -> 256`, `min_eq_points 64 -> 256`,
`gn_rel_tol 1e-3 -> 1e-4`, `gn_max_iters 3 -> 8`. ROM-only; no retrain.

## Summary of results after all fixes

| case        | ROM rel-L2 | AE recon rel-L2 | verdict        |
|-------------|------------|-----------------|----------------|
| heat2d_n64  | 4.87e-2    | 4.72e-2         | AE-limited     |
| heat3d_n32  | 1.01e-1    | 1.26e-1         | AE-limited     |

In both cases the ROM error tracks the autoencoder reconstruction floor
once the operator-consistency fixes (defects 1-3) and sufficient EQ
training data (defect 4) are in place. The ROM/EQ layer is correct; the
remaining accuracy headroom is entirely in the autoencoder (CP decoder
rank/separability, mean-pool encoder, training length).

## Accuracy push (2026-05-21 night)

### hi-accuracy AE retrain (job 879470, heat3d_n32_hi)

4x data (2000 trajectories), CP rank 512->1536, ViT embed 96->160 + 6
layers, CLS-token attention pooling, 150k epochs. Result:

* AE val loss 3.75e-2 -> 1.27e-2; AE reconstruction as low as 2.4e-3 on
  smooth (high-kappa) fields. The AE improvement worked.
* ROM mean rel-L2 stayed ~0.10 -- UNCHANGED.

So the bottleneck moved off the autoencoder. Per-trajectory: AE
reconstructs u_T to 2.4e-3 but the ROM only reaches 3.8e-2 -- a 15x
gap. The latent Gauss-Newton solve, not the AE, is now the limiter.

### the closure problem (ROM-knob sweep, job 880277)

Sweeping GN iterations made the ROM WORSE, not better:
  gn=8,  tol=1e-4 -> rel-L2 8.67e-2
  gn=20, tol=1e-6 -> rel-L2 9.52e-2

Converging harder to the EQ-residual minimum moves the latent code
*away* from the field the AE should produce. The latent that zeroes
the PDE residual is not the latent the encoder would assign -- a
reconstruction-trained AE gives no such guarantee. This is the NM-ROM
closure problem; knob-tuning cannot fix it.

### latent-cycle-consistency retrain — did NOT help (job 880280)

lam_cyc=0.25 adds a penalty on encode(decode(encode(u))) drifting from
encode(u), training the encoder as a stable left-inverse of the
decoder so the GN-found latent is self-consistent.

Result: AE val loss 1.23e-2 (≈ the plain hi retrain), but ROM mean
rel-L2 **5.94e-2** — WORSE than the EQ=1024 baseline (4.22e-2). The
cyc config also carried gn=20 / EQ=512, i.e. the regimes already
shown to hurt. The cycle-consistency penalty does not close the
closure gap; EQ density does. Abandoned this lever.

### EQ density is the lever — extended sweep (jobs 880291 / 880329)

| EQ nodes | ROM rel-L2 (mean) | median  |
|----------|-------------------|---------|
| 256      | 8.67e-2           | —       |
| 512      | 5.42e-2           | —       |
| 1024     | 4.22e-2           | 2.72e-2 |
| 2048     | 3.81e-2           | 2.47e-2 |

At EQ=2048 the ROM (3.81e-2) drops BELOW the AE reconstruction floor
(3.91e-2): the operator/EQ layer is fully healthy and the ROM tracks
whatever the manifold represents. heat3d_n32_hi locked at EQ=2048.
(The first 4096-node sweep crashed — scipy nnls maxiter too low for
the larger active-set search; fixed by scaling maxiter with problem
size, re-run as job 880329 for 4096/8192.)

## N=64 extension (2026-05-22)

Extending the pipeline to 64^3 surfaced five scale-up bottlenecks, all
fixed (resolution-independent):

1. `np.savez_compressed` single-threaded zlib stalls ~1h on the 45 GB
   dataset -> plain `np.savez` above 4 GB.
2. CP decoder's 4-operand `einsum("r,ri,rj,rk->ijk")` materializes a
   (rank,N,N,N) = 2.1 TB intermediate -> CUDA_ERROR_ILLEGAL_ADDRESS.
   Staged the contraction one axis at a time.
3. `eval_loss` vmapped the whole 2550-snapshot val set -> 85 GB OOM.
   Chunked eval (eval_chunk=64).
4. NNLS candidate pool was 60^3 = 216k columns -> hour-long solve.
   Capped at max_candidates (uniform subsample).
5. NNLS design matrix built row-by-row (2048 un-jitted dispatches);
   row count n_eq_samples=2048 also bloated the solve. Batched the
   residual build (vmapped chunks) and cut n_eq_samples to 512.

### first N=64 result (job 880702) — ROM 8.1e-2, worse than N=32

| trajectory group     | rel-L2      |
|----------------------|-------------|
| high-kappa (smooth)  | 1.0-1.9e-2  |
| low-kappa (sharp)    | 4e-2-1.8e-1 |
| mean (12 traj)       | **8.1e-2**  |

This is NOT a ROM/EQ failure. The AE training history is the smoking
gun: train loss **4e-4** vs val loss **3.5e-2** -- an ~85x overfit
gap, val flatlined from epoch 142k. The rank-2048 AE has ample
capacity (train ~0); 800 trajectories is too thin a manifold, so it
memorizes train and never generalizes. The ROM faithfully tracks that
3.5e-2 AE floor. High-kappa fields ROM well (1e-2); low-kappa sharp
fields the overfit AE cannot represent drag the mean up.

Fix (heat3d_n64_hi2): data + regularization, not capacity --
2x trajectories (800->1600), 2.5x weight decay (2e-3->5e-3).

### N=64 hi2 retrain (job 880711) — ROM 5.97e-2, still short of N=32

2x data (1600 traj) + 2.5x weight decay (5e-3) + 200k epochs:

| metric              | hi (880702) | hi2 (880711) |
|---------------------|-------------|--------------|
| AE train loss       | 4e-4        | 1e-3         |
| AE val loss         | 3.5e-2      | 2.46e-2      |
| overfit gap         | ~85x        | ~25x         |
| ROM mean rel-L2     | 8.1e-2      | 5.97e-2      |
| ROM median rel-L2   | ~6e-2       | ~4.5e-2      |

More data + regularization shrank the overfit gap (85x->25x) and the
ROM improved 8.1e-2 -> 5.97e-2, but the rank-2048 AE STILL overfits
and 64^3 does not beat N=32 (3.81e-2) on the mean. The hi2 median
(~4.5e-2) is close, and one trajectory (kappa=0.082) diverged to
0.247, inflating the mean. The remaining limiter is AE generalization.

### N=64 hi3 retrain (job 892699) — ROM 3.65e-2, BEATS N=32

3200 trajectories + cube-symmetry batch augmentation (48-element
symmetry group: axis flips x permutations), 250k epochs.

The augmentation closed the overfit:

| run  | data            | AE train | AE val  | gap  | ROM mean |
|------|-----------------|----------|---------|------|----------|
| hi   | 800 traj        | 4e-4     | 3.5e-2  | ~85x | 8.1e-2   |
| hi2  | 1600 traj       | 1e-3     | 2.46e-2 | ~25x | 5.97e-2  |
| hi3  | 3200 traj + aug | 2e-3     | 2.45e-3 | ~1x  | 3.65e-2  |

With train ~= val the AE finally generalizes; val loss 2.45e-3 is an
order of magnitude below hi2 and below the N=32 hi AE (1.27e-2).

Final comparison -- 64^3 beats 32^3 on BOTH axes:

| metric          | N=32 hi  | N=64 hi3 |
|-----------------|----------|----------|
| ROM mean rel-L2 | 3.81e-2  | 3.65e-2  |
| ROM median      | 2.47e-2  | ~2.2e-2  |

10 of 12 N=64 trajectories land at 8e-3..4e-2 (several ~1e-2); two
sharp-field outliers (~0.11-0.12) remain but no longer dominate.
The finer grid, with a non-overfitting AE, delivers the more
accurate ROM -- the extension to 64^3 is justified.

Decisive lesson: at 64^3 the rank-2048 AE has ample capacity but
overfits a thin trajectory set; cube-symmetry augmentation (a free
48x manifold expansion, valid because the heat Laplacian is
isotropic and the cube + Dirichlet BC are symmetry-invariant) is
what closes the generalization gap.
