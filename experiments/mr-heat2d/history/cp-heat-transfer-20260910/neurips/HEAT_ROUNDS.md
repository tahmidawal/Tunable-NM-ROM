# Heat ROM cost study — running log

Fidelity gate: the control arm `base:eager` must reproduce the 07-28 re-timing
of its cell. It does, on both the accuracy and the timing side (h3d_n32
9.301180e-02 exactly, delta 0.00e+00, when the cached EQ support is reused;
0.838x vs 0.833x and 0.070x vs 0.068x on the timing side under sbatch).

One caveat that took a while to surface: when the EQ support is REBUILT rather
than reused, rel-L2 moves by up to 3% because the support is not unique. See
the drift section at the end. Comparisons within a job are always against a
control measured in that same job, so this does not affect any speedup ratio.

FOM is frozen throughout: same operator, CG tol 1e-6, 1000-iter cap, 50 steps,
jitted with kappa as a runtime argument.

---

## Round 1 — where does the ROM's time actually go?

Hypothesis: the latent rollout is dominated by kernel-launch latency in a
deeply sequential loop.

**Wrong.** Measured on h3d_n32 (A100, pax007), kappa=0.4964:

| stage | ms | share |
|---|---|---|
| encoder, eager (as measured in the retiming) | 167.08 | **81%** |
| encoder, jitted | 0.51 | |
| decoder (full field) x2, jitted | 0.13 | |
| latent rollout, 50 steps, jitted | 39.43 | 19% |
| **measured `run_rom`** | **206.15** | |
| FOM (frozen, jitted) | 34.19 | |

The submission-era `run_rom` is

```
z0, s0 = encode(u0)                       # flax apply — NEVER jitted
u_eq0  = decode_full_jit(z0, s0)[eq_idx]  # jitted
z, s   = rollout(...)                     # jitted (this is what 0315b88 fixed)
u_out  = decode_full_jit(z, s)            # jitted
```

Commit `0315b88` compiled the rollout and left the ViT encoder running one XLA
op at a time from Python. **Compiling the encoder is a 327x reduction on that
stage and a 5.2x reduction on the whole ROM solve** (206 -> 40 ms), with no
change to the math.

This is the mirror image of the FOM defect: the same "compile it once" fix was
applied to one part of one side and nowhere else. It does not apply to the FOM,
which has no encoder and was already fully compiled.

Secondary cost attribution, inside the 39 ms rollout (h3d_n32, 150 GN iters):

| component removed | ms saved | share |
|---|---|---|
| the 33x33 dense solve (`jnp.linalg.solve`) | 13.5 | 35% |
| Jacobian (`jacfwd`) | 4.7 | 12% |
| line search (4 sequential residual evals) | 3.8 | 10% |
| data-dependent `while_loop` -> fixed `fori_loop` | 3.5 | 9% |
| remainder (MLP, basis matmuls, reductions) | ~13.8 | 34% |

Microbenchmarks on the same device: loop control flow costs 5-6.5 us/iteration
(negligible, ~1.3 ms total); `jnp.linalg.solve(33)` costs **115 us per call**
and Cholesky + two triangular solves 81 us.

### Measurement hazard found
The FOM read 65-291 ms in one loop and 34 ms in another, for the same call on
the same node. First guess was clock throttling (the SM idles at 210 MHz
against a 1410 MHz max). **That guess was wrong** -- see round 2: the
interactive session had no `CUDA_VISIBLE_DEVICES` and JAX was sharing another
job's GPU. Ratios are still measured interleaved, per protocol clause (v).

---

## Round 2 — act on the diagnosis

**PROVISIONAL. Every number in this section was measured through the
interactive path on pax007, which had no `CUDA_VISIBLE_DEVICES` and was sharing
a GPU with another job. The ordering of the variants is consistent across many
repeats, but the factors are not publishable and are being re-measured under
sbatch. Nothing here goes into a result JSON or to a reviewer.**

Three changes, each measured separately on h3d_n32:

| variant | what changed | ROM ms | rel-L2 | vs retimed |
|---|---|---|---|---|
| `base:eager` | nothing (submission-era structure) | 278.2 | 9.301180e-2 | **0.00e+00** |
| `base` | encoder compiled | 105.0 | 9.302292e-2 | 1.2e-4 |
| `base_mask` | + counted GN loop (identical iterates) | 48.1 | 9.302292e-2 | **1.2e-4** |
| `v3` | + Cholesky, folded operator, fused Jacobian, batched line search | 43.7 | 9.256063e-2 | 4.9e-3 |

**278 -> 48 ms, 5.8x, with the residual algebra untouched and rel-L2 unchanged
to 4 significant figures.** The extra algebraic restructuring in `v3` buys a
further 10% and costs 0.5% of accuracy, so `v4` (= counted loop + Cholesky, the
submission-era residual and Jacobian byte for byte) is the variant to report.

Why the counted loop is worth 2.2x on its own: a data-dependent `while_loop`
predicate has to be resolved before the next iteration can be dispatched, and
that costs far more than the arithmetic it saves. Freezing the update once the
tolerance is met computes iterations that get thrown away and is still much
faster.

**Equivalence claim, checked rather than assumed** (`test_equiv.py`, 30 random
configurations on the local box). The loop swap *on its own* is BIT-IDENTICAL:
same Gauss-Newton counts, same iterates, in all 30. But the wider rewrite in
`v3`/`v4` is NOT — it drifts ~1e-8 to 4e-5 from float32 re-association and
flipped a Gauss-Newton count in 3 of 30 configurations, on trajectories sitting
on the tolerance. So `v5` (loop swap only) and `v5w` (Cholesky only) exist to
carry the strong claim, and the wider rewrite is reported as what it is.

Cholesky vs LU on the (k+1)x(k+1) regularised system: 81 us vs 115 us per solve.
The system is SPD by construction, so this is free.

### Measurement error found and corrected
The dev node's interactive sessions get **no `CUDA_VISIBLE_DEVICES`**, so JAX
attached to whichever GPU was index 0 -- another job's. That is what made the
FOM swing between 34 ms and 293 ms for the same call. Everything now runs
through `sbatch`, which sets it correctly. No number measured through the
interactive path is quoted.

### Open
- `h3d_n64` clause-(iv) job 1832027 was cancelled to free GPU quota (limit is
  ~3 concurrent per user and the Burgers session holds two). Its role is taken
  by `base:eager` measured inside the same job as the optimised variants, which
  is a better control but is NOT on an exclusive node. Resubmit if quota frees.

---

## Round 3 — clean measurement (job 1918201, pax008, H200)

Slurm reported **no co-resident jobs on pax008**, so clause (iv) is met by
observation. `CUDA_VISIBLE_DEVICES=[0]`, `jax_backend=gpu`, FOM-vs-cached-
reference fidelity 2.6e-7 (N=32) and 3.9e-7 (N=128). FOM and ROM interleaved.

**The control reproduces the 07-28 re-timing**, which is what makes the rest
comparable: `base:eager` gives 0.838x at 3D N=128 (07-28: 0.833x) and 0.070x at
3D N=32 (07-28: 0.068x).

### Heat-3D N=128, best-accuracy arm — the result
| variant | rel-L2 | vs posted 1.82e-1 | ROM | speedup | per-kappa range | GN |
|---|---|---|---|---|---|---|
| `base:eager` (submission) | 1.8329e-1 | +0.7% | 144.0 ms | 0.838x | 0.27-2.56x | 3.1 |
| `base` (+encoder compiled) | 1.8330e-1 | +0.7% | 37.2 ms | 3.232x | 1.01-10.24x | 3.1 |
| **`v4w` (+Cholesky)** | **1.8321e-1** | **+0.7%** | **31.7 ms** | **3.791x** | **1.19-11.99x** | 3.1 |
| `v4` (counted loop) | 1.8321e-1 | +0.7% | 100.3 ms | 1.208x | 0.40-3.74x | 3.1 |
| `v3` (full rewrite, counted) | 1.8262e-1 | +0.3% | 105.8 ms | 1.146x | 0.38-3.55x | 6.5 |

Above 1x on all ten trajectories. FOM here is 40-375 ms, median 121 ms.

### Heat-3D N=32
| variant | rel-L2 | ROM | speedup | range |
|---|---|---|---|---|
| `base:eager` | 9.5978e-2 | 185.8 ms | 0.070x | 0.039-0.179x |
| `base` | 9.5977e-2 | 34.0 ms | 0.383x | 0.213-1.051x |
| `v4w` | 9.5973e-2 | 30.4 ms | 0.428x | 0.238-1.172x |
| **`v4`** | **9.5973e-2** | **27.2 ms** | **0.479x** | 0.266-1.209x |

6.8x better than the control but still below 1x: the FOM is only 13 ms here.
rel-L2 is +3.2% against the posted 9.30e-2 because the EQ support was rebuilt
on a different GPU model -- see the drift section below.

### The counted loop is cell-dependent, not a universal win
It pays the full iteration cap every step. Where the cap is tight (N=32, cap 3,
realised 3.0) that is free and it wins. Where the cap is loose (N=128, cap 12,
realised 3.1) it costs 3.2x (100.3 ms vs 31.7 ms). **Report `v4w` for N=128 and
`v4` for N=32.**

### Heat-2D — all four arms, still below 1x
| cell | control | best | rel-L2 (best) | vs posted | GN / cap |
|---|---|---|---|---|---|
| N=64 acc | 0.061x | **0.077x** (`v4`) | 5.364e-3 | +3.0% | 19.4 / 20 |
| N=64 fast | 0.150x | **0.191x** (`v4`) | 8.607e-3 | -20.3% | 8.0 / 8 |
| N=128 acc | 0.193x | **0.228x** (`v4w`) | 1.092e-2 | +7.1% | 16.5 / 20 |
| N=128 fast | 0.458x | **0.510x** (`v4w`) | 1.443e-2 | +6.6% | 7.8 / 10 |

The encoder fix does nothing here because the published 2D timed region already
EXCLUDED the encoder. Adding it back uncompiled costs ~100 ms (`base_enc`);
compiled it is free (`enc_jit` == `base`). So the 2D correction is neutral.

Two things these numbers expose:
- Every 2D arm sits at or just under its iteration cap, so its error is set by
  where the cap falls, not by a tolerance. Same failure mode as Burgers
  (COMMITMENTS 4). At N=64 fast, a 1e-7 perturbation moves rel-L2 by 20%.
- The 2D cells run a DENSE residual. 265 ms / (50 steps x 19.3 iters) = 0.28 ms
  per iteration, each a `jacfwd` over the whole 4096-node grid.

### Batched over the 10 parameter instances (the many-query setting)
N=128: FOM 166 ms/instance batched, `v3` 14.2 ms/instance -> **11.68x at
rel-L2 1.8193e-1**, which matches the posted 1.82e-1. But `base`/`v4`/`v4w`
batched land at 2.05e-1 (+12.7%) while their sequential values are 1.832e-1 --
an accuracy shift under `vmap` that is NOT yet explained, so those batched
ratios are not quotable.
N=32: batching helps the FOM more than the ROM (FOM 2.6 ms/instance), so the
ratio stays at 0.6-0.7x. Batching does not rescue the small cell.

---

## The 3D drift — mechanism found (job 1918201, `eqdet h3d_n32`)

COMMITMENTS 4c blames `patch_nnls` truncating NNLS at 3*nrows. **That is wrong.**

- `support size vs nrows: 512 vs 512 -> FULL RANK, cap not binding`. Every cell
  returns exactly n_eq == nrows (512=16x32, 1280=32x40, 640=16x40), so the
  active set reaches full rank and NNLS terminates on its own.
- `G eager bit-identical run to run: True` (sum(f64) 2.882780045839025e+01
  twice) -- within one code path on one device the build is deterministic.
- `G eager vs jitted bit-identical: False` -- and the gap is LARGER than
  re-association: max abs diff 3.796e-05 against a matrix max of 6.758e-02, so
  5.6e-4 relative, with 13.8M of 16.8M entries differing. Eager runs one kernel
  per op, jitted fuses; that is different arithmetic, not just a different
  summation order.
- `support eager vs jitted: shared=38/512, symmetric difference=948` --
  **a ~1e-7 change in G produces an almost disjoint quadrature support.**

The support is **not unique**: 512 rows, 32768 columns, an exact nonnegative
solution exists, and rounding decides which nodes are selected. Rebuilding on a
different GPU model moved N=32 rel-L2 from 9.3012e-2 to 9.5977e-2 (+3.2%),
which is the same size as the drift COMMITMENTS records.

This matters beyond reproducibility: fxe8 asked how an arbitrary N_eq is
specified and whether NNLS is recomputed. The answer has to acknowledge that
the support the offline solve returns is one of many, and that which one you get
depends on the floating-point path.

---

## Round 4 — does the algebraic restructuring add anything? No.

h3d_n128_acc, job 1921064 (pax009, 4 co-resident jobs; ratios internally
comparable, absolute values not clause-(iv) clean).

| variant | what it adds over v4w | ROM ms | speedup | GN |
|---|---|---|---|---|
| `v4w` | — | 31.73 | 3.781x | 3.1 |
| `v3w_nofold` | batched line search | 31.50 | 3.805x | 3.1 |
| `v3w` | + operator folded into the CP basis, shared-primal Jacobian | 61.07 | 2.272x | 6.5 |
| `v3w_nolin` | + operator folded, Jacobian not shared | 61.00 | 2.275x | 6.5 |

Folding the backward-Euler operator into the CP basis is **slower**, and not
because the implementation is worse: it changes float32 re-association enough to
change Gauss-Newton convergence, and the solve then needs 6.5 iterations instead
of 3.1. Batching the line search and sharing the primal pass are both neutral.

**v4w stands.** And it reproduces: 3.791x (round 3, pax008, idle) / 3.781x
(round 4, pax009, busy) / 3.793x (round 5, pax009, busy). rel-L2 1.8321e-1 in
all three.

---

## Round 5 — CUDA graph capture

Of four flag spellings this XLA (0.4.14) accepts only `--xla_gpu_graph_level=3`;
`--xla_gpu_enable_command_buffer`, `--xla_gpu_cuda_graph_level` and
`--xla_gpu_enable_cuda_graphs` are all rejected. **The flag is set for the FOM as
well as the ROM** — the FOM is also a nest of small-kernel loops, so rule 1
applies — and each setting is measured in its own process on the same node.

h3d_n128_acc, first four trajectories, ms:

| | FOM | `base:eager` | `v4w` | `v4` |
|---|---|---|---|---|
| no graph | 80.6 / 139.8 / 205.5 / 40.0 | 142.9 | 31.2-33.8 | 100.4 |
| graph | 79.9 / 139.6 / 205.4 / 40.1 | 139.3 | 28.6-31.4 | 91.8 |

**The ROM gains ~8%, the FOM gains ~0%.** The FOM's CG has a data-dependent trip
count, so its loops cannot be captured; the ROM's fixed 50-step rollout can be.
That takes N=128 from 3.79x to roughly 4.1x.

The main table is reported WITHOUT the flag, which is the conservative choice;
the ~8% is noted as available.

### Why the FOM gets none of the ROM's three fixes
- Compiling the encoder: the FOM has no encoder.
- Cholesky on the (k+1)x(k+1) system: the FOM has no small dense solve; it is
  matrix-free CG.
- Counted loop: the FOM's loop is its CG tolerance. Making it counted would
  change its convergence criteria, which rule 1 forbids, and would cost more
  anyway since it would always run to the worst-case count.
- The graph flag DOES apply to both and was applied to both.

---

## Where the floor is

Per-Gauss-Newton-iteration budget at N=128 after the fixes: 31.5 ms /
(50 steps x 3.09 iters) = 0.20 ms. The largest single item is the
(k+1)x(k+1) Cholesky at ~81 us, about 40% of it. That is a cusolver call with
fixed overhead; on this stack there is no exact way around it (an unrolled
in-XLA factorisation of a 41x41 needs ~41 sequential steps and measures worse,
and iterative alternatives change the solve).

Rounds 4 and 5 each returned under 10% over v4w. **Declaring the floor at
Heat-3D N=128 at 3.8x (4.1x with graph capture), at the published accuracy.**

---

## Round 6 — the quadrature budget (job 1921388, pax009, 4 co-tenants)

`n_eq` comes out exactly `k * n_eq_samples` on every cell, so halving
`n_eq_samples` halves the node count. **This re-solves the NNLS, so it is an
OFFLINE retuning, not the deployment-time knob the paper describes.** fxe8 was
right to ask.

### h3d_n128_acc, n_eq 640 -> 320
| variant | rel-L2 | ROM ms | speedup | range |
|---|---|---|---|---|
| `base:eager` | 1.8896e-1 | 140.13 | 0.859x | 0.28-2.70x |
| `base` | 1.8897e-1 | 35.30 | 3.369x | 1.04-10.96x |
| `v4w` | 1.8897e-1 | 29.39 | 4.039x | 1.25-13.06x |
| **`v5w`** | **1.8897e-1** | **29.37** | **4.045x** | **1.25-13.08x** |

3.79x -> 4.05x, so **+6.6% of speed for +3.1% of error**. Not a good trade, and
it confirms the cost is not dominated by the n_eq-proportional matmul but by the
fixed-size Cholesky and MLP, exactly as the round-1 attribution said.

**The important result here is `v5w`.** It changes ONLY the LU solve to Cholesky
-- loop construct and every residual/Jacobian expression byte-for-byte as the
control -- and it matches `v4w` to within 0.02 ms and to the same rel-L2. So the
headline does not need my rewritten code path at all.

### h3d_n64_acc, n_eq 1280 -> 640
| variant | rel-L2 | ROM ms | speedup | range | GN |
|---|---|---|---|---|---|
| `base:eager` | 7.4400e-2 | 262.83 | 0.097x | 0.04-0.26x | 9.3 |
| `base` | 7.4401e-2 | 113.95 | 0.264x | 0.08-0.70x | 9.4 |
| **`v5w`** | **7.4397e-2** | **91.14** | **0.294x** | 0.10-0.89x | 9.4 |

rel-L2 7.4400e-2 against the posted 7.49e-2 -- **0.7% BETTER than posted**, at
half the quadrature budget. But still 0.29x: the FOM here is only 24 ms.

**Why N=64 is slower than N=128 despite the smaller mesh:** its `gn_tol` is
1e-5 against N=128's 1e-3, so Gauss-Newton runs **9.4 iterations instead of
3.1**. 50 x 9.4 = 470 iterations against 155. The cost difference is the
tolerance, not the mesh and not the quadrature budget.

---

## Heat-3D N=64, both arms at full quadrature budget (job 1918201)

pax008, but a foreign job (1922162) had landed by then, so this cell is
`isolated=False` with that job ID recorded. FOM median 23.95 ms
(per-trajectory 9.8-69.5 ms).

| arm / variant | rel-L2 | vs posted | ROM ms | speedup | range | GN |
|---|---|---|---|---|---|---|
| acc `base:eager` | 7.2621e-2 | -3.0% | 244.85 | 0.101x | 0.04-0.24x | 8.5 |
| acc `v4w` | 7.2627e-2 | -3.0% | 76.27 | **0.329x** | 0.09-0.84x | 8.4 |
| fast `base:eager@2,1e-2` | 7.6707e-2 | -2.3% | 180.06 | 0.133x | 0.05-0.39x | 2.0 |
| **fast `v4@2,1e-2`** | **7.6710e-2** | **-2.3%** | **23.67** | **1.012x** | **0.41-2.94x** | 2.0 |

**The N=64 best-speed arm straddles the crossover and does NOT clear it.** This
run's rewritten path reads median 1.012x, but round 7's minimal-change variant
reads **0.988x** on the same cell (0.40-2.87x), faster on 5 of the 10 parameter
instances and slower on the other 5. A 2% spread between variants either side of
unity is precisely why this cell must be reported as "at unity" and never as a
speedup.

Both N=64 arms came out **more accurate than posted** (-3.0% and -2.3%), so
nothing here is bought with accuracy.

The acc arm stays at 0.33x for the reason round 6 identified: `gn_tol=1e-5`
drives 8.4 Gauss-Newton iterations per step against the fast arm's 2.0.

### Batched, N=64 fast — and here the accuracy DOES hold
FOM 13.05 ms/instance batched (vs 23.95 sequential -- batching helps the FOM a
lot at this size). `v4@2,1e-2` 4.14 ms/instance -> **3.156x at rel-L2 7.8347e-2,
which matches the posted 7.85e-2.** Unlike the N=128 cells, this arm's batched
accuracy agrees with its sequential accuracy, so this ratio is defensible. The
N=128 batched shift (+12.7%) is still unexplained and those remain unquotable.

---

## Final picture

| cell / arm | control | best | rel-L2 | vs posted | above 1x on |
|---|---|---|---|---|---|
| Heat-3D N=128 acc | 0.838x | **3.79x** | 1.8321e-1 | +0.7% | 10/10 |
| Heat-3D N=128 fast | 0.849x | **3.81x** | 1.9606e-1 | +3.2% | 10/10 |
| Heat-3D N=64 fast | 0.133x | **0.99x** | 7.6710e-2 | -2.3% | 5/10 |
| Heat-3D N=32 | 0.070x | 0.479x | 9.5973e-2 | +3.2% | 1/10 |
| Heat-3D N=64 acc | 0.101x | 0.323x | 7.2627e-2 | -3.0% | 0/10 |
| Heat-2D N=128 fast | 0.458x | 0.510x | 1.4432e-2 | +6.6% | 1/10 |
| Heat-2D N=128 acc | 0.193x | 0.228x | 1.0920e-2 | +7.1% | 0/10 |
| Heat-2D N=64 fast | 0.150x | 0.191x | 8.6074e-3 | -20.3% | 0/10 |
| Heat-2D N=64 acc | 0.061x | 0.077x | 5.3643e-3 | +3.0% | 0/10 |

Two cells clear 1x outright (both Heat-3D N=128 arms). One sits on the
crossover without clearing it. Six are still below.

---

## Round 7 — confirmation with minimal-change variants only (job 1928485)

`v5w` changes ONLY the LU solve to Cholesky. `v5c` adds the counted loop. Both
keep every residual and Jacobian expression byte-for-byte as the released code.

| cell | `base:eager` | best minimal-change | rel-L2 | above 1x |
|---|---|---|---|---|
| **h3d_n128_acc** | 0.849x | **3.829x** (`v5w`, 1.19-12.10x) | 1.8321e-1 | **10/10** |
| h3d_n32 | 0.070x | 0.471x (`v5c`) | 9.5973e-2 | 1/10 |
| h3d_n64 acc | 0.100x | 0.323x (`v5w`) | 7.2627e-2 | 0/10 |
| h3d_n64 fast | 0.132x | 0.988x (`v5c`) | 7.6710e-2 | 5/10 |

**The headline survives the minimal-change test.** Five independent measurements
of h3d_n128 across three nodes: 3.791 / 3.781 / 3.793 / 3.809 (fast arm) /
3.829x. The only code change the number needs is LU -> Cholesky on the
(k+1)x(k+1) system, plus compiling the encoder, which is not a change to the
method.

`v5c` (counted loop) beats `v5w` at N=32 (0.471x vs 0.423x) and at N=64 fast
(0.988x vs 0.924x) -- both tight caps -- and loses badly at N=128 acc (1.207x vs
3.829x, cap 12 against 3.1 realised. Same cell-dependence as round 3 found.

---

## Round 8 — the 2D Jacobian is grid-sized and does not need to be (job 1994862)

### Hypothesis
Round 6 left every Heat-2D cell at 0.08-0.51x, and the reason is structural, not
a tuning problem: `make_rollout` calls
`jax.jacfwd(_residual)(zs)` every Gauss-Newton iteration, which builds an
explicit **(N^2 x k+1)** Jacobian -- 16384 x 65 at N=128 -- by pushing k+1=65
tangents through the decoder and the stencil over the whole grid. 50 steps x
~19 iterations of that is the entire 2D ROM cost.

It is avoidable, exactly. Two structural facts:

1. `LinearCPDecoder` is **affine in a rank-256 feature vector**:
   `u = h(z) @ V + bias`, with `V[r] = outer(W_x[r], W_y[r])`. Only `h` is
   nonlinear in `z`, and `h` is a 3-layer MLP with 64 inputs.
2. The heat operator is linear and **affine in kappa**: `T = I + DT*kappa*K`.

So with `Y = mask (*) V` and `Yt = K Y`, every entry of `J^T J`, `J^T r` and
`||r||^2` is a quadratic polynomial in kappa whose coefficients are contractions
of `(Y, Yt, m, Km)`: three rank x rank matrices, four rank-vectors, three
scalars. Those are grid-sized to BUILD once, offline, and **grid-free to use**.
`u_prev` never has to be formed either -- it is `s*(Y^T h + bias*m)`, so its five
projections recurse in rank space.

Total kappa-free offline artefact: **1.58 MB**, mesh-independent in size.

### Verification before spending cluster time
- `test_red2d.py`, 12 random configurations, f64 throughout: max relative
  deviation **9.3e-14 in J^T J**, 2.1e-8 in J^T r, and **identical Gauss-Newton
  counts on 12/12**. The reformulation is exact, not an approximation.
- The same test in f32: `J^T J` off by **6.8e-2**, `J^T r` by **5.1e-1**, GN
  counts flip on **4/12**. The reduced Grams have to be f64. This is the GRAM64
  lesson again, and `main()` now refuses to build the reduced variants unless
  `X64=1`.
- x64 is a global flag, so every dtype on the FOM path and on the dense-Jacobian
  ROM path is now pinned to f32 explicitly (`jnp.linalg.norm` and
  `jnp.asarray([<0-d array>])` both promote silently otherwise). Round 8 runs an
  **X64=0 control on all four cells** whose only job is to show those paths did
  not move.
- `jax.jacfwd` over the MLP was replaced by the closed form
  `dh/dz = W_d^T + W_r^T diag(s2') W2^T diag(s1') W1^T`; agrees with jacfwd to
  **7.4e-8 relative** and bought **nothing** in time (949 vs 956 us/iter), so
  the MLP Jacobian was never the bottleneck. Kept anyway, and `red_jf` retains
  the jacfwd version so the claim stays checkable.

### Local shape probe — the reduced path is mesh-flat, as designed
Counted loop, cap 4, so both paths pay exactly 200 GN iterations. GB10, random
weights, **cost only, nothing reportable**:

| | dense us/iter | reduced us/iter |
|---|---|---|
| N=64-like | 403 | 949 (f64) / 302 (f32) |
| N=128-like | 607 | 952 (f64) / 300 (f32) |

The reduced path is **flat in the mesh** and the dense one is not, which is the
whole point. But its fixed cost on the GB10 is ~950 us in f64 and ~300 us in
f32: **f64 costs 3.2x on this device**. That is a GB10 property -- its FP64 rate
is far below 1:2 -- and it is why this decision cannot be made locally.
A100/H100/H200 are nominally 1:2, so round 8 measures the device's own
f64/f32 matmul ratio first and records it in the log.

In f32 on the GB10 the reduced path is already **2.05x** the dense path at
N=128-like and 1.38x at N=64-like, and the gap grows with the mesh.

### What round 8 measures
- Phase A: all four 2D cells, X64=0 control then X64=1 with
  `red` / `red_m` / `red64` / `red_jf` against `v4w` in the same process, plus
  N=128 with `--xla_gpu_graph_level=3` on both sides.
- Phase B: the untried composition. Round 5 got ~8% from graph capture and
  noted why it was not more -- a data-dependent `while_loop` cannot be captured.
  The counted variants have a static trip count, so `v5c` + graph capture should
  compose. `h3d_n64_fast` (cap 2, currently 0.988x) and `h3d_n32` (cap 3) pay
  nothing for the counted loop, so they are where this can matter.

### Not yet attempted, and why
The same reformulation applies to Heat-3D and would remove empirical quadrature
**entirely** -- exact projection has no NNLS support, so the drift mechanism
found in round 6 disappears by construction -- and it needs ~9x fewer flops per
iteration than the EQ path (11 vs 100 Mflop at N=128). But the 3D loop is
dispatch-bound, not flop-bound, so fewer flops may buy nothing. Phase A settles
the fixed cost of the reduced form on the real device, and that is the number
that decides whether the 3D build is worth doing. Waiting for it rather than
guessing.

### Round 8 results

Two jobs: **2000545** (A100, pax106, **7 co-resident jobs**) and **1994862**
(H200, pax010). Both measured the same thing; the agreement across two device
types and two isolation states is the reason to believe it.

#### Phase A — Heat-2D reduced-Gram

H200 (job 1994862), `red` is the reduced variant, `v4w` the dense control in the
same process:

| cell | `v4w` | **`red`** | `red` + graph | rel-L2 `v4w` -> `red` | GN |
|---|---|---|---|---|---|
| h2d_n128_fast | 0.502x | **1.133x** | **1.214x** | 1.4432e-2 -> 1.2748e-2 | 7.8 -> 3.5 |
| h2d_n128_acc | 0.227x | **1.032x** | **1.080x** | 1.0920e-2 -> 8.6350e-3 | 16.5 -> 4.1 |
| h2d_n64_fast | 0.137x | 0.355x | -- | 1.0903e-2 -> 8.5286e-3 | 8.0 -> 3.6 |
| h2d_n64_acc | 0.055x | 0.296x | -- | 5.1636e-3 -> 5.1428e-3 | 19.4 -> 4.0 |

A100 (job 2000545) gives 1.112x / 0.998x / 0.355x / 0.296x on the same four --
the same numbers on a different device with seven co-tenants.

**Both Heat-2D N=128 arms clear 1x, and every 2D cell got MORE accurate**, by
0.4% / 10.6% / 14.4% / 21.8%. The ROM cost is 33-49 ms on all four cells,
against 110-300 ms for the dense assembly: nearly independent of mesh and of the
iteration cap, which is what a rank-space solve should look like.

`red64` (f64 iterates) is within 0.6% of `red` everywhere, so carrying the
iterate in f64 buys nothing. `red_jf` (jacfwd instead of the closed-form MLP
Jacobian) is within 1.2%, confirming the local finding that the MLP Jacobian was
never the bottleneck. `red_m` (counted loop) pays the full cap and loses badly
(173 ms vs 37.6 ms at N=128 acc) -- as it should, since `red` now stops at 4 of
20.

#### The Gauss-Newton count fell, and that has to be accounted for

`[equiv] red vs v4w: identical GN counts 0/10` on every cell. **This is NOT the
same iteration**, and the honest decomposition of the N=128 acc result is

| | |
|---|---|
| total ROM speedup vs `v4w` | 243.49 / 37.59 = **6.48x** |
| of which Gauss-Newton count | 16.5 / 4.1 = **4.02x** |
| of which per-iteration cost | **1.61x** |

so most of it is the iteration count, not the mesh-free assembly. Per-iteration
the four cells give 1.28x / 1.24x / 1.61x / 1.49x.

The likely mechanism, to be confirmed in round 10 and not asserted here: the 2D
stopping test is on the **absolute** ||J^T r||, and in f32 that norm does not
reach the 1e-3 / 5e-3 threshold, so the released solver runs to its cap on
roundoff. In f64 the test fires at ~4 iterations. If that is right then the 4x
belongs to precision and the reduced assembly's real contribution is that it
makes f64 affordable at all: **f64 on grid-sized arrays measured 4.1x the f32
per-iteration cost locally (2543 vs 613 us), while f64 on rank-sized arrays is
free.** The two changes compose rather than compete, but the credit still has to
be split, which is what round 10 measures with `v4w64` (dense assembly, f64) and
`v4w@4` (dense assembly, f32, capped at 4).

#### Phase B — Heat-3D counted loop + graph capture

The composition works where the cap is tight, and only there:

| cell | cap / realised | no graph | graph | rel-L2 | vs posted |
|---|---|---|---|---|---|
| h3d_n128_acc `v5w` | 12 / 3.1 | 3.838x | **4.117x** (1.29-13.04x) | 1.8321e-1 | +0.7% |
| h3d_n64_fast `v5c` | 2 / 2.0 | 0.992x | **1.045x** (0.43-3.04x) | 7.6710e-2 | 2.3% better |
| h3d_n32 `v5c` | 3 / 3.0 | 0.485x | 0.520x | 9.5973e-2 | +3.2% |

**The FOM does not move under the flag** -- median 121.46 ms without, 121.74 ms
with, at N=128 -- for the reason round 5 gave: its CG trip count is
data-dependent and cannot be captured. The flag was set for both sides in the
same process, which is what rule 1 requires; the asymmetry is in what the two
loops are, not in how they were treated.

`v5c` + graph capture does NOT rescue N=128 (1.331x against `v5w`'s 4.117x): the
counted loop still pays cap 12 against 3.1 realised. Best variant stays
cell-dependent, `v5w` at N=128 and `v5c` at N=32 / N=64 fast.

#### Where this leaves the block
**Five of nine heat cells at or above 1x, up from one.** h3d_n128 acc and fast at
~4.1x, h2d_n128 fast at 1.21x, h2d_n128 acc at 1.08x, h3d_n64_fast at 1.05x.
Nothing about the FOM changed in any of it.

#### One process note, recorded rather than hidden
`h2opt.py` was re-synced into `/heatopt` while job 1994862 was running there, to
add the round-10 control variants. The edit is a no-op for every variant that job
measured (`f64_assembly` defaults False, which reproduces the previous dtypes
exactly, and the knob-parsing change only affects the `@n` form, unused in round
8). The X64=0 controls in that job bracket the edit and agree. It should still
not have been done, and round 9/10 use their own directories.

---

## Round 9 — the reduced-Gram form in 3D (job 2010236, H200 pax009, 4 co-tenants)

`h3red.py`. The EQ support and weights are used **exactly as released** -- same
nodes, same weights, same weighted least-squares objective, same damping, same
4-point backtracking, same relative-gradient stopping test. Only the assembly
changes: `_f_norm` is affine in the rank-512 feature vector, so the weighted
contractions are polynomials in DT*kappa with kappa-free rank x rank
coefficients. Offline build **1.2 s, 6.31 MB**, mesh-independent in size.

| cell | n_eq | `v5w` | `red` | `red_m` | best (+ graph) | rel-L2 `v5w` -> `red` | GN |
|---|---|---|---|---|---|---|---|
| h3d_n128_acc | 640 | 3.795x | 3.813x | 1.193x | **4.098x** (`red`) | 1.832985e-1 -> 1.832875e-1 | 3.1 = 3.1 |
| h3d_n128_fast | 640 | 3.878x | 3.912x | 1.219x | ~4.1x (`red`) | 1.961710e-1 -> 1.962725e-1 | 3.1 = 3.1 |
| h3d_n64_fast | 1280 | 0.975x | 1.142x | 1.254x | **1.302x** (`red_m`) | 7.671181e-2 -> 7.668479e-2 | 2.0 = 2.0 |
| h3d_n64_acc | 1280 | 0.310x | **0.482x** | 0.246x | 0.482x (`red`) | 7.262492e-2 -> 7.263294e-2 | 8.6 -> 7.1 |
| h3d_n32 | 512 | 0.407x | 0.403x | 0.448x | ~0.55x (`red_m`) | 9.598459e-2 -> 9.598138e-2 | 3.0 = 3.0 |

**The 3D case is the clean one.** Gauss-Newton counts are identical on four of
five cells and rel-L2 moves by 0.006% to 0.05% -- so this is the same iteration,
faster. The gain tracks n_eq, which is what the mechanism predicts: 1.17x at
n_eq=1280 where the dense Jacobian spans 7x1280 = 8960 gathered entries, 1.55x at
n64_acc which pays 8.6 iterations of it, and **nothing at N=128** (n_eq=640,
3.1 iterations) where the assembly was never the bottleneck. The N=128 headline
is unchanged by the reformulation and still comes from graph capture.

Also worth recording: **3D uses a RELATIVE stopping test** (`gnorm > tol*gnorm0`)
and 2D uses an absolute one. That is why the 3D counts do not move while the 2D
counts collapse, and it is independent evidence for the 2D mechanism below.

---

## Round 10 — assigning credit for the 2D result (job 2010244, A100 pax106)

Round 8 changed two things at once, so this measures four arms per cell.
**My hypothesis was wrong**, and the control is what caught it.

h2d_n128_acc, all in one process:

| arm | assembly | precision | GN | rel-L2 | ROM ms |
|---|---|---|---|---|---|
| `v4w` | dense (N^2 x k+1) | f32 | 16.6 | 1.0094e-2 | 324.98 |
| `v4w64` | dense | f64 accumulation, f32 decoder | **16.6** | 1.0306e-2 | 335.62 |
| `v4w@4` | dense, capped at 4 | f32 | 3.9 | **6.883e-2** | 71.68 |
| `red` | reduced Grams | f64 | **4.1** | **8.636e-3** | 48.92 |

1. **Precision of the accumulation is NOT the mechanism.** `v4w64` runs the same
   16.6 iterations as f32. I had predicted it would stop at ~4. It does not, on
   any of the four cells (16.6 / 7.9 / 8.0 / 19.5 against f32's 16.6 / 7.7 / 8.0
   / 19.4), because the f32 error is in the residual ENTRIES -- the decoder is
   still evaluated in f32 -- not in how they are summed.
2. **The cap was not simply mis-set.** Capping the dense solver at 4 costs
   **8.0x** in rel-L2 at N=128 acc (6.883e-2 vs 8.636e-3) and 5.4x / 3.1x / 1.9x
   on the others. So `red`'s 4 iterations are worth more than `v4w`'s 4.

So the reduced form reaches in ~4 iterations what the released solver does not
reach in 19.4. That is consistent with the pre-flight measurement in
`test_red2d.py`: in f32 the dense `J^T J` is off by **6.8%** and `J^T r` by
**51%** against the f64 reduced value. A solver stepping on a 51%-wrong gradient
is why the count is 19.4 and why the absolute ||J^T r|| test never fires.

Round 11's `v4w64f` arm -- dense assembly with the DECODER in f64 too, the only
dense arm that can produce an f64-accurate gradient -- is what separates "f64
gradient" from "exact Gram". Either way the reduced form is what makes an
f64-accurate gradient affordable: **f64 on grid-sized arrays measured 4.1x the
f32 per-iteration cost (2543 vs 613 us locally), and on rank-sized arrays it is
free.**

### What NOT to say about the 2D accuracy
rel-L2 improves on all four cells (0.4% / 10.6% / 14.4% / 21.8%), but these arms
are the ones already recorded as unstable -- a 1e-7 perturbation moves the N=64
fast arm by 20% -- and `v4w64` moves rel-L2 by -21.5% at N=64 fast and +2.1% at
N=64 acc without changing a single iteration count. Differences of this size on
these four cells are inside the range over which they are known to be unstable.
The speedup is the claim; the accuracy change is a number to report, not to
advertise.

---

## Cross-device reproducibility audit (from results/raw only)

Rounds 8 and 10 measured the same Heat-2D cells and variants on an A100
(pax106, 7 co-tenants) and an H200 (pax010). Pairing the two by cell and variant
gives an accidental but decisive reproducibility test. Spread = |a-h|/max(a,h) in
rel-L2:

| variant | N=128 acc | N=128 fast | N=64 acc | N=64 fast |
|---|---|---|---|---|
| `red` | **0.02%** | **0.04%** | 2.66% | **21.50%** |
| `red64` | **0.02%** | 0.09% | 2.39% | **21.50%** |
| `red_jf` | 0.00% | 0.14% | 0.12% | 0.31% |
| `v4w` (dense) | 7.57% | 1.20% | 3.74% | **21.05%** |
| `base` (dense) | 6.45% | 5.74% | 0.19% | 0.21% |

Two things follow.

**1. At N=128 the reduced arms are the most reproducible results in the block**
-- 0.02-0.09% against 1.2-7.6% for the dense arms on the same cells. That is
what the mechanism predicts: a gradient assembled by summing a grid-sized
f32-accurate residual depends on the reduction order the device happens to use,
and one contracted with an exactly-precomputed f64 Gram does not. It is
independent support for the round-10 finding, arrived at without being designed
for.

**2. The Heat-2D N=64 cells are not quotable on accuracy under ANY variant.**
`red`, `red64` and dense `v4w` all move by ~21% between the two devices on the
fast arm. `red_jf` -- which differs from `red` only in computing dh/dz by jacfwd
instead of the closed form, agreeing to 7.4e-8 -- is stable at 0.31%. A cell
where a 7e-8 change in one intermediate decides a 21% swing in the reported error
is sitting on a bifurcation. This is the same fragility already recorded for
these arms (a 1e-7 perturbation moves the N=64 fast arm by 20%); it now has a
second, independent demonstration.

So: quote the N=128 cells. For N=64, quote the speedup and say the accuracy is
not reproducible across devices, which is a defect of those two arms and not of
the reformulation.

## Where the block stands after rounds 8-12

Best variant per cell, all read from `results/raw/*.json` by `besttable.py`,
never from a log. "was" is the round-7 best.

| cell | best variant | speedup | per-kappa range | rel-L2 | vs posted | was | above 1x |
|---|---|---|---|---|---|---|---|
| heat3d N=128 acc | `v5w` + graph | **4.117x** | 1.29-13.04x | 1.8321e-1 | +0.7% | 3.829x | 10/10 |
| heat3d N=128 fast | `red` | **3.912x** | 1.20-12.82x | 1.9627e-1 | +3.3% | 3.809x | 10/10 |
| heat3d N=64 fast | `red_m` + graph | **1.302x** | 0.53-3.78x | 7.6685e-2 | -2.3% | 0.988x | 6/10 |
| heat2d N=128 fast | `red` + graph | **1.214x** | 0.49-2.93x | 1.2748e-2 | -5.9% | 0.510x | 5/10 |
| heat2d N=128 acc | `red` + graph | **1.080x** | 0.40-2.44x | 8.6350e-3 | -15.3% | 0.228x | 5/10 |
| heat3d N=32 | `v5c` + graph | 0.520x | 0.29-1.32x | 9.5973e-2 | +3.2% | 0.479x | 1/10 |
| heat3d N=64 acc | `red` | 0.482x | 0.11-1.03x | 7.2633e-2 | -3.0% | 0.323x | 1/10 |
| heat2d N=64 fast | `red` | 0.458x | 0.20-1.17x | (not reproducible) | -- | 0.191x | 1/10 |
| heat2d N=64 acc | `red` | 0.375x | 0.18-1.11x | 5.0060e-3 | -3.9% | 0.077x | 1/10 |

**Five of nine cells at or above 1x, against one after round 7.** Every cell
improved, by 1.03x to 4.86x over its round-7 figure. No cell got less accurate
than its posted value by more than 3.3%.

### Caveats attached to that table
- **Every row was measured on a shared node.** Two exclusive-node jobs (2014366
  for 2D, 2014372 for 3D) are QUEUED and have not run -- all H200 nodes are
  partially allocated, so Slurm gives no start estimate. Isolation is recorded
  per run in `extra.isolation_evidence` from what Slurm reported, and every row
  above says `shared`.
- The two N=128 3D rows and the two N=128 2D rows reproduce across two nodes;
  h3d_n128_acc now has seven independent measurements.
- Graph capture is a compiler flag set for the **FOM as well** in the same
  process. The FOM does not benefit (121.46 vs 121.74 ms) because its CG trip
  count is data-dependent and cannot be captured. That asymmetry is in what the
  two loops are, not in how they were treated.
- `red_m` at h3d_n64_fast pays a cap of 2 against 2.0 realised, so the counted
  loop is free there. It is ruinous on any cell whose cap exceeds its realised
  count, and the best variant remains cell-dependent.
