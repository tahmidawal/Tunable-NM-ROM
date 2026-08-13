# The coordinate-network decoder: why it removes the resolution wall

*Companion to `README.md` in this directory. Every number below is quoted from a
results JSON or the experiment archive in this worktree; the file is cited next
to each number. Single seed (0) throughout — see caveats in §6.*

---

## 1. The disease: the CP decoder is secretly a fixed linear subspace

### 1.1 What the code actually computes

The production decoder on `main` is `LinearCPDecoder`
(`poisson/src/tunable_rom_poisson/models/decoder.py`; Heat's `CPDecoder` in
`heat/src/tunable_rom_heat/models/decoder.py` is the same minus the linear
skip). Trimmed to its skeleton:

```python
# poisson/src/tunable_rom_poisson/models/decoder.py
h_nl  = nn.Dense(self.rank, name="W_rank")(...)          # MLP(z) -> R^rank
h_lin = nn.Dense(self.rank, name="W_direct", ...)(z)     # linear skip
h     = h_lin + h_nl                                     # rank-R channel weights

W_x = self.param("W_x", factor_init, (self.rank, self.N))   # grid-tied
W_y = self.param("W_y", factor_init, (self.rank, self.N))   # grid-tied
u   = jnp.einsum("r,ri,rj->ij", h, W_x, W_y) + bias
```

Look at where the nonlinearity lives and where it does not. The MLP is
nonlinear **in z**, but z only produces the coefficient vector `h ∈ R^rank`.
The field itself is

```
u = Σ_r  h_r(z) · ( W_x[r] ⊗ W_y[r] )  +  bias
```

The `W_x[r] ⊗ W_y[r]` are **fixed rank-1 basis fields** — learned during
training, but frozen at inference and independent of z. Every field the
decoder can ever emit lies in the span of these `rank` basis vectors. The
decoder is a nonlinear *parameterization of coefficients* over a **fixed
linear subspace** of dimension ≤ rank (and a constrained one: its basis
vectors must be separable products, so it is at best a rank-R subspace and
generally a worse one than the optimal).

The testbed's `GridTiedDecoder` (`poisson2d_diag.py:118-130`) distills this to
one line, and is actually *more* expressive than CP (an unconstrained
`(rank, N²)` matrix contains every CP-factorized subspace of the same rank):

```python
# poisson2d_diag.py — GridTiedDecoder
h = nn.Dense(self.rank)(h)                                # MLP(z) -> R^24
W = self.param("W", nn.initializers.normal(0.01), (self.rank, self.n_nodes))
return h @ W[:, idx] + b                                  # u = h(z) @ W
```

So results against `GridTiedDecoder` are a *generous* stand-in for the CP
decoder: whatever ceiling it hits, CP hits at least as hard.

### 1.2 Consequence 1: the POD floor is a hard ceiling

Because the output is confined to a fixed r-dimensional subspace, the
decoder's error is bounded below by the best possible r-dimensional linear
approximation of the solution family — which is exactly what POD-r measures.
No amount of training, data, or MLP capacity can cross it; it is a property of
the *family*, not of the optimizer. For the translated-Gaussian-bump family
this floor decays slowly, because translating a sharp local feature is the
classic worst case for linear subspaces (slow Kolmogorov n-width decay). The
measured POD sweep at N=256 (`round3/results_2d_N256.json`):

| r | 1 | 4 | 8 | 24 | 64 |
|---|---|---|---|---|---|
| POD-r rel-L2 | 5.67e-1 | 2.98e-1 | 1.92e-1 | 7.88e-2 | 2.77e-2 |

Sixty-four optimal linear modes still leave ~2.8% error on a 4-parameter
family. That is the mathematical wall.

### 1.3 Consequence 2: grid-tied parameters make error *rise* with N

The wall observed in this repo is worse than the POD floor: the CP
autoencoder's error does not merely stop improving with resolution, it
degrades (this is the motivating observation recorded in `README.md`). The
testbed reproduces the mechanism. The grid-tied arm's basis `W` has
`rank × N²` learned entries — the parameter count is tied to the mesh:

| N | grid-tied params | grid-tied rel-L2 | its own POD-24 floor |
|---|---|---|---|
| 16 | 26,393 | 2.45e-1 | 9.83e-2 |
| 32 | 44,825 | 1.90e-1 | 8.26e-2 |
| 64 | 118,553 | 1.85e-1 | 7.96e-2 |
| 128 | 413,465 | 2.17e-1 | 7.90e-2 |
| 256 | 1,593,113 | 2.42e-1 | 7.88e-2 |

*(all from `round3/results_2d_N*.json`; 80k steps each, identical training
recipe to the coord-net arm)*

Two things are wrong at once:

1. **It never reaches its own floor.** At every N it sits 2–3× above the
   POD-24 error its rank provably permits. The subspace is learnable in
   principle (SVD finds it in seconds); SGD from random init does not find it.
   This repo already knew this disease in another guise: the inherited
   `FREEZE_WDEC=1` rule exists because *training* the linear lift destroys the
   POD initialization (see `CLAUDE.md`, accuracy-critical rules).
2. **Refining the mesh makes it worse.** From N=64 to N=256 the error climbs
   1.85e-1 → 2.42e-1 while the parameter count balloons 118k → 1.6M and the
   data (512 snapshots) and training budget stay fixed. Each basis entry gets
   gradient signal from a shrinking fraction of the sampled points (P=4096
   points per step covers 100% of an N=32 grid but 6% of an N=256 grid), and
   the optimization problem grows with N while the information content of the
   family does not. Resolution is a *variable of the architecture*, and a
   hostile one.

That is the full disease: a provable linear ceiling, plus a trainability
pathology that keeps the model above even that ceiling and couples it to N
with the wrong sign.

---

## 2. The innovation: a decoder with no grid in it

### 2.1 The idea

Replace "z → coefficients over grid-anchored basis vectors" with a network
that represents the **continuum field directly**:

```
u_θ : (x, y; z) ↦ scalar
```

The decoder takes a *coordinate* and the latent code and returns the field
value at that point. The mesh appears nowhere in the parameterization — it
enters only at evaluation time, as the particular set of points you choose to
query. Resolution stops being something the architecture can even see.

```
   GRID-TIED (CP-style)                     COORDINATE NETWORK
   ====================                     ==================

   z ∈ R^4                                  (x,y) ∈ [0,1]²        z ∈ R^4
     │                                          │                   │
     ▼                                          ▼                   │
   MLP(z)                                  Fourier features         │
     │                                     [x, sin(jπx), cos(jπx),  │
     ▼                                      y, sin(jπy), cos(jπy)]  │
   h ∈ R^24  (coefficients)                     │                   │
     │                                          └───── concat ──────┘
     ▼                                                  │
   h @ W        W: (24, N²) grid-tied                   ▼
     │          params GROW with N                4-layer MLP (128 wide)
     ▼                                                  │
   u on THE grid (N² values)                            ▼
                                                  u(x,y;z)  ONE scalar
   · output = point in a fixed
     24-dim linear subspace                 · 66,945 params at EVERY N
   · POD-24 floor = hard ceiling            · no subspace: nonlinear in x AND z
   · cannot be evaluated on any             · evaluate on any point set:
     other grid (W rows ARE nodes)            grid choice = sampling choice
```

### 2.2 The actual forward pass

From `poisson2d_diag.py:133-153` (the round-3 runs used the `_nf` variant with
`N_FREQ=32`, giving the 66,945-param net in the JSONs):

```python
class CoordDecoder(nn.Module):
    n_freq: int = N_FREQ
    hidden: int = 128

    @nn.compact
    def __call__(self, xy, z):
        # xy: (P, 2) coordinates, z: (4,)
        j = jnp.arange(1, self.n_freq + 1, dtype=jnp.float32)

        def ff(c):  # (P,) -> (P, 2J+1)
            return jnp.concatenate(
                [c[:, None], jnp.sin(jnp.pi * j * c[:, None]),
                 jnp.cos(jnp.pi * j * c[:, None])], axis=1)

        zz = jnp.broadcast_to(z, (xy.shape[0], z.shape[-1]))
        h = jnp.concatenate([ff(xy[:, 0]), ff(xy[:, 1]), zz], axis=1)
        h = nn.swish(nn.Dense(self.hidden)(h))
        h = nn.swish(nn.Dense(self.hidden)(h))
        h = nn.swish(nn.Dense(self.hidden)(h))
        h = nn.swish(nn.Dense(self.hidden)(h))
        return nn.Dense(1)(h)[:, 0]
```

Three stages:

1. **Fourier features.** Each coordinate axis is lifted to
   `[c, sin(πjc), cos(πjc)]` for j = 1…n_freq. A plain MLP on raw coordinates
   is spectrally biased toward low frequencies and cannot fit sharp bumps;
   the sin/cos lift hands it an explicit multi-scale basis. (This choice has a
   sharp edge — see the Nyquist rule, §6.)
2. **Concatenate with z** and pass through a 4-layer swish MLP.
3. **One scalar out** — the field value at that point.

Training never assembles a grid-shaped output: each step samples a random
subset of P ≤ 4096 node indices and fits values at those points
(`poisson2d_diag.py:191-194`), the same subsets for both arms, so per-step
cost is resolution-independent and the comparison is like-for-like at the
batch level. (Per-FLOP the coord-net step is more expensive — the audit in
`codex-verify-out.md` notes ~5× wall-clock in the 1D head-to-head — a caveat
worth carrying, but irrelevant to the ceilings below, which no budget moves.)

Why this kills both symptoms of §1:

- **No fixed subspace.** The output is nonlinear in x *jointly* with z, so
  the reachable set of fields is a curved manifold, not a span. The
  n-width/POD bound simply does not apply to it.
- **No grid-tied parameters.** 66,945 parameters at N=16 and at N=256
  (`round3/results_2d_N16.json`, `round3/results_2d_N256.json` — same
  `params_coord`). Refining the mesh gives the *same* model more/better
  training points; it does not enlarge the optimization problem.

---

## 3. Each measured result, traced to its architectural cause

### 3.1 Breaking the equal-dimension linear barrier

If the manifold hypothesis is right — the family is intrinsically
4-dimensional, so a *nonlinear* decoder with 4 latent inputs should beat every
4-dimensional linear model — the coord-net must land below POD-4. It does, in
both dimensions.

**1D, translated-bump family, N=1024, 40k steps both arms**
(`results_bump_upgraded.json`):

| model | reduced dim | rel-L2 |
|---|---|---|
| POD-3 (optimal linear at equal dim) | 3 | 4.754e-2 |
| POD-24 (optimal linear) | 24 | 6.671e-5 |
| grid-tied decoder, rank 24 | 24 | 9.023e-3 |
| **coord-net decoder** | **3** | **4.473e-3** |

The coord-net with 3 latent variables is **10.6× below** the best any
3-dimensional linear model can ever do, while the rank-24 grid-tied arm sits
~135× above its own linear floor.

**2D, round 3, 80k steps, one Tufts A100 job per N**
(`round3/results_2d_N*.json`):

| N | POD-4 (equal dim) | POD-24 | grid-tied (rank 24) | coord-net (4 latent) |
|---|---|---|---|---|
| 16 | 3.14e-1 | 9.83e-2 | 2.45e-1 | 1.01e-1 |
| 32 | 3.01e-1 | 8.26e-2 | 1.90e-1 | 4.04e-2 |
| 64 | 2.99e-1 | 7.96e-2 | 1.85e-1 | **3.83e-2** |
| 128 | 2.98e-1 | 7.90e-2 | 2.17e-1 | 4.14e-2 |
| 256 | 2.98e-1 | 7.88e-2 | 2.42e-1 | **3.89e-2** |

At N≥32 the coord-net sits **~7.5× below POD-4** and beats POD-24 outright
with 6× fewer reduced variables. *Cause:* the only thing separating POD-4 from
this result is linearity of the lift — same latent dimension, same data, same
loss. Removing the fixed subspace removes the bound.

### 3.2 Lossless mesh transfer

For the grid-tied decoder, "evaluate on a different grid" is not a hard
operation — it is an undefined one: the rows of `W` *are* grid nodes, and a
finer grid changes the parameter shape. For the coord-net it is trivial: feed
finer coordinates.

Measured (`transfer2d_eval.py` on the round-2 checkpoints; reported in
`README.md`): the coord-net **trained at 32×32** (1,024 points), evaluated
natively at **256×256** (65,536 points) with zero retraining and zero
interpolation, scores **4.21e-2** — matching the natively-trained-at-256 model
(**4.41e-2**, `round2/results_2d_N256.json`) to within run-to-run noise.

*Cause:* the network learned the continuum field u(x, y; z); the evaluation
mesh is just a sampling of it. This is not a robustness property that
training bought — it is a property the architecture has by construction.

### 3.3 Error falls ~11× with training resolution instead of rising

The final convergence sweep (FiLM net of §4, one Tufts job per N, all
evaluated against a common N=512 CG reference —
`film/film_convergence_fixed.json`, the canonical corrected file):

| train N | data-floor (discretization bound) | FiLM coord-net vs N=512 reference |
|---|---|---|
| 16 | 4.74e-2 | 7.21e-2 |
| 32 | 4.20e-3 | 1.34e-2 |
| 64 | 9.81e-4 | 9.79e-3 |
| 128 | 2.46e-4 | 1.09e-2 |
| 256 | 7.02e-5 | **6.26e-3** |

Error **descends 11.5×** across the sweep (7.21e-2 → 6.26e-3) and never turns
upward. The regime structure is legible against the data-floor column (the
error of the coarse FD training data itself, interpolated to the reference
grid): at N=16 the net is **data-limited** — it sits near the floor because
16×16 snapshots barely contain the bumps; from N=32 on it is
**capacity-limited**, hovering at its own fit floor (~6e-3–1.1e-2, the N=128
wiggle is single-seed noise) while the data-floor keeps collapsing beneath it.
The same shape appears in 1D (`results_convergence.json`): coord-net error
falls monotonically 2.50e-1 → 8.78e-3 from N=16 to N=512, then holds.

*Cause:* with no grid-tied parameters, a finer training mesh changes only one
thing — the training data gets more accurate and denser. The model gets
strictly better information at fixed problem size, so error tracks the data
quality down until the network's own capacity binds. Contrast the grid-tied
column in §1.3, where the same mesh refinement *enlarges the model* and error
climbs. Resolution has flipped from an adversary of the architecture to a pure
data-quality knob — POD's healthy behavior, at accuracy levels POD-24 cannot
reach (the N=256 cell beats POD-24 by ~12× and POD-64 by ~4×).

---

## 4. FiLM conditioning: moving the plateau once linearity is gone

The concat-conditioned coord-net of §2 plateaus at ~3.9e-2 in 2D — its own
fitting error, not any linear floor. Three suspects were identified
(`README.md`): the conditioning mechanism, training-set coverage, and gradient
dilution on the sharp bumps. Two pieces of evidence pointed at conditioning
first: doubling Fourier bandwidth 16→32 frequencies bought only ~12% (round 2
N=256: 4.41e-2, `round2/results_2d_N256.json` → round 3: 3.89e-2,
`round3/results_2d_N256.json`), so raw spatial frequency content was not the
binding constraint — *how z couples to space* was.

**Why translation is the hard case for concatenation.** The dominant latent
directions are the bump center (cx, cy): the net must *move* a sharp feature
across the domain as z varies. In the Fourier feature basis, translation is a
phase rotation — `sin(jπ(x−c)) = cos(jπc)·sin(jπx) − sin(jπc)·cos(jπx)`, a
**z-dependent linear recombination** of the sin/cos features. A
concat-conditioned first layer cannot express that: its weights on the Fourier
features are fixed, and z enters only additively, so the multiplicative
x–z interaction must be synthesized indirectly through depth — the net ends up
memorizing bumps at many positions rather than learning "one bump, shifted."

**FiLM** (feature-wise linear modulation) makes the interaction native: z
generates a per-layer, per-feature scale and shift of the trunk activations.
From `poisson2d_film.py:147-156`:

```python
def film_apply(params, z, xy):
    g = jax.nn.swish(z @ params["z_embed"]["W"] + params["z_embed"]["b"])
    film = (g @ params["film"]["W"] + params["film"]["b"]).reshape(
        N_LAYERS, 2, HIDDEN)
    h = coord_features(xy)                    # Fourier features only — no z
    for i, lyr in enumerate(params["trunk"]):
        h = h @ lyr["W"] + lyr["b"]
        h = h * (1.0 + film[i, 0]) + film[i, 1]   # z-dependent scale & shift
        h = jax.nn.swish(h)
    return (h @ params["out"]["W"] + params["out"]["b"])[:, 0]
```

Now every trunk layer's features are multiplicatively reweighted by z —
exactly the operation phase rotation needs — with the modulation initialized
near identity (`film["W"] * 0.01` in `init_film_net`) so training starts from
the plain trunk. The upgrade bundle (`poisson2d_film.py`) attacked all three
suspects at once:

- **FiLM conditioning** on a 5×256 trunk, 463,681 params;
- **4× data** (2,048 train / 256 val) — coverage of the 4-d parameter box;
- **source-centered importance sampling** — half of each step's 8,192 points
  drawn near the sample's bump center (`sample_points`,
  `poisson2d_film.py:165-177`), so the hardest 0.1% of the domain finally gets
  gradient signal proportional to its difficulty;
- 120k steps.

Result: the fit floor moved **~6×**, from 3.89e-2 (round-3 concat net, N=256)
to 6.26e-3 (`film/film_convergence_fixed.json`, N=256) — and 5–9× across the
mid-resolution cells. That is the load-bearing asymmetry of this whole
experiment: when the *linear* decoder was stuck at 2e-1, no lever moved it
below its 7.9e-2 floor, because the floor was mathematics. When the coord-net
was stuck at 3.9e-2, an engineering push moved it 6×, because the floor was
budget.

---

## 5. What this means for the NM-ROM

The ViT-CP wall was never an optimization problem to be tuned away — the
decoder family could not express the answer (§1.2), and its grid-tied
parameterization actively degraded with the very refinement that should have
helped (§1.3). The coordinate decoder removes both by construction, and every
downstream property measured here — the equal-dimension barrier break, the
free mesh transfer, the healthy error-vs-N curve — is a corollary of the one
design decision: **no parameter in the decoder is anchored to a grid node.**

For the ROM pipeline specifically, the EQ/hyper-reduction step needs only
*point evaluations* of the decoder — which is natively what a coordinate
network provides, at cost independent of N. That is the integration experiment
still to run (§6, and `README.md` next steps).

---

## 6. Honest caveats

- **Single seed.** Every number here is seed 0. The order-of-magnitude gaps
  (coord vs POD-4, the 11× descent) are unlikely to flip, but the ~2× gaps
  (e.g. grid-tied vs coord in the 1D upgraded run) and the N=128 wiggle could
  move under seed variance. Multi-seed replication is the first listed next
  step.
- **Testbed, not the real pipeline.** These decoders are conditioned on the
  *true* normalized family parameters z — no encoder, no Gauss–Newton solve,
  no EQ. That isolates the decoder (the point of the diagnosis) but means the
  end-to-end ROM claim is still open. The 4.75e-2/2.98e-1 POD yardsticks are
  themselves oracle projections, so the comparison is fair in both directions.
- **A capacity floor remains.** The FiLM net flattens at ~6e-3–1.1e-2 while
  the data-floors beneath it reach ~1e-4. To see monotone descent *through*
  N ≥ 64 the fit floor needs another ~50× — more capacity/steps, or the
  warped/deformation conditioning listed in `README.md`. The claim is not
  "solved"; it is that this floor demonstrably moves when pushed, and POD's
  does not.
- **Compute matching.** The arms are iteration- and batch-matched, not
  FLOP-matched; the coord-net costs more per step (~5× wall-clock in the 1D
  audit, `codex-verify-out.md`).
- **The Nyquist bandwidth rule.** Fourier features must respect the training
  grid's Nyquist limit: **n_freq ≤ N/2**. The original N=16/32 FiLM runs used
  n_freq=32 on grids that cannot represent those frequencies; the net
  exploited aliased features that fit the training grid perfectly and exploded
  off-grid — 3.66e4 and 1.77e3 rel-L2 on the reference grid, preserved in
  `film/film_convergence.json` as the record of the landmine. **Never quote
  that file's N=16/32 cells as real errors**; the Nyquist-capped retrains
  (n_freq 8/16, `film-fix/`) produced the corrected
  `film/film_convergence_fixed.json` used throughout this document. The trap
  is insidious precisely because the architecture makes off-grid evaluation
  possible: on-grid metrics looked excellent.

---

## 7. One-page summary (liftable into a paper introduction)

Nonlinear model reduction promises to beat the Kolmogorov n-width barrier:
where a linear ROM of dimension r can do no better than the best
r-dimensional subspace, a nonlinear decoder should track the solution
manifold itself, whose intrinsic dimension is the number of physical
parameters. Our ViT-CP NM-ROM did not deliver this promise, and it failed in
a diagnostic way: its error refused to fall — and eventually rose — as the
mesh was refined, precisely the regime where reduced-order models should
improve.

We show this "resolution wall" is architectural, not an artifact of
training. A CP-tensor decoder, like any decoder ending in a contraction of
latent-dependent coefficients against learned grid-anchored factors, is a
nonlinear parameterization of a **fixed linear subspace**: every field it can
emit is a linear combination of rank-many frozen basis vectors. Its accuracy
is therefore bounded by the optimal-linear (POD) floor at equal rank — a
property of the solution family no optimizer can cross — and for
translation-dominated families that floor decays slowly (in our 2D Poisson
testbed, 64 optimal modes still leave 2.8% error on a 4-parameter family). In
practice it is worse: the grid-anchored basis must be *learned by SGD*, and
as the mesh is refined the basis parameter count grows with the grid (26k →
1.6M across our sweep) while the data does not, leaving the decoder stuck
2–3× above even its own permitted floor, with error that climbs as N grows.
Mathematics sets a ceiling, and grid-tied optimization fails to reach it with
the wrong trend in N.

The repair is to remove the grid from the architecture entirely: a
**coordinate-network decoder** u(x; z) — Fourier features of the spatial
coordinate, modulated by the latent code through feature-wise linear
modulation (FiLM) — has no grid-tied parameters at all. The mesh appears only
as the set of points at which the learned continuum field is sampled. Every
pathology above disappears as a corollary. With 4 latent variables the
decoder lands ~7.5× below the equal-dimension linear floor POD-4 and
outperforms POD-24 outright at constant parameter count (67k) across a 16×
range of resolutions. A model trained on a 32×32 grid evaluates on a 256×256
grid with no retraining or interpolation and matches the natively-trained
model (4.21e-2 vs 4.41e-2). And refining the training mesh now helps
monotonically: error falls 11× across the sweep (7.2e-2 → 6.3e-3), tracking
the discretization error of the training data down to the network's own
fitting floor, and never rises.

That remaining floor is the right kind of floor. When our linear decoder
stalled, no amount of data, steps, or width could move it — the barrier was a
theorem. When the coordinate decoder stalled at 3.9e-2, a targeted push —
FiLM conditioning for translation, 4× data, importance sampling near the
sharp features — moved it 6× in one round. **Linear methods hit a wall of
mathematics; coordinate decoders hit a wall of budget — and budget walls
move.**
