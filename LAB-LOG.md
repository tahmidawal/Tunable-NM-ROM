# Lab log — Tunable NM-ROM

**This is the single canonical log for the project.** It lives on `main` at the repository root
and every session appends to *this* file by absolute path, whichever worktree it is working in:

```
/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/LAB-LOG.md
```

There is no per-worktree copy. "Where things stand" is rewritten by each session; the chronology
below it is append-only, oldest first.

---

# Where things stand — 2026-08-18

## The method

Solve a PDE by finding a small latent vector `z` on a learned manifold and decoding it. The
decoder is a FiLM coordinate network `u(x;z)` — meshfree, no grid-tied parameters. `z` is found
by minimising the PDE residual **projected onto M smooth sine test modes** (a weak form),
evaluated at `m` NNLS empirical-quadrature points rather than the whole grid. One solve touches
256 points, forms 64 equations, solves for 8 unknowns — independent of grid size. Minimising the
pointwise residual instead does not work: the discrete Laplacian amplifies grid-scale decoder
error by ~N².

## What is true right now

**Accuracy holds.** At k=8, N=64, held-out relative L2 against the decoder's own ceiling:
Poisson 7.65e-3 (ceiling 7.11e-3), Heat 1.87e-2 (1.16e-2), Burgers 1.65e-2 (1.15e-2). Wave
fails outright, 8.78e-1 against a 1.72e-1 ceiling, for a structural reason. POD at the same
latent size is 1.3e-1 to 2.1e-1 and **cannot reach 1% error at any number of modes** — it
saturates near 5%.

**Cost is mesh-independent.** Iterations to tolerance at k=8 are 10.1 / 10.4 / 10.3 / 10.3 / 9.4
across N = 32…512. The same 256 quadrature points are 28% of the coarsest grid and 0.1% of the
finest with error flat at 8.4–8.6e-3.

**Speed, measured on one GPU against a full solver run only as far as our own accuracy:**
Poisson 0.35 / 0.65 / 1.26 / 1.73 / 3.39× at N = 32…512 — we cross over between 128 and 256.
Burgers 0.18 / 0.21 / 0.55 / 0.53×, losing everywhere. On 2-D Poisson a **direct sparse solve is
494× faster than the iterative one**; a reviewer said so and was right.

**The warm-start hybrid does not pay** on either PDE. Linear extrapolation from the previous two
time steps beats it and costs nothing.

**Decoding is now the bottleneck** — 84% of online cost at N=512, against 9% for the latent solve.

## What has been retracted — do not re-derive these

1. **Both classical baselines were over-solved.** Burgers ran a fixed 8 Newton iterations per
   step (~4× more work than needed); Poisson ran CG at the tolerance used to *manufacture the
   truth data*. Corrected Burgers ladder 0.72/1.57/4.46/7.96× → 0.19/0.36/0.93/1.83×. Poisson's
   factor **depends on the assumed deployment tolerance and the tolerance must always be named**:
   ~1.16× at 1e-10, ~1.31× at 1e-8, ~1.56× at 1e-6.
2. **"The solver stalls at particular latent dimensions" is withdrawn.** It was a
   mean-over-16-cases artefact; the median sits at the ceiling at every k. Root cause is an
   unglobalised latent LM, fixed by a trust region. The same optimiser is used on every PDE, so
   **accuracy numbers across the project are probably pessimistic**.
3. **`m ≈ 4M` is a hard rule.** At `m = M` the quadrature collapses.
4. **The heat-rollout freeze is a port regression in the public repo, not in the archive.** The
   published heat numbers are not impeached by it. Separately, `_build_eq_weights` has zero call
   sites under `best-results/` — the archived Heat-2D benchmark did no hyper-reduction while its
   README advertises EQ as the mechanism. Unexplained.

## Where the work is

Everything through 18 August is on **`exp/2026-08-18-codex-handoff`**, in
`worktrees/2026-08-18-codex-handoff` — seven experiment cells, four written reports with their
figure generators, and every Codex audit. `main` carries only this log and the rules.

## What to do next, in order

1. **Finish `cost-to-tolerance`**: §7 Verdict and §8 Caveats are placeholders and the results
   Codex audit never ran. The tables are trustworthy; the prose around them is unchecked.
2. **Apply the trust-region fix upstream** (`ms_autodecoder.lm_solve`, `pro_common.lm_generic`)
   and re-measure all four PDEs.
3. **Attack the decode** — 84% of online cost at the finest grid.
4. **Resolve the Burgers correction denominator** — the two cells ladder different knobs, so it
   is formally open. A pre-registered cross-check is set up.
5. Reviewer baselines: tuned FNO, POD-DeepONet, an L-shaped domain, 3-D, preconditioned and
   direct classical solvers.
6. Decide whether `fix/heat-rollout-warm-start` merges into `main`.

---

## 2026-05-14 — 2026-05-18 (prologue, reconstructed from remote-only branches)

Not in `worktrees/`. This work lives only on `origin/decoder-explorer` and
`origin/inr-siren-speed-accuracy`, under `Experiments/` and `claude-lab/sessions/*.tex`,
authored `tawal01` before the August worktree convention existed. It is included because
two of its results are re-derived, apparently independently, in August.

### The ViT-mirror decoder ablation (2026-05-14)

Six configs, Poisson-3D and Heat-3D at N=64, swapping the CP/LinearCP decoder for a
symmetric MAE-style ViT decoder (`mirrored-vit-ablation/STATUS.md`). Jobs 776380–776386.

**The autoencoder was fine and the ROM still died.** Poisson `C_wider` reached AE val
rel-L2 5.84e-3, *beating* the LinearCP baseline's 6.62e-3 at a third of the epochs — and
its ROM solve landed at 1.25, i.e. no solution at all. All three Poisson configs sat at
ROM rel-L2 1.12–1.25 against a 6.62e-3 baseline; all three Heat configs blew up over the
50-step rollout to 8.61–13.04 against 1.76e-2. Speedup collapsed to 0.00–0.02× in every
cell.

Two failure modes, not one. The expected one was cost: self-attention couples all output
tokens, so each GN iteration must run the decoder on all N³ = 262,144 nodes and EQ
hyper-reduction buys nothing. The unexpected one was that **decoder Jacobian geometry
breaks the latent solve independently of cost** — cold-start GN on Poisson and warm-started
rollout GN on Heat both fail regardless of AE accuracy or decoder size. This is what put
"per-node evaluability" into the paper's architecture argument as a solver-conditioning
requirement, not just a speed one.

### Multiresolution and the FAS V-cycle (2026-05-15)

An autonomous session on a shared-latent multi-resolution AE (`claude-lab/sessions/2026-05-15.tex`).

Option A, nested-iteration warm start, was **exhausted and abandoned**: warm-starting fine
GN at N=128 with the coarse solve's z* made things *worse* (1.69e-2 warm vs 1.49e-2 cold),
even though the shared latent is a near-perfect restriction operator
(cos(z₆₄, z₁₂₈) = 0.9999 across 20 source frequencies). Retraining the AE for N=128 capacity
improved reconstruction (1.81e-3 → 1.11e-3) and *degraded* the ROM (1.49e-2 → 2.5–2.9e-2).
Diagnosis: the limiter is which basin EQ-NNLS picks, not the initial guess or the manifold.

Option B, a learned residual encoder, also failed (1.47e-2 at best, did not generalise from
training to inference distribution). What worked was FAS — reusing the *existing* coarse
EQ-GN solver as the coarse-correction operator on the defect equation. 1.49e-2 → **1.18e-3**
with no retraining and no new model, saturating there (v6c 15 cycles and v7a 30 cycles tie
at 1.18e-3).

Counter-intuitive rules that came out of it: **skip the post-smooth** (a fine GN smoother
after the coarse correction pulls 2–3e-3 back up to ~1e-2 by re-attracting the latent to the
EQ-NNLS-128 basin), and skipping the pre-smooth too is better still.

⚠ **Caveat the session states about its own headline.** The per-case trajectory *bounces*
between 1–4e-3 over cycles rather than converging, and the 1.18e-3 mean is obtained by
"best-across-cycles harvesting". Selecting the best cycle per case requires knowing the true
solution, so this number is not obviously deployable. Nothing in the August tree revisits it.

### SIREN cold-start, and an INR decoder three months early (2026-05-18)

`origin/decoder-explorer` → `origin/inr-siren-speed-accuracy`. Poisson-2D, N=128 (16,384
unknowns), forcing `A·∏sin(kᵢπxᵢ)`, kᵢ∈[1,3], FOM = 5-point Laplacian + CG at ~270 ms.
Training data generated by that same CG FOM — 700/140/160 — explicitly to keep operator and
data consistent.

Start state: a modulated-SIREN decoder behind a ViT encoder trained cleanly to AE val
rel-L2 8.88e-3, but **cold-start ROM rel-L2 = 1.565 for every test parameter and every
n_eq**, against the CP decoder's 3.23e-3 at 183×.

The diagnosis is the part that matters. Four probes killed the obvious hypotheses in turn:
the Jacobian at z=0 is well conditioned (not rank deficiency), the residual lies in range(J),
and the loss landscape does have good descent directions 1.5 units out. The actual cause is
that **z=0 sits in a spurious basin** — GN converges, with a small final gradient norm, to a
non-solution stationary point. Seeding at ‖z‖≈1.5 reaches rel-L2 0.043.

Seven iterations of training-recipe and solver changes took cold-start from 1.565 to
**2.88e-3**, matching the CP reference — but at **1.24× FOM, not 183×**. Accuracy was
recovered; the speed was not.

That is what produced the `LinearAffineZDecoder` push: `u(x;z) = Φ(x)W_A z + b(x)` with a
SIREN feature trunk, so the **Jacobian is constant in z** and GN converges in 2 iterations.
Final Pareto (`SUMMARY.md`), cold-start, median rel-L2 / speedup vs FOM CG:

| arm | rel-L2 median | p90 | speedup |
|---|---|---|---|
| iter-7 SIREN baseline | 2.88e-3 | 6.61e-3 | 0.13× |
| Fast SIREN ic10/if20 | 3.02e-3 | 1.00e-2 | 4.17× |
| linaff_v5 | 9.81e-3 | 1.42e-2 | 139× |
| linaff_v3 | 1.14e-2 | 1.81e-2 | 247× |
| linaff_v2 | 1.81e-2 | 3.16e-2 | 320× |
| affine_v1 (broken) | 1.95e-1 | 5.88e-1 | 194× |

Abandoned inside this cell: **affine with a nonlinear A(z)** (v1–v4, all rel-L2 > 0.2 — a
non-constant low-rank Jacobian recreates the z=0 basin), and **removing the anchor/L2(z)
regularisers**, which made it worse (0.76 vs 0.33). The fix was linearity in z, and the
encoder/ROM consistency gap was closed by adding a Laplacian-residual training loss.

**Two things here bear directly on August.** First, an INR/coordinate decoder was tried in
May and its accuracy ceiling was already known; the August framing that the FiLM coordinate
decoder is a new replacement for ViT-CP is true of the *architecture and the resolution
argument*, not of "INR decoder" as an idea. Second, "GN converges to a spurious stationary
point because the iterate leaves the trained region" is exactly the root cause the
2026-08-18 k-spike investigation lands on — **found on 18 May, re-found on 18 August**, with
no cross-reference between them.

⚠ These May speedups are measured against a CG FOM and, on 2-D Poisson, are the same class
of number the 17 Aug baseline audit deflated. They predate that audit and have **not** been
re-derived under it. Treat 320×/139× as unaudited.

---

## 2026-08-12

### The resolution wall gets an architectural diagnosis

Session `e60431d0`; the tree was promoted from a scratchpad into
`exp/2026-08-12-coord-decoder` on 08-13. All numbers seed 0, single seed, every GPU run
verified `jax_backend=gpu` at `JAX_DEFAULT_MATMUL_PRECISION=highest`.

**The problem being reacted to.** The published table on `main` *is* the wall. Poisson-2D
val rel-L2 goes 4.84e-4 → 1.08e-2 → 1.20e-1 as N goes 64 → 128 → 256; Heat-2D goes
5.21e-3 → 1.02e-2 → 1.08e-1. Error rises 20–250× with refinement while POD-only ROMs
improve as expected. Reviewer fxe8 and the third reviewer had both pressed on exactly this
(Heat-3D "degrades", and "the paper never exhibits a regime where the nonlinear manifold is
the enabling ingredient rather than an expensive alternative to linear POD").

**The hypothesis: the wall is architectural, not a training problem.** The CP decoder's
factor matrices are z-independent, so its image is a fixed low-dimensional subspace with
grid-tied basis vectors. A "nonlinear" manifold ROM that can never leave a rank-R subspace
cannot beat rank-R POD, and refining the mesh adds parameters to a basis it then has to
learn by SGD.

Everything was run as a standalone parametric-Poisson testbed (translated Gaussian bumps,
FD/CG truth) with decoders conditioned on the true family parameters, deliberately outside
`heat/` and `poisson/` so `main` stayed untouched.

**1D proof of concept** (local GB10, N=1024, 40k steps both arms,
`results_bump_upgraded.json` — verified against the JSON):

| model | dim | rel-L2 |
|---|---|---|
| POD-3 | 3 | 4.75e-2 |
| POD-24 | 24 | 6.67e-5 |
| grid-tied (CP-style) | 24 | 9.02e-3 |
| coord-net | 3 | 4.47e-3 |

The coord-net beats the equal-dimension linear model by 10.6×. The damning number is the
grid-tied row: at rank 24 it sits ~135× above the 6.67e-5 floor its own rank provably
permits. That is the trainability half of the disease — SGD cannot find the basis SVD hands
you for free.

**2D, the decisive test** (Tufts A100s, round 3, 80k steps, `round3/results_2d_N*.json` —
table verified digit-for-digit):

| N | POD-4 | POD-24 | grid-tied (rank 24) | coord-net (4 latent, 67k params) |
|---|---|---|---|---|
| 16 | 3.14e-1 | 9.83e-2 | 2.45e-1 | 1.01e-1 |
| 64 | 2.99e-1 | 7.96e-2 | 1.85e-1 | 3.83e-2 |
| 256 | 2.98e-1 | 7.88e-2 | 2.42e-1 | 3.89e-2 |

Coord-net with 4 latent variables sits ~7.7× below POD-4 and beats POD-24 with six times
fewer reduced variables. The grid-tied control sits *above* its own POD-24 floor at every
resolution while its parameter count balloons 26,393 → 1,593,113 and buys nothing. Mesh
transfer, `round2/transfer2d_results.json`: trained at 32², evaluated natively at 256² →
4.21e-2, against 4.41e-2 for the natively-trained model.

**The FiLM push.** The concat-conditioned coord-net plateaus at its own fitting error
(~3.9e-2). FiLM conditioning on every trunk layer, 464k params, n_train 2048,
source-centred importance sampling, 120k steps, graded against a common N=512 CG reference
(`film/film_convergence_fixed.json`):

| train N | coarse-interpolation discrepancy | FiLM coord-net |
|---|---|---|
| 16 | 4.74e-2 | 7.21e-2 |
| 32 | 4.20e-3 | 1.34e-2 |
| 64 | 9.81e-4 | 9.79e-3 |
| 128 | 2.46e-4 | 1.09e-2 |
| 256 | 7.02e-5 | 6.26e-3 |

Error falls 11.5× across the sweep into a capacity floor. **Not monotone**: N=64→128 rises
9.79e-3 → 1.09e-2, an 11.3% increase.

### The Nyquist landmine

Fourier-feature bandwidth must respect the training grid's Nyquist limit. The original
N=16/32 FiLM runs used n_freq=32, learned aliased features that look perfect on-grid, and
**blew up by four orders of magnitude when evaluated off-grid** — rel-L2 3.66e+04 and
1.77e+03 against the N=512 reference, preserved in `film/film_convergence.json`. Retrained
at n_freq 8/16 in `film-fix/`. Cost one round.

The general form of the lesson: any in-resolution metric is blind to aliasing, and a
coordinate decoder's whole selling point is off-grid evaluation, so **off-grid validation is
not optional for this architecture**.

Jobs (Tufts `gpu`, A100, one isolated job dir per N): round 2 = 2277176–2277180,
FiLM = 2325289–2325294 (+2325324 for the N=512 reference), Nyquist fixes = 2329167/2329168.
Job dirs deleted after pulling artefacts.

### Operational landmine — login-node /tmp

Tufts login nodes are load-balanced with node-local `/tmp`. A job script staged through
login `/tmp` silently ran round 2 with a *stale config*; caught only by a parameter-count
mismatch. Stage directly into the shared paralab path. Round 2 is kept in the tree for the
record rather than deleted.

---

## 2026-08-13

Four cells in one day, all forked from `07a4d7d`, plus an adversarial review pass over the
whole tree and two release bugs found by accident. The worktree convention
(`worktrees/YYYY-MM-DD-<slug>`, `exp/` branches, one job dir per cell) was set up this day
and has held since.

### Online cost — is a ROM iteration mesh-free? (`exp/2026-08-13-cost-scaling-{cp,coordnet}`)

Hari's question: does the online solve cost depend on the mesh at all? Two arms, both timed
**sequentially in one job on one A100** (node `pax106`), random weights of the true
architecture shapes, median+IQR over 30 timed solves after ≥3 warm-ups, plus XLA
`cost_analysis()` FLOPs.

The CP arm (job 2345628) drives the *real* `NMROMSolver` with `m` pinned at 640 and
`gn_rel_tol=0` so the loop runs exactly 10 iterations. Per-GN-iteration median is flat across
N = 32…512 at every k — spread 0.27–2.13%, worst at k=4 — while FOM CG grows
0.198 ms → 31.148 ms, a regression slope of n^0.94. FLOPs per iteration are **bit-identical
across all five meshes** at fixed k. The coord-net arm (job 2345393) is a skeleton, not a
trained model, and is flatter still (≤0.44%); it asserts in code that the GN-step jaxpr shape
multiset and the FLOP count are identical for N=32 vs N=512, and both checks passed.

**This claim was published overclaimed and walked back the same evening** (commit `55ad708`,
prompted by the adversarial review). Three separate errors:

1. "Online cost depends **only on k**" → narrowed to "**the hyperreduced GN iteration kernel
   is n-free, at fixed EQ count m, rank and iteration count**". The one-shot final decode is
   O(rank·n) and sits outside the loop; whether the *required* m or iteration count grows
   with n at fixed accuracy is a different study; and **the repo's own configs co-vary rank
   and m with N**, so this says nothing yet about the shipped trained ROMs.
2. "Both are **4–6 orders below the FOM** at N=512" → actually ~1 order: a 10-iteration
   kernel is 1.55–2.19 ms (CP) against 31.1 ms for one FOM CG solve, i.e. **20.0×–14.2×**.
   An error of about 10⁵.
3. Cross-arm absolute times were being compared. They are not comparable — different m
   (640 vs 100), different decoder, weighted LM with line search vs unweighted normal
   equations without. Only each arm's own flatness in N is the result.

**Not in the README, worth recording:** the decode was *measured* and is flat, 64.2 → 72.7 µs
across a 256× range in n. At these sizes it is latency-bound, not FLOP-bound. The figure's
"O(n) contrast" is carried entirely by the FOM CG line; `figure_cost_scaling.py` never plots
the decode at all.

**The k=4 FLOP anomaly is a lowering artefact, not physics.** Both arms report an
anomalously low count at k=4 (CP 1.67e5 vs 3.12e5 at k=2; coord-net 6.20e7 vs 8.38e8). The
local GB10 smoke disagrees with the cluster's FLOP counts wholesale (coord-net k=4: 1.37e9
on GB10 vs 6.20e7 on A100), so `cost_analysis()` is backend- and jax-version-dependent. The
load-bearing check is across-N bit-identity at fixed k, never monotonicity in k. Excluding
the GB10 smoke from the figure was correct — it is also 2–4× slower and noisy enough to be
non-monotone in N (CP k=4 is "faster" at N=64 than at N=32).

**Job 2345392 died on `ModuleNotFoundError: No module named 'flax'`** — the cluster venv had
jax but not flax, so `poisson/src/.../models/encoder.py` could not import. Fixed by
`--no-deps` installs of flax 0.12.8 + msgpack/typing_extensions/rich/pyyaml, jax verified
unchanged at 0.10.2. Note `yaml` comes from the `pyyaml` package.

### Iterations to tolerance (`exp/2026-08-13-gn-tolerance-sweep`)

The other half of the cost question: a flat *per-iteration* cost proves nothing if the
*number* of iterations grows with the mesh. `measure_gn.py` builds each package's real
solver from a trained checkpoint and records GN iterations to reach relative gradient-norm
tolerances {1e-2, 1e-3, 1e-4} at N ∈ {32,64,128,256}. Probe cap 30. A mandatory per-cell
self-check that derived counts equal the solver's own returned counts passed **8/8**.

**Abandoned approach, recorded so nobody retries it:** a `scan`-based replica of the GN loop
was written to extract histories cheaply and thrown away — the line search's discrete argmin
amplifies scan-vs-while floating-point differences into diverging chains. Histories now come
from the real solver via a `(tol=0, max_iters=j)` sweep of the same program, never from a
re-implemented loop. Residual jit-vs-eager step-0 mismatches (4/2/7/9 across the heat cells)
are stored in the JSONs rather than hidden.

**Heat, the controlled arm** (k=64 and rank=256 fixed across meshes), median warm-step
iterations over 49 warm steps:

| N | tol 1e-2 | tol 1e-3 | tol 1e-4 | rollout rel-L2 |
|---|---|---|---|---|
| 32 | 2 | 4 | 8 | 4.12e-2 |
| 64 | 2 | 4 | 9 | 5.01e-2 |
| 128 | 2 | 3 | 9 | 4.55e-2 |
| 256 | 3 | 5 | **28** (49% capped) | 6.46e-2 |

Flat across a 64× range in n at 1e-2 and 1e-3. Cold step-0 costs 9.5–25.5 iterations, so
warm-starting is where the cheapness lives. The N=256 @ 1e-4 blow-up is attributed not to
mesh-dependence of GN but to the tolerance dropping below what that model can deliver — its
rollout error 6.35e-2 is the worst of the sweep. **That is the resolution wall showing up in
the end-to-end ROM rather than in a decoder testbed.**

⚠ **The heat arm is less controlled than its README claims.** k and rank are fixed, but
`n_eq` is 100/100/128/256 across N = 32/64/128/256, because the shipped configs set
`min_eq_points` 64/128/256. So m grows roughly linearly in N here too — the exact confound
the README calls out for Poisson. Training budget also varies (`num_epochs` 80k/150k/100k).
The "genuine fixed-architecture sweep" wording is too strong.

**Poisson, the not-controlled arm**, median cold-solve iterations: 9 → 11 → 15 → 17 at 1e-2,
with capping reaching 15.7% at N=256 and 75.7% at 1e-4. The README deliberately annotates
the figure "NOT controlled" rather than claiming an N-dependence: the shipped configs
co-vary k (8/8/12/16), m (640/640/960/1280) and rank (128/128/768/768) with N, every solve
is a cold start, and the models are trained and graded against data that turned out to be
broken. The defensible statement is "the shipped configuration family's end-to-end cost rises
with N", not "GN iterations scale with the mesh".

Jobs, Tufts `gpu`, one A100 each, `jax_backend=gpu` in every log: Poisson 2346808–2346811,
Heat 2346896/2346897/2346899/2346900. **Job 2346898 has no artefact** — the README's
provenance line claims five heat ids but only four logs exist. Presumably a failed submission
resubmitted as 2346899; not recorded either way.

### Time as a coordinate (`exp/2026-08-13-heat2d-coord-decoder`)

Does the coord-decoder result survive a time-dependent PDE? The architectural move is to
treat **time as another coordinate**: `u(x,y,t;z)` with Fourier features in x, y *and* t, a
5×256 trunk FiLM-modulated by (z,t). Heat FOM mirrored from the repo (backward Euler,
dt=0.005 × 50 steps, CG per implicit step), 1-blob family `z = (cx,cy,w,a,log κ)`. Tufts jobs
2341579–2341583, one A100 and one isolated job dir per N, `jax_backend=gpu` in every log.

In-resolution, mean rel-L2 over 64 held-out trajectories × 51 slices:

| N | POD-6 | POD-24 | POD-64 | grid-tied (r24) | FiLM |
|---|---|---|---|---|---|
| 16 | 1.99e-1 | 4.20e-2 | 7.97e-3 | 1.14e-1 | 7.59e-3 |
| 64 | 1.98e-1 | 4.45e-2 | 9.38e-3 | 9.99e-2 | 6.94e-3 |
| 256 | 1.98e-1 | 4.53e-2 | 9.74e-3 | 8.96e-2 | 7.13e-3 |

The coord-net beats equal-dimension POD-6 by 26–32× and POD-24 by 5.5–7.2×, flat in N. The
grid-tied control sits **1.98–2.71× above the POD-24 floor its own rank permits**, flat in N
while its parameter count grows 21k → 33k with the grid. That is a cleaner disease signature
than Poisson gave. Per-time error is worst at t=0 (the sharp IC) and settles to ~5e-3; no
drift along the time axis. Convergence against a 512² reference: 4.58e-2 → 9.53e-3, a 4.81×
fall into a ~9.5e-3 capacity floor; a net trained on 32² evaluates natively on 512² at
2.17e-2.

**Overclaims corrected the same evening** (`1229d7e`, from the adversarial review — README
only, no numbers changed):

- "65,536× more eval points" → **1,024×** (512²/16²). A 64× inflation of the headline.
- "POD-6 is the best any equal-dimension linear model can do" → a train-fitted SVD basis, not
  a certified floor — *and* a newly disclosed mismatch: **POD rows were fitted and evaluated on
  every 4th time slice while the neural columns average all 51**. The head-to-head table was
  not apples-to-apples. (This is why the Burgers and Wave cells the next day both run POD at
  `POD_TIME_STRIDE=1`.)
- "Error falls monotonically… it never rises" → scoped to the six-time *aggregate*, with the
  disclosure that the final N=128→256 step improves by **0.2%** (9.5508e-3 → 9.5304e-3) while
  per-time errors at t-index 20/30/40/50 individually **worsen 6.7/11.1/15.7/13.1%**.
- "identical parameter count" → 442k–468k, varying via the Nyquist-capped feature width; and
  sampled points are `min(8192, N²)`, not a flat 8192.
- "f64 evaluation" → f32 inference with f64 norms.

⚠ One inaccuracy the review missed and this backfill found: the README still says the coord-net
"roughly matches POD-64 (~7-8e-3 in-res)". POD-64 is 7.97e-3 only at N=16; it is 9.4–9.7e-3 at
N≥64, so the coord-net actually *beats* POD-64 at every mesh. This one errs conservatively.

### Release bug 1 — the heat rollout is frozen after step 1

Surfaced by the GN-tolerance work: on `main`,
`heat/src/tunable_rom_heat/solver/nm_rom.py` hands the next rollout step
`u_new_eq = s * f_norm_eq(z_new)` — the backward-Euler **operator applied to the new state**,
which at convergence is exactly the *previous* step's target. Every step after the first
therefore begins at (near-)zero residual and z never moves. `rollout(u0,k,1)` and
`rollout(u0,k,50)` are **bitwise identical**. Affects Heat 2D and 3D; Poisson has no rollout
and is unaffected.

Fix on `fix/heat-rollout-warm-start` (`704cfe7`, +14 lines): add `u_center_eq(z)`, the decoded
**state** at EQ centres with no implicit operator applied, and hand that forward. Two
regression tests in `heat/tests/test_rollout_advances.py` keyed on state-change *magnitude*
(frozen chains move ~5e-7, i.e. float noise; real chains move 0.2–0.9). Neither iteration
counts nor bitwise comparison are reliable detectors — GN can burn `max_iters` on ~1e-7
cross-program float noise without moving z an ulp. Verified two-sided: the suite passes on
the fix and the regression tests fail on `main`.

This is also why the merge into the GN-tolerance branch (`202aabc`) had to happen *before*
the heat numbers meant anything: on the unfixed solver every warm step would have reported a
trivial 1-iteration exit and "flat in N" would have been an artefact of the bug. Warm steps
doing 2–9 real iterations is the fix visible in the data.

**What it does to the published numbers.** The repro on the fixed solver (job 2346901) gives
**mean rel_l2 = 2.83e-2 and median speedup 0.17× at N=64** (per-trajectory 7.75e-3…5.92e-2,
0.13×…0.36×), against the published 5.21e-3 and 39.6×. Accuracy is 5.4× worse and the
"speedup" is the ROM running about 6× *slower* than the FOM. On the fixed solver accuracy
also worsens with mesh, 2.8e-2 → 6.5e-2 — the resolution wall, live, end-to-end.

⚠ **Correction to the 08-13 diagnosis, established during this backfill.** The contemporaneous
note concluded the published 39.6× "benefited from 49 trivial steps", i.e. that the paper runs
were themselves frozen. **The archived code does not have the bug.**
`best-results/Heat-2D/N64/ACC/experiment.py:385` reads
`u_n = sc_n * constrained_decode_normalised(z_n)` — the decoded state, exactly what the fix
restores — and the same holds in FAST, in N128/N256, and in the Heat-3D archives. The freeze
is a **port regression introduced in the clean-package rewrite**, not a defect in the runs that
produced the paper numbers. The published accuracy figures are therefore not impeached by the
freeze itself, and the memory note's inference should be revised.

That makes the 39.6× → 0.17× gap need a different explanation, and two candidates are visible
in the archived source, neither flagged anywhere in the tree:

1. **The archived Heat-2D benchmark does no hyper-reduction at all.** `_build_eq_weights` is
   *defined* in every Heat-2D `experiment.py` and has **zero call sites** anywhere under
   `best-results/` — verified by grep for this entry. The archived residual solves on the full
   grid. The clean package builds a real 100-node NNLS EQ and drives `jax.jacfwd` through the
   stencil basis every GN iteration. Those are different algorithms, and yet `heat/README.md`
   advertises EQ hyper-reduction as the mechanism behind the published table.
2. **The two benchmarks are not the same measurement.** Archived: 10 val trajectories, *mean*
   speedup, absolute GN tolerance. Clean: 5 trajectories, *median* speedup, `gn_rel_tol=1e-3`,
   `gn_max_iters=12`. And `best-results/Heat-2D/N64/ACC/experiment.py` ships
   `gn_max_iters: int = 8` with the comment "Run1: gn_max_iters 20→8 — speedup push" — the ACC
   directory's committed source carries the **FAST** setting, contradicting its own README. The
   39.63× cannot be re-derived from the ACC directory as committed.

**The merge is still open** as of this backfill: `git branch --contains 704cfe7` returns only
`fix/heat-rollout-warm-start` and `exp/2026-08-13-gn-tolerance-sweep`. `main` still ships the
frozen rollout. It was nominally gated on the repro establishing true numbers; the repro landed
the same evening at 21:31 and nobody returned to it. The likelier real blocker is that the
branch touches only `nm_rom.py`, the new test and the repro artifacts, and corrects **no**
documentation — merging as-is yields a repository whose code produces 2.83e-2/0.17× while its
root README, `heat/README.md`, all six heat config headers and the `best-results/Heat-*` cell
READMEs still advertise 5.21e-3/39.6× through 269.3×. An honest merge needs either a re-run of
all six heat cells on the fixed package (never launched) or a rewrite of every heat table with
a provenance caveat. That blocker is not written down anywhere.

### Release bug 2 — Poisson trains and grades on physics it does not solve

Found while validating the same sweep, and verified again for this backfill directly from
source on `main`. All shipping Poisson **2D** configs (`poisson2d_n{64,128,256}.yaml`) use
`data_source: analytical`. `sample_parameters` draws `freqs = rng.uniform(1.0, 3.0)` —
**continuous, non-integer** — and `_analytical_u` evaluates
`u = A/(π²Σk²)·∏sin(kᵢπxᵢ)` and then multiplies by the boundary mask. For non-integer k that
masked field solves neither the continuous Dirichlet problem nor the discrete FD system;
measured `‖u_analytic − u_cg‖/‖u_cg‖ ≈ 1.0` at N=128.

The ROM's GN/EQ solve enforces the FD system, so `run_rom` grades an FD-consistent answer
against an inconsistent "truth". The repo's own recipe at N=128 gives mean rel_l2 **0.647**
against the config header's claimed **1.08e-2**. The autoencoder itself trains fine (val MSE
~1e-5) — the inconsistency is data-vs-solver, not a training failure.

Consequences for the record: the `rel_l2_per_tol` columns in all four
`gn_poisson_N*.json` (0.600/0.905/0.725/0.681) are **not ROM accuracy** and must not be
quoted as such; the iteration counts from the same files remain valid, since they measure GN
convergence of the trained models as they are.

Also observed and unresolved: `fom.cg_solve` **NaN'd on 4 of 10 N=128 sources**. Only
`poisson3d_n256_cg.yaml` uses the CG path, so 2-D CG generation appears never to have been
exercised. Validate it before adopting `data_source: cg` as the repair.

**Both bugs are second and first reproducibility failures of the public release.** Between
them, the published Heat-2D and Poisson-2D numbers in `main`'s headline table cannot be
reproduced by `main`'s own code.

### The adversarial-review pass

A Codex CLI review was run over the whole tree
(`experiments/coord-decoder/ADVERSARIAL-REVIEW-2026-08-13.md`, 17 findings, 5 blockers) and
its corrections were committed the same evening across three branches (`28dacf7`, `1229d7e`,
`55ad708`). The findings that changed claims rather than wording:

- **"Rank-R POD hard floor" is not established.** The production decoders carry a learned
  scalar bias, so the image is an *affine* set of dimension ≤ R+1; and SVD is optimal for
  train-set absolute Frobenius error while the tables report val-set mean per-sample
  *relative* L2. The POD rows are strong reference baselines, not certified lower bounds.
  The architectural diagnosis survives in weakened form — a genuine linear-width limitation —
  and the 7–30× margins are far larger than either gap.
- **"Monotonic / never rises" is false in both canonical sweeps** and was retracted.
- **The convergence studies do not hold architecture or budget fixed across resolution** —
  the Nyquist cap varies n_freq and parameter count per cell and sampled points are
  min(8192, N²). A fixed-bandwidth, fixed-budget, multi-seed rerun is required before any
  convergence *law* is claimed. This was queued and, as far as this backfill can tell, never
  run.
- **"Data floor" is not a lower bound** — it is the error of bilinearly interpolating coarse
  FD solutions, and a parametric model can learn discretisation correction and beat it.
  Renamed "coarse-interpolation discrepancy".
- **No artifact demonstrates the coordinate decoder fixes the actual NM-ROM.** Every accuracy
  test at this point hands the decoder the true physical parameters — no encoder error, no
  inferred latent, no GN solve, no EQ error, no time-stepping stability. This is the finding
  that set the whole subsequent direction.
- **"Lossless mesh transfer"** → "interpolation-free native evaluation", and the missing
  transfer artifact was archived (`round2/transfer2d_results.json`).
- Minor but embarrassing: "f64 evaluation" was f32 inference with an f64 norm; "65,536× more
  points" was 1,024×.

The working pattern that came out of this and held for the rest of the project:
**implementation → two-reviewer adversarial gate (Codex read-only + a fresh general-purpose
agent on the same brief) → fix → cheap control → full run.**

---

## 2026-08-14

### Two new PDE classes, built to answer a reviewer

`exp/2026-08-14-burgers2d-coord-rom` and `exp/2026-08-14-wave2d-coord-rom`, built and swept
in one day. The motivation is explicit in the rejected paper's reviews: one reviewer asked for
"Navier-Stokes, Euler, Bergers, Schrodinger, or anything nonlinear" and a less well-conditioned
operator; another wrote that the evaluation is "restricted to diffusion-dominated PDEs —
precisely the setting where low-dimensional linear subspaces are already known to perform
well", so the paper "never exhibits a regime where the non-linear manifold is the enabling
ingredient". Burgers (nonlinear, advection-dominated) and Wave (hyperbolic, conservative)
extend the claim across the classical types.

Both cells are decoder-expressivity testbeds, **not ROMs**: the decoder is handed the true
5-parameter z, there is no encoder, no latent solve, no EQ. Both are seed 0 only, N ∈
{16,32,64,128,256} plus a 512² f64 reference, 512 train / 64 val trajectories, POD fitted and
evaluated on all 51 slices.

**Burgers-2D** — `u_t + u(u_x+u_y) = ν∆u`, backward Euler with full Newton and matrix-free
BiCGStab, `z = (cx,cy,w,a,log ν)`, ν ~ logU(0.01,0.1). Held-out rel-L2:

| N | POD-6 | POD-24 | grid-tied (r24) | FiLM |
|---|---|---|---|---|
| 16 | 2.64e-1 | 5.16e-2 | 7.87e-2 | 2.98e-3 |
| 64 | 2.83e-1 | 6.38e-2 | 8.61e-2 | 3.19e-3 |
| 256 | 2.91e-1 | 6.92e-2 | 8.64e-2 | 3.57e-3 |

The widest margin of the four PDEs. **The commit says "~100× vs POD-6" and the README says
"90–110×"; the JSON says 81.6–108.8×** — N=256 (81.6×) and N=16 (88.4×) both fall outside the
stated band. Convergence against the 512² reference is strictly monotone, 1.41e-1 → 8.71e-3,
a verified **16.2×**, with the net sitting *on* its data floor (slightly below it at N=16/32,
1.34× above at N=256) — no capacity plateau reached, this family is data-limited everywhere.

**Wave-2D** — `u_tt = c²∆u` as a first-order (u,v) system, Crank–Nicolson, SUBSTEPS=80,
traj-RMS metric. FiLM 2.80e-2 → 3.51e-2 across the ladder against POD-6 at 4.41e-1 → 4.52e-1,
i.e. **12.9–15.8×** (the commit's "13–16×" rounds the N=256 cell up from 12.9). Convergence
strictly monotone, 2.33e-1 → 4.37e-2 = **5.34×**, but the net gives ground back to its floor
as the mesh refines: within 9.3% at N=16 and **1.69× above floor at N=256**. POD floors here
are roughly 4× worse than heat's at matched rank — the slow Kolmogorov decay a hyperbolic
transport problem is supposed to have, confirmed.

Self-convergence is clean: Burgers observed orders 0.98–1.56 (between 1 for upwind advection
and 2 for centred diffusion, as designed), Newton ≤3 iterations with zero unconverged steps
against a fixed budget of 8. Wave orders 1.19–2.31, energy drift ≤ ~2e-11.

⚠ Two prose-vs-JSON errors worth not repeating. The Burgers README says ViT-CP "cannot even
hold POD-24's 5.2-6.9e-2 at fine N" — **the JSON says ViT-CP beats POD-24 at every N**
(ratios 0.76→0.94); the true and arguably stronger statement is that a rank-256 decoder's
margin over a rank-24 linear basis collapses from 1.49× to 1.06×. And the README cites
"POD-256" as the relevant ceiling, but `POD_RANKS` tops out at 64 — **POD-256 was never
computed**. The Wave README likewise claims ViT-CP "lands in the same band as the grid-tied
control"; the JSON has ViT-CP better than grid-tied at every N except, marginally, N=256.

### Three landmines, all found the same afternoon

**(a) POD Gram OOMs an 80 GB A100.** Fitting POD on all 51 slices (itself the heat round's
review fix) makes the snapshot matrix 512 × 51 = **26,112 rows**; the f64 Gram is 26,112² ≈
**5.45 GB**, and cuSOLVER's `eigh` workspace on top of a 13.7 GB device copy of the snapshots
blows past 80 GB at N ≥ 128. Fix: do the Gram, the eigh, the basis lift, the orthonormality
check and the reconstruction arithmetic in **host NumPy f64**. This is the local dialect of the
inherited `GRAM64=1` rule, and it holds — measured orthonormality deviations are 3.6e-14 to
1.2e-11, never the ~1e-4 that signals an f32 Gram.

**(b) BiCGStab breaks down on an already-converged residual and emits a NaN.** Newton is a
fixed-length `lax.scan` of 8 iterations and cannot exit early. Newton actually converges in
≤3, so iterations 4–8 hand BiCGStab a right-hand side at machine epsilon; the `rho` and
`omega` inner products underflow and `rho/omega` produces a **NaN step**, which is added to u
and propagates through the remaining time steps into the stored data. It is
**non-deterministic** — the identical replay came back clean. Damage: the N=128 training cell
and the first 512² reference, both regenerated. Guard: skip the update when
`‖r‖ ≤ 1e-12·‖u‖` and reject any non-finite step.

**(c) The audit that hid (b).** The per-chunk residual accumulator was
`res_max = max(res_max, float(...))`, and Python's builtin `max(x, nan)` returns **x** — so a
chunk that went NaN was silently discarded and the job printed a healthy
`max Newton rel residual 1e-16` while shipping corrupted trajectories. Fixed to a
NaN-propagating accumulate in both the trainer and the refgen, in both PDEs. **This is the
most dangerous of the three**: (a) crashes loudly and (b) is at least visible downstream, but
(c) silently converts a corrupted run into a clean-looking one.

The diagnostic logs preserve the evidence: at N=128, trajectory 421 — a completely benign
mid-box parameter with `umax=0.268` — runs a flat 1.3e-15 residual for 40 steps and then NaN
for the last 10. `BAD TRAJECTORIES: 1`. The 512-trajectory replay: `BAD TRAJECTORIES: 0`.

⚠ **A side effect worth knowing.** Post-guard the converged Newton residual floors at ~1e-12
instead of polishing to ~1e-16. That is a clean pre/post fingerprint in the logs, and it means
the Burgers sweep is **not solver-homogeneous** — N=128's training data and the 512² reference
were produced by a different code version, with a 10³× looser converged residual, than
N=16/32/64/256. Immaterial against a 3e-3 model error, but real. The N=16/32/64/256 cells ran
*before* the audit fix, so their own printed `res_max` cannot be trusted; the clean-data
argument for them is indirect (a finite POD Gram over every element of the data implies finite
data) and should be stated as inference, not audit.

---

## 2026-08-15

### Timing and the published-decoder arm

Both testbeds got an online-timing sweep and a verbatim import of the published `CPDecoder`
and `LinearCPDecoder` at rank 256, run as (z,τ)-conditioned arms under the testbed's own best
conditions.

**The resolution wall reproduces on both new PDEs.** Burgers ViT-CP 3.91e-2 → 6.50e-2 and
Wave 2.40e-1 → 3.46e-1, both monotone *worsening* as N goes 16 → 256, while FiLM stays flat.
FiLM is 13.1–19.6× better on Burgers and 8.3–10.3× on Wave. **LinearCP is identical to CP
within a few percent** on both — the linear skip changes nothing, so the disease lives in the
grid-anchored factors, not in the skip. That puts the wall on four PDE classes:
elliptic, parabolic, nonlinear-advective and hyperbolic.

⚠ Caveat that belongs with every quotation of this: the ViT-CP arm is fed the true z⊕τ at
latent dim 6, not a ViT-encoder latent at dim 64 — **no encoder is trained**. It is the
published *decoder* under the testbed's best case, not the published *pipeline*. And on Wave
the published arm is trained under the published per-sample relative loss, which up-weights
near-silent snapshots, while the FiLM arm is not — an arm asymmetry that inflates the measured
8–10× by an unquantified amount. That is the weakest link in the Wave comparison.

**Timing (A100, batch 1, median of 5 after warmup, one GPU sequentially).** Two different
comparisons, and the commit messages conflate them:

| | Burgers | Wave |
|---|---|---|
| native resolution (net on its own grid vs same-N FOM) | 5.30–13.25× | 2.15–20.70× |
| full field at 512² vs the 512² FOM | **3.93–4.30×** | **0.80–0.86×** |

The Burgers commit says "5-13x **full-field** surrogate speedup" — 5.3–13.25× is the
*native-resolution* number, where the net is not producing a 512² field at all. The genuine
full-field figure is 3.9–4.3×. The Wave 0.8× was logged honestly as a negative: the wave FOM
is *cheap* (SPD operator, fast CG), so decoding 13.4M coordinate queries through a 477k-param
MLP costs more than solving the PDE. If anything it is understated — the net is timed in f32
against an f64 FOM.

Both are **surrogate-inference** numbers with the decoder conditioned on the true z, decoding
every grid point at every slice. Neither is a ROM-solve speedup. Also: the 13–15 ms floor at
N ≤ 64 is kernel-launch latency across 51 sequential dispatches, so the small-N "speedups" are
launch-bound on both sides and should not be quoted as compute speedups. And the Wave FOM
series is non-monotone (N=128 at 490 ms is *slower* than N=256 at 426 ms), unexplained, which
makes those two speedup cells not strictly comparable.

### Multistage precision — the line that was killed (`exp/2026-08-14-multistage-precision`)

Branch named 08-14, all work on 08-15. **This is the highest-value negative in the pre-16-Aug
record and the reason nobody should re-run staged-residual training on a parametric family.**

**Hypothesis.** Wang & Lai multi-stage training (JCP 504 (2024) 112865) drives a network to
machine precision by fitting the residual of the previous stage with a frequency-scaled fresh
network. Does that precision survive into a parametric decoder, and then into a ROM? Poisson-2D
bump family, N=64 (N=128 for the single-function case), f64 throughout, seed 0.

**It works spectacularly on one function.** Four stages take the relative residual
**3.07e-4 → 3.33e-11**. With the corrected frequency units, one stage does
**1.84e-4 → 7.58e-11**, a 2.4e6× gain.

**And it dies as a power law in the number of training samples.** The control ladder holds
everything fixed but family size, feeding the decoder the true parameters:

| n_train | 1 | 4 | 16 | 64 | 256 | 512 |
|---|---|---|---|---|---|---|
| stage-1 gain | 2.42e6× | 1448× | 53.7× | 13.1× | 4.28× | 2.68× |

Fitting over n_train ≥ 16 gives **gain ≈ 542·n^−0.865, R² = 0.994** — essentially gain ∝
1/N_TRAIN, reaching 1× (staging buys literally nothing) somewhere around n ≈ 400–1400. That is
*below* any realistic ROM training set.

**Every alternative explanation was killed with a control**, which is what makes this
conclusive rather than suggestive: 2× steps at constant LR gives 30× where plain gave 54×;
300 full-batch L-BFGS iterations give 52× vs 54×; full-batch-in-samples at 512 gives 3.15× vs
2.68×; forcing n_freq ≥ 32 leaves 512/256 unchanged and makes n=64 *worse* (5.65× vs 13.1×);
Fourier features on the conditioning input do not help train and make validation **6.6× worse**.
Verdict as written in the cell: "**REPRESENTATIONAL, in the conditioning direction — not an
optimizer/budget artifact.**"

**The mechanism.** Fields are smooth in parameter space (nearest-neighbour correlation 0.95),
but the residual after stage 1 has correlation 0.375, after stage 2 0.082, after stage 3
**−0.013**. The residual becomes *uncorrelated between neighbouring parameter points*, so no
smooth function of (x,z) can represent it. Its spatial spectral centroid climbs 2.7 → 16.7
cycles/unit. **Held-out error never improves with staging at any family size** — 4.94e-3 →
4.51e-3 → 4.52e-3 at n=512, and identical to three digits below that. The gains at n ≤ 16 are
pure memorisation: those arms' held-out error is 0.51–1.18, i.e. 50–120%.

**Then the ROM killed it a second way.** Auto-decoder latents learned with stage 0, frozen,
then an LM Gauss–Newton solve on held-out sources. Three floors at K=8: manifold/train
**2.10e-3** → finite-budget inferred latent **8.38e-3** → **ROM 6.25e-2** (1 stage), **8.55e-2**
(3 stages). More staging makes the ROM *monotonically worse*, because later stages' corrections
are higher-frequency.

**The decisive observation**, holding in every nearest-init arm of every report: **the LM
solution attains a lower PDE residual than the oracle latent while having ~8× larger field
error** (‖r(z_LM)‖ = 0.82 vs ‖r(z_oracle)‖ = 1.01, field 6.25e-2 vs 7.78e-3). The FD Laplacian
multiplies grid-scale decoder error by ~(N−1)² ≈ 4000, so the minimum-residual point on the
manifold is not the minimum-error point. Re-running at 5× the LM budget changed the headline
arms by under 1%. Verdict: "**The ROM floor is the OBJECTIVE, not the solver and not the
decoder's data precision.**"

**This is the single most consequential result in the pre-16-Aug record.** Conclusion 4 of the
cell names the untried remedy — a residual metric that does not amplify grid-scale decoder
error, i.e. an H⁻¹/energy-norm or Galerkin projection onto smooth test functions — and that
proposal is exactly what becomes `exp/2026-08-16-poisson2d-rom-objective` the next day and the
"minimise the weak form, not the pointwise residual" rule at the top of this log. The 8× gap
quoted in the 16 Aug entry is *this* 8×, first measured here.

**Corrections applied at a two-reviewer adversarial gate** (Codex + a fresh agent on the same
brief), commit `77aadb9`. The substantive ones:

- The **frequency probe was biased to the Nyquist ring** — it summed amplitude per annulus, and
  annulus area grows with radius, so white noise returned the last bin. Replaced by a radial
  *mean*, with a unit test on known sinusoids and a white-noise regression check.
- **Fourier units were 2× off.** Every pre-fix number, including the 3.33e-11 headline, is in
  the old units; the corrected single-function number is the c1 arm's 7.58e-11.
- The **ROM residual was not the FOM operator** — it concatenated a weighted boundary-penalty
  block, so no decoder could zero it. Replaced by a ghost-zero Dirichlet residual with wall
  neighbours zeroed in the stencil.
- **Unweighted MSE fitted only the loud samples** (snapshot energies span >100×) → inverse-energy
  weights and dual metrics.
- **The "oracle floor" was not a floor** — relabelled a finite-budget inferred-latent error,
  explicitly "not a lower bound", computed with the same solver, starts and budget as the ROM.
- `ms_rom_solve.py` deleted; held-out-derived latents quarantined into `*_TAINTED.pkl` so they
  can never reach a deployable checkpoint.

⚠ Three README claims in this cell do not survive its own JSON: "the 300-attempt rerun
converged 16/16 in every arm" (only 7/12 and 8/12 arms did; the rest still hit the budget);
"changed no number by more than a few %" (true for the headline arms, but a mean-init m512 arm
moves −32.6%); and "both metrics agree to ~10% everywhere" (train yes, but at n=256 the two
validation metrics are 69% apart). The *conclusions* survive on the arms that carry them; the
blanket statements do not.

⚠ Provenance gap: the README claims Tufts A100 job directories, but **no SLURM job id, GPU
model or node name appears in any log or JSON in this cell**. The only corroboration is a ~10×
wall-clock gap consistent with GB10-vs-A100. Treat the A100 attribution as unverified prose.
The whole cell cost minutes of GPU time, not hours.

**Status: abandoned.** The branch is contained in no other branch and was never merged.

---

## 2026-08-16 (before the objective-fix wave)

### The generalizable cascade — stopped at its own review gate

`exp/2026-08-16-cascade-nmrom`, one commit, 2,795 lines. The README's first line is literal:
*"Status: IMPLEMENTATION READY FOR REVIEW. No full run has been launched."* There is no
`runs/` directory, only `smoke/`, and a fully written "Planned runs (NOT launched)" section.

The design tried to rescue the multistage line by fixing its two diagnosed failures at once: an
**encoder over the PDE input** on a fixed 16×16 lattice, frozen after stage 0 (so later stages
have a fixed target and the ROM has a legitimate online-known conditioning vector), plus f64
frequency-scaled residual stages, a **probe gate** that stops stacking when the residual's
effective rank or its nearest-neighbour correlation in z says another stage cannot help, and
two levers for extra conditioning and encoder smoothness. Behind it, an LM NM-ROM with three
Poisson objectives — `fd` (plain residual), `hinv` (residual preconditioned by the exact
discrete inverse Laplacian) and `hinvK` (K CG iterations) — and Burgers latent time stepping.

The `hinv` objective is precisely the multistage cell's Conclusion 4 implemented, and the
README is admirably explicit that it is a **diagnostic, not a cheaper ROM**: for a linear
operator `L⁻¹(Lu(z) − f) = u(z) − u_FOM`, so the exact-`hinv` objective *is* the field misfit,
at FOM-solve cost.

The smoke runs are labelled meaningless by design (N=16, 8–32 training samples, 100–200 steps)
and they are, but two things in them are informative. The predicted `hinv` ≡ inferred-latent
identity reproduces to three digits (2.396e-1 vs 2.392e-1), which validates the implementation.
And **the gate fires immediately in every single arm** — nearest-neighbour correlation collapses
from 0.67–0.71 on the fields to 0.04–0.13 on the residual after stage 0, below the 0.2
threshold. That is the multistage roughness finding reproducing on first contact: the design's
own diagnostic telling it the cascade will not stack.

**It was superseded within hours by its own two halves.** The two sibling worktrees created the
same day — `exp/2026-08-16-poisson2d-rom-objective` and
`exp/2026-08-16-burgers2d-rom-latent-stepping`, the Wave 1 cells at the top of this log — are
exactly cascade-nmrom's Poisson-objective study and Burgers latent-stepping plan, pursued
*without* the cascade superstructure, and they carry six-plus commits each through full sweeps
and review passes. The objective fix was the part that mattered; the staging was not.

⚠ That reading is inferred from branch topology and dates. **Nothing in the tree records the
decision to stop**, which is the gap this log exists to prevent.

---

## 2026-08-16

### Wave 1 — the objective fix, Poisson and Burgers

Two agents, one worktree each: `exp/2026-08-16-poisson2d-rom-objective` (namespace `pobj/`) and
`exp/2026-08-16-burgers2d-rom-latent-stepping` (`blat/`).

**The problem.** The latent solve stalled about 8× above the decoder's own accuracy ceiling.
Traced to the objective: minimising the pointwise FD residual amplifies grid-scale decoder error
by ~N², and the minimiser it finds has a *lower* residual than the true latent while producing
8× worse fields.

**The fix.** Minimise the residual projected onto M low smooth sine test modes, `Λ⁻¹`-weighted,
evaluated at NNLS empirical-quadrature points fitted on decoder-output snapshots. Reaches the
ceiling and is insensitive to the initial guess. Verified by giving it 5× the budget and seeing
nothing change.

Established: `M > k` comfortably or the objective collapses; `m ≈ 4M` is the knee; meshfree
candidate pools work as well as grid nodes; hyper-reduce the cold start or the online path stays
grid-bound.

### Wave 2 — Heat and Wave ports

`exp/2026-08-16-heat2d-rom-latent-stepping` (`hlat/`), `exp/2026-08-16-wave2d-rom-latent-stepping`
(`wlat/`).

Held-out error at k=8, N=64 — ceiling / ours / POD: Poisson 7.11e-3 / 7.65e-3 / 1.77e-1;
Heat 1.16e-2 / 1.87e-2 / 1.29e-1; Burgers 1.15e-2 / 1.65e-2 / 2.09e-1;
Wave 1.72e-1 / 8.78e-1 / 3.42e-1.

**Wave fails structurally.** On a fixed subspace the Newmark/CN recurrence is time-reversible
and therefore conservative; on a nonlinear manifold that symmetry is gone. End-time energy ratio
0.27 against POD's 1.000003. Refining the ROM time step 5× lowers the time-discretisation floor
28.9× and *raises* error 7.5%. Not fixable by tuning — the open problem is structure-preserving
latent stepping.

**Heat is accurate but does not pay**: a direct reduced POD-Galerkin solve is 13–38× faster than
the FOM, so nothing iterative competes on a linear parabolic problem.

---

## 2026-08-17

### Consolidation

Merged the four completed cells into `exp/2026-08-17-inr-rom-consolidated` with the reports
pipeline. Pruned `__pycache__`, smoke artefacts, and staged code copies.

**Retracted later:** this tree pruned the trained decoders, which is how they ended up tracked
nowhere. See 2026-08-18.

### Two experiments launched

`exp/2026-08-17-cost-to-tolerance` (`ctol/`) and `exp/2026-08-17-rom-warmstart-fom` (`wsfom/`),
one agent each.

**Incident — `scancel` killed both fleets.** A `scancel --name=a,b,c` matched nothing, degraded
to an empty selector, and cancelled every job on the shared account. ~16 min of compute lost, no
corruption. `cluster/cancel.sh` now refuses names, globs and user selectors.

**Correction to the record — every published speedup was inflated.** Both classical baselines did
more work than their accuracy required. Burgers ran a fixed 8 Newton iterations per step (400
Newton steps and 400 BiCGStab solves per rollout, ~4× over-converged); Poisson ran CG at
`tol=1e-13`, the tolerance used to *manufacture the truth data*.

- Burgers ladder 0.72 / 1.57 / 4.46 / 7.96× becomes **0.19 / 0.36 / 0.93 / 1.83×**. N=128 moves
  from clearly winning to break-even.
- Poisson depends on the assumed deployment tolerance and **the tolerance must always be named**:
  ~1.16× at 1e-10, ~1.31× at 1e-8, **~1.56× at 1e-6**. At 1e-6 the archived N=256 figure changes
  *sign*, 1.40× → 0.91×.
- Poisson's factor is near-constant in N so one scalar is defensible per tolerance; Burgers'
  varies 28% across meshes and must be applied per mesh or not at all.

**Retracted mid-session:** ~1.16× was first published as "the" Poisson factor. It is the tightest
of three columns and the one most flattering to the archived numbers. Erratum issued.

**The warm-start hybrid does not pay.** No Poisson crossover in any of 75 configurations, best
0.933×; Burgers 0.15–0.49×, losing 12/12 on wall clock. Linear extrapolation from the previous
two time steps beats it and costs nothing. Different mechanisms: on Poisson our guess is poor in
the norm CG actually contracts (4.97e-2 against a 9.27e-3 field error); on Burgers the guess is
good but costs 273 ms against a 48–228 ms solve.

**A crossover that did not exist.** The pre-fix panel showed 1.02× at N=512. It was a 17%
device-clock-ramp bias after a long host-bound NNLS fit. Burn in before every timed block.

**Bugs caught by the two Codex passes.** A missing BiCGStab alpha half-step test that *discarded
converged iterates*; `N_POD_TRAJ=128` pinned in the batch environment, which would have silently
reinstated a POD handicap the audit had just removed; invalid commit provenance (`git -C` walking
into an unrelated ancestor repo); unpersisted timing repetitions, which forced withdrawing a
"best configuration" claim rather than defending it.

**My grid-spec error.** I specified `M=256` with `m=256` for k≥32, landing on the `m = M` corner
and violating the project's own `m ≈ 4M` rule. Quadrature 660× worse, worst-row error 8.5e+05.
The entire k=32 column was an artefact of our own settings until fixed.

---

## 2026-08-18

### The k-spike retraction

A dedicated investigation (`experiments/k-stall-diagnosis/`) overturned the "our solver stalls at
particular latent dimensions" finding.

**It was a mean-over-16-cases artefact** dominated by 1–5 diverging solves. The median sits within
0.96–1.24× of the ceiling at *every* k, and k=32's median is the best on the ladder. Running all
64 held-out cases shows k=8 and k=16 failing too — the 16-case panel simply drew none of them.
The powers-of-two pattern was coincidence read off eight noisy points.

**Root cause: the latent LM has no globalisation.** `lam0=1e-6` gives an essentially undamped
Gauss–Newton first step, 10–350× the norm of the latent itself, and any residual decrease is
accepted. The iterate leaves the region the decoder was trained on and converges to a spurious
stationary point. Discriminator, exact on 9/9 traced cases: iterate norm after the first accepted
step, relative to the training-cloud radius — 0.77–1.32× succeeds, 1.73–4.60× fails.

**Fix**: trust region at the training-cloud radius. Ratios across the ladder go from
1.11/7.61/1.10/7.85/1.26/3.40/15.49 to **1.09/1.05/1.09/1.18/1.28/1.45/1.50**, nothing regresses,
iterations fall at large k. Not yet measured through the EQ + timing path.

**The same optimiser is used on every PDE**, so the project's accuracy numbers are likely broadly
pessimistic by an unquantified amount.

### Consolidation for Codex

Created `exp/2026-08-18-codex-handoff` with all seven cells, the reports, `AGENTS.md` and
`CODEX-START-HERE.md`.

**Two things were nearly lost, and are why the lab-log rule now exists:**

1. The k-stall investigation existed only in a `/tmp` scratchpad — session-scoped, and the most
   actionable result of the two days.
2. The trained decoders were `.gitignore`d and tracked on no branch. The Burgers k-ladder existed
   in git nowhere at all. A clone could be read but not rerun. All 16 are now tracked.

### Open

- **`cost-to-tolerance` is unfinished.** The single-GPU consolidation run never happened, so every
  ROM-vs-FOM wall-clock ratio pairs an A100-80GB against an A100-40GB — a 3.7× apparent speed-up
  from hardware alone. §7 Verdict and §8 Caveats are placeholders; no results audit. **Until this
  lands no speed number in the tree is publishable.**
- The Burgers correction denominator is formally open — the two cells ladder different knobs
  (a Newton tolerance vs the testbed's fixed `NEWTON_ITERS`). A pre-registered cross-check with a
  stated sign convention is set up.
- Apply the trust-region fix upstream and re-measure all four PDEs.
- Decoding is now 84% of online cost at N=512 while the latent solve is 9%.
- On 2-D Poisson a direct sparse solve is 494× faster than the iterative baseline. Say it before
  a reviewer does.
