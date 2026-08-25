# Lab log — Tunable NM-ROM

**This is the single canonical log for the project.** It lives on `main` at the repository root
and every session appends to *this* file by absolute path, whichever worktree it is working in:

```
/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/LAB-LOG.md
```

There is no per-worktree copy. "Where things stand" is rewritten by each session; the chronology
below it is append-only, oldest first.

---

# Where things stand — 2026-08-25

*(The 2026-08-22 block this replaces described the FiLM-decoder era; it is in git history,
commit 4c2cf0d and earlier.)*

## Read this first

**Consolidated line of descent:** worktree `worktrees/2026-08-25-sepdec-consolidated`, branch
`exp/2026-08-25-sepdec-consolidated`, with `START-HERE.md` at its root. It merges
`exp/2026-08-22-separable-decoder` → `exp/2026-08-23-n256-push` → `exp/2026-08-25-burgers-accuracy`.
Those three, and the scaling arms `exp/2026-08-23-sepdec-n{128,256,512,1024}`, are read-only
archives. **New work branches from the consolidated branch, never from `main`.**

## The method (separable EQ-decoder)

`u(x;z) = bc(x)·⟨g(x), h(z)⟩`: a 512-feature spatial track `g` (random-Fourier lift → SiLU
MLP, last layer linear, so `G_HIDDEN ≥ 2R` or the bank is rank-capped at `G_HIDDEN+1`) and a
K-dim latent track `h` (SiLU MLP + linear skip). `g` never sees `z`, so it is evaluated once
at the m≈256 NNLS quadrature points into a cached table and the online trust-region LM solve
is `table @ h(z)` — the grid never enters the loop. Weak form on M=4K sine test modes;
incumbent discretization reproduced bit-for-bit (gate 0 ≤1e-12, measured 0–3e-15). Pure
neural; SVD appears only as a diagnostic. Auto-decoder training (codes are free variables),
point-subsampled AdamW+EMA, banks/data as explicit jit arguments (a ~2 GB closed-over array
costs +10 GB RSS and 16 s compile per jit).

## What is true right now

**Speed — the architecture's prediction is confirmed.** ROM online cost is flat in N;
classical cost grows. K=16/R=512, optimized solver (Gram-space IC 18→1.9 ms; adaptive stall
tolerance), matched-accuracy paired AB/BA vs a swept Helmholtz-preconditioned Newton ladder
(`CROSS-RESOLUTION.md`, generated): N=128 0.38×, N=256 0.39×, N=512 0.60×, **N=1024 1.61×
(27.1 vs 42.9 ms)**; batch-16 upper bounds 0.38× / 0.91× / 2.37× / 11.74×. Profile: the solve
is kernel-dispatch bound (`PROFILE.md`), not bandwidth or compute bound.

**Accuracy — the binding rung is `h`'s generalisation.** Solver = weak-EQ optimum ≈ oracle
with no error compounding; the span floor (2.6e-4 at N=1024) is far below target; `h` reaches
~1/25–1/35 of it, resolution-independently. Best rollout errors: K=16 dense_mid 8.96e-3 at an
unchanged classical ratio (round 5, N=256); **K=32 h512x3 at N=1024: 5.14e-3 (M=256 EQ) /
6.85e-3 (M=64)** — but timed only on the un-optimized solve path (98.9 ms vs matched classical
62.9 ms = 0.66×). 1e-3 is NOT reached; it needs an oracle ≈5e-4, ~5× below the best measured.

**Retracted (do not quote):** the N=64 "3.3×" (IC fit excluded, over-solved baseline); the
N=1024 "5.2×" (compared against a rung 24× more accurate — `lin_tol` was carrying it; honest
figure 1.61×); "K is nearly free online" (K=48 = 3.81× wall time); "h is capacity-limited"
(it is generalisation-limited); "codes may not have converged" (1.000×); "more span helps"
(R=1024 widened the gap to 70.6×); multi-scale Fourier features (worse than single-scale).

**Quadrature — measured for the first time (EQ fidelity ladder, 2026-08-25).** The sampled
weak residual differs from the exact full-grid one by tens of percent on the solver path
(control set) and by of order one at the solution; the Jacobian is sampled well (Hessian rung
~1e-4–1e-2), so the error enters through `R` and corrupts the gradient (cosines ~0.3–0.4
control, ~0.6–0.8 fine) and the step. The larger share is in the LINEAR terms (`Φᵀu`), which
the separable form can compute exactly with no quadrature. The NNLS "rel fit" (5e-3 / 5e-4) is
not a certification of anything the solver uses. Resolution-independent (N=256 ≈ N=1024).

**Soft:** the Poisson 28× at N=1024 is against an **unpreconditioned** `jax.scipy` CG; a
preconditioned arm has not been run. An exact spectral solve does 0.64 ms at 7e-15 on this
constant-coefficient Poisson. Burgers is the real target.

## The next experiment

**Exact linear terms in the weak residual** (report `2026-08-25-eq-fidelity-ladder.md`, §3
item 1): replace `Φ_qᵀ(u−uⁿ)` and the Laplacian term by `(ΦᵀG_int)(h(z)−h(zⁿ))` — one
precomputed M×R matrix, no quadrature, zero online cost change — then refit the m nodes for
the advection term only, and re-run the ladder to confirm rungs (b)/(c1) drop. Needs gate 0
redefined against the full-grid weak residual (log it as a rule change). After that, the
round-4 speed sweep on the K=32 N=1024 checkpoint (`runs/push_r4a6/out/…K32_R512_h512x3.pkl`,
`cluster/run_r4s1024.sbatch` with `CKPT` swapped) and pulling `dn1024` (2837430) /
`dn256b` (2837431) in `burgacc/`.

## Open

Exact linear terms + advection-only EQ refit (the ladder's #1 fix) · K=32 optimized timing ·
dn1024/dn256b · preconditioned Poisson CG arm · certify EQ sets with the ladder
(`sep_eq_ladder.py`), not the NNLS rel-fit or row tail · `h` generalisation (capacity +
μ-density together; test-time residual refinement of `h` untried) · merge decisions for the
consolidated branch and for `exp/2026-08-25-eq-fidelity-ladder`.

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

---

## 2026-08-19

### Hybrid NM-ROM → FOM progress review

Read-only review of the final hybrid cell on `exp/2026-08-18-codex-handoff`; no new GPU jobs or
numerical experiments were run. Re-ran `experiments/rom-warmstart-fom/wsf_verify.py` with the
repository venv: **146 checks passed, 0 failed**. The recorded conclusion is unchanged: the
decoded NM-ROM field is handed to the same FOM solver and both arms reach `fom_tau=1e-6`, but the
hybrid is slower at every measured mesh. Poisson reaches at best 0.933× across 75 configurations;
Burgers reaches 0.15–0.49× and loses 12/12 wall-clock comparisons. The core result remains final
and audited. Scope remains Poisson-2D and Burgers-2D only; applying the trust-region latent-solver
fix through the production EQ/timing path and testing whether operator-aware training improves
the Poisson A-norm warm start remain open follow-ups.

Follow-up feasibility analysis from the consolidated `hybrid_points.json`: the hybrid was not
architecture-optimised for warm-start utility. It froze the standalone-ROM K=8 checkpoints and
varied ROM/FOM stopping tolerances; the upstream cells had swept K and M/m, but not decoder shape
or an FOM-convergence-aware training loss. At Poisson N=512 and `fom_tau=1e-6`, the best row spends
7.434 ms on preprocessing + ROM + decode and saves 3.007 ms in the FOM stage, so preserving the
same solver-stage saving while reducing the warm-start cost by 2.47× would break even. The
corresponding requirement is 3.02× at N=256 and grows rapidly on coarser meshes. This makes a
targeted Poisson optimisation push plausible against the current unpreconditioned iterative
baseline, though not against the measured direct solver. Burgers is much less promising at the
fixed `dt=0.005`: at N=256, `fom_tau=1e-6`, warm-start construction costs 479.57 ms and saves only
15.15 ms of FOM work, requiring 31.66× cheaper construction at unchanged guess quality. Any next
round should pre-register total-cost gates, target the norm/work the FOM contracts rather than
field L2 alone, and compare Burgers against linear extrapolation.

### Autonomous hybrid optimisation through N=1024 (in progress)

User approved two isolated tracks from `exp/2026-08-18-codex-handoff`:
`exp/2026-08-19-poisson-hybrid-1024` in `worktrees/2026-08-19-poisson-hybrid-1024`
with cluster namespace `hybp1024/`, and `exp/2026-08-19-burgers-hybrid-1024` in
`worktrees/2026-08-19-burgers-hybrid-1024` with namespace `hybb1024/`. Results below are
feasibility diagnostics, not final claims; the required single-GPU N=32…1024 ladders have not
run yet.

**Poisson early gates.** A train-only RBF map from the known four source parameters to the frozen
K=8 auto-decoder latents failed (calibration standardised latent MSE 0.838); its N=64 decoded
guess was worse than zero in both L2 and A-norm, so this path was abandoned rather than tuned on
test cases. The tracked parameter-aligned decoder is far better (local smoke: 3.32e-3 L2,
2.55e-2 A-norm ratio), but construction did not pay at N=64. An exact 8×8 low-sine-mode
classical control dominated every learned arm locally; cluster timing and larger N remain open.
This reinforces that source-aligned training and solver-norm quality matter, while also setting a
strong non-learned control the NM-ROM must beat.

**Burgers oracle gate.** Job 2653184, H100 PCIe, f64/highest, `jax_backend=gpu`, checksums matched,
zero health failures, remote directory deleted after pull. On its recorded fresh four-case seed-1
cohort at N=256 and outer tolerance 1e-6, linear extrapolation has mean guess L2 2.72e-3. Removing
75% / 90% / 99% of that error with a nondeployable oracle reduced mean BiCGStab work from 2311 to
2110 / 2042 / 2014, but the trajectory-0 paired-median FOM budgets were only about 1.7 / 3.4 /
2.0 ms and were noisy/nonmonotone; only the exact next state caused a discontinuous zero-Newton
finish. Therefore a practical predictor must remove roughly 75–90% of extrapolation error and cost
only a few milliseconds per entire 50-step rollout. This gate is preliminary: drawing `m=4` from
`sample_params` is not the prefix of the canonical `m=16` draw because parameter arrays are sampled
sequentially. Future runs lock the full draw before selecting cases. The final baseline will also
calibrate the inherited inner BiCGStab tolerance instead of retaining the over-solved 1e-10 value,
and final timing will cover all test trajectories rather than timing trajectory 0 while averaging
work over four.

### Autonomous hybrid optimisation through N=1024 — final

The two approved isolated tracks are complete and clean but **not merged**:
`exp/2026-08-19-poisson-hybrid-1024` at `0ffc0d8` and
`exp/2026-08-19-burgers-hybrid-1024` at `1752d9e`. All real measurements ran on the Tufts `gpu`
partition with `jax_backend=gpu`, f64/x64, and `JAX_DEFAULT_MATMUL_PRECISION=highest`. Each final
job had an isolated paralab directory, regenerated its cohort from the recorded seed, persisted
raw repetitions and same-invocation cost/accuracy/work, passed checksum/source-hash audits, and
had its explicit remote directory deleted after pull.

**Poisson architecture and solver gates.** The frozen K=8 latent chart is not predictable from
source parameters by a simple train-only RBF (best standardised latent MSE 0.838), so that path
was stopped. Fixed-N=64 decode and trust-region LM reduced construction to about 4.25 ms at the
finest mesh. A cached nonlinear K=16 GroupFiLM remained neutral/losing as a pure warm start.
GroupFiLM plus q8 cut CG work, but the matched spectral-only arm was tens of milliseconds faster;
learning was not responsible for the apparent combined win. A proposed compact transported tail
was hard-stopped when the classical sine-mode ladder reached sub-millisecond total cost.

The first six-mesh Poisson panel, job 2662802 on an A100-80GB, was numerically healthy but its
15-arm cyclic timing order was not balanced: six cases times seven repetitions covered only
offsets 0–11. Its claim of 1.04–1.08× K8 speedup at N=1024 is **retracted**. The corrective job
2664551 used fresh seed 20260820, eight cases, twelve repetitions, and exact burn-AB/reburn-BA
balance (48 first and 48 second positions per method). At N=512, K8 versus zero-start counting CG
is 1.004× [0.984,1.058] at 1e-6, 0.998× [0.973,1.018] at 1e-8, and a supported loss 0.965×
[0.941,0.981] at 1e-10. At N=1024 it is a supported 1.026× [1.014,1.077] at 1e-6, an
inconclusive 1.014× [0.993,1.031] at 1e-8, and an inconclusive 0.994× [0.980,1.010] at 1e-10.
The loose row is the **only** defensible genuine NM-ROM→FOM crossover.

This learned crossover is not the production winner. In job 2662802 at N=1024, the locked
spectral control takes 0.759/0.814/0.931 ms at FOM tolerances 1e-6/1e-8/1e-10 versus
228.212/269.776/302.598 ms for zero-start counting CG: 300.9/331.3/325.2×. FFT-DST is an eligible
exact control. Dense DST is faster at loose tolerances but its measured N=1024 residual fails the
1e-10 gate; it remains correctly labelled as a separate direct baseline. Native library CG
misses the tight true-residual gate at N=512/1024, so the reviewed counting solver is
authoritative and the solver sensitivity is disclosed.

**Burgers architecture, solver, and history gates.** The inherited fixed inner tolerance was
over-solving. A same-job calibration locked exact Dirichlet Helmholtz preconditioning with inner
tolerances 1e-2/1e-4/1e-5 for outer tolerances 1e-6/1e-8/1e-10. Cubic live-history prediction
beat linear, higher-degree blends did not improve it, and IMEX Euler/AB2 lost. A corrected dynamic
oracle showed that removing 75–90% of extrapolation error could buy only a few to about fourteen
milliseconds at N=256. Fixed correction POD was not compact (rank-32 validation remaining ratio
0.528), and even translation/width alignment plus a deployable predictor left 0.620 globally;
those learned-correction paths were stopped.

The retained genuine control is the audited K=8 weak FiLM NM-ROM with M=64, m=256, exact upwind,
two Jacobians per step, fixed-N=64 decode/prolongation, an exact full-FOM residual guard against
the live cubic guess, and every candidate/guard/fallback/FOM cost charged. Cold-start features now
use a fixed coarse endpoint sample rather than scanning the target grid. The provisional seed-1
panel is superseded because its uncertainty treated timing repetitions as IID and its cohort had
informed early controls.

Final Burgers job 2664725 ran on an A100-40GB with untouched seed 20260819, canonical draw 16
then indices 0–3, seven repetitions per trajectory, and trajectory-clustered bootstrap. It has 54
rows and 1,512 timed records; all are finite, with zero flags/breakdowns, returned residual/tau at
most 0.997803, references at most 9.999924e-13, and 18/18 counting/JAX checks passing. Cubic beats
linear in all 18 mesh/tolerance conditions; guarded FiLM loses to cubic in all 18. At N=1024,
linear/cubic/FiLM times are 587.983/390.409/535.287 ms at 1e-6,
963.710/586.169/728.815 ms at 1e-8, and 1204.624/851.428/1016.865 ms at 1e-10. FiLM accepts a
median four of fifty learned candidates there and cannot repay roughly 149–169 ms of overhead
versus cubic.

**Artifacts and remaining decision.** The cross-PDE report is generated by
`reports/build_2026_08_19_hybrid_warm_starts_through_1024.py` into
`reports/2026-08-19-hybrid-warm-starts-through-1024.md`. Source artifacts are the checksummed
`runs/pairfinal1/` and `runs/final1/` directories in the Poisson worktree and `runs/confirm2/`
plus the correction gates in the Burgers worktree. The only open session action is the user's
decision whether to merge the two experiment branches; no merge was performed implicitly.

---

## 2026-08-19

### Pure nonlinear coordinate-decoder architecture search — final

Completed the autonomous standalone NMROM search in
`worktrees/2026-08-19-nonlinear-decoder-architecture` on
`exp/2026-08-19-nonlinear-decoder-architecture`, branched from
`exp/2026-08-18-codex-handoff`. Real runs used the isolated Tufts namespace
`/cluster/tufts/paralab/tawal01/nmrom_nonlinear_decoder/`; it is empty after checksummed pulls,
and no jobs remain queued. The final branch commit is `9c330bd`.

Every candidate is a pure nonlinear coordinate decoder, not POD plus a nonlinear corrector.
Selection required all three seeds—not only the mean—to pass. Poisson gates were decoder/full/EQ
≤6e-3 and EQ/full ≤1.05; Burgers gates were decoder ≤8e-3, full/EQ ≤1e-2, and EQ/full ≤1.05.
The selector then minimized quadrature points and parameter count. Fully converged objective
selection and practical-tolerance end-to-end deployment were kept separate.

**Poisson selection.** Group-FiLM H98×4/g2, 58,419 parameters, removes 51.6% of the saved
H128×4 FiLM's parameters. At M=112,m=448 its three-seed decoder/full/EQ means are
4.808e-3 / 5.232e-3 / 5.411e-3, with every seed passing. The next cheaper M=108,m=432 arm fails:
worst-seed EQ is 6.073e-3 and maximum EQ/full is 1.080. The saved control remains more accurate
in a fair M=128,m=512 comparison, so the claim is a size/speed tradeoff, not better accuracy.
On the A100-PCIE-40GB same-job deployment measurement, M=128,m=512,tau=1e-2 gives compact
4.568 ms / 9.924e-3 error versus saved architecture 4.387 ms / 9.741e-3 and iso-accuracy FOM
5.013 ms. Both learned arms are uncensored, each retains 144 raw timings, and both have zero
timing outliers. The compact decoder is therefore 1.041× slower than the saved architecture but
1.098× faster than the FOM. The cheaper M=112,m=448 point did not produce an uncensored
1%-accurate stopping point in the measured tolerance bracket.

**Burgers selection.** Group-FiLM H160×4/g2, 140,449 parameters, removes 69.7% of the saved
H256×5 FiLM's parameters. At M=128,m=592 its three-seed decoder/full/EQ means are
7.446e-3 / 9.580e-3 / 9.860e-3; every seed passes. The closest cheaper m=576 arm fails only the
robust degradation gate, with maximum EQ/full 1.057. On the H200 kernel benchmark, the compact
Jacobian is 2.061× faster raw and 2.101× faster with coordinate caching; all timing outlier
counts are zero. On the A100-80GB end-to-end job at tau=5e-2, it is 1.633× faster than the saved
architecture (336.159 versus 548.938 ms), but error is 1.791e-2, censoring is 45.0%, and it is
3.054× slower than the iso-accuracy FOM. At tau=2e-2 it is 2.197× faster than the saved
architecture but still has 1.267e-2 error and 100% censoring. Thus no selected Burgers row is
both uncensored and 1%-accurate, and no FOM crossover is claimed.

**Closed architecture brackets.** Residual-FiLM failed both PDE targets. Burgers H160/g8 fails
on seed 0; H160/g4 and H159/g3 are seed-unstable; H144/g2 fails on seed 0. H192/g2 did not
improve the decoder or M64 ROM enough to justify width. The final H176/g2 run (job 2653521,
L40S) improved the decoder floor to 7.504e-3 but worsened actual M64 full/EQ256/EQ512 errors to
1.128e-2 / 1.267e-2 / 1.207e-2, so no extra seeds were warranted. Better reconstruction alone
did not select a better ROM.

**Audit and retractions.** The independent audit passes across 73 pulled cells: checksums,
staged launch provenance, GPU backend, f64/highest, completion markers, and accepted raw timing
arrays/medians all agree. Five cells remain visible but excluded: `nda_pbench_g98_r8` used the
wrong post-fit burn/warm order; `nda_pe2e_g98_r11`, `nda_be2e_g160_r14`, and
`nda_be2e_g160m640_r21` did not retain every raw timing repetition; and
`nda_be2e_g160_r12_failed` aborted after losing compact-decoder metadata. Corrected runs replace
all timing claims. The generated final report is
`reports/2026-08-19-nonlinear-decoder-architecture.md`; tables, prose numbers, and figures are
regenerated from the run JSONs, and the SVG output is deterministic.

**Open.** This is final only for N=64,k=16 and the recorded held-out families. Resolution scaling
of the selected standalone models remains unmeasured. On Burgers, further H/group-size tuning is
not justified by this bracket; progress now requires an online objective/solver/stopping change.
The branch is deliberately unmerged pending the user's decision.

---

## 2026-08-19

### Heat-AmgX preregistration stopped before execution; next target narrowed to Burgers only

Created `exp/2026-08-19-heat-amgx-nmrom` in
`worktrees/2026-08-19-heat-amgx-nmrom` from the final pure-nonlinear architecture commit
`9c330bd`. The isolated intake found no Tufts PETSc, AmgX, or hypre module/package, so the
preregistered route was an isolated AmgX source build under the assigned paralab namespace.
That finite search plan is committed at `f76ff22`.

The user then explicitly narrowed the next experiment to **Burgers first and Burgers only**.
The Heat agent was interrupted before implementation or submission. No Heat GPU job was ever
submitted, `/cluster/tufts/paralab/tawal01/heat_amgx_nmrom/` was never created, and the Heat
worktree is clean. There are therefore no Heat numerical results, timing claims, or cluster
artifacts to retain or retract. The stopped preregistration branch remains unmerged. A fresh
Burgers-only goal/worktree still requires the durable Heat goal to be cleared and the proposed
Burgers base/worktree/namespace to be confirmed.

---

## 2026-08-20

### Burgers 1e-3 / 10x — transported-Hermite Phase 1 final negative

Executed the preregistered Burgers-only Phase 1 in
`worktrees/2026-08-19-burgers-1e3-10x` on
`exp/2026-08-19-burgers-1e3-10x`, based on audited Burgers hybrid/FOM commit
`1752d9e`. The Phase-1 closure is commit `3d3fe09`. Real jobs used only
`/cluster/tufts/paralab/tawal01/burgers_nmrom_1e3_10x/`; D0 and FOM artifacts
were checksum-verified before remote deletion, and the assigned namespace is empty. No merge was
performed.

**Representation hard stop.** D0 job 2667377 ran on an H200 with GPU backend, f64/highest, empty
stderr, and fixed seed-0 train/selection indices 0:512 / 512:576. HG4's aligned representation
oracle mean/worst are 3.657e-2 / 7.632e-2; HG5's are 2.103e-2 / 5.267e-2. Both miss the fixed
2e-4 / 7e-4 gate by orders of magnitude. The simple ridge predictor is worse: mean/worst
1.048e-1 / 4.441e-1 for HG4 and 1.220e-1 / 5.731e-1 for HG5. The authoritative nearest-wall
squared representation-error fractions are 0.374408 and 0.380216, below the 0.5 conditional
HG5S license. Therefore no failed concept was trained, D1 was not run, and weak/EQ, scaling,
all-seed, model-validation, and untouched-confirmation gates were not opened. The active floor is
representation, not latent prediction or weak optimization.

**Final FOM denominator.** Corrected job 2667536 ran on an A100-PCIE-40GB and used an audited
cubic-history/exact-Helmholtz reference at outer/inner 1e-12/1e-7, checked against an independent
3e-13/3e-8 chain. The maximum cross-chain difference is 5.234e-13 and all chains are finite with
zero flags/breakdowns. The fastest eligible calibration is outer 3e-3, inner 1e-1 at every N.
At N=256/512/1024 its mean errors are 9.348e-4 / 7.003e-4 / 6.584e-4, worst errors
1.374e-3 / 1.382e-3 / 1.400e-3, and 50-step medians 26.599 / 66.450 / 238.049 ms. N=1024's 10x
point budget is 23.805 ms and its 8x threshold is 29.756 ms. No ineligible decoder was timed and
promoted against those budgets.

**Exclusions and artifacts.** Job 2667476 is retained but excluded: it produced provisional
N=256/512 rows, then failed before N=1024 because its driver used the legacy fixed-eight-Newton
training-data rollout as truth. Jobs 2667361, 2667374, and 2667531 had zero scientific output.
Local smokes are execution-only. The generated report is
`reports/2026-08-20-burgers-1e3-10x.md`; its machine tables, figures, and passing rerunnable audit
are generated from run JSONs. A direct-predictor/one-bounded-weak/EQ driver was implemented and
smoked but deliberately not submitted after the representation hard stop.

**Open Phase 2.** The overall Burgers objective remains active. The next proposal is a finite
transported compact-support cubic B-spline hyperdecoder bracket that separates the spline
control-grid size from a small online latent dimension and must pass free-spline representation,
small-k coefficient-manifold, and N=1024 cost oracles before training. It preserves every
development/model-validation/confirmation lock and remains pure nonlinear; its preregistration is
not yet committed pending audit of the finite bracket.

### Burgers 1e-3 / 10x — adaptive spline Phase 2 preregistered, S0 ready

Phase 2 is now prospectively locked at commit `73fd1d1` and amended/implemented at clean commit
`3d9adb7`, before any Phase-2 scientific job. This is a new adaptive preregistration after the
Phase-1 hard stop, with a maximum of 12 new scientific cells; the prior three plus these 12 are
cumulative bookkeeping, not retroactive authorization. Seed-0 selection data are exposed
development data, while model-validation indices 576:640 and the seed-20261031 confirmation draw
remain unopened.

The finite bracket is `(R,k,M,m)=(24,12,64,256)`, `(32,16,64,256)`, and
`(48,24,96,384)`. It uses a fixed `[-4.5,4.5]^2` full-covariance transported, clamped cubic
B-spline hyperdecoder with exact 16-point local support, zero outside the aligned domain, exact
binary physical boundary, and no POD/linear corrector. The conservative analytic tail-plus-cubic
bounds are 2.272e-3 / 6.388e-4 / 1.266e-4, so R=48 is the credible guard arm. Training and
selection mixtures are locked at N=64/128/256 and use all 51 time slices. All three training
seeds must completely retrain the hyperdecoder, autolatents, and direct predictor; seed 11 is the
single-k deployable policy, while seeds 29/47 are pass-all robustness retrainings.

The excluded local S0 smoke completed in 28.83 seconds on GPU/f64/highest. It exercised finite
direct, mandatory-weak, and maximum-one-update paths, exact clamped-span agreement at knots and
adjacent f64 values, and a hyper-reduced 3x3 log-quadratic cold recovery. On the four locked
seed-20260822 N=1024 initial conditions, that recovery read exactly 4096 samples and had relative
parameter errors between 4.30e-15 and 3.16e-14 without constructing an N^2 coordinate array.
These are execution/identity checks, not scientific accuracy or timing results.

S0 is implemented but not yet scientifically submitted. It will execute all 17,136 free-oracle
fits and pair the N=1024 structural direct/50-weak/max-one-update paths against a live healthy,
accuracy-eligible cubic/exact-Helmholtz FOM in the same job. Eligibility uses only the paired
same-job FOM median/10 and clustered speedup ratio; the earlier 23.805 ms value is planning
context, not a cross-job clock gate. The audited request is one isolated H200, 8 CPUs, 64 GB,
and six hours under the existing Burgers namespace. The local per-fit bracket implies about 1.42
hours serial projection work; scientific S0 uses an ordered eight-worker map and may not truncate
any cohort.

---

## 2026-08-20

### Autonomous Poisson/Burgers hybrid speed push — final

Continued the two completed hybrid tracks autonomously in their existing approved worktrees:
`exp/2026-08-19-poisson-hybrid-1024` and `exp/2026-08-19-burgers-hybrid-1024`. The final branch
commits are `57329c0` and `559583c`; both worktrees are clean and remain unmerged. All real jobs
used isolated Tufts GPU directories, GPU backend, f64/highest precision, same-invocation timing,
accuracy and work telemetry, retained repetition arrays, checksum-verified pulls, and explicit
remote cleanup.

**Poisson architecture/objective closure.** Job 2667580 tested one weak Gauss--Newton update on
the physical-parameter-aligned K=4 conditional decoder with alpha=1. Job 2667673 then tested one
separately preregistered alpha=0.5 energy/Ritz update, M=24,m=96,c64, on a new development seed.
Both routes reduced their projected objective and accepted every update but worsened global
A-error, counting-CG work, and same-job total against the direct parameter surrogate at both
N=64 and N=256. The alpha=0.5 comparison is 5.391 versus 5.125 ms at N=64 and 20.801 versus
20.312 ms at N=256. The frozen gate therefore stopped N=1024 confirmation and any additional
alpha/M sweep. This does not retract the earlier balanced K=8 result: the only supported genuine
NM-ROM crossover remains N=1024,tolerance 1e-6 at 211.996 versus 217.578 ms, 1.026× with 95%
interval [1.014,1.077], against counting CG. Dense/spectral rectangular solvers remain hundreds
of times faster and are the production choice.

**Burgers learned closure.** Job 2667575 tested the bounded one-shot transported trajectory
representation. Its q=64 remaining correction ratio grows from 0.0797 at N=64 to 0.3717 at
N=256, failing the frozen 0.25 scalability gate. No weak wrapper or training round was promoted.
The earlier guarded FiLM hybrid remains a clean 18/18 loss against cubic-history FOM; no learned
Burgers speedup is claimed.

**Burgers practical solver result.** A separate classical full-grid residual-plus-exact-
Helmholtz correction passed development selection, a disjoint N=256 AB/BA gate (job 2667675),
and the untouched final panel (job 2667698, A100-80GB, seed 20260825). The final has 18 cells,
1,728 timed records, 864 immediate burns, exact AB/BA balance, zero solver flags/breakdowns, and
all returned residuals within tolerance. It supports 10/18 cells: all six tolerance-1e-6 meshes,
none at 1e-8, and N=32/256/512/1024 at 1e-10. At N=1024,tolerance 1e-6, optimized cubic takes
308.432 ms and the corrected FOM takes 226.403 ms, 1.362×; paired saving is 81.583 ms with
trajectory-clustered 95% interval [19.396,95.499]. This is explicitly classical, nonlearned,
and not an NM-ROM. Wall time charges 50 full-grid residual evaluations and 50 exact Helmholtz
inverses; finishing Newton/BiCGStab counters exclude them. Timings are warmed steady-state online
times, not compile/load/first-query latency.

The independent audit found no remaining validity blocker. The generated cross-PDE report is
`reports/2026-08-20-hybrid-speed-push.md`, built from the retained JSONs by
`reports/build_2026_08_20_hybrid_speed_push.py`. The two result branches were deliberately not
merged; that remains an explicit user decision.

### Burgers 1e-3 / 10x — adaptive spline Phase 2 final S0 hard stop

Scientific S0 job `2667808` ran in the isolated `s0_spline_r1` directory from commit `3d9adb7`
on an NVIDIA H200. It completed in 26:29 with GPU backend, f64/highest, empty stderr, and no
health-warning strings. The staged manifest-file hash was `4d685e0e0010b50b1eb0bdd7ec912af18f7b8f5f7ef5e0dd97f7a118548d0b5c`.
All remote/pulled checksums and the independent numerical audit passed. The original audit
checker incorrectly demanded that a scientific health gate pass instead of verifying the
recorded negative outcome; commit `e2cb52d` corrected that integrity logic without changing the
immutable scientific JSON/NPZ. Local checksums then passed and the exact remote directory was
deleted. The assigned namespace and queue were clean afterward.

The same-invocation N=1024 cubic/exact-Helmholtz FOM is healthy and accuracy eligible, with
mean/worst trajectory error 6.584e-4 / 1.400e-3 and H200 median 98.732 ms. The tight/tighter
reference difference is at most 4.452e-13 with zero flags/breakdowns. These H200 times supersede
the earlier A100 number only within this paired S0 comparison; no cross-job wall-clock claim is
made.

The R=48/k=24 guard arm C is the only credible representation row. Its pooled free-spline
oracle mean/worst are 1.015e-4 / 9.968e-4. N=64 and N=256 pass the 2e-4 / 7e-4 accuracy gate,
but N=128 worst error is 9.968e-4 and fails. All three meshes fail coefficient-solve health:
1847/3264, 707/1632, and 313/816 fits pass the 1e-8 normal test, every median/max iteration count
is 500, and worst normal residuals are 8.728e-7, 2.751e-6, and 1.791e-6. This is classified as
a numerical-oracle-fit failure plus a remaining N=128 accuracy miss, not as proof that the
mathematically fixed spline space itself fails.

Arm C's charged direct, mandatory-weak, and maximum-one-update medians are 14.966, 15.078, and
25.935 ms. Mandatory speedup is only 6.548x with trajectory-clustered 95% interval
[5.161,8.029]; both the 10x point gate and 8x lower-bound gate fail despite passing memory and
FOM-denominator gates. No arm was promoted, so no Phase-2 training/model-validation/EQ/scaling/
confirmation job was submitted. Two local synthetic-only model-validation drafts remain
explicitly excluded and uncommitted.

**Open bounded Phase 3.** The only licensed direction is a finite joint repair that preserves
the exact R=48/k=24 transported spline: compare augmented column-preconditioned LSMR against
sparse normal-equation LU for coefficient health, and compare mathematically identical
span-polynomial/fused sequential-time decode kernels for at least a 1.53x mandatory-path speed
gain. A diagnostic may license at most one full C-only rerun; training remains blocked unless
all-fit normal residuals are <=1e-8, N=128 worst error is <=7e-4, and a new live paired H200 panel
passes median >=10x and clustered lower bound >=8x.

### Burgers 1e-3 / 10x — Phase 3 exact solver/kernel repair final negative

The single preregistered P3-D job `2668417` ran from commit `bff4472c` in isolated directory
`p3_d_r1` on an NVIDIA H200. It completed in 3:20 with GPU backend and f64/highest. Its raw
214-byte stderr retains exactly two classified Tufts/JAX hwloc binding lines. The manifest-file
SHA-256 is `c7bacf66d326e4511be17db2af7b46be57bd45832781f5c11d3e93d048dc7107`.
Remote and local checksums, Slurm completion, staged source hashes, and the independent
negative-aware audit all pass; the exact validated remote directory was deleted and the assigned
namespace/queue are empty.

**Algebraic repair succeeded, representation still failed.** S1 augmented column-preconditioned
LSMR and S2 sparse normal-equation SuperLU each have 128/128 healthy fixed-subset fits, pass the
per-snapshot no-regression rule, and have worst relative normal residuals 5.672e-14 and 1.341e-15.
Their N=128 draw-530 trajectory errors are 7.888286332e-4 and 7.888286339e-4, improving the S0
9.968316650e-4 witness by 20.9% but missing the fixed 7e-4 gate by 1.1269x. No solver is selected;
the active floor is the locked spline representation, not numerical oracle health.

**Kernel repair succeeded.** The live paired cubic/exact-Helmholtz FOM is healthy and eligible at
mean/worst 6.584e-4 / 1.400e-3 and median 98.886 ms. K0 is 17.259 ms, 5.729x with clustered 95%
interval [4.518,7.011], and fails. Exact sequential polynomial K1 is 8.867 ms, 11.152x
[8.821,13.701]; chunk-three K2 is 8.605 ms, 11.492x [9.093,14.152]; block-128 f64 Pallas K3 is
2.535 ms, 39.005x [30.285,46.755] and is selected. All identity, exact boundary/support, memory,
canonical finite exactly-50 weak evaluation, exact timing balance, and zero-outlier gates pass.

The joint Phase-3 license requires both a solver and a kernel. Its immutable decision is
`selected_solver=null`, `selected_kernel=K3`, `run_P3_F=false`, and hard stop. Therefore P3-F,
hyperdecoder/direct-predictor training, model validation, EQ, N=256/512 scaling, and untouched
confirmation were not run. This closes the finite Phase-3 bracket negatively without weakening
the 1e-3/10x headline or any supporting gate. Synthetic-only model-validation drafts and the
stopped prospective P3-F draft remain excluded and uncommitted.

### Burgers 1e-3 / 10x — Phase 4 hierarchical diagnostic selects H1

The final provenance-correct P4-D job `2668794` ran from commit `ae80f8f5` in isolated directory
`p4_d_r3` on an NVIDIA H200. It completed in 26:33 with GPU backend and f64/highest. Its stderr
contains exactly two classified Tufts/JAX hwloc binding lines. The root manifest SHA-256 is
`67a3bf8d0d8c755ec39cd4056a0bca0e852b4ba17493e8f052a0b288e63a201a`; JSON and NPZ SHA-256
are `ff425dfa1f73ac2d8559df2d780ef09dc1ade179ed5636458e53e0f389f5d617` and
`720c5890b22709c83858f46e43f18fa3d28bb1305544f02ad9aea71625a2228f`. Remote/local checksums,
Slurm completion, nested S0/P3 provenance, immutable source hashes, and the independent
negative-aware audit pass. The exact remote directory was deleted after preservation.

**The exposed representation gate now passes.** H1 (global R=48 plus central P=32, k=24) has
pooled free-oracle trajectory mean/worst `5.8054779105e-5 / 3.8589189381e-4`. Its N=64,128,256
means/worsts are `5.4254925523e-5 / 3.5931759898e-4`, `6.2987244194e-5 /
3.8589189381e-4`, and `6.3389263258e-5 / 2.9951210958e-4`; all `3264/1632/816`
fits are healthy with worst independent normal residual `3.033e-15`. Exact boundary, C2
partition, maximum support 32, full S0 no-regression, and the fixed P3-subset no-regression rules
pass. The targeted N=128 draw-530 trajectory is `2.8852054843e-4`. H2 also passes but is larger,
so the preregistered smallest-pass rule selects H1.

**Mandatory speed passes; maximum-one speed does not.** The same-invocation tight/tighter
cubic/exact-Helmholtz FOM is healthy/eligible at mean/worst `6.5840644023e-4 /
1.4000013197e-3` and median `99.256895` ms. H1 mandatory median is `3.816482` ms,
`26.007434x` with trajectory-clustered 95% interval `[20.138590,31.646479]`. H1 maximum-one
median is `14.065942` ms, only `7.056541x` with interval `[5.577488,8.672268]`. All full-field,
stencil, previous-state, residual, rho, boundary, support, memory, exactly-50 weak-evaluation,
balanced-order, and zero-outlier checks pass. The immutable decision is H1 selected, seed-11
k24 training licensed, and correction class `conditional-zero-or-occasional-attempt`. A later
corrected rollout may proceed only if its actual thresholded policy passes the unchanged 10x/LB8
and zero-failure gates; mandatory timing alone is not a deployable claim.

Jobs `2668601` and `2668613` are excluded provenance attempts. The first failed before science
on a legacy dependency-audit schema mismatch. The second completed but was preserved without
scientific inspection because the root manifest omitted the nested P3 manifest. Both exact
remote directories were checksummed, retained locally, classified, and deleted. Neither altered
the Phase-4 methods, gates, or cap. The next and only licensed cell is H1/k24 seed-11 training;
model-validation and untouched confirmation remain unopened.

The licensed H1/k24 seed-11 trainer, independent negative-aware audit, and exact
make/launch/pull path are implemented at commit `7b21066a`. The final exact-staged excluded local
execution plus audit completed on the GB10 in 36 seconds with GPU backend, f64/highest, a
3,328-value hierarchical coefficient output, three retained selection-oracle optimizer states,
and no locked data access. The audit independently recomputed state/autolatent/predictor
consistency, deterministic schedule IDs and point seeds, metadata/global ranges, histories, and
all negative scientific gates. The exact local staging manifest is
`edbfc03235de0a2fbe9787c5fa4ee3418f765ea910ff3a5a6c447f8b7bd2b431`; it binds the P4-D
root manifest `67a3bf8d...` and the git-clean batch script. The prospective cluster request is
H200/8 CPU/64 GiB/12 h. No training job has been submitted pending implementation audit.

### Burgers 1e-3 / 10x — Phase 4 H1/k24 seed-11 learned-manifold hard stop

The only licensed Phase-4 training cell, job `2668956`, ran from exact commit `7b21066a` and
manifest `edbfc03235de0a2fbe9787c5fa4ee3418f765ea910ff3a5a6c447f8b7bd2b431` in isolated
directory `p4_h1_s11_r1` on an NVIDIA H200. It completed in 8:37 with exit 0, GPU backend,
f64/highest, 7.21 GiB maximum RSS, and exactly two classified Tufts/JAX hwloc binding lines.
All remote and local checksums, Slurm state, P4-D dependency chain, staged source hashes, and the
independent negative-aware audit pass. The JSON, NPZ, and checkpoint SHA-256 values are
`9963b1bff49d3e4f4deffd364d326f2ec2a578c66bbeb039e72272fd75eae1f8`,
`968452abc765a78d63a21c001cec6b7d3508ce6eec5f771fdae602ecb59812b4`, and
`2982cec3653260374b5c01d44e0518f727117779d87f75cac3adc5ad8e71899c`. The exact remote
directory was deleted after preservation, and the assigned namespace and queue were clean.

**The dense coefficient manifold fails decisively.** The complete 30,000-step H1/k24
hyperdecoder/autolatent fit ends at sampled relative field loss `1.8403763717e-3`; the
20,000-step direct predictor ends at relative field loss `4.9071821961e-3`. Three independent
10,000-step selection-autolatent starts were evaluated, and start 1 was selected by the locked
pooled mean snapshot-relative-L2-squared criterion. Its learned-manifold oracle pooled
trajectory mean/worst are `0.4309173579 / 0.6021214615`. Per mesh they are
`0.4257203251 / 0.6021214615` at N=64, `0.4343099942 / 0.5310786191` at N=128, and
`0.4449202165 / 0.5508343065` at N=256. The direct predictor is better but still far outside
its gate: pooled `0.0813062370 / 0.1933868523`, with per-mesh mean/worst
`0.0791735201 / 0.1933868523`, `0.0829357135 / 0.1703658247`, and
`0.0865781516 / 0.1689190210`. Every output is finite and preserves the exact binary boundary.

The representation-oracle, direct-predictor, seed-promotion, loss-revision, and k32 near-miss
gates all fail. The immutable next decision is `hard stop: learned manifold misses the 2x
bracket`. Therefore seeds 29/47, k32, model validation, full weak/EQ, N=256/512 scaling,
N=1024 rollout, and untouched confirmation were not licensed or run. The positive free H1
oracle and mandatory K3 timing remain structural evidence only; the training cell did not time
a deployed rollout. The active Phase-4 floor is the learned small-k coefficient generator, whose
dense affine output confines all generated coefficient grids to a fixed low-dimensional affine
subspace, not the H1 spline representation or its mandatory online kernel.

The complete immutable training bundle is committed on the experiment branch at `8d8e613`.
The generated final Phase-4 report and global rerunnable audit include this negative result and
rebuild byte-identically. Seven scientific cells were consumed through the Phase-4 hard stop;
no downstream Phase-4 job remains licensed. The two untracked synthetic-only model-validation
drafts remain excluded and neither model-validation nor confirmation data was accessed.

### Burgers 1e-3 / 10x — Phase 5 nonlinear-generator diagnostic hard stop

The single authorized P5-D job `2669249` ran from exact scientific commit
`e18edda9b124be8f7fa21c07ef21804c8dbecc48` and staged-manifest SHA-256
`6135791d3a5cca08b0ff1c424d93451579f3dd1048bef2a5cf314e2b8bf317d6` in isolated directory
`p5_d_r1` on an NVIDIA H200. It completed in 31:45 with exit 0, GPU backend, f64/highest, 7.24
GiB maximum RSS, and exactly three classified Tufts/JAX hwloc binding lines. The independent
negative-aware audit passes. Its JSON/NPZ/AUDIT SHA-256 values are
`97f8bc6bb9e1d67d0baf4652bd57e6fb69dab484fc8f99ce12018e9f6c1d0c96`,
`5235b81b19c4ed459e7fda4291fe67eb3f4b87ba07413eb36e861a0b147dfe54`, and
`c84ee29e1b9fe84f5e90949e18be26c07a6c54c00320a2f7a82bd1cb8dee0eff`. A local audit-only
patch at `eba2295` allows the one-ULP cluster/local JAX difference in generated viscosity and
its normalized feature while retaining bitwise checks on all other parameters/features. The
scientific artifacts are unchanged. All remote/local checksums passed and the exact remote
directory was deleted after preservation.

**The full training-target generation succeeds.** All 35,904 S2/H1 coefficient targets in the
locked N=64/128/256, case-major, all-51-time mixture are finite and healthy. All 16 chunks pass
exact boundary, partition-of-unity, support, RHS/prediction/coefficient finiteness, and metadata
checks; the worst normal residual is `3.2782920288e-15`. Train-only coarse/fine head RMS values
are `0.8494597791 / 0.6797529720`.

**The nonlinear generators avoid the diagnosed affine-output collapse and have structural speed
headroom.** On the deterministic nonconstant screen both G1 and G2 have numerical output rank
127 at relative tolerance 1e-10; their median curvature values are `0.0550704125` and
`0.0572025868`. Against the same live eligible H200 FOM (mean/worst error
`6.5840644023e-4 / 1.4000013197e-3`, median `99.177700` ms), G1 mandatory is `4.538580` ms,
`21.852144x` with clustered 95% interval `[17.003472,26.385599]`; G2 mandatory is `6.586756`
ms, `15.057138x` with interval `[11.837359,18.486060]`. Canonical work, memory, timing balance,
zero-outlier, noncollapse, and eligible-denominator gates pass. Worst-case maximum-one remains
nondeployable at `1.628804x` for G1 and `0.559453x` for G2.

**The unchanged identity gate hard-stops both arms before training.** K3 and Cox full fields
agree to worst relative L2 `2.8580109302e-16` for G1 and `2.8732871822e-16` for G2. Their weak
residuals amplify that roundoff: live-case relative discrepancies range from
`3.2730455159e-14` to `4.8255283343e-14`, above the locked `2e-14` identity tolerance. Every
case therefore fails identity, so both training licenses are false and `next_seed11_arm=null`.
No G1/G2 training, weak/EQ, scaling, model validation, or untouched confirmation cell was run.
The result is a numerical identity-screen hard stop, not a measured learned-generator accuracy
failure. The complete immutable bundle is committed at `d6ba627`; the generated report/global
audit are committed at `67b8c7e` and rebuild byte-identically. Eight scientific cells were
consumed through Phase 5. The two untracked synthetic-only model-validation drafts remain
excluded and untouched.

### Burgers 1e-3 / 10x — Phase 6 G1 actual-route identity repair passes

The single authorized P6-D job `2669652` ran from exact scientific commit
`3f454057c2aa5234ae1f589d1f7443a994ebf740` and staged-manifest SHA-256
`f8932a6a4304a14b93bfdf6783e900a47a9a1d45ba03f9915941bb00770bfb40` in isolated directory
`p6_d_r1` on an NVIDIA H200. It completed in 2:36 with exit 0, GPU backend, f64/highest, 6.44 GB
maximum RSS, and exactly three classified Tufts/JAX hwloc binding lines. The independent
negative-aware audit passes. JSON/NPZ/AUDIT SHA-256 values are
`9fe2d49bbb0324fd08ef5da906c3afab0338a1f3bbfb6dfc1ef73f89603c139a`,
`9f0372daba8c12e3aff86efde3201d3ae0612aba8cd297aa37559aa286967381`, and
`9e017b37709bf37fc8c8b87bbbb70461cba8afb47603a5b65dd2618c901ad2b4`. All remote/local
checksums, nested immutable P5 provenance, source hashes, Slurm state, backend/health, and
scientific recomputations pass. The exact remote directory was deleted after preservation; the
audited bundle is committed on the experiment branch at `de5b450`.

**The arithmetic repair succeeds without changing the learned method.** R0 reproduces the
Phase-5 polynomial-weak/K3-full discrepancy, with live-case identity maxima
`3.533e-14--4.826e-14`. The sole candidate R1 evaluates the same fixed cubic basis with
Cox--de Boor arithmetic for weak current/previous queries, then retains K3/Pallas for the final
full-grid decode. R1's four Cox-control identity maxima are `2.831e-16--2.858e-16`, and a
separate execution of the actual timed route agrees with the identity route to at most
`6.851e-16`. Exact boundary, basis/support, finite output, exactly 50 weak evaluations, exactly
51 coefficient-grid evaluations, zero Jacobian/trial evaluations, zero failures, and the
496,002,376-byte compiled-memory gate all pass.

**The repaired actual route clears the same-job speed gate with ample margin.** The live
cubic/exact-Helmholtz FOM is healthy and eligible at mean/worst trajectory error
`6.5840644023e-4 / 1.4000013197e-3`, with median `98.760372` ms and zero timing outliers. R1 has
median `4.798337` ms, paired speedup `20.582209x`, and trajectory-clustered 95% interval
`[16.204135,25.159861]`, also with zero outliers. The tight/tighter reference chains are healthy
and differ by at most `4.4518333659e-13`. The immutable decision is
`repair_licensed=true`, `phase6_hard_stop=false`, and `training_authorized=false`.

Phase 6 consumed the ninth scientific cell in the finite search. It licenses only a separate,
prospectively audited G1 seed-11 training proposal; it does not authorize training itself and
does not establish any learned reconstruction, rollout, EQ, N=256/512 scaling, or N=1024
headline result. Model-validation indices 576:640 and seed-20261031 confirmation remain unopened.
The generated report and global rerunnable audit incorporate Phase 6 and rebuild byte-identically.

### Burgers 1e-3 / 10x — Phase 7 G1 seed-11 final hard stop

Phase 7 prospectively fixed one complete G1/seed-11 training cell: 10,000 mirrored-encoder/G1
warmup updates with bitwise handoff, 30,000 joint G1/autolatent updates, 20,000 direct-predictor
updates, and three fixed exposed-selection q-oracle starts. It bound the immutable 35,904 P5
coefficient targets, the Phase-6 Cox-weak/K3-full route, every seed/schedule/loss/gate, and kept
model validation, confirmation, weak/EQ, scaling, and G2 inaccessible.

Two retries are preserved as zero-science infrastructure failures. Job `2669861` ran
GPU/f64/highest on an H200 but stopped before output or a training update because it compared P5's
physical affine transport directly with the normalized decoder state. The checksummed failure is
at `runs/p7_g1_s11_r1`; its exact remote directory was deleted. After locking the pre-existing
physical-to-normalized mapping, job `2669975` ran on pax011/H200 from commit `825107e` and
manifest `9e8e722b...`, but again stopped before output/update because it required bitwise equality
between P5's batch-4 generation and Phase 7's batch-8 regeneration. A same-node read-only audit
found first N64 draw-0 difference `3.1734496141e-18`, trajectory maximum
`4.4408920985e-16`, and first-64-trajectory maximum `8.8817841970e-16`; exact feature columns
and immutable physical targets remained intact. The retracted assumption is that separately
regenerated f64 scalar exp/log paths must be bitwise equal even on the same node. The prospectively
fixed replacement is finite componentwise absolute difference `<=2e-15`, no relative tolerance,
followed by exact immutable-target replacement before any update. The regression covers all
35,904 immutable rows, exercises the actual N64 regeneration path, and rejects a `2.1e-15`
perturbation. No data, architecture, target, training input, loss, schedule, or scientific gate
changed.

The sole complete cell, job `2673185` (`ctol_b10_p7_g1_s11_r3`), ran from scientific commit
`b85a0b06d3dd2fe460e41a801f51050f271fcf5b` and manifest SHA-256
`a95fc4621a90cef13071df1ad1db363187deac0f7fc7c8c774c0fedd7c9c3a19` on
pax008/NVIDIA H200 with GPU backend, f64/highest, eight CPUs, and 96 GiB. It completed in 10:46,
exit `0:0`, with empty stderr and 8,708,316 KiB peak batch RSS. The exact remote checksum manifest
was reverified after pull. The complete local bundle at `runs/p7_g1_s11_r3`, including the
independent audit, is sealed by `LOCAL.sha256` whose SHA-256 is
`43b1bd33314a021dd492c20ba8d1fb0bc6d80022b4da2689bb42d38daa552acc`; the exact remote
cell was then deleted and absence verified.

The first local independent audit correctly stopped before metric inspection because a local
NumPy scalar exp/log recomputation of the normalized physical target was not bitwise identical to
the cluster artifact. Read-only diagnosis found first difference `[6,3]` at
`3.469446951953614e-18`, maximum `2.220446049250313e-16` at `[16128,4]`, and 1,915 differing
values among 179,520. The immutable physical target, feature array, artifact normalized affine,
regenerated affine, and checkpoint bindings were intact. Audit-only commit
`ae99d3b5898265d7f62aca8db5c317b11a261e5d` therefore independently recomputes the mapping
with the already-preregistered `2e-15` absolute/no-relative portability ceiling while retaining
bitwise physical, feature, checkpoint, and artifact-state checks. It changes no scientific JSON,
NPZ, checkpoint, method, or gate. The full negative-aware audit then passed and records local
maximum `2.220446049250313e-16` plus the exact-versus-portable classification.

**The trained G1 accuracy result is decisively negative.** Learned-oracle trajectory mean/worst
relative-L2 values are `0.3877976038 / 1.2732774798` at N64,
`0.3564755671 / 1.1429050819` at N128, `0.3477184635 / 0.9032960927` at N256, and
`0.3731228590 / 1.2732774798` pooled. Direct-predictor mean/worst values are
`0.0798188159 / 0.1936914758`, `0.0828073612 / 0.1766724003`,
`0.0871985709 / 0.1777454954`, and `0.0817269367 / 0.1936914758`, respectively.
Direct/oracle mean ratios are `0.205825965`, `0.232294633`, `0.250773485`, and
`0.219034923`, so the degradation ceiling passes; every output is finite with exact binary
boundary, and route identity passes. But both fixed accuracy gates fail at every N and pooled.
The immutable decision is `g1_seed11_pass=false`, `phase7_hard_stop=true`,
`seeds29_47_proposal_licensed=false`, `training_authorized=false`, and `next_action=hard stop`.
No G2, further retry, weak/EQ, model validation, scaling, or confirmation is licensed or run.
This tenth scientific cell closes the Burgers-only search without a 1e-3/10x learned Pareto
point. The generated final report and global rerunnable audit incorporate the result and rebuild
byte-identically; the two unrelated untracked synthetic model-validation drafts remain excluded.

---

## 2026-08-20

### Read-only review of the 18 August results and plots

Reviewed the canonical log, the consolidated `exp/2026-08-18-codex-handoff` worktree, its
reports, raw-result inventories, and the headline figures without running a new numerical
experiment or changing an experiment branch. Re-ran the completed warm-start cell's independent
verifier with the repository venv: **146 checks passed, 0 failed**.

The durable 18 August results are the weak-form accuracy result on Poisson/Heat/Burgers (with the
Wave negative), mesh-independent iteration/quadrature behavior, the k-spike retraction and
full-grid trust-region diagnosis, and the final negative ROM-to-FOM warm-start result. The
cost-to-tolerance single-GPU consolidation JSONs are present and generated the
0.35/0.65/1.26/1.73/3.39x Poisson iso-accuracy ladder, but that cell still has placeholder
Verdict/Caveats sections and no `CODEX-REVIEW-RESULTS.md`; treat its speed/Pareto claims as
historical/provisional until that audit is finished. In particular,
`reports/2026-08-18-solve-cost-quadrature-and-the-hybrid.md` still carries the older cross-GPU
warning and a now-retracted k-specific stall interpretation, while the later regenerated
`talk_figs/1_crossover` uses the consolidation timings. No statement in the top-level current
status changed as a result of this review.

### Generated two-PDE results and plot audit through 18 August

Created the generated report `reports/2026-08-18-two-pde-results-and-plot-audit.md`, its canonical
artifact JSON, its self-contained HTML reader, and
`reports/build_2026_08_18_two_pde_results_and_plot_audit.py` on `main`. The builder reads only the
archived Poisson-2D and Burgers-2D JSONs in `exp/2026-08-18-codex-handoff`; no experiment worktree
was changed and no numerical job was launched. A second build reproduced the Markdown and artifact
byte-for-byte. The portable artifact validates with six native charts, five tables, five headline
metrics, and complete source metadata. Its packaged structural verification passes; interactive
browser verification was unavailable because the local machine has no working supported Chromium
binary.

The report makes the main inconsistencies explicit rather than combining them into one status:
weak-form full/EQ accuracy is close to the nonlinear decoder ceiling on both PDEs, `m=M` is poor
while `m=4M` is near the full-grid accuracy region, the old latent-dimension stalls are withdrawn
because eleven divergent source-solves distorted means and the trust-region repair leaves zero,
the single-GPU standalone Poisson cost curve remains provisional, and the independently verified
original ROM-to-FOM hybrid remains slower than baseline through the archived meshes. Burgers has
zero uncensored coordinate points in that archived strict cost cell. Open work is to apply the
same trust-region solver and aggregation rules to both PDEs, finish the cost-cell audit, persist
timing repetitions, and compare learned warm starts with the strongest classical history arm in
the same solver invocation.

### Fresh-seed N=2048 Poisson and Burgers extensions

Extended the independently audited hybrid-solver studies from N<=1024 to N=2048 on two new
branches created from the completed speed-push commits rather than `main`:
`exp/2026-08-20-poisson-hybrid-2048` from `57329c0` and
`exp/2026-08-20-burgers-hybrid-2048` from `559583c`. Both branches remain clean and unmerged at
final commits `2c33188` and `a081472`. All accepted jobs used GPU backend, f64,
`JAX_DEFAULT_MATMUL_PRECISION=highest`, isolated paralab directories, same-invocation
cost/accuracy/work telemetry, persisted repetitions, immediate burns, checksum-preserving pulls,
and exact remote cleanup.

**Poisson final.** Job `2670159` ran on an A100-80GB using fresh seed `20260826`, eight timed
cases, 12 exactly balanced AB/BA repetitions per learned/zero method and case, and a separately
balanced five-method production block. The genuine fixed-N64 K=8 trust-region NM-ROM plus
counting CG is supported only at tolerance 1e-6: `1472.985849 / 1547.144049` ms for learned and
zero, `1.050345x` with case-clustered ratio interval `[1.018431,1.066654]`, paired saving
`53.329957` ms with interval `[33.637275,84.795740]`, and all eight cases faster. The 1e-8 row
is `1.025872x [0.993227,1.052155]` and the 1e-10 row is
`0.996423x [0.984538,1.003211]`; neither is supported. All 576 learned/zero and 1,200
production-control records pass true-residual, boundary, flag, position-balance, and provenance
gates, with zero timing outliers and peak device fraction 0.3658.

The strongest eligible Poisson production control is classical `spectral_q1024` at every
tolerance. Its `1.921111/2.469948/3.733728` ms totals at 1e-6/1e-8/1e-10 are
`798.29/723.79/533.26x` faster than their same-block zero-start counting-CG medians. Dense and
FFT DST direct solves are eligible at 1e-6 and 1e-8 but fail the measured 1e-10 true-residual
gate; both spectral arms remain eligible. Three independent audit layers pass. The learned
crossover is retained as a real counting-CG
result, but any claim that it is the fastest Poisson route on this rectangle is rejected.

**Burgers learned sensitivity.** Job `2672536` ran on an H200 using fresh seed `20260828`, four
trajectories, and exactly balanced within-job comparisons. The genuine K=8 weak FiLM NM-ROM uses
M=64, m=256, at most two latent Jacobians per step, fixed-N64 decode/prolongation, an exact
residual guard, and the same exact-Helmholtz FOM finish as its controls. It has zero supported
wins in three comparisons against cubic and zero against the classical dynamic correction; all
six point estimates are slower. Against cubic, control/FiLM ratios are
`0.877106/0.947231/0.948679x` at 1e-6/1e-8/1e-10. Against the dynamic control they are
`0.827790/0.888724/0.862581x`. The guard accepts a median four of 50 steps while the arm pays
100 reduced Jacobians per trajectory. All 576 timed records, 288 burns, accuracy, residual,
work, equivalence, memory, and provenance checks pass. This confirms the learned Burgers route
as a negative result rather than a practical warm start.

**Burgers reference failures and repair.** Primary jobs `2672535` and `2674703` are excluded
pre-science with zero timing rows: the inherited fixed public-JAX reference freezes at residual
`1.215e-2` on development seed `20260827`, even with 25 fixed Newton iterations. Diagnostic job
`2676167` is also excluded: an unbatched replay changed the public execution topology, and its
batch script used a false-success `producer && checker` pattern. Corrected batch-one diagnostic
`2677878` reproduces the failure: all 25 active public updates at trajectory 3, step 41 are
nonfinite. Its healthy exact-Helmholtz candidate could not self-license: the update-match gate is
null because every active public update is nonfinite, and the separately frozen outer-1e-13
strict route hit the maximum Newton count. Seed `20260829` was then excluded
after an N=32 implementation smoke touched only reference/equivalence health, with no N=2048
field or method timing. None of these attempts contributes a speed claim.

The final reference design was preregistered on never-used seed `20260830`: two separately
compiled exact-Helmholtz routes at outer tolerance 1e-12 and inner tolerances 1e-8/1e-10 had to
pass on all four trajectories before online compilation or timing. Both pass with worst actual
outer residual `9.537719e-13`, maximum step agreement `4.682753e-14`, maximum trajectory
agreement `1.793184e-14`, and zero flags/breakdowns.

**Burgers classical final.** Job `2680178` then ran on an H200 with 12 AB/BA repetitions per
arm and trajectory at all three tolerances. The full-grid residual-plus-exact-Helmholtz
correction is supported only at 1e-6: cubic `532.816699` ms versus corrected `343.673245` ms,
`1.550358x`, with paired saving `178.174519` ms and trajectory-clustered interval
`[54.945137,195.983786]`. At 1e-8 the ratio is `0.996064x` and paired-saving interval
`[-27.190926,67.302173]`; at 1e-10 it is `0.984962x` with interval
`[-20.894478,266.999483]`. Neither tighter row is supported. All 288 timing records and 144
burns pass health, work, accuracy, equivalence, balance, provenance, and independent raw
recomputation. The method is classical, nonlearned, and not an NM-ROM. Its wall time charges 50
full-grid exact-upwind residuals and 50 exact Helmholtz inverses; finishing iteration counters
exclude those extra operations.

The generated cross-PDE report is
`reports/2026-08-20-hybrid-warm-starts-at-2048.md`, built by
`reports/build_2026_08_20_hybrid_warm_starts_at_2048.py` directly from the six final raw/audit
JSON inputs. Repeated generation is byte-identical. The result branches were not merged; an
explicit user decision remains required.

### Added the audited N=2048 extension to the two-PDE plot audit

Updated the user-requested generated report
`reports/2026-08-18-two-pde-results-and-plot-audit.md` without replacing its archived August-18
sections. The new `Audited N=2048 extension` subsection sits inside `Cost and hybrid evidence`
and adds the final Poisson learned, Poisson spectral, Burgers learned-FiLM, and Burgers classical
tables plus two static plots:
`reports/figs/two-pde-n2048-learned-speedup.png` and
`reports/figs/two-pde-n2048-classical-speedup.png`.

The report builder now binds the three N=2048 raw JSONs to their independent-audit SHA-256
values before rendering. Both plots use only dimensionless within-job speedups, label the
parity threshold, distinguish learned and classical methods, and state the case/trajectory and
GPU context. The classical figure uses separate Poisson and Burgers panels because their
speedup scales differ by hundreds-fold. No absolute timing is compared across jobs or GPU
types. The linked HTML companion remains explicitly labeled as the archived through-August-18
surface; the requested extension lives in the Markdown report.

Two complete builds under the required local `jaxrun`, f64, highest-precision wrapper produced
byte-identical Markdown, artifact JSON, and PNG files. The generated report SHA-256 is
`29f53cd8d04c3af3826e49b45fee05db359c2ee2100bbff062057b677b87f5e9`; the learned and
classical plot SHA-256 values are `8979c59475fae646c04faebf0789a89e421d45026435dda3614e3569786b17b3`
and `4ece1dcae86d629ecafb9380d3ed0ff2b52ec400c9ef6ad370f659a1fbcedef5`.

### Paired Poisson/Burgers k=4--64, m=4M sweep launched

Created the confirmed isolated branch/worktree
`exp/2026-08-20-two-pde-k64-eq4m` / `worktrees/2026-08-20-two-pde-k64-eq4m`
from corrected standalone-architecture commit `9c330bd`, not from `main`. The experiment and
generic timing/trust changes are committed at `ae92252`. The frozen N=64 coordinate-NM-ROM
ladder is k=4,6,8,12,16,24,32,48,64 with M=4k and m=4M=16k for both Poisson-2D and
Burgers-2D. Tau=1e-3 is primary for comparability with the retracted k-spike study; tau=1e-2 is
a secondary deployment point sharing the same per-k EQ fit. Existing seed-0 checkpoints through
k=32 are reused, while k=48/64 are trained in the job with the matching recorded recipe.

Both PDEs run sequentially in one job on one physical GPU. The whole path, isolated decoder,
and isolated latent solve each persist 16-by-9 or nine raw timing repetitions after three
warm-ups and an explicit GPU burn. Burgers additionally persists its hyper-reduced cold-start
timings. Whole-path cost and relative-L2 come from the same final timed invocation. Poisson uses
a one-training-cloud-radius trust step; Burgers uses the independently accepted 0.01-radius
factor and retains the FOM-exact upwind weak advection. Every EQ fit records global and worst-row
diagnostics.

Static parsing, shell syntax, generated-schema tests, both staged GPU imports, and a synthetic
trust-rejection test passed locally. A bounded one-case Poisson end-to-end smoke reached `DONE`
and populated all new timing fields. The analogous Burgers smoke reached regenerated truth and
the EQ fit but was interrupted during first pipeline compilation at the repository's one-minute
local-smoke ceiling; it produced no scientific result and no failure was inferred from the
interrupted compilation.

The checksummed 67 MB stage was copied directly to
`/cluster/tufts/paralab/tawal01/k64-eq4m/ctol_k64_eq4m_r1`. Local and remote manifest SHA-256 are
both `c07a22985adbe206a3aa285abd2e1ebae3b6bb6b3f6f422230ea546b29cedd43`. Queue checks were
performed before and after submission. Job `2689547` requests the `gpu` partition, one generic
GPU, eight CPUs, 96 GB host memory, and 18 hours; at launch it was pending for `Priority` while
unrelated job `2686475` remained running. A subsequent check found job `2689547` running on
`pax007` with `jax_backend=gpu`; both new Poisson checkpoints had trained and Burgers k=48
training was in progress. No paired k-sweep result exists yet. Open: wait for completion, pull
with checksums, audit health and raw repetitions, delete the exact remote job directory,
generate the requested plots/report, and ask whether to merge this experiment branch after it
finishes.

## 2026-08-20

### Closed stale Codex session writer

Diagnosed failed resume of thread `01a01b95-3dad-7773-9c9e-0f6696c0a2c1`. The live writer was
the Codex process group led by PID `1356196`, with writer PID `1356203`, attached to the VS Code
terminal `pts/34`. At the user's request, sent `SIGINT` to that process group. Both processes
exited cleanly, and a follow-up `lsof` check found no remaining holder of the thread-writer lock.
No repository experiment, result, branch, worktree, cluster job, or scientific project state was
changed.

## 2026-08-20

### Closed remaining stale Codex terminal sessions

At the user's request, enumerated the remaining Codex TUI processes and closed the two sessions
other than the active conversation: process group `1357652` on `pts/33` and process group
`1359834` on `pts/32`. Both exited cleanly after `SIGINT`. Verification found none of PIDs
`1357652`, `1357659`, `1359834`, or `1359841` still running. The only remaining Codex writer
and lock belong to the active conversation on `pts/9`; surrounding VS Code and shell terminal
processes were intentionally preserved. No scientific project state changed.

---

## 2026-08-22

### Read-only audit of the actual latest experiments

Reviewed the canonical log first, then reconciled it against all branches and found that it was
stale in two important ways: paired sweep job `2689547` had completed, and the Burgers 1e-3/10x
branch had continued from Phase 8 through a Phase-12 preregistration after the canonical Phase-7
hard stop. No numerical job was launched, no cluster output was deleted, and no experiment
worktree was modified in this audit.

**Burgers Phase 8--12 disposition.** Verified the local `LOCAL.sha256` seals for P8-D job
`2686475`, the accepted Phase-9 terminal-field audit job `2737342`, Phase-10 scientific job
`2739690` and accepting audit-only job `2750135`, and Phase-11 job `2754129`; every listed file
passes. P8-D remained invalid under its locked trust-health rule and, even descriptively, its
best two-start G1 selection mean/worst was `2.3116452e-2 / 5.4601627e-2`. Phase 9's exact-field
G1 training completed all 528,768 updates, but its accepted terminal pooled train field was
`1.3055157e-2 / 4.8406660e-2`, with exact boundary and worst K3/Cox identity `2.974e-15`; the
`2e-4 / 7e-4` gates fail. Its tangent-capacity record is retracted because independent CG
repeats disagreed on thousands of iteration counts and one convergence label, so it cannot
license G2.

Phase 10's fixed-generator final-q-only globalization is a valid train diagnostic and improves
pooled mean/worst from `1.3055157e-2 / 4.8406660e-2` to
`9.7831427e-3 / 3.4629417e-2`, with zero optimizer updates in the diagnostic and zero boundary
violations. It still misses the train gates by `48.916x / 49.471x`; the fixed G1 generator is
not train-representable at the target. Phase 11 stopped before update 1. Post-closure source
reconciliation proved that its timed G2 route returned an unintended second 51-state Cox full
field in addition to the intended 50 Cox weak and 51 K3 full evaluations. Its `3.0472336x`
speed and `[2.3967067,3.7521239]` clustered interval are therefore retracted as overcharged and
canonical-work-invalid; they do not prove that the corrected G2 route fails. Phase 12 fixes that
route prospectively but has only preregistration plus untracked implementation files. Current
queue and accounting queries find no Phase-12 job.

This continuation did not produce a learned Burgers Pareto point, a valid G2 speed result, any
G2 training, predictor, selection, weak/EQ, scaling, model-validation, or confirmation result.
It also conflicts with the still-canonical Phase-7 instruction that G2 and further retry were
not licensed. Resolve that governance/state conflict before any Phase-12 submission.

**Paired k-sweep disposition.** Scheduler accounting shows job `2689547` completed `0:0` after
`11:59:39` on pax007/A100-80GB with peak batch RSS `7,665,336 KiB`. Its log records GPU backend,
x64, highest precision, `ALL-DONE`, and only seven classified `hwloc_set_cpubind` lines in
stderr. The remote `RESULTS.sha256` verifies both result JSONs, the original validation JSON,
and copied manifest; the manifest SHA-256 `c07a22985adbe206a3aa285abd2e1ebae3b6bb6b3f6f422230ea546b29cedd43`
matches the local stage and every staged input verifies. A temporary read-only pull was audited
without changing the experiment worktree. All 36 rows reproduce their whole-path, solve,
decode, Burgers cold-start, error, and stored truth-baseline speed aggregates from the persisted
nine-repetition arrays. The recorded 5-MAD rule flags 13 of 576 per-source timing medians.

The strengthened uncommitted validator fails on the first Burgers speed assertion only because
it incorrectly compares stored `speedup_e2e` to `fom_iso_accuracy_ms`. The stored field is
documented and exactly reproduces `fom_rollout_ms / time_ms`; the separately recomputed draft
report ratio uses `fom_iso_accuracy_ms / time_ms`. After applying the documented distinction in
the read-only audit, every row passes. This is a validator bug, not result corruption.

Scientifically the sweep is negative/inconclusive. At tau `1e-3`, Poisson mean relative-L2
improves monotonically from `2.3401651e-2` at k=4 to `2.9323133e-3` at k=64, while whole-path
time increases from `4.115183` to `56.288242` ms. Every tight row is censored; 15/16 cases are
censored for each k from 16 through 64. The loose tau `1e-2` row first reaches mean error below
1% at k=32 (`9.0336771e-3`, `4.800909` ms) and is uncensored, but maximum case error is
`2.9982541e-2`. Burgers is censored in all 18 rows. Its best tight mean is
`1.0137654e-2` at k=24 with `1,692.781856` ms whole time; k=32 and 48 do not improve it. The
new k=64 Burgers checkpoint diverged during training after 10,000 updates and finished at
`0.4541673` train mean relative-L2 versus `0.00475747` at k=48; its EQ global fit degrades to
`0.1621591` and its ROM mean to `0.2714355`. That row diagnoses a failed training recipe/seed,
not latent-dimension convergence.

Both result JSONs contain only one FOM row: Poisson CG at the `1e-13` truth-manufacturing
tolerance (`7.528091` ms) and Burgers at the known over-solved fixed eight Newton iterations
(`350.706853` ms). Therefore neither `fom_iso_accuracy_ms` field is an actual deployment-
accuracy ladder, and the untracked report builder's "same-job like-for-like FOM" wording is
false. Even before that baseline defect, no uncensored Burgers row reaches the 1% mean target.
Do not make a speed/Pareto claim from this sweep. Its official result pull, exact remote cleanup,
corrected validator, generated report, branch commit, and merge decision remain open.

## 2026-08-22

### Meeting-note interpretation and sample-aware decoder ideation

Reviewed the supplied Hari/Sanjeev meeting transcript against the current Burgers implementation,
the completed paired k-sweep artifact, and primary prior work. No code, checkpoint, numerical job,
branch, worktree, cluster state, report, or scientific result was changed. The transcript asks for
a staged algorithmic investigation, not yet an immediate unconstrained end-to-end training run:
run full and sampled paths side by side; measure sampled-value, integral/weak-residual, and
Gauss--Newton gradient/Jacobian/Hessian errors; establish scaling in both N and k; begin with a
fixed sample rule; then consider sensitivity-driven or lazily updated samples. Hari does ask for a
decoder whose partial reconstruction is intrinsic, but does not explicitly require joint decoder
and quadrature optimization in the first experiment.

The existing FiLM coordinate decoder is already intrinsically queryable: `CoordDecoder` accepts
arbitrary coordinates and the weak Burgers path calls it only on selected five-point upwind
stencils. The missing property is solver-aware efficiency and fidelity. Current decoder training
minimizes weighted pointwise reconstruction plus latent regularization/smoothing. Current NNLS EQ
is fitted afterwards on decoder-output fields and their induced exact-upwind advection projections;
it does not preserve the Gauss--Newton gradient, normal operator, or update. The online weak path
uses `jax.jacfwd` on the sampled residual (and another decoder Jacobian in the Galerkin route).
Because m=4M=16k and exact upwinding evaluates 5m stencil entries, forward differentiation makes
the sampled solve strongly superlinear in k even though it is independent of N.

The completed N=64 Burgers tau=1e-3 rows make that distinction concrete. From k=4 to k=64, the
one-time full decode remains approximately 10.7 ms, while the directly isolated 50-step latent
solve grows from 121.55 ms to 22,698.07 ms and time per Jacobian from 0.358 to 20.805 ms. Thus the
current bottleneck is repeated sampled-decoder differentiation and increasingly difficult/censored
Gauss--Newton solves, not the unavoidable final full-field readout. These rows remain scientifically
censored and are used here only as a component-cost diagnosis, not a deployment-speed result.

The proposed research direction is a weak-operator-aware sparse decoder with a shared latent trunk,
cached coordinate features, a point/stencil head evaluated only on the union of requested upwind
stencils, and efficient JVP/VJP or structured latent derivatives. Train it against full-grid field
accuracy and full weak-solver geometry: weak residual/integral error, gradient norm and direction,
normal-operator actions on probe directions, and optionally the damped Gauss--Newton step. Learn or
alternate a positive, fixed-budget quadrature rule while penalizing the cardinality of the union of
stencils rather than only the number of centers. Keep smooth weak test modes and the FOM-exact
upwind operator fixed initially; prevent decoder/quadrature collusion by always auditing full-grid
teacher quantities and held-out trajectories. Test one static rule jointly across N=64/128/256 at
fixed k and M before any adaptive rule. Random/importance/off-grid strong-form sampling is not the
primary arm because the project has already rejected those on localized families.

This broad area has material prior art. Shen et al., *High-order Differentiable Autoencoder for
Nonlinear Model Reduction* (arXiv:2102.11026), alternate a GCN element selector and a nonnegative,
state-dependent cubature-weight network for nonlinear reduced elastic mechanics. Their online
weight network first expands the latent state to the full displacement, so it does not supply the
sample-only N-independent path required here. Deep-HyROMnet (arXiv:2202.02658) instead learns
reduced residuals and Jacobians directly. Classical DEIM/GNAT and newer gradient-preserving
hyper-reduction also establish that preserving nonlinear/Jacobian or gradient structure is an
existing objective. Therefore the plausible contribution is not merely "neural quadrature," but
co-designing a query-only weak PDE decoder and positive stencil-aware quadrature to preserve the
Gauss--Newton solve with measured N independence and like-for-like runtime.

---

## 2026-08-22

### Separable-decoder first cell: designed, audited, built, and run

Reviewed the state of the project after the paired k-sweep and the Burgers Phase 8-11 push.
Two findings from other sessions were confirmed from primary sources and are recorded here so
they are not lost: paired k-sweep job `2689547` COMPLETED on 2026-08-21 07:12 (11h59m,
A100-80GB, `VALIDATION-PASSED rows=36`) but its `out/` JSONs remain UNPULLED and its 104 MB
cluster directory `k64-eq4m/ctol_k64_eq4m_r1` still exists; its log shows a k=4..64 ladder in
which error floors near 1e-2 by k=16-24 on both PDEs, cost grows ~170x across the ladder, and
the Burgers k=64 cell collapsed in TRAINING (not only in EQ). Separately, Burgers 1e-3/10x
Phases 8-11 ran on 2026-08-21 in `exp/2026-08-19-burgers-1e3-10x` (jobs 2702357..2754129, six
FAILED): every surviving decision JSON says `scientific_promotion_allowed: false`; Phase 11
G2 hard-stopped at the structural preflight with zero optimizer updates; Phase 12 is
preregistered but NOT run, and its scripts are untracked in that worktree.

New work this session, on `exp/2026-08-22-separable-decoder` (worktree of the same name,
branched from k-sweep commit `ae92252` with user confirmation):

**Design.** `reports/2026-08-22-separable-eq-decoder-design.md` on `main` (uncommitted)
specifies a separable decoder u(x;z) = bc(x) (g(x)^T h(z)) - a Fourier-feature MLP feature
bank in x times a nonlinear MLP head in z, no POD anywhere - whose weak-NM-ROM iteration is
cached dense algebra with no spatial network in the loop. An independent adversarial audit
(OpenAI Codex) found two blockers that were incorporated: the Burgers cache must reproduce
the FOM sign-upwind stencil exactly (not autodiff continuum derivatives), and quadrature
fidelity must be measured on the GN map (J^T r, J^T J, damped step), not Jacobian Frobenius
norm. Prior art is closer than assumed (Shen et al. learned cubature, CROM, Deep-HyROMnet,
gradient-preserving DEIM); the novelty claim is narrowed to preserving the incumbent
discrete weak GN geometry with a query-only separable manifold.

**Implementation.** `experiments/separable-decoder/`: `sep_common.py` (model + auto-decoder
training), `sep_poisson.py`, `sep_burgers.py`. Both PDEs run TWO arms through the INCUMBENT
solvers (`ctol_tol.lm_tau_poisson`; `blat_common` weak upwind ops / `lm_step_jit` /
`rollout_jit`): a meshfree arm evaluating the network in-loop, and a cached arm using
feature banks at the EQ/stencil nodes. GATE 0 asserts the arms' weak residual/Jacobian agree
<= 1e-12 relative, so the discretization is provably unchanged. Local N=16 smokes passed
(gate-0 dev 0.0 and 1.6e-15).

**Run.** Slurm job `2802238`, isolated namespace `sepdec/sepdec_r1`, H200 (pax010),
`jax_backend=gpu`, f64, highest matmul precision, COMPLETED in 6m47s. Four cells at N=64,
seed 0: (K,R) = (8,32) and (16,64), M=4K, m=16K NNLS-EQ grid nodes, 40k training steps.
Results pulled with matching SHA-256, remote directory deleted, JSONs/checkpoints committed
at `2215bd9`. Gate 0 on the cluster: 0.0 (Poisson x2), 1.50e-15 / 2.79e-15 (Burgers).

**Numbers (within-job unless noted).** Training cost collapsed: 19-21 s (Poisson) and
44-47 s (Burgers) for 40k steps. Poisson K=16: test-oracle rel-L2 4.43e-2, weak-EQ solve
3.75e-2 at tau=1e-3 in 2.06 ms (cached) with 13.3 Jacobians - the solve SATURATES the
manifold, so accuracy is capacity/training-limited, not solver-limited; same-job FOM CG at
tol=1e-1 reaches 3.24e-2 in 1.31 ms, so Poisson remains a classical win at N=64, as
expected. Burgers K=16: all 4/4 fresh-seed test trajectories complete 50/50 steps, zero
blowups, ~315 total Jacobians (incumbent FiLM k=16 used ~507 in the k-sweep log); trajectory
rel-L2 1.38e-2/2.10e-2/2.58e-2 plus one 1.47e-1 outlier whose IC fit was poor (3.6e-1);
cached rollout median 54.1 ms/trajectory versus the SAME-JOB, SAME-GPU truth-generating FOM
Newton rollout at 176.1 ms - 3.3x faster. K=8 cached: 44.5 ms, errors 3.2e-2..5.2e-2 plus
the same-IC outlier 2.01e-1.

**Caveats, stated before anyone quotes the speedup.** The 176 ms Burgers baseline is the
fixed-Newton truth rollout, NOT the optimized cubic-history/inexact-Newton solver, which at
N=64 is expected to be substantially faster; no matched-tolerance classical arm ran in this
job, so this is NOT yet a supported win against the strong baseline. Four trajectories, one
GPU, no AB/BA repetition protocol, no clustered intervals. Cross-job comparisons to the
FiLM k-sweep (A100 vs H200) are indicative only. The cached-vs-meshfree gap within this job
is small (~12%) because the separable network is itself small; the large speedup vs FiLM
comes from the architecture, and a same-job FiLM arm is needed to quote it properly.
Trajectory 1's IC-fit failure (auto-decoder init from mean/train codes) is the clearest
defect to fix.

**Open, in order.** (1) Pull/validate/clean k-sweep job `2689547` and generate its report -
still owed. (2) Same-job comparison cell: separable vs FiLM vs optimized classical Burgers
baseline at matched tolerance with the repetition protocol; add IC-fit repair (better init
search or encoder). (3) Poisson capacity push (r, steps, ff_scale) toward the FiLM ceiling
~6.5e-3 before any Poisson claim. (4) The design doc's gates 1-2 (span ceiling, coefficient
oracle) at N=128-512 to test mesh-independence of accuracy. (5) Decide whether Phase 12
(preregistered, unrun) is ever launched given Phases 8-11; its scripts are untracked. (6)
The two hand-edited generated reports on `main` still disagree with their builders.

## 2026-08-23

### Adversarial audit of the separable-decoder cell; N-scaling round launched

**Audit.** A Codex adversarial verification of `experiments/separable-decoder/` ran against
the committed sepdec_r1 checkpoints (report committed as
`experiments/separable-decoder/AUDIT-2026-08-23.md` on `exp/2026-08-22-separable-decoder`,
commit 4a2c186). Verdicts: nonlinearity PASS (superposition failure 0.07-0.70, Jacobians
vary with z, union tangent ranks exceed k, no POD/SVD; g/h/h_lin trained from random init);
online solve PASS (no path for test truth into either solver; mutation test — replacing the
Burgers truth by 7*U+3 changed only the reported error, solution bits identical); gate 0
PASS, independently recomputed at 0.0 (Poisson) / 1.8e-15 (Burgers); metrics PASS with
caveats; **timing FAIL**; claim-sanity SUSPICIOUS (numbers match JSONs, two prose claims
false).

**Retracted/corrected.** (1) README/EXPERIMENT claimed "cost and error from the same
invocation" — false: error came from an untimed invocation, cost from 7 timed repetitions
of the same jitted call (audit measured 3e-10 latent agreement between the two, but the
claim as written was wrong). (2) README called the Fourier matrix B "fixed" — B is in the
optimized tree and drifted 7-11% from init. Both fixed in 4a2c186. (3) The 3.3x Burgers
number is NOT a supported speedup: the timed ROM path excludes the online IC fit (~8 s
warm local for 9 starts) and returns latents while the timed FOM returns full fields; the
176 ms baseline is the over-solved fixed-8-Newton truth generator; raw timing reps were
discarded; all Burgers steps stopped on 'stalled' and Poisson tau=1e-3 was 100% censored.

**Decision.** The architecture and solves are real and unfabricated; the failures are
measurement methodology. Proceeded with the confirmed N-scaling round with the audit's
fixes made mandatory: `experiments/separable-decoder/HANDOFF.md` (same commit) now carries
binding measurement rules — end-to-end timing incl. IC fit, symmetric decoded outputs, raw
reps retained, balanced ordering, strong classical Burgers baseline in-job, censoring
reported, fresh-seed Poisson cohort arm, per-row EQ diagnostics.

**Launched.** Four worktrees `2026-08-23-sepdec-n{128,256,512,1024}` (branches
`exp/<same>`) cut from `exp/2026-08-22-separable-decoder` @ 4a2c186, one sub-agent each,
cluster namespaces `/cluster/tufts/paralab/tawal01/sepdec_n<N>/`, budget <=4 cluster jobs
per agent. Mission per agent: best accuracy and honest speed-vs-classical at its N, both
PDEs, f(K) with >=2 K values, per HANDOFF.md.

**Also this session.** The first audit launch hung for ~6 h on a stdin-read (codex exec
inherited an open socket as stdin and waited on it; killed, relaunched with stdin from
/dev/null — the fix worth remembering).

**Open.** Unchanged from 2026-08-22 items (1)-(6); plus: scaling-round results to collect
and merge decision for the four sepdec-n* worktrees when they finish.

### N=512 separable-decoder scaling arm (sepdec_n512 agent) — complete, wrapped

Branch `exp/2026-08-23-sepdec-n512` (worktree `worktrees/2026-08-23-sepdec-n512`), cluster
namespace `/cluster/tufts/paralab/tawal01/sepdec_n512/` (now deleted). Jobs: j1 = 2825500
(Poisson, A100 pax106, COMPLETED 2h55m), j2 = 2826612 (Burgers, H200, COMPLETED 2h28m,
after two FAILED data-generation attempts 2825764 and 2826193), j4 = 2827874 (Poisson
capacity push, CANCELLED by user decision at 1h15m when the project redirected to a
focused N=256 push; no reportable numbers, logs preserved). j3 (Burgers refinement) was
never submitted. All results pulled with verified SHA-256, committed (5725733, 15a5a11,
1461b86), summary tables generated only by `experiments/separable-decoder/summarize_n512.py`
(`SUMMARY-N512.md`).

**Measurement rules.** Every AUDIT-2026-08-23 fix ran in-job: Burgers timing is END-TO-END
(IC latent fit + 50-step rollout + full-grid decode of all 51 states, split reported);
Poisson timing includes the source-to-f_m projection (proved identical to
`pc.weak_source_term`, dev 0.0) and full-grid decode; errors extracted from a timed
invocation's own outputs (timed-vs-error deviation 0.0 Poisson, ~1e-10 vs the incumbent
untimed rollout for Burgers); raw timing repetitions retained; balanced alternating-order
schedules; stop-reason histograms next to every error; fresh-seed Poisson cohort;
per-row EQ diagnostics; gate 0 asserted per cell (0.0 Poisson, <=2.4e-15 Burgers).

**Poisson N=512 (16 held-out + 16 fresh-seed sources, tau=1e-3, cached arm).** f(K):
K8 5.61e-2 @ 4.19 ms, K16 3.15e-2 @ 3.06 ms, K24 2.71e-2 @ 3.88 ms, K32 2.72e-2 @ 7.76 ms.
Error equals the representation oracle at every K — the accuracy plateau ~2.7e-2 for K>=24
is decoder capacity/training-limited, not solver- or K-limited. Fresh-seed cohort ~1.4x
worse (4.5e-2 at K24), same ranking — no cohort-contamination signal. Same-job CG ladder:
26.4 ms (tol 1e-1, err 7.1e-3) to 61.8 ms (tol 1e-6). The ROM is 6.8-8.6x faster than the
CHEAPEST CG rung — the N=64 "classical win" has flipped in wall time — but CG's loosest
rung is already 3.8x MORE accurate than the ROM's plateau, so there is NO iso-accuracy
crossover claim at N=512; the win is speed-at-the-ROM's-achievable-accuracy only. Caveat:
tau=1e-3 rows are 100% censored (LM stops 'converged'/stalled, not tau-reached); tau=1e-2
rows are 69-81% censored. The cancelled j4 was closing the capacity gap (train rel-MSE
1.75e-5 at 108k steps vs 5.4e-5 at j1's K16 end) — the capacity push works and is the
obvious N=256 lever.

**Burgers N=512 (6 fresh-seed trajectories, 50 steps, end-to-end, cached arm).** f(K):
K8 err 6.12e-2 @ 65.6 ms, K16 2.94e-2 @ 69.9 ms, K24 2.61e-2 @ 80.1 ms (split ~7-17 ms IC
fit + ~60-64 ms rollout+decode). 6/6 trajectories complete at every K, zero blowups; the
N=64 IC-fit failure mode did NOT recur — the nearest-decoded-t0 init bank (mean + top-3
nearest training t=0 codes, incumbent-identical jitted LM) fixed it (worst IC 9.7e-2 at
K16 vs 3.6e-1 at N=64). All ROM steps stop 'stalled' (100% censored). **Honest speed
verdict: the ROM LOSES.** The same-job strong classical arm — tolerance-terminated
inexact Newton with the exact-Helmholtz-preconditioned BiCGStab — reaches err 3.1e-4 in
15-22 ms (ntol 1e-2) and 6.9e-8 in ~71 ms (ntol 1e-6): 3-4x faster AND ~100x more accurate
than the ROM. The ROM beats only the over-solved fixed-8-Newton truth generator (357-360 ms,
labelled, never a headline). This extends "Burgers is won by solver and history tuning,
not by the NM-ROM" to N=512 with a same-job, same-GPU, end-to-end protocol.

**Failures and fixes (the part that matters).** (1) Truth generation FAILED twice at
N=512: the incumbent generator's unpreconditioned BiCGStab (tol 1e-10, maxiter 2000)
stalls — job 2825764 (chunk 64) and job 2826193 (chunk 8, disproving my first batch-lane
hypothesis; 7/576 trajectories, mostly high-nu 0.06-0.09, worst Newton rel residual
1.37e-1). Fix adopted from the N=1024 agent (commit b5159a3 on its branch): exact
sine-basis Helmholtz preconditioner patched into the STAGED burgers2d_film only
(discretization, Newton guard, and 1e-8 truth gates unchanged and remain the arbiter);
verified here at N=32 against the original generator to 9.6e-16. Data then converged to
1.0e-12. (2) A near-miss worth recording: I ran a broad `pkill -f sep_burgers.py` that
could have killed sibling agents' local smokes — it did not, but kill by exact PID only.
(3) j2's tol-Newton baseline carries the same preconditioner (stated in its config); the
"fom_truth_newton8" timing key name is retained for compatibility even though the staged
generator is preconditioned — the config records this.

**Open items for the N=256 push (from this arm's evidence).** (a) Decoder capacity is the
binding constraint on BOTH PDEs (err==oracle everywhere): R=8K + n_ff=256 + 150k+ steps
is the first lever (j4's truncated training curve supports it). (b) The Burgers EQ
worst-row diagnostics grow with K (row_rel_max 2.0e3 at K24 vs 2.6e2 at K16) — watch.
(c) Any Burgers speed claim at any N must now face the preconditioned tol-Newton arm,
which exploits the same separable-rectangle structure the spectral Poisson warm start
does. (d) The e2e ROM cost is rollout-dominated (~60 ms = ~300 LM iterations' kernel
overhead), K-flat; shrinking it needs fewer implicit-step iterations, not smaller K.

## 2026-08-23

### N=256 separable-decoder scaling arm — closed (branch `exp/2026-08-23-sepdec-n256`)

**What ran.** Four cluster jobs (all A100, namespace `/cluster/tufts/paralab/tawal01/sepdec_n256/`,
now deleted): j1=2825729 (Poisson K8/K16-R64/K16-R128/K32, 100k steps), j2=2825730 (Burgers
K8/K16, 60k steps), j3=2827130 (Burgers K24/K32, 40k steps, 1h58m), j4=2827131 (Poisson
K16-R64 at 200k steps + K24, plus looser CG rungs 5e-1/3e-1). All COMPLETED 0:0; results
sha256-verified and committed under `experiments/separable-decoder/runs/n256_j{1..4}/`.
The session that ran j1/j2/j4 was lost mid-study; this closing session was a handoff that
re-verified everything from the JSONs. All numbers below are from the committed generated
tables `runs/SUMMARY-n256.md` / `runs/CROSSOVER-n256.txt` (scripts `summarize_n256.py`,
`crossover_n256.py`); per-cell gate0 = 0.0 (Poisson) / ≤3.4e-15 (Burgers), timed-vs-error
invocation latent deviation ≤~1e-8, recorded per HANDOFF measurement rules.

**Poisson N=256 — the ROM is on the winning side of the crossover, with caveats.** Best
config K=16 R=64 at 200k steps: cached solve 2.539 ms at mean rel-L2 2.893e-2 (heldout
seed-0 cohort, tau=1e-3). The cheapest same-job CG rung at or below that error is tol=1e-1
(8.156 ms, err 1.065e-2) -> ROM 3.21x faster at iso-accuracy-or-better, and j4's looser CG
rungs close the escape hatch: CG 3e-1 is 6.4 ms at 6.58e-2 and CG 5e-1 is 5.2 ms at
1.41e-1 — both slower AND less accurate than the ROM point, so no CG rung dominates it.
Caveats, ranked: (1) the matched CG rung is ~2.7x MORE accurate than the ROM — this is a
Pareto claim, not an equal-error win; (2) tau=1e-3 solves are 100% censored (all stop on
budget/stall, none on tolerance; tau=1e-2 is 56-75% censored at nearly identical error);
(3) fresh-seed-777 cohort degrades to 3.529e-2 mean (and K16-R128big degrades worst:
2.988e-2 -> 5.478e-2 with max 3.97e-1 — bigger R overfits the seed-0 family); (4) 16-case
cohorts, median-of-reps timing, single GPU type. Best raw accuracy: K=32 R=128 at 2.565e-2
heldout / 3.230e-2 fresh, but at 5.4 ms it loses most of the speed margin.

**Burgers N=256 — classical wins, cleanly, at every K.** Best ROM cell K=16 R=64: cached
end-to-end 106.85 ms (ic 17.77 + roll 86.66 + dec 0.19) at traj err 2.453e-2. Same-job
tolerance-Newton at 1e-3 is 60.5 ms at err 3.015e-4 -> ROM 0.57x (slower AND ~80x less
accurate); even Newton 1e-2 (50.3 ms, err 3.050e-2 mean but max 1.85e-1) matches the ROM
error class at half the cost. K=24/32 make it worse: e2e 139.0/175.1 ms (ratios 0.45x/0.35x)
because the IC latent fit grows superlinearly with K (cached ic ms 8.0/17.8/36.5/66.9 for
K=8/16/24/32). All rollout steps stop on 'stalled' (a few 'budget'), 0 blowups; the
over-solved truth generator (~1.6 s) was never used as the headline baseline.

**f(K) across all cells.** Error saturates by K~24-32 on both PDEs while cost keeps rising:
Poisson heldout 5.37e-2 / 3.14e-2 / 2.92e-2 / 2.57e-2 and Burgers 4.72e-2 / 2.45e-2 /
2.13e-2 / 2.09e-2 for K=8/16/24/32. The K16->K32 accuracy gain is ~1.2x on both PDEs for
~2x (Poisson) to ~1.6x (Burgers) more online cost. The binding constraint is decoder/EQ
quality, not latent capacity. EQ worst-row errors grow with K (Poisson row max 7.7e1 at K8
-> 2.1e3 at K32; Burgers 3.0e1 -> 1.8e3), confirming the inherited N=64 concern scales.

**Failures/retractions this arm.** None retracted; nothing exceeded budget (4 jobs
submitted, 4 completed). Known weak points recorded, not hidden: 100% censoring at Poisson
tau=1e-3; several K=24/32 Burgers IC fits landed at 7e-2-1.5e-1 latent-fit error (rollout
errors stayed <=4.2e-2, but the N=64 IC-fit fragility persists in milder form); fresh-seed
Poisson degradation above.

**Open items.** (1) Censoring: no tau reaches tolerance-terminated stops at N=256 — the
stopping rule needs rework before any tolerance-style claim. (2) Burgers IC fit is the
scaling bottleneck (an offline IC encoder is the obvious fix). (3) EQ row-max growth with
K unexplained. (4) Merge decision for the four sepdec-n* worktrees is pending the other
three arms. Per user redirect the four-resolution round is ENDING; a new focused N=256
push will run in a separate worktree by a different agent — this branch is the frozen
archive of the scaling arm.

## 2026-08-23

### N=128 separable-decoder scaling arm (sub-agent): flat Poisson f(K), honest Burgers loss to tolerance-Newton, j2 cancelled by redirection

One of the four per-resolution scaling agents (HANDOFF.md contract). Branch
`exp/2026-08-23-sepdec-n128`, worktree of the same name, cluster namespace
`sepdec_n128` (now deleted). Every number below is regenerated by
`experiments/separable-decoder/summarize_n128.py` into `SUMMARY-N128.md` on that
branch — read it there, not here, if precision matters.

**Measurement rules implemented (all 8 audit fixes, both PDEs).** Poisson's timed
pipe now charges the online source projection (verified == `weak_source_term` to
<1e-12) and the full-grid decode; Burgers timing is END-TO-END = batched-LM IC
fit (u0 only; inits = nearest training t=0 codes via offline QR of the readout
bank) + incumbent `rollout_jit` + full-grid decode of all 51 states, with the
(ic, roll, dec) split reported. Errors/counters come from the LAST TIMED
invocation; first-vs-last-rep deviation recorded (0.0 Poisson, ~1e-10 Burgers vs
untimed incumbent rollout). All raw reps retained; balanced unit ordering
(reversed on odd reps); stop-reason distributions in every row; fresh-seed
Poisson cohort (seed 1) beside the held-out seed-0 cohort; strong classical
Burgers baseline in-job = tolerance-terminated Newton (same residual/BiCGStab/
NaN-guard as the truth generator) beside the fixed-8 truth rollout (labelled
over-solved, never a comparator). The new IC fit matches the incumbent
`bc.fit_ic` to 4 digits on every checked trajectory. GATE 0 asserted in all 9
cells: 0.0 (Poisson x4-of-9), 9.7e-16..2.9e-15 (Burgers).

**j1 = Slurm 2825804** (A100 pax010-class node pax105, COMPLETED 1h44m, 8/8
cells, N=128 seed 0, 60k steps, R=6K, n_ff=128, REPS=7). Findings:

- *Poisson f(K) is FLAT.* Held-cohort cached solve at tau=1e-3: 4.99e-2 (K=8,
  3.2 ms), 3.06e-2 (K=16, 3.1 ms), 2.99e-2 (K=24, 5.2 ms), 2.86e-2 (K=32,
  6.3 ms). Same-job CG ladder: cheapest rung tol=1e-1 is 3.75 ms at 1.73e-2.
  So at N=128 the RAW wall-clock crossover has happened (K=16 ROM is 1.21x
  faster than the cheapest CG rung; it was slower at N=64) — but iso-accuracy
  still favors CG, since the ROM error is ~2x the loosest rung's. tau=1e-3
  remains 100% censored (all 'stalled').
- *Poisson error is objective/representation-limited, with a cohort split.*
  Held oracle ~3.5-3.7e-2 =~ solve error (manifold-limited); FRESH-cohort
  oracle is much lower (1.20-1.64e-2) yet the fresh solve still lands at
  ~2.8-2.9e-2 — on fresh sources the weak-EQ objective loses ~2x vs the
  representation ceiling. Raising K does not close it.
- *Burgers f(K) improves with K and tracks the training floor.* End-to-end
  cached: 3.17e-2 @ 91 ms (K=8), 2.25e-2 @ 85 ms (K=16), 2.13e-2 @ 97 ms
  (K=24), 1.63e-2 @ 118 ms (K=32); recon floors 3.4/1.9/1.5/1.2e-2. Split:
  ic ~6-12 ms, decode ~0.2 ms, latent rollout dominates. No IC-fit failures
  (worst ic_rel 0.183 vs the N=64 cell's 0.36 outlier) — the nearest-t0-code
  init selection fixed the inherited defect. All steps still stop 'stalled'
  (394-400 of 400); zero blowups, 50/50 steps, 8/8 fresh-seed trajectories.
- *The honest Burgers speed verdict at N=128 is a LOSS.* Tolerance-Newton at
  ntol=1e-3 does the same 50-step rollout in 68.4 ms at 3.0e-4 error — faster
  AND ~50x more accurate than every ROM cell. ntol=1e-2: 68 ms @ 3.16e-2
  (comparable error to the ROM, still faster). Only the over-solved fixed-8
  truth generator (908-948 ms) is slower than the ROM; quoting THAT ratio
  (~11x) is exactly the N=64 audit's mistake and is not a supported win. The
  N=64 3.3x headline does not survive at N=128 against a strong baseline.
- *EQ per-row diagnostics grew with N* (recorded per HANDOFF): Poisson row
  p95 0.10-0.27 (rel fit 6.6e-4..1.9e-3), Burgers p95 0.33-0.44 with row max
  up to 1.0e+4 (K=16) despite small global fits.

**j2 = Slurm 2827226 (accuracy push: 200k steps, R=8K, 16k snapshots,
Poisson m=32K probe): CANCELLED at 2h35m by user decision** — the study was
redirected to a focused N=256 push mid-run. Cell 1 of 5 had completed
(Burgers K=16 R=128 200k/16k-snaps): recon 1.93e-2 -> 1.37e-2 and rollout err
mean 2.25e-2 -> 1.67e-2 at 80.5 ms e2e — j1's K=32 accuracy at K=16 cost, so
longer training DOES buy Burgers accuracy (sublinearly vs the recon floor).
Partial results pulled and committed as `runs/sepdec_n128_j2/` with a
CANCELLED.md; its K=32-200k Burgers cell and all three Poisson-200k cells
(incl. the m=32K EQ probe) never ran. Remote namespace
`/cluster/tufts/paralab/tawal01/sepdec_n128/` fully deleted.

**Failures/retractions this session.** (1) First submission 2825787 died in
2 s on a bash expansion-order bug (`local k=$1 r=$((6*k))` evaluates the
arithmetic before k is assigned) — nothing ran, fixed and resubmitted. (2) My
earlier in-flight read that "200k training barely moved rollout error" was
wrong — it was based on the first 4 trajectories scrolling by; the full-cell
mean moved 2.25e-2 -> 1.67e-2. (3) Operational landmine for multi-agent
sessions: the four sibling agents share one scratchpad, and our local smoke
outputs clobbered each other in `scratchpad/smoke/` until I moved to an
agent-specific subdir — sibling smoke JSONs overwrote mine mid-validation.

**Open.** N=128 arm is closed (2 of 4 job slots used). Unrun but staged-ready:
Burgers K=32-200k, Poisson 200k cells, Poisson m=32K probe (run_j2.sbatch on
the branch documents them). The Poisson fresh-cohort objective gap (~2x above
oracle) is the most interesting unexplained number. Merge decision for the
four sepdec-n* worktrees still pending with the user.

### N=1024 separable EQ-decoder scaling arm (sub-agent session)

Branch/worktree `exp/2026-08-23-sepdec-n1024`, cluster namespace
`/cluster/tufts/paralab/tawal01/sepdec_n1024/` (pulled, checksummed, committed, DELETED).
Six submissions, ended early by user redirect to a focused N=256 push: 2825734 (Poisson,
FAILED 3m, truth-guard), 2825838 (Poisson, TIMEOUT 4h05, 2/3 cells complete), 2825735
(Burgers, FAILED 28m, truth not converged), 2826213 (Burgers, COMPLETED 1h50, both cells),
2827675 (Burgers push, OOM 15m), 2827878 (Burgers push, user-CANCELLED at 1h14, cell 1 at
34k/60k train steps, no artifacts saved). Results in
`experiments/separable-decoder/runs/sepdec_n1024_{j1,j2,j3}/` on the branch; generated
tables in `SUMMARY-N1024.md` (script `summarize_n1024.py`, no hand-typed numbers). All
mandatory measurement rules from AUDIT-2026-08-23/HANDOFF implemented: end-to-end timing
(Burgers includes the IC fit, split reported; Poisson includes source->weak projection,
verified vs incumbent at 2.6e-16), full-grid decoded outputs on the timed paths, raw
timing reps retained, balanced AB/BA pairs, timed-vs-error deviation recorded (0.0
Poisson, <=3.8e-9 Burgers), stop-reason distributions, fresh-seed Poisson cohort, per-row
EQ diagnostics. Gate 0: 0.0 (Poisson x2), 8.4e-15/4.8e-15 (Burgers).

**Poisson N=1024 (job 2825838, A100, K=8 R=32 and K=16 R=64, 50k steps): the CG crossover
flips hard, but the exact spectral control still wins.** Cached end-to-end solve (source ->
projection -> trust-LM -> full 1022^2 decode) K=16: 3.0-3.5 ms, err 3.48e-2 held-out /
5.06e-2 fresh-seed cohort (oracle 4.22e-2/6.39e-2 on n=4 -- solver saturates the manifold;
solve errs are n=16). Same-job balanced-pair CG ladder: loosest (tol 1e-1) 98.4 ms at err
4.9e-3, tightest (1e-6) 223 ms -> the ROM is 31x faster than the CHEAPEST CG run (which is
still 7x more accurate than the ROM); ROM cost is N-independent (2.06 ms at N=64 -> 3.2 ms
at N=1024) while CG grew 1.31 -> 98 ms. BUT the exact dense-spectral solve (same job,
balanced-paired) is 0.64 ms at err 7e-15: the ROM LOSES 5x to the structure-exploiting
classical solver, consistent with main's spectral-warm-start finding. Caveats ranked: (1)
no CG point exists at the ROM's error level, so "31x" is at-unmatched-accuracy (ROM worse);
(2) tau=1e-3 rows 100% censored (stalled), tau=1e-2 69-88%; (3) fresh-seed errors ~1.45x
the held-out cohort's (mild model-selection contamination of the seed-0 cohort is visible);
(4) truth CG hits the f64 floor at rel residual 4.3e-10 (guard made explicit at
FOM_RES_TOL=1e-9; CG tol/maxiter unchanged). f(K): K=8 err 5.82e-2 at 3.5 ms; K=16 3.48e-2
at 3.2 ms -- error improves with K at flat cost; K=16 R=128 cell died at the 4h wall,
K=24/32 never ran. Meshfree arm solves are 14.7 ms vs cached 3.0 ms: caching is now a ~5x
lever at N=1024 (was ~12% at N=64).

**Burgers N=1024 (job 2826213, H200, K=16 R=64 and K=8 R=32, 40k steps, 104 trajectories,
2048 states): honest end-to-end 5.1x over the STRONG classical baseline, but accuracy is
training/coverage-limited and got worse with N.** Cached end-to-end (u0 -> span-split IC
fit -> 50 implicit LM steps -> full 51-state decode): K=16 59.7 ms = 3.96 IC + 56.4
roll+decode; K=8 53.8 ms. Balanced-paired against the tolerance-terminated Newton ladder
(Helmholtz-preconditioned BiCGStab, same residual/stencil as truth; at its 1-Newton-per-
step floor ntol=1e-2 it reaches err 4.9e-4 in 310 ms): 5.1x (K16) / 5.7x (K8); vs its
ntol=1e-4 rung (err 1.1e-4, 599 ms): 9.9x. The truth generator (fixed-8-Newton) at 1587
ms/traj is recorded as OVER-SOLVED reference only, never a headline. Caveats ranked: (1)
ROM err 1.17e-1 (K16) / 1.57e-1 (K8) vs baseline 4.9e-4 -- nowhere near iso-accuracy, and
WORSE than N=64 (5.2e-2), because training coverage collapsed (104 traj/2048 states vs
576/8192 at N=64; memory-driven subsampling, indices recorded) and IC fits degraded
(4.3e-2 .. 5.5e-1 per trajectory); (2) all 200 steps stopped on 'stalled', zero on 'tol',
zero blowups; (3) 4 test trajectories, one GPU; (4) EQ worst-row error grew to 1.5e3 (K16;
92 at N=64) -- the inherited worst-row growth is real for Burgers; (5) f(K) here: K=16
beats K=8 on error at ~equal cost.

**Fixes that cost jobs (shareable to the other scaling arms).** (a) Poisson truth guard:
`ms_parametric.build_snapshots` asserts rel residual < 1e-10, unreachable at N=1024 (f64
CG floor 4.31e-10) -> env-configurable FOM_RES_TOL patch applied by stage.sh to STAGED
copies only. (b) Burgers truth generation: unpreconditioned BiCGStab (LIN_TOL 1e-10,
maxiter 2000) stalls at N=1024 (max Newton rel res 8.67e-2, job 2825735) -> staged
burgers2d_film patched with the exact sine-basis Helmholtz preconditioner
(I+dt*nu*(-lap_h))^-1 on the interior; discrete residual, Newton guard, and <=1e-8 truth
checks unchanged; verified vs the original generator at N=64 to 7.6e-16 state agreement;
post-fix N=1024 truth residual 9.95e-13. Same preconditioner strengthens the tol-Newton
baseline. (c) OOM (job 2827675, RSS 245 GB): the training jit closed over the 25.7 GB
data array (captured-constant landmine) and sep_burgers held a 71.6 GB S_all beside the
71.6 GB U -> data/coords now explicit jit args, direct row gather, U freed.

**Retracted/incomplete.** Poisson K=16 R=128 (timeout) and the whole Burgers accuracy
push (K=16 R=128, K=24 R=96, 168 traj/3072 states/60k steps -- j3, cancelled) produced no
numbers; j4 (Poisson capacity push + K=24/32) never ran. The Burgers accuracy story at
N=1024 is therefore an under-trained lower bound, not a capability ceiling. No speedup
claim here is iso-accuracy; every ratio above is at the ROM's own (worse) error level.

**Open for whoever picks this up.** (1) Burgers coverage/capacity push (j3 config
committed as `cluster/run_j3.sbatch`, all three fixes in the committed stage) -- the
N=256 arm reported 2.5e-2 errors, so the gap is training, not method. (2) Poisson
K=16 R=128 rerun + K>=24 f(K) points. (3) IC-fit representation failure (worst traj
5.5e-1) persists -- more trajectories should fix it before any encoder work. (4) Merge
decision for the four sepdec-n* branches is with the coordinator/user.

## 2026-08-23

### N=256 push session (branch exp/2026-08-23-n256-push) — round 1 + POD floors, rounds 2 in flight

Dedicated push session on worktree `worktrees/2026-08-23-n256-push`, cluster namespace
`/cluster/tufts/paralab/tawal01/n256_push/`. Mission evolved mid-session by user
directive: first "push accuracy+speed at N=256", then "target rel-L2 ~1e-3
(architecture+optimization authorized, PURE NEURAL, SVD as diagnostic only)", then
"BURGERS is the primary 1e-3 target; Poisson becomes the control". All in
`experiments/separable-decoder/PUSH-PLAN.md` on the branch.

**Round 1 (jobs 2828682 Poisson, 2828683 Burgers, A100s, pulled+committed, remote dirs
deleted).** Solver-termination repairs (budget 60->300, restart-on-stall x6, adaptive
trust region), EQ arms (m 256->512, M 64->128, tail-reweighted NNLS, per-query adaptive
quadrature), IC repairs (48-candidate/3x-budget fit, offline u0->z encoder), and
2-step safeguarded warm-start extrapolation. Verdict, both PDEs: **accuracy moved by
NONE of it** — solver output == weak-EQ optimum == representation oracle (Poisson: M=128
weak-opt equals the L2 oracle to 4 digits; Burgers: single-step weak-opt from the true
previous state's oracle equals the per-state oracle at every probed t; no error
compounding over 50 steps). REPRESENTATION is the only binding rung at K16/R64
(Poisson solve 2.89e-2 held / 3.53e-2 fresh; Burgers traj 2.47e-2). Speed levers that
ARE real, identical error: Poisson f_m->z encoder init 2.8->2.0 ms; Burgers encoder IC
20.9->3.5 ms (jac 136->9) and extrapolated warm start 94.5->55.4 ms rollout (jac
352->241). Adaptive-TR restarts: 3x jacobians for nothing — dropped.
**Retraction note:** the r1 "IC fit is a major accuracy lever" hypothesis (from j2's
ic_rel up to 1.6e-1) is DEAD — the IC fits sit exactly ON the t0 representation floor;
high-budget IC changed nothing.

**POD-floor diagnostic (f64 Gram; `pod_floor_n256.py`; Poisson local GB10, Burgers
cluster job 2829709 after the GB10 attempt proved too slow and was killed).**
DIAGNOSTIC ONLY — no POD in any model. Poisson fresh-cohort floors onto the
512-snapshot training span: R=64 3.2e-2 (the r1 oracle sits ON it), R=256 4.0e-3,
R=512 8.1e-4; onto a 2048-sample dense family: R=256 2.7e-3, R=512 4.8e-4 mean.
Burgers fresh test-state floors (8192-state training basis): R=64 1.7e-2 (r1 oracle
on it), R=128 4.6e-3, R=256 7.9e-4, **R=512 7.0e-5 mean / 2.3e-3 max**, R=1024 1.9e-6.
So Burgers 1e-3 is single-span feasible at R=512 — no multi-bank needed. Structural
note recorded in-session: ANY cached-bank architecture (multi-bank, z-modulated mixing
included) evaluates as G(x)·H(z), so its oracle is bounded by the POD floor at total
bank width; multi-bank buys trainability, never a better floor.

**Round 2 in flight at session-log time:** r2a/r2b Poisson control (2829746/7, A100,
R=512, S=512+2048 seed-4242 extras, multi-scale-vs-single-scale FF pair, v2 trainer
with point-subsampled AdamW+EMA, EQ certification M in {64,128,256} + tail) and r2
Burgers primary (2829766, H100, R=512, 16384 states with all t<=5 early states,
multi-scale FF, STEP_TOL ladder {1e-9,1e-6,1e-5}, per-step error tables, per-set
single-step weak-opt vs oracle). Results land on the branch when pulled.

## 2026-08-25

### N=256 push, rounds 3-4: the span fix, the h diagnosis, a retracted crossover, and the speed win

Branch `exp/2026-08-23-n256-push` (carries N=1024 work despite the name), cluster
namespace `n256_push`. Continues the round-1/2 entry above. Reports on `main`:
`reports/2026-08-24-separable-decoder-architecture-and-results.md` (commit ee6b3e4);
on the branch: `PROFILE.md`, `CROSS-RESOLUTION.md`, `runs/SUMMARY-R3.md`, all generated.

**Round 3 (r3a 2835788, r3b 2835789, r3c 2835790, r3d 2835794 — all COMPLETED).**
Fixed the rank cap found at the end of round 2 (g's last layer is linear, so span rank
<= g_hidden+1; nominal R=512 runs had numerical rank exactly 257). `G_HIDDEN=2R` gives
rank 512/512. Span least-squares floor on fresh test states is now 2.60e-4 (N=1024) /
2.12e-4 (N=256, r3a) / 1.15e-4 (R=1024) — all BELOW the 1e-3 target.
**But accuracy did not improve**, which is the informative result: R was never the
limitation. The binding rung moved to **h**: oracle 8.24e-3 against a span floor of
2.60e-4 at N=1024, i.e. the K=16 map reaches ~1/32 of what its own span allows. The same
gap is 36.8x at N=256 — **resolution-independent**, so it is h's capacity/optimization,
not anything about the grid, discretization or quadrature.

**A landmine re-found and fixed.** A ~2 GB device array closed over by a `jit` is lowered
as an HLO literal: +10 GB host RSS and 16 s compile PER JIT, vs 0.087 s as an explicit
argument. `train_autodecoder_v2` closed over the training array (8.6 GB at N=1024) and the
driver closed over `G_all` (4.3 GB) in ~8 jits. This is almost certainly what OOM-killed
the earlier N=1024 arm at 245 GB RSS and forced its ~5x coverage cut. Banks and data are
now explicit jit arguments everywhere; numerics unchanged. N=1024 then trained with FULL
coverage and its error fell 1.17e-1 -> ~6e-3.

**RETRACTION — the N=1024 Burgers crossover (5.2x) reported on 2026-08-24 is withdrawn.**
That number compared the ROM against `nt=1e-2, lt=5e-4` (310 ms), a rung ~24x MORE
accurate than the ROM and over-tight in its INNER tolerance; `lin_tol`, not `newton_tol`,
was carrying it. Against the properly swept (newton_tol, lin_tol) ladder in the same job,
the cheapest classical rung at least as accurate as the ROM was 63.1 ms vs ROM 72.9 ms =
**0.86x, classical wins**. Cause: the report generator only looked for baselines in
`rows[]`, while the n512/n1024 arms store them under `fom_newton_tol`/`timing.summary` and
`fom_tolnewton`; fixed, and recorded as a retraction in the report itself.

**Round 4 speed (r4s1024 2837055, r4s256 2837056 — no retraining, frozen r3 checkpoints).**
Profiled first rather than guessing. The rollout is **kernel-count / dispatch bound**, not
bandwidth or compute bound: per-LM-iteration marginal cost 0.140 ms, residual+Jacobian
0.123 ms while moving ~5 MB (~70x above the bandwidth bound), dozens of fusions for
~0.1 MFLOP. Both earlier hypotheses (naive host relaunch; bandwidth-bound skinny matmuls)
were wrong. Two levers, on a frozen decoder:
 - **Gram-space IC fit**: 18.08 -> 1.93 ms (~9x), matching the incumbent full-grid fit to a
   latent deviation of 3.27e-13.
 - **Adaptive stall tolerance** (1e-3): cuts iterations the solver was spending chasing a
   tolerance representation error makes unreachable (399/400 steps stopped 'stalled',
   0 on 'tol').
Result at N=1024: ROM e2e 60.8 -> 24.6 ms (2.5x) at unchanged error (0.15% shift).
**Matched-accuracy paired AB/BA: ROM 26.76 ms vs tol-Newton 43.10 ms = 1.61x, ROM wins.**
Batch-16: ROM 4.19 vs classical 49.17 ms/query = 11.74x. At N=256 the same recipe gives
0.40x single-query and 1.01x at batch 16 — the crossover is genuinely a large-N effect.
**Honest negative:** hard-capping LM iterations is NOT a valid lever (budget<=5 collapses
error to 6-8e-1); only the adaptive stall criterion is safe. Extrapolation order 1.0 beats
0 and 1.5.
**Caveat recorded in the generated file:** both sides of the batched columns are vmapped,
and a vmap of a `lax.while_loop` runs until every lane finishes, so each batched solve pays
its batch's worst-case iteration count — which penalises the classical solver more than the
ROM (visible as classical per-query cost worsening from B=1 to B=2). **11.74x is an upper
bound**, not a measurement against best-possible classical batching. The single-query 1.61x
is unaffected.
**Provenance error caught:** the first N=256 speed point used r3b's checkpoint
(`snap_norm=True`), off-recipe relative to the N=1024 point; being redone as r4s256a.

**In flight at checkpoint time (9 jobs, all sized to clear the 06:00 maintenance window):**
r4x128 (2837076) and r4x512 (2837077) — train + in-job speed sweep at the reference recipe
to complete a four-point resolution curve; r4s256a (2837078) — the recipe correction; and
six accuracy arms 2837061-2837066 attacking h (wider/deeper h, code refinement, K=24/32,
early-time loss weighting).

**Open / next.** Accuracy is the sole remaining gap: ROM rollout ~6e-3 (M=256 EQ set) to
~2.6e-2 (coarse M=64 set used in the speed sweep) against a classical rung at ~8e-4. To
reach 1e-3: (1) fix h — capacity, latent-code convergence, K=24/32 (nearly free, the solve
is dispatch-bound); (2) weight training toward the sharp early-time states, where the error
concentrates (oracle 3.0e-2 at t=0 vs ~5.5e-3 late; t<=5 mean 1.83e-2 vs 6.09e-3 after);
(3) fix the EQ ceiling — the M=64 set's own rel fit is 6.08e-3 with worst row ~9.8e2, so it
cannot support a 1e-3 claim; tail-capped NNLS and larger m are needed; (4) only then
re-tune the stall threshold. A dedicated worktree `2026-08-25-burgers-accuracy` continues
this; the four-point curve and the accuracy-arm verdict land on the n256-push branch.

### Round 5, Burgers accuracy campaign: the gap is generalisation in mu, and four inherited beliefs were wrong

Branch `exp/2026-08-25-burgers-accuracy` (disposable worktree
`worktrees/2026-08-25-burgers-accuracy`), cluster namespace `burgacc`. Generated
files on that branch: `experiments/separable-decoder/HFIT.md`
(`runs/gen_hfit.py`) and the report
`reports/2026-08-25-burgers-h-generalisation-wall.md`
(`reports/gen_2026-08-25-burgers-h-generalisation.py`). Thirty-plus h arms,
three end-to-end confirmations, two coefficient extractions, all pulled,
sha256-verified, `jax_backend=gpu`, stage manifests checked, remote dirs deleted.

**The method that made the round possible.** With the bank `G` frozen,
`Gram = G^T G`, `c*` the span least-squares coefficients of a target and `f` its
residual, `||G h(z) - u||^2 = (h(z)-c*)^T Gram (h(z)-c*) + f^2` EXACTLY, because
`G c*` is the orthogonal projection. So fitting h against full fields and
against precomputed coefficients in the Gram metric are the same problem, at
`O(S r^2)` instead of `O(S n_pts r)`, and reconstruction, the span floor and the
representation oracle all become computable from a 70 MB file instead of the
data (137 GB at N=1024). Whitening (`Gram = L L^T`, `q = L^T h`) folds exactly
into h's last linear layer, so the model class is unchanged while the badly
conditioned bank (cond(G) 2.64e4 at N=256) stops distorting the optimisation.
`sep_coeff_extract.py` runs once per (N, checkpoint) in 140 s at N=256 / 20 min
at N=1024; every h arm after that is minutes. The identity is gated by computing
the source checkpoint's reconstruction two independent ways on 64 retained
full-resolution states: mean deviation 3.67e-12 (N=256) / 6.52e-12 (N=1024).
The extractions reproduce the cluster's own numbers to every digit those jobs
printed — N=256 recon 7.171140e-03 against r3a's 0.007171139529778487, span
floor 2.124e-04 against r3a's 0.00021236444373757098; N=1024 span floor
2.601e-04 against r3d's 2.601e-04.

**FOUR INHERITED BELIEFS ARE WRONG, and each was retiring a lever or wasting one.**

1. *"The h gap is h's capacity/optimisation."* It is not: it is generalisation.
   On the frozen r3a N=256 bank, capacity moves the TRAINING fit by 5x
   (recon 7.17e-3 -> 1.39e-3 at h=2048x4) and the FRESH oracle by at most 1.2x
   (7.81e-3 -> 6.34e-3). Capacity past h=1024x3 reverses outright (h=2048x4
   gives the third-best training fit and 65.1x against the span floor).
2. *"The latent codes may not have converged, so h may be fitting noise."*
   False. Re-solving every training code exactly at frozen h changes the
   incumbent's reconstruction by a factor of 1.000. Adding a
   nearest-training-state initialisation chosen USING the test target changes
   no oracle either. Both were levers 2 and part of lever 1 in the handoff.
3. *"K is not the constraint — the family is intrinsically ~6-dimensional."*
   False, and K is the strongest single knob measured. Oracle / span floor at
   N=256, h=1024x3, 576 trajectories: 58.1x, 29.8x, 26.9x, 23.9x, 18.6x, 16.7x,
   13.0x, 9.7x for K = 8, 16, 24, 32, 48, 64, 96, 128 — monotone and still
   falling at K=128 (oracle 2.062e-3). The same slope appears at N=1024 on the
   r4a6 bank (32.6x / 28.1x / 23.5x for K=32/48/64). Intrinsic dimension bounds
   the DATA manifold; what the oracle needs is for h's IMAGE to pass near fresh
   states' coefficients, and a bigger image does that better.
4. *"K is nearly free online per the round-4 cost model."* False, and this is
   the one that decides the campaign. Job `cf256` ran the round-4 protocol
   unchanged on a K=48 decoder at N=256, control EQ set: rollout error
   2.751e-2 -> 9.232e-3 (a 2.98x gain that DOES survive from the oracle into
   the rollout) at 35.60 -> 135.75 ms, a **3.81x slowdown**; matched-accuracy
   paired 0.39x -> 0.12x. 254 Jacobians per trajectory against 164, each with
   three times the tangents. The N=1024 crossover is 1.61x and cannot absorb
   3.81x.

**The lever that is free at solve time is mu-density, and it is a power law
that saturates.** The canonical draw is 576 trajectories for a five-parameter
family plus time — about 3.5 samples per parameter dimension. Fitting the same
arm on the first F canonical trajectories and grading every F on one reserved
cohort (trajectories 432-575, fitted by no arm): held-out oracle 7.21e-2,
4.14e-2, 2.39e-2, 1.49e-2 at F = 72, 144, 288, 432 for h=1024x3 (fitted
exponent 0.86) and 2.02e-2, 1.36e-2, 1.08e-2, 9.47e-3 for h=256x2 (exponent
0.42). Note the crossover: at 72 trajectories the SMALL h generalises 3.6x
better than the wide one, at 432 the wide one is worse still — capacity has to
be scaled with density, not raised on its own.
Appending 4032 trajectories from seed 1000 to the canonical 576 (the canonical
draw and the cohort definitions untouched) and grading on trajectories
4096-4607: held-out oracle 1.375e-2 -> 8.017e-3 -> 6.013e-3 at 576 -> 1152 ->
4096 for h=1024x3, i.e. exponent 0.78 over the first doubling but only 0.22
over the last — **the density lever saturates once h's capacity binds again**,
and the same arm's training reconstruction degrades 2.22e-3 -> 5.85e-3 as it is
asked to fit seven times more states. K=24 at the same densities gives
1.265e-2 -> 6.851e-3 -> 4.915e-3.

**Best decoder of the campaign, and its price.** Job `cfd256`: 4608
trajectories, K held at the incumbent 16, h=1024x3, 200k steps, refit on the
frozen r3a bank, then the round-4 protocol with the control EQ set at N=256.
Oracle 2.747e-3 (12.9x the span floor, against the incumbent's 36.8x), IC fit
3.067e-2 -> 8.496e-3, **rollout error 2.751e-2 -> 5.022e-3, a 5.5x improvement**
— and 35.60 -> 73.99 ms, 2.08x slower, matched-accuracy paired 0.39x -> 0.23x.
It dominates the K=48 arm on both axes. The remaining slowdown is h's WIDTH,
not the data and not K (this arm also widened h from the incumbent's 256x2);
job `dr256` isolates that with h=512x2 and 256x2 at full density.

**The quadrature "hard blocker" is real but is not where the handoff said, and
the statistic that motivated it is misleading.** In the sibling's r4a6 job
(N=1024, K=32, tail-capped NNLS already applied, cap 3e-2, 2 rounds), the
single-step weak-EQ optimum tracks the L2 oracle as follows, by time index:
at M=256/m=1024, ratio mean 2.256 (max 9.774) at t=1, then 1.040, 1.017, 1.004,
1.001, 1.001, 1.001 at t = 2, 3, 5, 10, 25, 50. The control set (M=64, m=256)
is 3.271 at t=1 and 1.026 by t=5. So **objective truncation is confined almost
entirely to the FIRST STEP off the sharp blob initial condition**; from t=2
onward the EQ objective is not binding at all at the 3e-3 error scale. Meanwhile
the row-tail statistic that motivated the blocker points the WRONG WAY: the
M=256 set has a far worse worst row than the control (9.41e3 against 1.43e2)
and yet tracks the oracle 1.4x BETTER. Tail-capped NNLS was already in these
runs and did not remove the tails. The right certification metric is
weak-optimum-versus-oracle tracking, not the row tail, and the right fix is a
first-step-specific quadrature, not a globally finer one.

**Where the error lives inside a trajectory.** The bank's own span floor on
fresh test states, by time index at N=256: 1.049e-3 at t=0, 5.03e-4 at t=1,
3.53e-4 at t=5, 2.75e-4 at t=10, 1.09e-4 at t=50 (mean 2.124e-4; t<=5 mean
5.08e-4, t>5 mean 1.73e-4). At N=1024: 9.23e-4 at t=0 down to 1.03e-4 at t=50.
Every arm's oracle has the same shape an order of magnitude above it — even at
K=128 the oracle is 1.270e-2 at t=0 against 1.337e-3 at t>5. The sharp blob IC
is the hardest state for BOTH the bank and the map. Because a trajectory error
is a mean over 51 states this does not by itself block a 1e-3 mean, but it
blocks any per-state or worst-state claim, and t=0 plus t=1 is exactly where
both the span floor and the quadrature are worst.

**Levers measured and found neutral or harmful** (so nobody spends a retrain on
them again): latent Fourier features of the code (best training fit in the
table, oracle 260x-590x — an image that wiggles between training codes),
capacity beyond h=1024x3, code jitter at sigma = 0.02 and 0.05 (47.0x and
120.3x against 29.8x), code-only polish (43.6x), per-snapshot versus global
loss normalisation, weight decay at 1e-6/1e-4/1e-3 (oracle unchanged to four
digits at both K=16 and K=48), 25k-step early stopping (55.9x), early-time loss
weighting inside the coefficient fit (30.6x against 29.8x), better latent
initialisation, and more span (R=1024 widened the gap, an inherited result this
round confirms the mechanism for).

**Honest verdict on 1e-3.** Not reached, and not reachable by this round's
levers at preserved speed. The best measured rollout is 5.022e-3 at N=256, 5.5x
better than the incumbent, at 2.08x its cost; the classical ladder still wins
at N=256 (0.23x). Reaching 1e-3 needs the oracle at roughly 5e-4 (rollout is
1.4-1.9x the oracle in every configuration measured), which is another factor
of 5.5 below the best oracle here. Density alone will not deliver it — it
saturates — and K will, but at a cost the 1.61x N=1024 crossover cannot pay.
The open question the campaign hands on is whether capacity and density scaled
TOGETHER break the saturation at a cost the crossover can absorb; that is what
`dc256` (capacity sweep at 4096 trajectories) and `dr256` (cost-matched h at
full density) were run to answer, and what `dn1024` (2837322) and `dn256b`
(2837323) will answer at full scale — both queued with `--begin` after the
2026-08-25 06:00-12:00 maintenance reservation and NOT yet run.

**Engineering notes.** Three submission waves were lost to avoidable causes and
each is now fixed in the committed scripts: an identity gate set tighter than
f64 round-off allows given cond(Gram) = cond(G)^2 (fixed with one step of
iterative refinement through G, which took the deviation from 2.04e-8 to
1.37e-11, plus bars set to catch a formulation error rather than police
round-off); a stage script with an enumerated file list that omitted the driver
(now globs `sep_*.py`); and the captured-constant landmine, which
`sep_coeff_extract.py` avoids by passing the 4.3 GB bank as an explicit jit
argument.

**ADDENDUM (04:45), the result the entry above was written just before: a
genuine Pareto improvement, not a trade.** Job `dr256` isolated what `cfd256`
confounded — that arm changed the data AND h's width together. Holding K at the
incumbent 16 and h at 512x2, and spending the whole gain on 4608 trajectories:

```
decoder                        K   h        traj  rollout err  e2e ms  paired
round-4 incumbent (r4s256a)   16   256x2     576    2.751e-2    35.60   0.39x
round 5 dense, modest h       16   512x2    4608    8.961e-3    45.15   0.40x
round 5 dense, wide h         16   1024x3   4608    5.021e-3    73.99   0.23x
round 5 K=48                  48   1024x3    576    9.232e-3   135.75   0.12x
```

all four measured on the round-4 protocol at N=256 with the CONTROL EQ set and
matched-accuracy paired AB/BA against the swept classical ladder. **The ROM is
3.07x more accurate at an UNCHANGED position against the classical ladder**
(0.39x -> 0.40x): it costs 1.27x more, and so does the classical rung it now has
to be matched against (13.81 ms at 8.377e-4 becomes 18.26 ms at 3.538e-4). The
wide-h arm is the campaign's most accurate decoder, 5.48x better than the
incumbent, but pays 1.7x of the ratio for it. The K=48 arm is strictly dominated
— worse error than dense/512x2 at three times the time.

So the campaign's answer to "more accuracy without giving back the speed" is:
**mu-density plus a modest h widening, at the incumbent K**. 3.07x, free in
ratio terms; beyond that every further gain measured costs ratio.

Also corrected here: the entry above says "there is no lever that is free at
solve time"; that is too strong. Density is free, h's WIDTH is cheap-but-not-free
(256x2 -> 512x2 costs 1.27x), and K is expensive (K=48 costs 3.81x). The three
are ordered, and only the first two are usable at the current crossover.

The two post-maintenance jobs were re-aimed at this recipe before requeueing:
`dn1024` (2837430) and `dn256b` (2837431), both `--begin 12:10`, arms
`mid,wide,k32` with `EMIT=mid` and the fine EQ set for the accuracy grade. The
N=1024 confirmation matters most, because that is the only resolution where the
ROM currently wins (1.61x) and therefore the only one where a 1.27x cost has
room to be absorbed.

**ADDENDUM 2 (05:15), the fine-quadrature grade and a correction to this
entry's own quadrature verdict.** Job `fq256` re-measured the cost-matched
decoder (dense, h=512x2, K=16, 4608 trajectories) on M=256 / m=1024 with its own
timing and pairing in the same job:

```
EQ set          EQ rel fit   rollout err   e2e ms   matched ms   paired
M=64,  m=256      ~4e-3       8.961e-3      45.15     18.26      0.40x
M=256, m=1024     5.42e-4     6.184e-3      55.65     18.07      0.32x
```

gate 0 3.21e-15 on both. **The single-step diagnostic above understates what
the quadrature costs a ROLLOUT.** "Truncation is ~1.00 from t=2 onward" is true
of one step and false of fifty: the first-step error propagates, and the same
decoder is 1.45x more accurate on the fine set. The quadrature is worth about a
third of the remaining rollout error — and, like every other lever this round,
it is bought with speed. What survives from the original verdict is narrower
but still useful: the fix indicated is a FIRST-STEP-specific quadrature, and the
row-tail statistic remains the wrong certification metric.

**Final ladder for the campaign, N=256, all rows the round-4 protocol with
matched-accuracy AB/BA pairing:**

```
decoder                       K   h       traj  EQ     rollout err  e2e ms  paired
round-4 incumbent            16  256x2     576  M=64     2.751e-2    35.60   0.39x
round 5 cost-matched         16  512x2    4608  M=64     8.961e-3    45.15   0.40x
round 5 cost-matched, fineEQ 16  512x2    4608  M=256    6.184e-3    55.65   0.32x
round 5 wide                 16  1024x3   4608  M=64     5.021e-3    73.99   0.23x
round 5 K=48                 48  1024x3    576  M=64     9.232e-3   135.75   0.12x
```

Best fine-EQ-certified rollout of the campaign: **6.184e-03**. Best at an
unchanged classical ratio: **8.961e-03, 3.07x better than the incumbent**.

**Next session should:** collect `dn1024` (2837430) and `dn256b` (2837431),
which run the corrected recipe at full density after the maintenance window and
carry the only N=1024 confirmation — the resolution where the ROM actually wins
(1.61x) and therefore the only one where the cost-matched row's 1.27x has room
to be absorbed. Then decide whether to merge `exp/2026-08-25-burgers-accuracy`
into `main`: the branch was created disposable, but it produced a 3.07x accuracy
improvement at an unchanged ratio, four retractions, and reusable machinery
(`sep_coeff_extract.py`, `sep_hfit*.py`, `sep_speed_r5.py`, `sep_burgers_r5.py`)
that makes an h arm cost minutes instead of hours, so it should not be deleted.


### Consolidation session — pulled the round-4 close-out, completed the resolution curve, cut the consolidated worktree

The previous session's n256-push close-out agent died before pulling its eight completed jobs
(2837061-66 accuracy arms, 2837076/77 N=128/N=512 points); they sat on the cluster ~5 h.
Pulled, sha256-verified, committed on `exp/2026-08-23-n256-push` (524d920), remote
`n256_push/` deleted (now empty). `CROSS-RESOLUTION.md` regenerated (4187205) — the generator
takes the JSON paths as ARGUMENTS (`runs/gen_cross_resolution.py runs/push_*/out/*.json`);
run without them it prints empty tables, which briefly fooled this session.

**Four-point speed curve now complete** (K=16 reference recipe, optimized solver, paired):
N=128 0.38×, 256 0.39×, 512 0.60×, 1024 1.61×; batch-16 0.38/0.91/2.37/11.74× (upper bounds).

**Accuracy arms (N=256 unless noted; oracle / span-LS floor = the h gap):** r4a2 K=32 h512x3
full 300k: oracle 3.71e-3, rollout 5.05e-3 (M=256), gap 23.5×. **r4a6 K=32 h512x3 at N=1024,
full 300k: oracle 3.70e-3, rollout 5.14e-3 (M=256) / 6.85e-3 (M=64), gap 21.3×; e2e 98.9 ms
on the UN-optimized r3 solve path vs matched classical 62.9 ms (0.66×); batch-8 63.9 vs 52.9
ms/query.** r4a5 h1024x4+code-polish: recon 1.1e-3 (best ever) but oracle 7.96e-3, gap 50.6× —
capacity without generalisation, confirming round 5. r4a1 (h1024x3) time-capped at 100k/300k:
gap 22.4×; r4a3/r4a4 time-capped at ~40k on a shared L40S node — uninformative, not verdicts.

**Consolidation.** Worktree `2026-08-25-sepdec-consolidated` / branch
`exp/2026-08-25-sepdec-consolidated` cut from `exp/2026-08-25-burgers-accuracy` (the lineage
tip), `exp/2026-08-23-n256-push` merged cleanly (47d2bbd), `START-HERE.md` added (670380e).
"Where things stand" rewritten (the 08-22 block described the FiLM era).

**Pending on the cluster:** `dn1024` (2837430) and `dn256b` (2837431), namespace `burgacc/`,
`ReqNodeNotAv` — H200 nodes not yet back from the 06:00–12:00 maintenance. No watcher is
running; the next session must pull them.

**Next:** the round-4 speed sweep on the r4a6 K=32 checkpoint (see START-HERE) — the cheapest
experiment that could put accuracy (5e-3) and speed (>1×) in the same N=1024 model.

### EQ fidelity ladder — the quadrature error measured at every rung, and where it lives

Worktree `worktrees/2026-08-25-eq-fidelity-ladder`, branch `exp/2026-08-25-eq-fidelity-ladder`
(cut from `exp/2026-08-25-sepdec-consolidated`), cluster namespace `eqladder/` (now empty).
Driver `experiments/separable-decoder/sep_eq_ladder.py`; per-bucket tables `EQ-LADDER.md`
(`runs/gen_eq_ladder.py`); report on `main`: `reports/2026-08-25-eq-fidelity-ladder.md` with
tables from `reports/gen_2026-08-25-eq-ladder.py`. Trains nothing.

**Why.** Today's group meeting (`meeting_notes/aug25`, untracked): the advisor asked where the
quadrature weights come from, suggested the model learn positions and weights, and insisted the
L2, integral, gradient and Hessian errors be measured separately — "the gradient is the holy
grail". Design doc §6.1 had specified exactly that ladder in August and it had never been run.
Correction to carry back to the group: the weights are NOT uniform, they are NNLS-fitted (rel
fit 5e-3 control / 5e-4 fine); the open question was what they are fitted to, not whether.

**What ran.** Four arms, one checkpoint each, both EQ sets (control M=64/m=256 and fine
M=256/m=1024, built by the same code as every speed/accuracy job): lad256k16 (r3a, 2841798,
A100-40GB), lad256dm (round-5 dense_mid, 2841799, A100-80GB), lad1024k16 (r3d, the 1.61×
decoder, 2841800, H200), lad1024k32 (r4a6, 2841801, H200). N=64 smoke local first. All
`jax_backend=gpu`, stage manifests OK, sha256 OK, no captured-constant warning, remote dirs
deleted. Gates: gate 0 ≤7e-15 on every arm; a NEW gate F (full-grid reference vs
`make_weak_ops` on the whole interior with unit weights) ≤3e-16 at N=256, skipped at N=1024
(reference too large). dense_mid's NNLS rel fits reproduce `fq256`'s to the digit.

**Findings (numbers in the report's T-L1–T-L5, not retyped here).** (1) The sampled residual
differs from the exact one by tens of percent along the solver path on the control set, of
order one at the solution; single-digit percent / ~5–20 % on the fine set. Two orders of
magnitude above the NNLS rel fit — that statistic measures `Φᵀu` on snapshots, the residual is
a small difference. (2) The LINEAR part (`Φᵀ(u−uⁿ)`, Laplacian) is comparable to or larger than
the advection part everywhere and 2–4× it on the fine set — and it can be computed EXACTLY as
`(ΦᵀG)h(z)` with no quadrature at all. (3) The Hessian rung is fine (1e-4–1e-2): J is sampled
well; the error enters through R and corrupts the gradient (path cosines ~0.3–0.4 control,
~0.6–0.8 fine) and the LM step (cosines ~0.4–0.7). (4) At the oracle code the sampled gradient
is several times the true one on the control set — the sampled objective's minimiser is
elsewhere. This is the mechanism for rollout ≈ 1.3–1.9× oracle and for the fine-set 1.45× gain.
(5) The absolute gradient error is similar across decoders; the better decoders (K=32,
dense_mid) see a larger RELATIVE error because their true residual is smaller — the quadrature
is a floor the accuracy campaign runs into. (6) Rung (a), L2 on the nodes vs the full grid,
ratio ~0.98 everywhere: the sample points are representative; the state is not the problem.
(7) N=256 ≈ N=1024. Caveat: t=0 oracle rows have `uⁿ = u` so are structurally lower; compare
t≥5.

**What is wrong / retracted here.** No published number changes. Two beliefs the project has
been acting on are now measured false: that the NNLS rel fit certifies the EQ set (it certifies
nothing the solver consumes), and — implicit in the design — that the whole residual must be
sampled (the linear terms need not be). The meeting's "uniform weights" premise was also
incorrect and is corrected in the report.

**Engineering notes.** A first local smoke was killed by my own `timeout` while inside the
M=256 NNLS fit (Lawson–Hanson to support 1024 takes 12–23 min on any machine; ~700 s on H200,
~1000 s on A100) — not a code error; the M=256 fit dominates each job's wall time. A lab-log
edit with `str.replace` on an empty anchor scrambled the file once; restored from git, redone
with line anchors. Both are the reason this entry exists in git rather than a scratchpad.

**Verdict on "learn the quadrature".** Ranked by the ladder: (1) exact linear terms — free,
no learning, removes the dominant part; (2) same-target NNLS for the advection term on
residual/gradient-fidelity rows incl. off-manifold iterates — convex, no network change;
(3) learned nodes/weights only if the advection rung still binds after 1–2. Joint `(g,h,w)`
training is not motivated by anything measured.

**Next session:** implement (1) in a dated worktree from the consolidated branch (or from this
one), redefine gate 0 against the full-grid weak residual and log it as a rule change, refit the
advection-only EQ, re-run `sep_eq_ladder.py` to confirm, then the round-4 protocol for the
accuracy/speed effect. `dn1024` (2837430) / `dn256b` (2837431) were still RUNNING in `burgacc/`
at the end of this session — pull them. Merge decision for `exp/2026-08-25-eq-fidelity-ladder`:
ask.
