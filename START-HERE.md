# START HERE — separable EQ-decoder NM-ROM, consolidated line (2026-08-25)

This worktree/branch (`exp/2026-08-25-sepdec-consolidated`) is the **single line of
descent** for the separable-decoder work. It contains everything from
`exp/2026-08-22-separable-decoder` → `exp/2026-08-23-n256-push` → `exp/2026-08-25-burgers-accuracy`,
merged. The three source branches and the four scaling arms
(`exp/2026-08-23-sepdec-n{128,256,512,1024}`) are **read-only archives** from here on.
New work: branch a dated worktree from THIS branch (`CLAUDE.md` rules), never from `main`.

Read in this order:
1. This file.
2. `LAB-LOG.md` on `main` — "Where things stand" + the 2026-08-25 entries (retractions live there).
3. `reports/2026-08-24-separable-decoder-architecture-and-results.md` (main) — architecture,
   math, generated tables T1–T8. `reports/2026-08-25-poisson-architecture-and-results.md` for
   the plain-language picture with diagrams.
4. `experiments/separable-decoder/CROSS-RESOLUTION.md` and `PROFILE.md` (this branch) — the
   four-point speed curve and the dispatch-bound profile, both generated.
5. `reports/2026-08-25-burgers-h-generalisation-wall.md` (this branch) — the round-5 campaign.
6. `experiments/separable-decoder/HANDOFF-BURGERS-ACCURACY.md` — rules, levers, mechanics.

## Why this line of work exists — the separable decoder is a NEW architecture

Everything on this branch is about one architectural idea, introduced 2026-08-22. Before it,
the project's decoders were FiLM / coordinate networks in which the latent code `z` modulates
the network *as it processes the coordinate* `x`. That is flexible, but it means every time
the solver tests a new `z` it must re-run the whole network at every quadrature point — the
decode cost sits inside the online loop and the ROM cannot be cheap.

The advisor's ask (meeting notes, 08-22) was a decoder **designed for this task**: decode only
at the sample points, with the empirical quadrature built in. The separable decoder makes
that structural rather than approximate:

```
u(x; z)  =  bc(x) · ⟨ g(x) , h(z) ⟩         g: x → ℝ^R     h: z → ℝ^R
```

`x` and `z` never meet except in one inner product. Consequences that define the method:

- **Sampling is built in.** `g` never sees `z`, so it is evaluated ONCE at the m≈256 NNLS
  quadrature points into a cached table `G_q` (and a stencil bank `G_st` for Burgers
  advection). Online, the residual is `table @ h(z)` and its Jacobian `table @ ∂h/∂z`
  exactly — no spatial network in the loop, and **N never enters the loop**. That is why
  the solve costs ~2–35 ms at N=64 and at N=1024 alike.
- **Boundary conditions are exact by construction:** `bc(x)=16x₁(1−x₁)x₂(1−x₂)` is zero
  on the walls, so every output satisfies Dirichlet — never imposed, never learned.
- **It is genuinely nonlinear** (the manifold is the K-dim image of the SiLU MLP `h`
  inside the R-dim span of `g`), which is what makes it an NM-ROM: on Burgers, 16 nonlinear
  coefficients match what takes **117 linear POD modes** (report T7). Pure neural — no POD
  basis, no POD init, no linear corrector anywhere; SVD is a diagnostic only.
- **Its hard ceiling is a linear-algebra fact:** the manifold lives inside span{g}, so no
  variant of this family can beat the rank-R POD floor. And the last layer of `g` is
  linear, so the bank's rank is capped at `G_HIDDEN+1` — `G_HIDDEN ≥ 2R` is mandatory
  (this bug silently capped R=512 at 257 for three rounds).

Shapes and activations (round-3/4 config, `n_ff=64`, `G_HIDDEN=1024`, `R=512`, `K=16`):

```
   SPATIAL TRACK  g(x)                        LATENT TRACK  h(z)
   x = (x₁,x₂)  (2,)                          z  (16,)
        │ x @ B   B:(2,64) trained                 ├──────────┐
   ang          (64,)                              │          │ linear skip
        │ [sin(2π·), cos(2π·)]                     │          │ z·W (16→512)
   features     (128,)                             │          │
        │ 128→1024, SiLU                           │ 16→128, SiLU
                (1024,)                                   (128,)
        │ 1024→1024, SiLU                          │ 128→128, SiLU
                (1024,)                                   (128,)
        │ 1024→512, LINEAR                         │ 128→512, LINEAR
   g̃(x)         (512,)                                    (512,) ←┘
        │ × bc(x)·out_scale (a scalar)             │
   g(x)         (512,)                        h(z)        (512,)
        └──────────────┬─────────────────────────────┘
                       ▼
               ⟨g(x), h(z)⟩ = u(x;z)      one number
```

Training is auto-decoder style (one free code per snapshot, trained jointly with the weights),
relative-MSE + a feature-Gram conditioning term, point-subsampled AdamW + EMA. The online
solve is the incumbent weak-form trust-region LM on M=4K sine test modes — the discretization
is never changed, only how it is evaluated (gate 0 asserts this at ≤1e-12 in every job).

**Where the architecture is written up** (read these before changing the model):
- `reports/2026-08-22-separable-eq-decoder-design.md` (main) — the design doc: math, the
  cached-operator algebra derived from the discrete operators, cost model, gate ladder,
  prior art, and the audit record of what the first draft got wrong.
- `reports/2026-08-24-separable-decoder-architecture-and-results.md` (main) — the study:
  Parts I–IV are the architecture and integrity guarantees; Part V the generated tables
  T1–T8; VI.3 the retractions.
- `reports/2026-08-25-poisson-architecture-and-results.md` (main) — the plain-language
  version with the sheets-and-recipes picture and the offline-table/online-loop diagram.
- `experiments/separable-decoder/README.md` and `AUDIT-2026-08-23.md` (this branch) — how a
  run works end to end, and the adversarial audit that verified nonlinearity, no leakage,
  and gate 0 — and failed the first timing methodology.
- `PROFILE.md` (this branch) — why the online solve is kernel-dispatch-bound, which is what
  every speed decision since round 4 rests on.

## What is established (all numbers from generated tables — never retype)

**Speed, K=16 / R=512 reference recipe, optimized solver (gram IC + adaptive stall),
matched-accuracy paired AB/BA vs a swept Helmholtz-preconditioned Newton ladder:**

| N | ROM e2e ms | matched classical ms | single-query | batch-16 (upper bound) |
|---|---|---|---|---|
| 128 | 22.9 | 8.4 | 0.38× | 0.38× |
| 256 | 35.6 | 13.8 | 0.39× | 0.91× |
| 512 | 25.2 | 15.0 | 0.60× | 2.37× |
| **1024** | **27.1** | **42.9** | **1.61× ROM wins** | **11.74×** |

ROM cost is flat in N; the classical cost grows; the crossover is between 512 and 1024
single-query, between 256 and 512 batched. Batched ratios are **upper bounds** (a `vmap` of a
`lax.while_loop` runs to the slowest lane, penalising the classical side).

**Accuracy (Burgers, the real target).** Everything below the manifold is tight (solver =
weak-EQ optimum ≈ oracle, no compounding); the binding rung is `h`'s *generalisation* — not
capacity, not code convergence, not the span. Best measured:

| decoder | N | rollout err (M=256 EQ) | e2e ms (solver path) | matched classical | ratio |
|---|---|---|---|---|---|
| K=16 reference | 1024 | 2.57e-2 (M=64) | 27.1 (optimized) | 42.9 | 1.61× |
| K=16 dense_mid (round 5) | 256 | 8.96e-3 (M=64) / 6.18e-3 | 45.2 (optimized) | 18.3 | 0.40× |
| **K=32 h512x3 (r4a6)** | **1024** | **5.14e-3** (6.85e-3 @ M=64) | 98.9 (**UN-optimized** r3 path) | 62.9 | 0.66× |
| K=32 h512x3 (r4a2) | 256 | 5.05e-3 | 99.5 (un-optimized) | 16.5 | 0.17× |

Rollout sits 1.3–1.9× above the oracle everywhere, so 1e-3 needs an oracle ≈5e-4 — about
5× below the best measured (r4a2/r4a6 oracle 3.7e-3). See the round-5 report for the K
ladder (gap 58× at K=8 → 9.7× at K=128) and the μ-density saturation.

## Poisson (the control problem) — where it stands

Full write-up: `reports/2026-08-25-poisson-architecture-and-results.md` on `main`, tables by
`reports/gen_2026-08-25-poisson-summary.py`. Table below is pasted from that generator's
output (K=16, held-out sources, tau=1e-3; classical = cheapest CG rung at least as accurate):

| grid N | unknowns | ROM ms | ROM error | classical CG ms | CG error | who wins |
|---|---|---|---|---|---|---|
| 64 | 3,844 | **2.06** | 3.75e-02 | 1.31 | 3.2e-02 | CG 1.6x |
| 128 | 15,876 | **3.09** | 3.06e-02 | 3.75 | 1.7e-02 | **ROM 1.2x** |
| 256 | 64,516 | **2.54** | 2.89e-02 | 8.16 | 1.1e-02 | **ROM 3.2x** |
| 512 | 260,100 | **3.06** | 3.15e-02 | 26.36 | 7.1e-03 | **ROM 8.6x** |
| 1024 | 1,044,484 | **3.50** | 3.48e-02 | 98.31 | 4.9e-03 | **ROM 28.1x** |
| grid N | solver | ms | error |
|---|---|---|---|
| 1024 | spectral_dense | 0.64 | 7.0e-15 |

Best decoder accuracy (N=256, R=512, single-scale features): recon 6.23e-3, fresh-cohort
solve 9.46e-3. POD floor of the basis on fresh sources (diagnostic only):

| basis size R | best possible error, fresh sources |
|---|---|
| 64 | 3.22e-02 |
| 128 | 1.41e-02 |
| 256 | 4.01e-03 |
| 512 | 8.11e-04 |
| 1024 | 8.11e-04 |

So Poisson shows the same shape as Burgers: cost flat in N, crossover vs the iterative
baseline, and an `h`-generalisation gap (~9e-3 achieved vs ~8e-4 floor).

**Why the Poisson 28× is SOFT — read before quoting it:**
1. The CG baseline is `jax.scipy.sparse.linalg.cg` on a matrix-free 5-point Laplacian with
   **no preconditioner** (`M=` is passed nowhere). Its cost growth with N is partly the
   baseline being weak; a multigrid/IC-preconditioned CG would be far flatter. This arm has
   never been run — open item 3.
2. This Poisson has constant coefficients, so an exact spectral solve exists: **0.64 ms at
   7e-15** at N=1024, beating the ROM 5×. The ROM's claim is only against *iterative* solvers
   on problems with no fast direct method.
3. The Burgers baseline IS well-preconditioned (exact Helmholtz), so the Burgers numbers are
   the trustworthy ones. Poisson is the control, not the headline.

## CURRENT PHASE: SOLIDIFY (from 2026-08-25) — not the accuracy push

The 1e-3 Burgers push is a *later* goal. The phase now is to make the architecture and every
claim about it solid enough to build on: verified against the current code, reproducible from
a clean checkout, documented once, and compared against honest baselines. Nothing new gets
built until the checklist below is done.

### A. Close the open measurements (no new science, just finish what is half-measured)
1. Pull `dn1024` (2837430) and `dn256b` (2837431) from namespace `burgacc/` when they land
   (`runs/pull_*.sh` pattern: sha256, commit, delete remote). No watcher is running on them.
2. The K=32 N=1024 checkpoint (`runs/push_r4a6/out/…K32_R512_h512x3.pkl`, rollout 5.14e-3)
   was timed only on the un-optimized solve path. Run the round-4 speed sweep on it
   (`cluster/run_r4s1024.sbatch` with `CKPT` swapped; minutes, no training) so the N=1024
   accuracy/speed state is a measurement, not a projection.
3. Poisson: add a **preconditioned** CG arm (`M=` callable — multigrid or incomplete
   Cholesky) at N=256 and N=1024. The current 28× is against unpreconditioned
   `jax.scipy.sparse.linalg.cg` and is soft until this runs.

### B. Verify the architecture as it exists NOW
The only adversarial audit (`AUDIT-2026-08-23.md`) covered the N=64 cell. Since then: the
`G_HIDDEN ≥ 2R` rank fix, the v2 point-subsampled trainer, Gram-space IC fit, adaptive stall
tolerance, extrapolated warm start, batched paths, and the round-5 Gram-space `h` refit
machinery. None of it has been audited. Run a fresh adversarial audit (Codex or equivalent,
read-only, numerical checks on committed checkpoints) of the consolidated code: nonlinearity,
no test-truth leakage in every solve path incl. the encoder IC and the Gram refit, gate 0 on
every arm, timing symmetry, and that every generated table reproduces from the JSONs.

### C. Make it reproducible
- Can a fresh clone of this branch regenerate one cell end to end (data → train → EQ →
  solve → tables) from documented commands? Do it once at small N and record the recipe.
- Turn the gates (gate 0, EQ-bank-vs-meshfree, Gram-IC-vs-full, batched-vs-single) into a
  runnable test file, not assertions scattered across five drivers.
- The driver lineage `sep_burgers.py → _r2 → _r3 → _r4 → _r5` plus `sep_speed_r4.py` and the
  `hfit` tools duplicates logic. Consolidate to one documented driver set; keep the old ones
  only if a committed result depends on them, and say which.

### D. Document once, against the current code
- `reports/2026-08-22-separable-eq-decoder-design.md` predates the rank fix, the profile,
  the speed levers, the batched caveat, and the `h`-generalisation wall. Revise it (or write
  its successor) so the design doc describes the architecture that actually runs.
- Every retracted number must be marked as retracted wherever it appears. Grep the reports.
- One results table per PDE, generated, with the baseline named and its preconditioning
  stated beside every classical number.

### E. Baselines standardized
Same classical solver family, same `(newton_tol, lin_tol)` / CG-tolerance ladder, same
preconditioning, same paired AB/BA protocol at every N and both PDEs. The cross-N curves
were assembled from jobs with differing `lin_tol` and preconditioning; regenerate them
under one protocol before anyone plots them.

**Exit criterion for this phase:** an updated audit passes, one clean driver set reproduces a
cell from scratch, the design doc matches the code, and both PDEs' headline tables carry a
preconditioned baseline. Only then does the accuracy campaign resume (open items below).

## After solidifying — the accuracy campaign (deferred)

Burgers to ~1e-3 at preserved N=1024 speed. Levers, ranked by round-5 evidence: `h`
generalisation (capacity + μ-density must scale together; K is the strongest knob but costs
online time — K=48 = 3.81×); certify the EQ set (M=64's own rel-fit is 6e-3; use M=256);
test-time refinement of `h` on the PDE residual (no truth, untried). Merge decision for this
branch: ask, don't assume.

## Non-negotiables (unchanged)

Gate 0 ≤1e-12 on every arm; incumbent discretization/residual/Jacobian untouched; no
test-truth in solve paths; end-to-end timing incl. IC fit + full decode, raw reps kept,
balanced order, censoring reported; matched-accuracy comparisons only (a 5.2× claim was
retracted for `lin_tol` cherry-picking); never close a jit over a large array; one
verified-alive watcher per submission wave; summaries by committed script; pull + sha256 +
commit + delete remote dirs; append to `LAB-LOG.md` before the session ends.
