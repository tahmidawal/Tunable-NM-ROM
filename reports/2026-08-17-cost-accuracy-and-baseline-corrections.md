# ROM cost and accuracy — results as of 17 August 2026

Two experiments ran today, one agent each, both on Tufts A100s in f64 with `jax_backend=gpu`
asserted on every job. Between them they answered the two questions left open by the 16 August
work — *what does the latent solve cost as `k` and `N` vary*, and *can the ROM accelerate a
full-order solve* — and, unexpectedly, forced a correction to every speedup number the project
has published.

---

## Read this first: what is settled and what is not

| | status | trust |
|---|---|---|
| **Experiment 2** — ROM warm-start → FOM | **Final.** Complete, two Codex audits, 141/141 independent checks, pushed, cluster cleaned. | Quote freely. |
| **Correction to the record** — both FOM baselines were inflated | **Final** for Poisson; **open** for Burgers pending a cross-check. | Quote Poisson. Mark Burgers provisional. |
| **Experiment 1** — cost-to-tolerance / Pareto | **Surface complete**, single-GPU consolidation pulled, 873 points, tables rebuilt. Verdict prose and the results Codex pass still to come. | Numbers quotable; prose unaudited. |
| **Accuracy results** (all PDEs, both experiments) | **Unaffected by any correction.** | Quote freely. |

Everything the corrections touched was a **denominator**. No accuracy number moved.

---

## 1. The headline reversal: we are not faster than the full-order solver

The project has been quoting speedups against a full-order baseline that does far more work
than the accuracy it delivers requires. Measured against an **iso-accuracy** FOM — conjugate
gradients run only as far as the ROM's own accuracy, which is the only honest denominator —
the coordinate ROM looks like this on Poisson:

| N | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|
| coordinate ROM vs iso-accuracy FOM | 0.35× | 0.65× | **1.26×** | **1.73×** | **3.39×** |

*Final: measured on a single GPU by the consolidation runs of 18 August (Slurm 2516900 and
2516905), replacing the earlier cross-GPU pairings. The coarse meshes moved down as predicted;
N ≥ 256 moved **up**, because those panels had run on a slower A100 model.*

At coarse meshes the ROM is **slower than simply under-converging CG**, and it does not cross
1× until past N = 64. And on 2-D Poisson specifically, an exact **direct solve runs in
0.063–0.166 ms — 494× faster than the iterative FOM at N = 512**. On this problem the reviewer
who said a direct solver beats us was right, and we should say so in our own words.

**So the claim cannot be "the ROM is faster" without qualification.** What carries it is
**mesh-independence**: our cost runs 3.3 → 7.9 ms across a 256-fold change in unknowns while the
iso-accuracy FOM grows 1.2 → 26.7 ms, a factor of 23. We cross over between N = 128 and N = 256
and the advantage widens to 3.39× by N = 512. The scaling figure stops being a supporting
exhibit and becomes the entire result.

The honest scope of the direct-solver concession: this is 2-D with a cheap sparse
factorisation. It does not automatically transfer to 3-D, to nonlinear problems, or to settings
where refactorisation dominates.

---

## 2. What does survive: accuracy per dimension

Held-out relative L2 error at `k = 8`, `N = 64`. "Ceiling" is the error at the best possible
latent — the decoder's own floor, which no solver can beat.

| PDE | ceiling | coordinate ROM | POD, same `k` | ROM / ceiling |
|---|---|---|---|---|
| Poisson | 7.11e-3 | **7.65e-3** | 1.77e-1 | 1.08× |
| Heat | 1.16e-2 | **1.87e-2** | 1.29e-1 | 1.61× |
| Burgers | 1.15e-2 | **1.65e-2** | 2.09e-1 | 1.43× |
| Wave | 1.719e-1 | 8.783e-1 | 3.424e-1 | 5.11× — fails |

On the three that work the solve lands within **1.1–1.6× of the decoder's own ceiling**, so the
latent solve is essentially no longer the bottleneck. That is what the weak-form objective
bought.

**The Pareto frontier splits rather than POD dominating.** POD is 5–10× cheaper than us at equal
`k`, but its error *saturates* at ~5.1e-2 and does not improve at any `k` — more modes stop
helping. We reach ~1.2e-2. That is roughly **4.4× more accurate than POD can ever be** on this
family, and it is why POD owns everything looser than ~5e-2 while we own everything below it.

Accuracy is also flat under refinement: **1.155e-2 at N = 64 and 1.150e-2 at N = 128** at the
best uncensored operating point (`k = 16`, `τ = 1e-2`) — the same mesh-independence the cost
shows.

Two qualifications worth stating plainly. ~1.2e-2 is about **1 % relative error** —
engineering-grade, not high-precision; if you need 1e-6, this method does not offer it, and the
binding constraint is the decoder ceiling, not the solver. And the stopping tolerance costs real
accuracy: at `k = 8`, `τ = 1e-1 / 1e-2 / 1e-3` gives 8.9e-2 / 1.3e-2 / 8.5e-3, with `τ = 1e-3`
largely **unreachable** because the objective's own floor is ~5e-3 relative reduction (94–100 %
of those cells censor).

---

## 3. A new defect, now diagnosed and fixed

The reported error at some latent dimensions sits far above what the decoder is capable of —
6.6× at k=6, 7.7× at k=12, 12.3× at k=32, against 1.1–1.6× at k=4, 8 and 16. **That k-specific
reading was an artifact and is now withdrawn.**

The reported figure is a **mean over 16 test cases**, and at those k it is dominated by one to
five diverging solves. The median tells a different story:

| k | 4 | 6 | 8 | 12 | 16 | 24 | 32 |
|---|---|---|---|---|---|---|---|
| ceiling | 1.55e-2 | 8.84e-3 | 7.04e-3 | 6.24e-3 | 4.13e-3 | 3.98e-3 | 3.28e-3 |
| mean (as published) | 1.74e-2 | 5.85e-2 | 8.48e-3 | 4.79e-2 | 6.54e-3 | 1.49e-2 | 4.02e-2 |
| median | 1.55e-2 | 8.52e-3 | 7.25e-3 | 7.58e-3 | 5.05e-3 | 4.93e-3 | 3.66e-3 |
| diverged | 0/16 | 2/16 | 0/16 | 3/16 | 0/16 | 1/16 | 5/16 |

The median is within 0.96–1.24× of the ceiling at **every** k, and k=32's median is the best on
the ladder. Running all 64 held-out cases rather than 16 shows k=8 and k=16 failing too (3 and
4 cases). Our sixteen simply drew none of them. There is no powers-of-two structure; the
failure rate rises smoothly with k.

**Root cause: the latent optimiser has no globalisation.** It starts essentially undamped, so
its first Gauss–Newton step is 10–350× the norm of the latent itself, and it accepts any step
that decreases the residual at all — no sufficient-decrease or gain-ratio test. When an
overshoot happens to decrease it slightly, the iterate leaves the region of latent space the
decoder was trained on and converges to a spurious stationary point out there. The discriminator
is exact: the iterate's norm after the first accepted step, relative to the training-latent
cloud radius, is 0.77–1.32× for solves that succeed and 1.73–4.60× for solves that fail.

The objective is not at fault. A slice from the starting guess to the true answer is
monotonically downhill in every case tested, failures included; every successful solve ends
*below* the objective value at the oracle latent while every failed one ends 27–90× above it.
Reproduced on the full-grid objective with no hyper-reduction at all, which also rules out
quadrature.

**Fix: constrain the step to the training-latent cloud radius.** One line. Both rows below
were measured on the full-grid objective, so they compare to each other but not to the
hyper-reduced numbers above.

| k | 4 | 6 | 8 | 12 | 16 | 24 | 32 |
|---|---|---|---|---|---|---|---|
| ratio to ceiling now | 1.11 | 7.61 | 1.10 | 7.85 | 1.26 | 3.40 | 15.49 |
| with the fix | 1.09 | 1.05 | 1.09 | 1.18 | 1.28 | 1.45 | 1.50 |

No arm regresses, iterations fall at large k, and it holds at N=256. Two caveats: the fix has
not been re-measured through the production hyper-reduction and timing path, so those ratios are
a demonstration rather than replacements; and 3 of 64 cases at k=4 survive every fix without
ever leaving the trust region, which looks like a representation limit rather than a solver one.

**The same unglobalised optimiser is used on Heat, Burgers and Wave.** So the project's accuracy
numbers are broadly pessimistic, by an amount not yet quantified.

## 4. The ROM warm-start hybrid does not pay (final)

**The question:** hand the ROM's solution to the full-order solver as its initial guess and
finish to full accuracy. That would convert the ROM's accuracy ceiling from a limitation into a
non-issue, since the answer is FOM-exact and only cost remains.

**Poisson — there is no crossover `N`.** Across N ∈ {32…512}, three `τ_FOM` values and five ROM
tolerances (75 configurations), the hybrid never beats the plain FOM. Best anywhere **0.933×**.

An apparent 1.02× crossover in the first round was a **17 % device-clock-ramp artefact** — the
GPU ramping up after a long host-bound NNLS fit. Removing the bias removed the finding. Burn in
before every timed block.

*Mechanism.* The ROM's *field* error is 9.27e-3, but its **A-norm error — the quantity CG
actually contracts — is 4.97e-2**. So it saves only 1.3–3.7 % of iterations, and the head start
is **eroded, not diluted**: absolute savings fall 104 → 39 as the tolerance tightens. **30 of 75
configurations needed *more* iterations from the ROM start than from zero.** On top of that the
decode is structurally O(n), 0.118 → 4.56 ms across the ladder — precisely the cost the ROM
exists to avoid.

**Burgers — a one-line heuristic beats it.** More inner BiCGStab iterations in **12 of 12**
configurations, including the four where the Newton count improves; loses on wall clock 12/12;
hybrid totals **0.15×–0.49×**. Linear extrapolation `2u_{n-1} − u_{n-2}` beats the previous-step
start 12/12 on Newton and 10/12 on BiCGStab.

One exception, reported rather than averaged away: at N = 256, `τ` = 1e-6 the FOM stage alone is
faster from the ROM start (434.7 vs 449.9 ms). It does not rescue the hybrid — the ROM stage
costs far more than the 15 ms saved.

Solver health across the whole run: **zero BiCGStab breakdowns, zero non-zero Newton flags**, so
the result is not an artefact of a broken inner solve.

---

## 5. Correction to the record: every published speedup was inflated

Both full-order baselines did more work than their stated accuracy required, in the direction
that flatters us.

### Burgers — over-convergence 3.75×–4.79×

The baseline is a **fixed-8-Newton rollout**: 400 Newton steps and 400 BiCGStab solves per
50-step trajectory, by construction, landing at a per-step residual of ~9e-13. A tolerance-based
solve matched to what it actually reaches needs 105–120 Newton steps.

| N | fixed-8 | at `τ` = 1e-10 | factor |
|---|---|---|---|
| 32 | 197.3 ms | 52.6 ms | 3.75× |
| 64 | 412.0 ms | 94.7 ms | 4.35× |
| 128 | 1216.8 ms | 253.8 ms | 4.79× |
| 256 | 2159.4 ms | 496.8 ms | 4.35× |

**The published end-to-end ladder 0.72× → 1.57× → 4.46× → 7.96× becomes roughly
0.19× → 0.36× → 0.93× → 1.83×.** The 8.0× headline is ~1.8×, and **N = 128 moves from clearly
winning to break-even**. Because the multiplier varies with `N`, it flattens the ladder's
*slope* as well as its level — so the claim that the advantage grows steeply with refinement is
weakened on its own terms.

### Poisson — the factor depends on the tolerance you assume, and the tolerance must be named

The baseline is `jax.scipy.sparse.linalg.cg` at `tol = CG_TOL = 1e-13`, the same constant used
to *manufacture the truth data*. It attains 8.9e-14 at N = 32 but only **3.9e-12 at N = 512** —
at fine meshes it neither reaches its requested tolerance nor stops at a useful one.

| assumed deployment tolerance | 1e-6 | 1e-8 | 1e-10 |
|---|---|---|---|
| over-convergence factor | **~1.56×** | ~1.31× | ~1.16× |

Near-constant in `N` at each. **Two independent instruments measured 1.53–1.57× at `τ` = 1e-6
across all five meshes**, and a third run reproduced it.

This was itself a correction: ~1.16× — the *tightest* rung, the one most flattering to the
archived numbers — was first published as "the" Poisson factor. Quote the **1e-6 column** for a
deployment. Corrected archived Poisson end-to-end:

| N | archived | at 1e-6 |
|---|---|---|
| 128 | 0.74× | **0.47×** |
| 256 | 1.40× | **0.91×** — a sign change; stops beating the FOM |
| 512 | 3.56× | **2.33×** |

### How to apply these

- **Poisson: one scalar is defensible** — 2.8 % spread across meshes — but **only per assumed
  tolerance**, since the factor varies 1.34× across tolerances.
- **Burgers: it is not** — 28 % spread. A scalar bends the N-ladder's slope, which is the very
  quantity a mesh-independence claim rests on. Correct per mesh, or not at all.
- Prefer the **time** multiplier where the per-mesh anchoring check is tight; the iteration
  multiplier is a fallback, and it is *not* a bound in either direction.

**What licenses all of this:** the correction re-times the archived functions themselves rather
than modelling them. Agreement with the archive is 0.2 % / 0.1 % / 0.5 % on Burgers at
N = 32/128/256, and 1.8 % / 3.8 % on Poisson at N = 128/512. At N = 128 three independent
measurements agree to 2.3 % (15.145 / 14.86 / 14.795 ms).

**Still open:** the two experiment cells ladder *different knobs* on Burgers — a Newton tolerance
versus the testbed's `NEWTON_ITERS` — so the Burgers denominator is formally unsettled pending a
pre-registered cross-check with a stated sign convention.

---

## 6. Wave, N = 128 (folded in today)

The last 16 August job landed and completed the wave cell — all 16 coordinate variants, 15 POD
control arms, full timing block, zero blow-ups across all 31 arms.

Because the N = 64 and N = 128 cells ran on different A100s, only **within-cell ratios** are
admissible across them:

| within-cell ratio | N = 64 | N = 128 |
|---|---|---|
| coordinate step / POD `k`=8 step | 5.39× | 4.64× |
| coordinate step / POD `k`=64 step | 2.56× | 1.66× |
| coordinate speedup vs FOM rollout | 0.158× | 0.204× |
| POD `k`=64 speedup vs FOM rollout | 0.404× | 0.339× |

**Hyper-reduction's mesh-independence is confirmed on wave** — the half of the recipe that works.
The accuracy failure is untouched by resolution, so it is not a resolution artefact. The open
problem there remains structure-preserving latent stepping, not tuning.

---

## 7. Bugs and artefacts caught, and what they cost

Every one of these would have produced a plausible, publishable, wrong number.

| what | consequence if missed |
|---|---|
| **17 % GPU clock ramp** after host-bound NNLS fits | Manufactured a Poisson crossover at N = 512 that does not exist. |
| **BiCGStab missing alpha half-step test** | An exactly-converged `s` gave `t·t = 0`, mis-declared a breakdown, **discarding a converged iterate**. |
| **`N_POD_TRAJ=128` pinned in the batch env** | Silently re-imposed a POD training-set handicap the audit had just removed — every Burgers POD number unfair to the baseline, in our favour. |
| **Oracle ceiling stalled at large `k`** | Reported a `k`=32 "ceiling" of 7.26e-2 while the ROM reached 3.52e-2 — the ROM beating its own lower bound. Fixed init gives 3.83e-3. |
| **`m = M` quadrature corner** (my grid spec) | 660× worse quadrature; worst-row error 8.5e+05. The entire `k`=32 column would have been an artefact of our own settings. |
| **Tolerance-unlabelled correction factor** | The most flattering of three columns quoted as "the" factor; ~35 % error, and a sign change at N = 256. |
| **Invalid commit provenance** | `git -C <staged dir>` walked into an unrelated ancestor repo; the commit in all 11 reports was not an object in this repository. Metadata only — proven sound via content hashes. |
| **Timing repetitions not persisted** | Post-selection bias unbounded; the exact "best rung" claim was **withdrawn rather than defended**. |
| **Oracle ceiling OOM** (17 GiB `jacfwd`) | Killed the N = 512 panel. Fixed by chunking the Gauss–Newton accumulation; N = 512 recovered without a rerun. |

Verification used two adversarial Codex passes per cell — one on the harness before any real job
ran, one on the results against the raw JSONs. The results pass on the completed cell re-derived
2 052 table cells and every solver-health gate. Cross-agent review independently changed **five**
published claims that neither cell would have caught alone.

---

## 8. What is still open

**Closing within the hour** — Experiment 1's last Burgers panel, then the single-GPU
consolidation run (which produces the publishable ratios), generated tables, four figures, the
verdict, and the second Codex pass.

**Then, in the order I would run it:**

1. **Diagnose the `k` = 6/12/32 solver failure.** Reproducible across four meshes, not explained
   by any of the usual suspects, and the Jacobian spectra to settle it are already collected.
   Until it is understood, any published `k`-sweep has holes a reviewer will find.
2. **Decide the paper's framing.** The speed claim is gone; accuracy-per-dimension plus
   mesh-independence is what remains. Everything else should be planned around that choice.
3. **Apply the per-mesh Burgers multiplier** to the archived tables — mechanical, but it touches
   every prior speedup.
4. **Training-set-size ladder** on the 2- and 3-bump Poisson families, to separate the
   generalisation limit from an architectural one.
5. **Reviewer-requested baselines** — tuned FNO, POD-DeepONet, SMA cold start, preconditioned and
   direct classical solvers, an L-shaped domain, 3-D.
6. **Structure-preserving latent stepping** for wave.
7. **Heat and Wave Pareto frontiers**, dropped today for speed. A four-PDE frontier where we win
   two and lose two is more credible than a two-PDE one where we happen to win both.
8. **Decide the `fix/heat-rollout-warm-start` merge** into `main` — main's published heat numbers
   are currently unreproducible by its own public code.

---

*Branches: `exp/2026-08-17-cost-to-tolerance`, `exp/2026-08-17-rom-warmstart-fom`,
`exp/2026-08-17-inr-rom-consolidated`. Every run: f64, `JAX_DEFAULT_MATMUL_PRECISION=highest`,
`jax_backend=gpu` asserted, one job per directory, data regenerated on the cluster from seed,
results pulled with checksums and cluster directories deleted.*
