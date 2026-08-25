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

## The model in one line

`u(x;z) = bc(x)·⟨g(x), h(z)⟩` — a 512-feature spatial track `g` (Fourier lift → SiLU MLP,
`G_HIDDEN ≥ 2R` or the bank is rank-capped) and a K-dim latent track `h` (SiLU MLP + linear
skip). Because `g` never sees `z`, it is evaluated ONCE at the quadrature points into a cached
table; the online solve is `table @ h(z)` — the grid never enters the loop. Pure neural, no
POD anywhere; SVD is a diagnostic only.

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

## THE NEXT EXPERIMENT (cheap, decisive, do it first)

The K=32 N=1024 checkpoint `runs/push_r4a6/out/sep_burgers_r4_N1024_K32_R512_h512x3.pkl`
reaches **5.1e-3** but was timed only on the un-optimized round-3 solve path (98.9 ms). The
round-4 levers took the K=16 path from 71.7 → 27.1 ms (2.65×). **Run the round-4 speed sweep
on the K=32 checkpoint** — no training, minutes of GPU:

```
cluster/run_r4s1024.sbatch with CKPT= the r4a6 .pkl, N=1024, same STALLS/BATCHES/NEWTON ladder
```

If the same factor applies: ~37 ms vs the matched classical 62.9 ms ≈ **1.7× at 5–7e-3
error** — accuracy and speed in the same model at N=1024, which is the result this project is
for. Round 5 measured K=48 at 3.81× the K=16 cost, so K=32's optimized cost is the open number;
measure, don't extrapolate. Report the batched curve too.

Then: the two queued confirmations `dn1024` (2837430) and `dn256b` (2837431) in cluster
namespace `burgacc/` (pending on H200 availability after the 08-25 maintenance) — pull them
(`runs/pull_*.sh` pattern, sha256, delete remote), they are the full-density round-5 recipe
at N=1024.

## Open items, ranked

1. K=32 optimized-solver timing at N=1024 (above).
2. Pull `dn1024`/`dn256b`.
3. **Poisson baseline is unpreconditioned** `jax.scipy.sparse.linalg.cg` with no `M=`; the 28×
   Poisson claim is soft. Add a preconditioned (multigrid/IC) CG arm before quoting Poisson.
4. Certify the quadrature for any 1e-3 claim: the M=64 EQ set's own rel-fit is 6e-3; use M=256
   and report the EQ set beside every error.
5. `h` generalisation: capacity + μ-density must scale together (round 5); K is the strongest
   knob but costs online time. Test-time refinement of `h` on the PDE residual (no truth) is
   legitimate and untried.
6. Decide the merge to `main` for this branch (CLAUDE.md: ask, don't assume).

## Non-negotiables (unchanged)

Gate 0 ≤1e-12 on every arm; incumbent discretization/residual/Jacobian untouched; no
test-truth in solve paths; end-to-end timing incl. IC fit + full decode, raw reps kept,
balanced order, censoring reported; matched-accuracy comparisons only (a 5.2× claim was
retracted for `lin_tol` cherry-picking); never close a jit over a large array; one
verified-alive watcher per submission wave; summaries by committed script; pull + sha256 +
commit + delete remote dirs; append to `LAB-LOG.md` before the session ends.
