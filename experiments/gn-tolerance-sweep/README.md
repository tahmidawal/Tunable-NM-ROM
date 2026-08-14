# GN iterations-to-tolerance vs mesh (2026-08-13)

Completes the cost story: the per-iteration kernel is provably n-free
(exp/2026-08-13-cost-scaling-*), but does the NUMBER of iterations needed to
reach a fixed accuracy grow with the mesh? Trained ViT-CP ROMs, both packages,
N in {32, 64, 128, 256}, relative gradient-norm tolerances {1e-2, 1e-3, 1e-4}.
Figure: `iters_vs_N.png` / `.pdf`; per-cell data in `results/`.

## Headline

**Heat (the controlled experiment): iterations are flat in N.** The heat
configs hold k=64 and rank=256 fixed across all meshes, and the (fixed)
rollout warm-starts each step, so this is a genuine fixed-architecture sweep:

| N | median iters/warm step @1e-2 | @1e-3 | @1e-4 |
|---|---|---|---|
| 32 | 2 | 4 | 8 |
| 64 | 2 | 4 | 9 |
| 128 | 2 | 3 | 9 |
| 256 | 3 | 5 | 28 (49% hit the 30-cap) |

At practical tolerances the cost to reach fixed accuracy does not grow with
the mesh. The N=256 @ 1e-4 blowup is the tolerance dropping below what the
model can deliver at that mesh (its rollout error is ~6.5e-2, worst of the
sweep — the resolution wall showing up in the full ROM); iterating harder
cannot fix a capacity limit. Cold step-0 medians: 10-26 iters (30 at
N=256/1e-4) — the warm-start chain is what keeps the rollout cheap.

**Poisson (NOT controlled): iterations grow along the shipped config family.**

| N | median iters/solve @1e-2 | @1e-3 | @1e-4 |
|---|---|---|---|
| 32 | 9 | 12 | 15 |
| 64 | 11 | 14 | 19 |
| 128 | 15 | 25 (37% capped) | 30 (51% capped) |
| 256 | 17 (16% capped) | 30 (52% capped) | 30 (76% capped) |

Attribution caveats: the shipped configs co-vary k (8/8/12/16) and m
(640/640/960/1280) with N, every solve is a cold start (z=0), and these
models train/evaluate against the analytic-data inconsistency (below) — so
this shows the end-to-end cost of the SHIPPED configuration family rising,
not an N-dependence at fixed architecture. The controlled fixed-k poisson
sweep on repaired data is the follow-up.

## Data-integrity context (found during this study)

1. **Heat frozen-rollout bug** — the original motivation for re-running heat
   on `fix/heat-rollout-warm-start`: main's rollout never advanced past step
   1 (see the fix branch and memory). All heat cells here use the FIXED
   solver; warm steps doing 2-9 real iterations (vs the bug's frozen 1) is
   the fix visible in data.
2. **Poisson analytic-data inconsistency** — the shipped 2D configs train and
   grade against boundary-masked closed-form fields that are ~O(1) away from
   the FD solution the solver enforces (measured ||u_ana-u_cg||/||u_cg|| ~ 1
   at N=128; repo's own run_rom gives mean rel_l2 0.647 vs the header claim
   1.08e-2). The `rel_l2_per_tol` values in the poisson JSONs are against
   that inconsistent truth — DO NOT quote them as ROM accuracy. Iteration
   counts remain valid as convergence measurements of the trained models.
   Also: `fom.cg_solve` NaN'd on 4/10 N=128 sources during diagnosis — the
   2D CG path needs validation before a data repair.

## Method

- `measure_gn.py` builds each package's real solver from the trained
  checkpoint exactly as `run_rom.py` does. Histories come from the REAL
  solver itself (a (tol=0, max_iters=j) sweep — the while-loop carry's exit
  gnorm is gnorms[j-1]), never from a re-implemented loop (a scan replica
  diverged from the real body via line-search argmin flips on fp noise —
  abandoned).
- Heat runs the full jitted rollout separately per tolerance with the real
  stopping rule active (warm-start chains depend on the tolerance, so
  post-hoc derivation is invalid for heat); poisson solves are independent
  cold starts so any tolerance is derivable from one recorded history.
- MANDATORY self-check per cell: iterations derived from histories must
  exactly match the real solver's returned counts at the default tolerance —
  `VALIDATION OK` in every log (8/8). Small per-cell counts of jit-vs-eager
  step-0 discrepancies (fp near threshold) are recorded in the JSONs.
- Censoring: probe cap 30 iterations; capped fractions reported everywhere.

## Provenance

- Branch `exp/2026-08-13-gn-tolerance-sweep` (includes the merged rollout
  fix); harness commit `0c6bd0d`, fix merge `202aabc`.
- Tufts jobs (gpu partition, one A100 + isolated dir each, jax_backend=gpu
  and JAX_DEFAULT_MATMUL_PRECISION=highest verified in every log):
  poisson 2346808-2346811, heat 2346896-2346900. Full repo training recipe
  per cell (100k epochs poisson / 80k heat), data regenerated from seed in-job;
  n32 configs are copies of n64 with N: 32 (`configs/`). Cluster dirs deleted
  after pulling.
- Related: heat2d_n64 repro on the fixed solver (job 2346901, on the fix
  branch): rel_l2 2.83e-2, speedup 0.17x — the honest replacements for the
  unreproducible published 5.21e-3 / 39.6x.

## Files

| path | what |
|---|---|
| `measure_gn.py` | history-recording measurement harness (both packages) |
| `configs/` | added n32 configs (copies of n64 with N: 32) |
| `results/gn_{poisson,heat}_N*.json` | per-cell histories, iteration counts, rel-L2 per tolerance, slurm logs |
| `results/summary_stats.json` | medians/p90/capped fractions used in the tables |
| `figure_iters_vs_N.py` / `iters_vs_N.png|pdf` | the two-panel figure |
| `smoke/` | local GB10 smoke artifacts (not representative; note field inside) |
