# Start here

Branch `exp/2026-08-18-codex-handoff`. Everything the project has produced through
18 August 2026 is in this one tree. Read this file, then `reports/report_src.md` for the
science, then the cell README you are working in.

`main` is the frozen public baseline and its heat ROM rollout is known broken (frozen after
step 1), so **do not treat `main` as a reference for behaviour.**

---

## What the method is, in one paragraph

We solve a PDE by finding a small latent vector `z` on a learned manifold and decoding it to a
field. The decoder is a FiLM coordinate network `u(x;z)` — meshfree, no grid-tied parameters.
`z` is found by minimising the PDE residual **projected onto M smooth sine test modes** (a
weak form), evaluated at `m` NNLS empirical-quadrature points rather than the whole grid. So one
solve touches 256 points, forms 64 equations, and solves for 8 unknowns, independently of
whether the grid has 900 or 260,100 nodes. Minimising the pointwise residual instead does not
work — the discrete Laplacian amplifies grid-scale decoder error by ~N², and the solver chases
that noise.

---

## State of every cell

| cell | status | trust |
|---|---|---|
| `poisson2d-rom-objective` | complete, 2 Codex audits | quote freely |
| `burgers2d-rom-latent-stepping` | complete, 2 Codex audits — **the reference ROM implementation** | quote freely |
| `heat2d-rom-latent-stepping` | complete, 2 Codex audits | quote freely |
| `wave2d-rom-latent-stepping` | complete, 2 Codex audits; negative result | quote freely |
| `rom-warmstart-fom` | **complete**, 2 Codex audits, 141/141 self-checks, cluster cleaned | quote freely |
| `cost-to-tolerance` | **INCOMPLETE — see below** | tables yes, verdict no |
| `k-stall-diagnosis` | complete as a diagnostic; not a production measurement | see its README |

### `cost-to-tolerance` is the one unfinished cell

Done: all nine panels, the fine FOM ladder, both N=512 recovery cells, the single-GPU
ceiling+spectrum run, `runs/pareto_points.json` (698 rows), generated tables in §6, four figures.

Not done:
1. **The single-GPU consolidation run never happened.** There is no `ctol_consol_*` directory in
   `runs/`. This matters: every ROM-vs-FOM ratio currently pairs ROM times from fanned-out
   panels with FOM times from a separate job, i.e. **different GPUs**. N ≤ 128 ran on an
   A100-80GB and N ≥ 256 on an A100-40GB, which alone produces a 3.7× apparent speed-up in the
   latent solve. Iteration counts are hardware-free and fine; wall-clock ratios are not.
2. §7 Verdict and §8 Caveats are placeholders.
3. No `CODEX-REVIEW-RESULTS.md` — the second audit never ran.

To finish it, from `experiments/cost-to-tolerance/`:

```bash
python ctol_pick_configs.py
./cluster/make_cells.sh consolidate
./cluster/launch.sh ctol_consol_p && ./cluster/launch.sh ctol_consol_b
./cluster/pull.sh
python ctol_tables.py     # refuses to build unless the surface is complete
python ctol_figs.py
```

---

## The results, and what is safe to say

**Accuracy holds.** At k=8, N=64, held-out relative L2, against the decoder's own ceiling:
Poisson 7.65e-3 (ceiling 7.11e-3), Heat 1.87e-2 (1.16e-2), Burgers 1.65e-2 (1.15e-2). Wave
fails outright at 8.78e-1 against a 1.72e-1 ceiling. POD at the same latent size is 1.3e-1 to
2.1e-1, and **POD cannot reach 1% error at any number of modes** — it saturates near 5%.

**Cost is mesh-independent.** Iterations to tolerance at k=8 are 10.1 / 10.4 / 10.3 / 10.3 / 9.4
across N = 32…512. The same 256 quadrature points are 28% of the coarsest grid and 0.1% of the
finest, with error flat at 8.4–8.6e-3.

**Speed does not hold the way we claimed.** Against a full solver run only as far as our own
accuracy, we are 0.65× at N=64, 1.63× at N=256, 3.03× at N=512 (provisional, see the
consolidation gap above). So the claim is mesh-independence, not raw speed. On Burgers we lose
everywhere (0.25–0.55×). And on 2-D Poisson a **direct sparse solve is 494× faster than the
iterative solver** — a reviewer said so and was right.

**The warm-start hybrid does not pay**, on either PDE, in any of 75 Poisson configurations
(best 0.933×) or 12 Burgers ones (0.15–0.49×). Linear extrapolation from the previous two time
steps beats it and costs nothing.

---

## Three corrections already applied — do not re-derive them

1. **Both full-order baselines were over-solved.** Burgers used a fixed 8 Newton iterations per
   step, ~4× more work than its accuracy needed; Poisson used CG at `tol=1e-13`, the tolerance
   used to *manufacture the truth data*. Corrected Burgers ladder: 0.72/1.57/4.46/7.96× becomes
   0.19/0.36/0.93/1.83×. Poisson's factor **depends on the assumed deployment tolerance and the
   tolerance must always be named**: ~1.16× at 1e-10, ~1.31× at 1e-8, **~1.56× at 1e-6**.
   Correcting the archived Poisson ladder at 1e-6 changes a **sign** at N=256 (1.40× → 0.91×).
   Poisson's factor is near-constant in N so one scalar is defensible *per tolerance*; Burgers'
   varies 28% across meshes so it must be applied per mesh or not at all.
2. **The "solver stalls at particular k" finding is withdrawn.** It was a mean-over-16-sources
   artefact. See `experiments/k-stall-diagnosis/README.md`. The optimiser has no globalisation
   and a trust region fixes it; the same optimiser is used on every PDE, so the project's
   accuracy numbers are likely **broadly pessimistic**.
3. **`m ≈ 4M` is a hard rule.** At `m = M` the quadrature collapses — relative fit 1.5e-3 →
   6.75e-2 with a worst-row error of 8.5e+05. An earlier grid spec violated this and made an
   entire k=32 column an artefact.

---

## Landmines, each of which cost a run

- **Never `scancel` by name, glob, or user.** `scancel --name=a,b,c` matches nothing, degrades to
  an empty selector, and kills **every job on the account**. It killed two agents' fleets at
  once. Use `cost-to-tolerance/cluster/cancel.sh` with explicit numeric ids.
- **Burn in the GPU before every timed block.** A 17% clock-ramp bias after a long host-bound
  NNLS fit manufactured a Poisson crossover that does not exist.
- **Never compare wall clock across jobs.** Different panels land on different A100 models.
  Use within-job ratios, or one job on one GPU.
- **`git -C <staged dir>` on the cluster walks into an unrelated ancestor repo.** The commit
  recorded in one cell's 11 reports was not an object in this repository. Verify provenance by
  content hash against git history, not the working tree.
- **Persist timing repetition arrays** or post-selection bias is unbounded and "best rung"
  claims cannot be defended.
- **Never state a correction factor without its reference tolerance.** Doing so is exactly how a
  1.56× correction got published as 1.16×.
- Cluster home is at quota — all output to the paralab space. One job per directory. Check
  `squeue` before and after every submit. Regenerate data from seed on the cluster.

---

## What I would do next, in order

1. **Finish `cost-to-tolerance`** — the consolidation run, verdict, caveats, Codex audit. Until
   that lands no speed number in this tree is publishable.
2. **Apply the trust-region fix** in `ms_autodecoder.lm_solve` / `pro_common.lm_generic` and
   re-measure all four PDEs. Accuracy should improve everywhere; nobody knows by how much.
3. **Attack the decode.** At N=512 it is 84% of online cost while the latent solve is 9%. The
   thing the method optimises is no longer the bottleneck.
4. **Resolve the Burgers denominator.** The two cells ladder different knobs (a Newton tolerance
   vs the testbed's fixed `NEWTON_ITERS`), so the Burgers correction factor is formally open. A
   pre-registered cross-check with a stated sign convention is set up in `cost-to-tolerance`.
5. **Reviewer baselines**: tuned FNO, POD-DeepONet, an L-shaped domain, 3-D, and preconditioned
   and direct classical solvers.
6. Decide whether `fix/heat-rollout-warm-start` merges into `main`.

---

## What is and is not in this branch

Self-contained: all 16 trained decoders for the k-ladder (`cost-to-tolerance/ckpt_poisson`,
`ckpt_burgers`, 74 MB) are tracked here. They were `.gitignore`d in the source cell and existed
nowhere else in git, so a clone would have had no decoders to run. Result JSONs, Slurm logs,
sha256 manifests, figures and every Codex audit are tracked.

Not here: `runs/smoke/` interface-test artefacts, `cluster/stage/` copies of code already
tracked, and `__pycache__`. Data is regenerated from seed on the cluster and is never synced.

## Compute

`CLAUDE.md` in the repo root is authoritative and unchanged. In short: Tufts is the default for
real GPU work, the local GB10 is for sub-minute smokes with at most three concurrent `jaxrun`
jobs, always use the venv by absolute path, `gpu` partition only, assert `jax_backend=gpu` or
abort, f64 and `JAX_DEFAULT_MATMUL_PRECISION=highest` everywhere.

## Reports

`reports/` holds the written-up results: `report_src.md` (illustrated, four-PDE),
`Cost-Frontier-and-Hybrid.md` (cost, quadrature, frontier, hybrid),
`ROM-Cost-and-Accuracy-Findings.md` (the 17 August results and the corrections), and
`When-The-ROM-Pays.md` (slide deck). Figures in `reports/talk_figs/`, generators alongside them.
Every table in those documents is generated from the run JSONs, not hand-typed.
