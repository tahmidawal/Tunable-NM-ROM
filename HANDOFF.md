# INR NM-ROM — consolidated working tree

**Start here.** This branch (`exp/2026-08-17-inr-rom-consolidated`) merges the four
completed 16–17 Aug 2026 experiment cells into one tree so the next session works from a
single checkout. `main` stays the frozen public baseline.

Read `reports/report_src.md` first — it is the illustrated results report, written for a
reader who has not seen any of this. Then this file for state and mechanics.

---

## What is here

```
experiments/
  poisson2d-rom-objective/       objective study (the fix) + k/m/M ladders, complexity ladder
  burgers2d-rom-latent-stepping/ the REFERENCE ROM implementation + ladders, timing
  heat2d-rom-latent-stepping/    heat port + direct-POD baseline
  wave2d-rom-latent-stepping/    wave port + energy analysis (the negative result)
reports/                         illustrated report, figure scripts, figures
```

Each experiment dir has: `README.md` (design, tables, verdict, caveats, provenance),
`*_common.py` / `*_train_ad.py` / `*_rom.py` (the harness), `cluster/` (sbatch generator),
`runs/` (result JSONs, trained checkpoints, Slurm logs, sha256 manifests), and two
`CODEX-REVIEW-*.md` audit reports.

Removed from this tree as noise (still on the original branches): `__pycache__`, `smoke/`
interface-test artifacts, and `cluster/stage/` copies of code already tracked here.
`experiments/wave2d-rom-latent-stepping/deps/` is kept — it holds modules imported from
branches that are *not* merged here, so the wave scripts run as-is.

Source branches, if you need the unpruned history:
`exp/2026-08-16-{poisson2d-rom-objective, burgers2d-rom-latent-stepping,
heat2d-rom-latent-stepping, wave2d-rom-latent-stepping}`.

---

## The result in one page

We replaced the paper's ViT-CP decoder with a FiLM **coordinate decoder** (INR, `u(x,t;z)`,
no grid-tied parameters) and made the latent solve actually work.

**The fix.** Minimising the pointwise/FD PDE residual stalls ~8× above the decoder's own
ceiling, because the discrete Laplacian amplifies grid-scale decoder error by ~(N−1)².
Minimising the residual **projected onto M low smooth sine test modes** (a weak form,
`Λ^-1`-weighted) reaches the ceiling and is insensitive to the initial guess.

**At k = 8, N = 64** (held-out rel-L2; "ceiling" = error at the oracle latent):

| PDE | ceiling | coordinate ROM | POD, same k | ROM / ceiling |
|---|---|---|---|---|
| Poisson | 7.11e-3 | 7.65e-3 | 1.77e-1 | 1.08× |
| Heat | 1.16e-2 | 1.87e-2 | 1.29e-1 | 1.61× |
| Burgers | 1.15e-2 | 1.65e-2 | 2.09e-1 | 1.43× |
| Wave | 1.719e-1 | 8.783e-1 | 3.424e-1 | **5.11× (fails)** |

**Cost.** The latent solve is mesh-independent: Poisson 19.7–20.0 ms across a 289× DOF
range (FOM 5.6 → 96.0 ms); Burgers rollout 258–264 ms across 64× DOF (FOM 197 → 2148 ms,
end-to-end 8.0× at N=256) — but only after hyper-reducing the **cold start** as well as the
stepping.

> ### ⚠ EVERY SPEEDUP NUMBER ON THIS PAGE IS UNDER CORRECTION (opened 2026-08-17)
>
> **Do not quote a speedup from this tree until `exp/2026-08-17-rom-warmstart-fom` lands.**
> Accuracy numbers are unaffected — only the denominators are wrong, and they are wrong in the
> direction that flatters us.
>
> - **Burgers:** the FOM baseline is a *fixed-8-Newton* rollout (8 × 50 = 400 Newton steps and
>   400 BiCGStab solves per rollout, by construction), roughly **4× over-converged**. It does far
>   more work than the stated accuracy requires.
> - **Poisson:** the FOM baseline is CG at `tol=1e-13` — the tolerance used to *manufacture the
>   truth data*, not one a deployment would ask for. A fixed very-tight tolerance inflates the
>   baseline exactly as a fixed iteration count does. The archived run does not even reach it
>   (achieved 7.0e-11 at N=512), which makes the cost harder to justify, not easier.
>
> Affected: the Burgers 8.0× end-to-end and the 0.72× → 7.96× N-ladder; the Poisson 0.3× → 4.9×
> solve-only and 3.6× end-to-end ladders. **The correction multiplier is N-dependent, so it
> changes the SLOPE of the N-ladders, not just their level — never apply it as a single scalar.**
> Prefer the hardware-free multiplier (ratio of Newton/CG iterations actually performed against
> the fixed count) over the wall-clock ratio when correcting numbers measured on other GPUs.
>
> What survives: all ROM-side numbers (only the denominator changes), every accuracy and
> ceiling result, and the qualitative mesh-independence claim, which is about scaling — both
> baselines scale similarly, so the slope argument holds even as the levels move.

**Where it does not pay.** Heat: accurate but a direct reduced POD-Galerkin solve is
13–38× faster than the FOM, so nothing iterative competes on a linear parabolic problem.
Wave: fails structurally (below). Richer families: the advantage collapses (23× → 1.1×
going from 1 to 3 bump sources) because 512 training samples do not cover a 12-dim family —
a data limit, not a solver limit (the ROM stays within 1.0–1.6× of its own ceiling).

---

## Operating rules (each of these cost us a run)

1. Minimise the **weak form**, not the pointwise residual.
2. **M > k**, comfortably. At `M = k` the objective collapses (heat `M=16,K=16` → 9.0e-2
   against a 6.3e-3 ceiling; burgers POD `k=64,M=64` diverged).
3. **Hyper-reduce the cold start**, or the online path stays mesh-bound.
4. Fit NNLS quadrature weights on **decoder-output** snapshots, never residual snapshots.
   Meshfree candidate pools work as well as grid nodes. `m ≈ 4M` points is the knee.
5. **Never** use random / importance / off-grid strong-form collocation on localised
   families — every such arm failed (0.15–0.99 error) on every PDE.
6. More latent dimensions is **not** better past the family's intrinsic dimension.
7. On Burgers, keep the FOM-exact upwind operator inside the weak advection term; the
   continuum by-parts form converges to the continuum solution, drifting O(h) from the FOM.

---

## Why wave fails (do not re-litigate this with knobs)

On a **fixed** subspace the Newmark/CN recurrence reduces to `S(c_{n+1} + c_{n−1}) = 2 T c_n`
with `S`, `T` symmetric — the same matrix on both neighbours, i.e. time-reversible, which is
the structure that makes the scheme conservative. On a **nonlinear** manifold `S` and `T`
depend on `z`, that symmetry is gone, and nothing bounds the per-step energy change. Measured:
end-time energy ratio 0.27 (ours) vs 1.000003 (POD). Refining the ROM time step 5× lowers the
time-discretisation floor 28.9× and *raises* ROM error 7.5%, so smaller steps are not the fix.
The hyper-reduction half of the recipe works fine there (256 points beat the full grid by 1.9%).

The open problem is **structure-preserving latent stepping**, not tuning.

---

## State as of 2026-08-17

- All four cells complete, committed, pushed, cluster dirs deleted, results checksum-verified.
- `wad_n128_k8` (Slurm 2481538) **finished 2026-08-17** — COMPLETED, exit 0:0, 13:47:03. Pulled
  with matching sha256 manifests, folded into the wave README and SUMMARY_TABLES on both
  `exp/2026-08-16-wave2d-rom-latent-stepping` and this branch, cluster dir deleted. The cell now
  has all 16 coordinate variants, 15 POD control arms and the timing block; zero blow-ups across
  all 31 arms. New reading: hyper-reduction's mesh-independence **is** confirmed on wave — the
  coordinate/POD per-step ratio falls 5.39x -> 4.64x (k=8) and 2.56x -> 1.66x (k=64) from N=64 to
  N=128, and the coordinate speedup vs the FOM rises 0.158x -> 0.204x while POD k=64's falls
  0.404x -> 0.339x. Accuracy is flat in N on both arms, so the wave failure is not a resolution
  artefact. NOTE: those cells ran on different A100s, so only **within-cell ratios** are
  admissible across them; raw cross-cell milliseconds are not.

### Running now (2026-08-17)

Two experiments launched off this branch, one agent each, both Poisson-2D + Burgers-2D:

- `exp/2026-08-17-cost-to-tolerance` — the k x N cost surface and the ISO-ERROR PARETO. Stops on
  relative reduction of the weak-form objective, tau in {1e-1,1e-2,1e-3}; cost and accuracy from
  the same invocation. Cluster namespace `ctol/`.
- `exp/2026-08-17-rom-warmstart-fom` — ROM solution as the FOM's initial guess, finish to FOM
  accuracy; total cost vs ROM tolerance vs N. Cluster namespace `wsfom/`.

Two things learned the hard way on 2026-08-17, both now guarded:

1. **`scancel` is shared-account dangerous.** A `scancel --name=<comma,separated,list>` matched
   nothing, degraded to an empty selector, and cancelled EVERY job on the account — both agents'
   fleets at once. Only ever cancel an explicit list of numeric job IDs you submitted; check
   `squeue -u tawal01 -o '%.10i %.20j'` first. `cost-to-tolerance/cluster/cancel.sh` enforces this.
2. **A pinned batch env var silently undid a fix.** `N_POD_TRAJ=128` in the job environment
   overrode the driver default and would have re-imposed the POD training-set handicap that the
   Codex audit had just removed — an unfair-to-baseline result with nothing in the output to show
   it. Check that batch env blocks do not shadow the values you just corrected.

Early partial signal (Poisson, N<=256): cost(k) looks genuinely mesh-independent (normalised at
k=8 the curve is 2.67/1.13/1.21/1.00 at N=32 vs 2.38/1.09/1.11/1.00 at N=64), and tau=1e-3 is
largely unreachable because the objective's own floor is ~5e-3 relative reduction. Provisional
until the full grid lands.
- Every run: `jax_backend=gpu`, f64, `JAX_DEFAULT_MATMUL_PRECISION=highest`, one job per
  directory, data regenerated on the cluster from seed.
- Two Codex audits per cell (harness before the fan-out, results after). The results pass
  re-derived 1 311 table cells, 321 aggregates, 143 timing medians and 8 figures with no
  error in any generated table; 19 hand-written prose errors were fixed.

---

## Next, in the order I would run it

1. ~~**Iso-error cost comparison (the important one).**~~ **IN FLIGHT** — this is
   `exp/2026-08-17-cost-to-tolerance`, above. At equal `k` we are far more accurate, but POD's
   *per-step* cost is 4–8× lower (Burgers N=64: POD 35–73 ms vs coordinate 260–277 ms), so
   POD-64 may dominate us at N=64. No speed claim may be made until that cell lands.
2. **Training-set-size ladder** on the 2- and 3-bump Poisson families, to separate the
   generalisation limit from an architectural one.
3. **Structure-preserving latent stepping** for wave (symplectic / energy-projected).
4. Reviewer-requested baselines: tuned FNO, POD-DeepONet, SMA cold-start, preconditioned and
   direct classical solvers, L-shaped domain, 3D.
5. Decide the merge of `fix/heat-rollout-warm-start` into `main` (still open; main's public
   heat ROM rollout is frozen after step 1, so its published numbers are unreproducible).

---

## Running things

Compute rules are in `CLAUDE.md` (unchanged): Tufts is the default, local GB10 for
sub-minute smokes only, ≤3 concurrent local `jaxrun` jobs, one cluster job per directory,
`jax_backend=gpu` preflight mandatory.

```bash
# local smoke (one slot)
source /etc/profile.d/jax-mem.sh
PY=/home/tahmid/Dev/.venv/bin/python
cd experiments/burgers2d-rom-latent-stepping
N=16 K_LAT=4 AD_STEPS=400 JAX_DEFAULT_MATMUL_PRECISION=highest jaxrun "$PY" blat_train_ad.py smoke

# cluster cell: generate a job dir, scp it into paralab, submit (never stage via login /tmp)
cd experiments/<cell> && cluster/make_cell.sh <name> && cluster/launch.sh <name>

# rebuild the illustrated report after new results
cd reports && "$PY" make_report_figs.py && "$PY" build_report.py
```

`reports/build_report.py` inlines the figures into `Coordinate-ROM-Findings.md` for
publishing; `reports/report_src.md` is the editable source.
