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
- **One job still running:** `wad_n128_k8` (Slurm 2481538, wave N=128 K=8, ~12 h elapsed,
  15 of 16 arms logged and all consistent with N=64). When it lands: pull with checksums,
  fold the N-flatness row into the wave README/SUMMARY_TABLES, re-audit only the changed
  numbers, delete `/cluster/tufts/paralab/tawal01/wlat/wad_n128_k8`.
- Every run: `jax_backend=gpu`, f64, `JAX_DEFAULT_MATMUL_PRECISION=highest`, one job per
  directory, data regenerated on the cluster from seed.
- Two Codex audits per cell (harness before the fan-out, results after). The results pass
  re-derived 1 311 table cells, 321 aggregates, 143 timing medians and 8 figures with no
  error in any generated table; 19 hand-written prose errors were fixed.

---

## Next, in the order I would run it

1. **Iso-error cost comparison (the important one).** At equal `k` we are far more accurate,
   but POD's *per-step* cost is 4–8× lower (Burgers N=64: POD 35–73 ms vs coordinate
   260–277 ms). We have accuracy-vs-`k` and cost-vs-`N` but **no matched-accuracy Pareto
   curve**, and POD-64 may dominate us at N=64. This must exist before any speed claim.
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
