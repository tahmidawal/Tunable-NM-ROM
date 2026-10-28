# Speed/accuracy trade-off — final summary

Branch: `inr-siren-speed-accuracy`, started 2026-05-18 from `decoder-explorer`.
Cold-start only. No cheating (no per-sample warm-up, no oracle, no FOM-assist).
Poisson-2D, N=128, parametric forcing `A·∏sin(kᵢπx)`, FOM solver = CG float64.

## Headline Pareto curve

| Decoder | rel-L² median | rel-L² p90 | frac<1e-2 | Speedup vs FOM CG | wall (ms) |
|---|---|---|---|---|---|
| iter-7 baseline (proper warmup) | 2.88e-3 | 6.61e-3 | 0.94 | 0.13× | 2152 |
| Fast SIREN @ ic30/if20 | 2.88e-3 | 6.66e-3 | 0.94 | **3.44×** | 80 |
| Fast SIREN @ ic20/if10 | 2.89e-3 | 9.87e-3 | 0.91 | **3.78×** | 72 |
| Fast SIREN @ ic10/if20 | 3.02e-3 | 1.00e-2 | 0.89 | **4.17×** | 66 |
| **linaff_v5** | **9.81e-3** | **1.42e-2** | 0.53 | **139×** | **1.97** |
| linaff_v3 | 1.14e-2 | 1.81e-2 | 0.34 | 247× | 1.11 |
| linaff_v4 | 1.31e-2 | 1.99e-2 | 0.22 | 274× | 1.00 |
| linaff_v2 | 1.81e-2 | 3.16e-2 | 0.12 | 320× | 0.85 |
| affine_v1 (broken) | 1.95e-1 | 5.88e-1 | 0.02 | 194× | 1.42 |

## Net improvement vs iter-7 baseline

- @ rel-L² ~3e-3 (publication-grade): **3.4× FOM** (was 0.13× FOM) = **27× wall-clock improvement**
- @ rel-L² ~1e-2: **139× FOM** = **1100× wall-clock improvement**
- @ rel-L² ~1.8e-2: **320× FOM** = **2500× wall-clock improvement**

## What worked

### Lever A′ — `FastINRNMROMSolver` (vmap(jacrev) instead of jacfwd)
Replaces `latent_dim`-parallel forward passes with one fwd + one bwd per
EQ point. Single-stage Jacobian cost drops 3.2× (verified to z-agreement
6.7e-9 with baseline). Used for every SIREN result above.

### Lever C — Pareto sweep
240-cell grid revealed that the iter-7 SOTA config (c500/f8000, 30/30 iters)
was a pathological point on the Pareto curve. The frontier is at
**c200/f500** with iters in {ic20-30, if10-20}. Most of the speedup came
from realizing **fine_neq=500 is enough** — no need for 8000.

### Lever D — LinearAffineZDecoder + Laplacian-residual loss
`u(x;z) = V(x)@z + b(x)` where `V(x) = Φ(x)@W_A` and Φ is a SIREN feature
trunk. **Jacobian is constant in z** → GN converges in **2 iterations**.
- The naive affine_z with nonlinear `A(z)` failed (rel-L² 0.2-0.8): non-
  constant Jacobian + cold-start z=0 fell into a basin away from good z.
- Switching to **linear A(z)** alone got rel-L² 0.25 — manifold rich
  (oracle 3-24e-3) but encoder/ROM consistency gap.
- Adding **Laplacian residual loss** during training (`||K@decode - F||²`
  on M_lap=256-384 random interior stencils) closed the gap completely.
  After lap training, `||z_enc|| ≈ ||z_rom||` within 1% across all
  diagnostic samples — encoder embeds parameters at the residual-min z.

### Architecture/training scaling for linaff:
- v2 (Phi 512w/5L, 50k ep, λ_lap=0.1): 320× / 1.81e-2
- v4 (same arch, 100k ep, λ_lap=0.2): 274× / 1.31e-2 — training-time-limited
- v3 (Phi 640w/6L, 100k ep, λ_lap=0.05): 247× / 1.14e-2 — capacity helped
- **v5 (Phi 768w/5L, latent 24, 150k ep, λ_lap=0.15): 139× / 9.81e-3** — best
  Diminishing speed with growing accuracy = clean Pareto curve.

## What didn't work

### Affine with nonlinear `A(z)` MLP (affine_v1-v4)
Globally low-rank Jacobian (rank ≤ latent_dim) combined with **non-constant**
Jacobian created a basin near z=0 that GN couldn't escape. Tried bigger
hidden_dim (v2), bigger latent (v3), and no anchor (v4) — all rel-L² > 0.2.
The fix was the LINEAR A(z), which makes the Jacobian constant.

### Removing anchor + L2(z) regularizers (affine_v4)
Made things worse (rel-L² 0.76 vs v2's 0.33). Confirmed that for nonlinear
A(z), the problem is solver dynamics, not encoder over-regularization.

## Files of note
- `src/.../solver/nm_rom_fast.py` — `FastINRNMROMSolver` (lever A′)
- `src/.../solver/nm_rom_affine.py` — `AffineNMROMSolver` (lever D)
- `src/.../decoders/affine_z.py` — `AffineZDecoder` (nonlinear A)
- `src/.../decoders/linear_affine_z.py` — `LinearAffineZDecoder` (linear A, winner)
- `scripts/timing_harness.py` — per-phase wall-clock decomposition
- `scripts/pareto_sweep.py`, `pareto_sweep_fast.py`, `pareto_frontier.py`
- `scripts/diagnose_affine_manifold.py` — oracle vs ROM rel-L² split
- `results.csv` — every cell + meta line keyed by commit
- `runs/pareto/v1/results_pareto_v1.csv` — baseline-solver sweep (complete, 90 cells)
- `runs/pareto/fast/results_pareto_fast.csv` — fast-solver sweep (complete, 240 cells)
- `runs/rom/{affine,linaff}_v*/*.json` — per-checkpoint ROM eval JSONs
