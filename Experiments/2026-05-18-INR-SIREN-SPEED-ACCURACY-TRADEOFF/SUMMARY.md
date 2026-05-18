# Speed/accuracy trade-off — running summary

Branch: `inr-siren-speed-accuracy`, started 2026-05-18 from `decoder-explorer`.

## Headline

Cold-start NM-ROM speedup vs FOM CG @ N=128 Poisson-2D, no cheating:

| Decoder | rel-L² median | rel-L² p90 | Speedup | wall (ms) |
|---|---|---|---|---|
| iter-7 baseline (composite SIREN) | 2.88e-3 | 6.61e-3 | **0.13×** | 2152 |
| Fast SIREN (best @ <1e-2) | 3.02e-3 | 1.00e-2 | **4.17×** | 66 |
| **LinearAffine v2** | **1.81e-2** | **3.16e-2** | **320.7×** | **0.85** |

Three regimes on the Pareto frontier:
- **Accurate (<5e-3)**: Modulated SIREN with Fast solver → 3-4× FOM
- **Balanced (~1e-2)**: SIREN at frontier elbow → 4× FOM
- **Fast (~2e-2)**: **LinearAffine + Lap loss → 320× FOM** ← NEW TIER

vs. the iter-7 starting point, the LinearAffine tier is a **2500× wall-clock
improvement** at ~6× worse rel-L².

## Levers explored

### A′ — custom Jacobian SIREN forward (`FastINRNMROMSolver`)
Replace `jacfwd(residual_vec)` with `vmap(jacrev(decoder_scalar))`. For
scalar-output reverse-mode, 1 fwd + 1 bwd per point ≈ 2× residual; baseline
is `latent_dim × residual`. **Result: 3.2× per-iter; z agreement 6.7e-9.**

### B — coarse N_eq tapering
EQ files at n_eq in {200, 250, 500, 1000, 2000, 4000, 8000}. NNLS floor
≈ 200 in this dataset. All Pareto-optimal points used coarse_neq=200.

### C — Pareto sweep over (coarse_neq, fine_neq, coarse_iters, fine_iters)
- Baseline-solver sweep: 90 cells, frontier at 2.5× FOM @ 3.46e-3.
- Fast-solver sweep (in progress): 240 cells, frontier at **4.17× FOM @ 3.02e-3** (best so far).

### D — Affine-in-z decoders
`u(x;z) = Phi(x) @ A(z) + b(x)`. Phi precomputed once per solve → CP-decoder
asymptote at GN time.

- **affine_v1-v4** (nonlinear A): 80-194× speedup but rel-L² 0.2-0.8.
  Diagnosis: nonlinear A(z) → non-constant Jacobian; GN local quadratic model
  fails far from training z. Cold-start z=0 gets trapped in basin near 0.
- **linaff_v1** (LINEAR A, `u(x;z) = V(x)@z + b(x)`): 270× speedup but
  rel-L² 0.25. Manifold rich (oracle 3e-3) but encoder/ROM consistency gap.
- **linaff_v2** (LinearAffine + Lap loss): **320× FOM, rel-L² 1.81e-2 ✓**
  Lap residual loss closed the consistency gap.
- linaff_v3 in flight: bigger Phi (640w/6L), 2× epochs, lower λ_lap.

## Files of note
- `src/.../solver/nm_rom_fast.py` — FastINRNMROMSolver (A′)
- `src/.../solver/nm_rom_affine.py` — AffineNMROMSolver (D)
- `src/.../decoders/affine_z.py` — AffineZDecoder (D, nonlinear A)
- `src/.../decoders/linear_affine_z.py` — LinearAffineZDecoder (D, linear A)
- `scripts/timing_harness.py` — per-phase wall-clock decomp
- `scripts/pareto_sweep.py` / `pareto_sweep_fast.py` — Pareto grids
- `results.csv` — every cell + meta line keyed by commit
- `runs/pareto/v1/results_pareto_v1.csv` — baseline-solver sweep (complete)
- `runs/pareto/fast/results_pareto_fast.csv` — fast-solver sweep (in progress)
