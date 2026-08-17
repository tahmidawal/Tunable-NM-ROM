# Generated tables -- ROM-warm-started FOM

Every number below is produced by `wsf_summarize.py` from the JSONs in `runs/`.  Do not edit by hand.

## Poisson-2D

> WALL-CLOCK tables (P1-P3) use ONLY the `consolidated` run -- every N measured sequentially in one job on one GPU.  Iteration counts and accuracy (P4) come from the fanned-out per-N panels where available; P5 checks that the two agree.

## Burgers-2D

> Wall-clock rows (B2) use ONLY the `consolidated` run; iteration counts (B1) and solver health (B3) come from the per-N panels where available.

### B1. Newton and BiCGStab iterations per 50-step trajectory

| tau_FOM | N | Newton (prev) | Newton (extrap) | Newton (ROM) | BiCGStab (prev) | BiCGStab (extrap) | BiCGStab (ROM) |
|---|---|---|---|---|---|---|---|
| 1e-06 | 32 | 98.2 | 54.2 | 92.5 | 770 | 442 | 974 |
| 1e-08 | 32 | 100.2 | 83.5 | 100.2 | 789 | 699 | 1.07e+03 |
| 1e-10 | 32 | 104.8 | 101.5 | 120.8 | 834 | 876 | 1.27e+03 |

### B2. Wall clock (ms) and the hybrid total, consolidated run

| tau_FOM | N | FOM (prev) | FOM (extrap) | FOM (from ROM) | ROM IC | ROM rollout | decode | hybrid total | speedup |
|---|---|---|---|---|---|---|---|---|---|

### B3. Accuracy and solver health

| tau_FOM | N | ROM rel-L2 | hybrid err vs FOM | baseline err vs FOM | BiCGStab breakdowns | non-zero Newton flags |
|---|---|---|---|---|---|---|
| 1e-06 | 32 | 0.016 | 6.36e-07 | 2.35e-07 | 0 | 0 |
| 1e-08 | 32 | 0.016 | 5.11e-09 | 1.12e-09 | 0 | 0 |
| 1e-10 | 32 | 0.016 | 9.44e-11 | 1.51e-10 | 0 | 0 |

## Consolidated-run provenance (the sole source of every cross-N time)

