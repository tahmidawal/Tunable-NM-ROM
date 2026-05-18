# INR-SIREN Speed/Accuracy Trade-off

Started 2026-05-18 from decoder-explorer iter-7 SOTA (rel-L² 2.88e-3, 0.50× FOM).
Cold-start only. No cheating (no test-set leakage, no oracle warm-start, no FOM-assist).

## Baseline (commit 8b2dc44, iter-7 composite)
- coarse: `poisson2d_siren_anchor_v1.pkl`, N_eq=500, 30 iters
- fine:   `poisson2d_siren_lap_v1.pkl`,    N_eq=8000, 30 iters
- median rel-L² = 2.88e-3, median wall-clock = 551 ms, FOM median = 274 ms → 0.50× FOM
- median iters: coarse 30 (capped!), fine 5

## Levers being explored
- **B** — taper coarse N_eq aggressively (500 → 100 → 50 → 25)
- **C** — Pareto sweep over (N_eq_coarse, N_eq_fine, max_iters_coarse, max_iters_fine)
- **A′** — custom Jacobian-emitting SIREN forward (replace `jax.jacfwd`)
- **D** — affine-in-z decoder branch (separate model; CP-style Jacobian)

## Reproducibility
Results logged in `results.csv` (one row per evaluated config), keyed by git commit.

Checkpoints, EQ index files, and data are symlinked into
`Experiments/exploring-decoders/` and **not** committed here.
