# Poisson-2D N=256 — FAST variant

| Metric | Value |
|---|---|
| val_rel_l2 | 2.28e-1 |
| speedup | **294.65×** |

## Hyperparameters

| | |
|---|---|
| k | 16 |
| CP rank | 512 |
| GN max_iters | **4** |
| CG tol | 1e-3 |
| eq_mode | nnls |
| n_eq | ~1280 |
| epochs | 80 000 |

## Files

- `poisson_nmrom_sweep.py`
- `generate_poisson_data_2d.py`
- `config_used.json`
- `checkpoint_f0252520f7ca.pkl`

## Reproduce

```bash
python generate_poisson_data_2d.py --N 256
python poisson_nmrom_sweep.py --config config_used.json
```

## Source

`20260416-NeurIPS/Poisson-2D/sweeps/res256_maxiters4_eq/`. Highest speedup across all N=256 runs; ~2× rel-L2 penalty vs N256-ACC.
