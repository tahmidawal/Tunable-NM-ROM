# Poisson-2D N=128 — FAST variant

| Metric | Value |
|---|---|
| val_rel_l2 | 2.74e-2 |
| speedup | **320.60×** |

## Hyperparameters

| | |
|---|---|
| k | 12 |
| CP rank | 512 |
| GN max_iters | **4** |
| CG tol | 1e-3 |
| eq_mode | nnls |
| n_eq | ~960 |
| epochs | 100 000 |

## Files

- `poisson_nmrom_sweep.py`
- `generate_poisson_data_2d.py`
- `config_used.json`
- `checkpoint_da75e915a4bd.pkl`

## Reproduce

```bash
python generate_poisson_data_2d.py --N 128
python poisson_nmrom_sweep.py --config config_used.json
```

## Source

`20260416-NeurIPS/Poisson-2D/sweeps/res128_maxiters4_eq/`. ~2.5× rel-L2 penalty for 1.75× speedup over ACC.
