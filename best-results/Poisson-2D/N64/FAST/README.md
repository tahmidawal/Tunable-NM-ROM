# Poisson-2D N=64 — FAST variant

| Metric | Value |
|---|---|
| val_rel_l2 | 1.996e-2 |
| speedup | **324.35×** |

## Hyperparameters

| | |
|---|---|
| k | 8 |
| CP rank | 512 |
| GN max_iters | **4** |
| CG tol | 1e-3 |
| eq_mode | nnls |
| n_eq | ~640 |
| epochs | 100 000 |

## Files

- `poisson_nmrom_sweep.py`
- `generate_poisson_data_2d.py`
- `config_used.json`
- `checkpoint_b0d2124d5d95.pkl`

## Reproduce

```bash
python generate_poisson_data_2d.py --N 64
python poisson_nmrom_sweep.py --config config_used.json
```

## Source

`20260416-NeurIPS/Poisson-2D/sweeps/res64_maxiters4_eq/`.

max_iters=4 (vs 8 in ACC) trades a ~40× rel-L2 penalty for 1.8× more speedup.
