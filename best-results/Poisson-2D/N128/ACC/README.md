# Poisson-2D N=128 — ACC variant

| Metric | Value |
|---|---|
| val_rel_l2 | **1.08e-2** |
| speedup | 183.38× |

## Hyperparameters

| | |
|---|---|
| k | 12 |
| CP rank | 768 |
| GN max_iters | 8 |
| CG tol | 1e-3 |
| eq_mode | nnls |
| n_eq | ~960 |
| epochs | 100 000 |

## Files

- `poisson_nmrom_sweep.py`
- `generate_poisson_data_2d.py`
- `config_used.json`
- `checkpoint_d3986dad4261.pkl`

## Reproduce

```bash
python generate_poisson_data_2d.py --N 128
python poisson_nmrom_sweep.py --config config_used.json
```

## Source

`20260416-NeurIPS/Poisson-2D/sweeps/res128_rank768_eq/`. Higher k and rank are needed at N=128; speedup saturates around 183× due to FOM solve cost.
