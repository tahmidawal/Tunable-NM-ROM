# Poisson-2D N=256 — ACC variant

| Metric | Value |
|---|---|
| val_rel_l2 | **1.20e-1** |
| speedup | 149.35× |

## Hyperparameters

| | |
|---|---|
| k | 16 |
| CP rank | 768 |
| GN max_iters | 8 |
| CG tol | 1e-3 |
| eq_mode | nnls |
| n_eq | ~1280 |
| epochs | 80 000 |

## Files

- `poisson_nmrom_sweep.py`
- `generate_poisson_data_2d.py`
- `config_used.json`
- `checkpoint_02f6bad5ab09.pkl`

## Reproduce

```bash
python generate_poisson_data_2d.py --N 256
python poisson_nmrom_sweep.py --config config_used.json
```

## Source

`20260416-NeurIPS/Poisson-2D/sweeps/res256_rank768_eq/`.

Speedup drops at N=256 vs N=64/N=128; rel-L2 jumps an order of magnitude (250× worse than N=64-ACC). Latent dim k rises with N (8 → 12 → 16) to maintain accuracy.
