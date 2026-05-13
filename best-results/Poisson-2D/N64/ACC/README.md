# Poisson-2D N=64 — ACC variant

| Metric | Value |
|---|---|
| val_rel_l2 | **4.84e-4** |
| speedup | 180.24× |

The lowest validation error across all Poisson-2D resolutions — 128 CP rank gives a tight reconstruction at N=64.

## Hyperparameters

| | |
|---|---|
| k (latent dim) | 8 |
| CP rank | 128 |
| GN max_iters | 8 |
| CG tol | 1e-3 |
| eq_mode | nnls |
| n_eq | ~640 |
| epochs | 100 000 |

## Files

- `poisson_nmrom_sweep.py` — training + ROM driver
- `generate_poisson_data_2d.py` — generates FEM-based Poisson-2D snapshots (700 train + 140 val)
- `config_used.json` — exact hyperparameters used for the trained checkpoint
- `checkpoint_92f776b3428d.pkl`

## Reproduce

```bash
python generate_poisson_data_2d.py --N 64
python poisson_nmrom_sweep.py --config config_used.json
```

## Source

`20260416-NeurIPS/Poisson-2D/sweeps/res64_rank128_eq/`.
