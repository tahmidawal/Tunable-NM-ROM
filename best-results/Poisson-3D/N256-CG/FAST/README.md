# Poisson-3D N=256 (CG) — FAST variant

| Metric | Value |
|---|---|
| val_rel_l2 | 2.32e-2 |
| speedup | **34.21×** |

CG-data variant only — the non-CG N=256 sweep is contaminated (analytical training data, rel-L2 floor ~0.8).

## Hyperparameters

| | |
|---|---|
| k | 4 |
| CP rank | 512 |
| GN max_iters | 8 |
| eq_mode | nnls |
| FOM solver | CG |

## Files

- `poisson3d_nmrom_sweep.py`
- `generate_poisson_data_cg.py` — CG-based Poisson-3D snapshots (only valid data source at N=256)
- `config_used.json`
- `checkpoint.pkl`

## Reproduce

```bash
python generate_poisson_data_cg.py --N 256
python poisson3d_nmrom_sweep.py --config config_used.json
```

## Source

`20260416-NeurIPS/Poisson-3D/sweeps/res256_k4_cg/`. k=4 minimizes decoder cost at the price of rel-L2.
