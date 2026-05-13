# Poisson-3D N=256 (CG) — ACC variant

| Metric | Value |
|---|---|
| val_rel_l2 | **8.44e-3** |
| speedup | 15.28× |

The lowest rel-L2 across all Poisson-3D N=256 variants.

## Hyperparameters

| | |
|---|---|
| k | 24 |
| CP rank | 512 |
| GN max_iters | 8 |
| eq_mode | nnls |
| FOM solver | CG |

## Files

- `poisson3d_nmrom_sweep.py`
- `generate_poisson_data_cg.py`
- `config_used.json`
- `checkpoint.pkl`

## Reproduce

```bash
python generate_poisson_data_cg.py --N 256
python poisson3d_nmrom_sweep.py --config config_used.json
```

## Source

`20260416-NeurIPS/Poisson-3D/sweeps/res256_k24_cg/`. k=24 compresses to 8.44e-3 but speedup drops to 15.28× as O(k·N³) decoder cost dominates.
