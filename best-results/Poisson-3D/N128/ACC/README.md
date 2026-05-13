# Poisson-3D N=128 — ACC variant

| Metric | Value |
|---|---|
| val_rel_l2 | **1.05e-2** |
| speedup | 35.15× |

## Hyperparameters

| | |
|---|---|
| k | 16 |
| CP rank | 512 |
| GN max_iters | 8 |
| eq_mode | nnls |

## Files

- `poisson3d_nmrom_sweep.py` — training + ROM driver from the older sweep era
- `generate_poisson_data.py`
- `config_used.json` — exact hyperparameters
- `checkpoint.pkl`

## Reproduce

```bash
python generate_poisson_data.py --N 128
python poisson3d_nmrom_sweep.py --config config_used.json
```

## Source

`20260416-NeurIPS/Poisson-3D/sweeps/res128_k16/`.

**Note:** The FAST variant (`../FAST/`) has val_rel_l2 within noise of this entry (1.07e-2 vs 1.05e-2) but 3.4× more speedup. For most use cases prefer FAST. ACC is included for reproducibility of the older sweep headline.
