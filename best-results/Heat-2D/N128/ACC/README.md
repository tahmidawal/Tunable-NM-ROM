# Heat-2D N=128 — ACC variant

| Metric | Value |
|---|---|
| val_rel_l2 | **1.02e-2** |
| speedup | 37.83× |

## Hyperparameters

| | |
|---|---|
| k / CP rank | 64 / 256 |
| patch_size / embed_dim | 8 / 96 |
| n_heads / n_layers | 4 / 4 |
| epochs | 150 000 |
| batch_size | 16 |
| lr / weight_decay | 1e-3 / 2e-3 |
| GN tol | 5e-3 |
| seed | 0 |

## Files

- `experiment.py` — training + ROM driver
- `run.py`
- `generate_heat_data_2d.py` + `upsample_data.py` — upsample N=64 → N=128
- `checkpoint_N128_k64_r256_h256_p8_e96_nh4_nl4_ep150000_bs16_lr0.001_wd0.002_s0.pkl`

## Reproduce

```bash
python generate_heat_data_2d.py --N 64
python upsample_data.py --src training_data_2d_64.pkl --dst training_data_2d_128.pkl --N 128
python experiment.py
```

## Source

`20260423-NEURIPS/Autoresearch/Heat-2D/N128_fixed/` (commit 24dc331 — Pareto-dominates the earlier Run3 at 1.05e-2).

Looser GN tolerance (5e-3) regularizes convergence at low condition numbers, matching the N=64 pattern.
