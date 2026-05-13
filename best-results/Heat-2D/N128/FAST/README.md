# Heat-2D N=128 — FAST variant

| Metric | Value |
|---|---|
| val_rel_l2 | 2.18e-2 |
| speedup | **64.77×** |

## Hyperparameters

Same trained AE as `../ACC/` — `checkpoint.pkl` is a symlink. Only ROM-time settings differ.

| | |
|---|---|
| k / rank / patch / embed | 64 / 256 / 8 / 96 |
| epochs | 150 000 |
| GN max_iters | **8** |
| GN tol | 1e-3 |
| AE-reuse | yes |

## Files

- `experiment.py` (configure `max_iters=8` before running)
- `run.py`
- `generate_heat_data_2d.py` + `upsample_data.py`
- `checkpoint.pkl` → `../ACC/checkpoint_N128_k64_r256_h256_p8_e96_nh4_nl4_ep150000_bs16_lr0.001_wd0.002_s0.pkl`

## Reproduce

```bash
# data already generated for ACC; reuse it
python experiment.py    # gn_max_iters=8, gn_tol=1e-3
```

## Source

`20260423-NEURIPS/Autoresearch/Heat-2D/N128_fixed/` (commit 1724e07 — Run5, AE-reuse speedup anchor).
