# Heat-2D N=64 — FAST variant

| Metric | Value |
|---|---|
| val_rel_l2 | 1.08e-2 |
| speedup | **68.04×** |

## Hyperparameters

Same trained AE as `../ACC/` — `checkpoint.pkl` is a symlink to the ACC checkpoint. Only ROM-time settings differ.

| | |
|---|---|
| k / rank / patch / embed | 64 / 256 / 8 / 96 |
| epochs | 80 000 |
| GN max_iters | **8** (vs default in ACC) |
| AE-reuse | yes |

## Files

- `experiment.py` — same as ACC (configure `max_iters=8` in the hyperparameter block before running)
- `run.py`
- `generate_heat_data_2d.py`
- `checkpoint.pkl` → symlinked to `../ACC/checkpoint_N64_k64_r256_h256_p8_e96_nh4_nl4_ep80000_bs32_lr0.002_wd0.0005_s0.pkl`

## Reproduce

```bash
# data already generated for ACC; reuse it
python experiment.py    # ensure gn_max_iters=8 in the config
```

## Source

`20260423-NEURIPS/Autoresearch/Heat-2D/N64_fixed/` (commit 4b280a5 — Run1, AE-reuse with gn_max=8).

Trades ~2× rel-L2 for 1.7× speedup.
