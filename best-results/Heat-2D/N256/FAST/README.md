# Heat-2D N=256 — FAST variant

| Metric | Value |
|---|---|
| val_rel_l2 | 1.36e-1 |
| speedup | **55.61×** |

## Hyperparameters

Same trained AE as `../ACC/` — symlinked checkpoint. ROM knobs differ.

| | |
|---|---|
| k / rank / patch / embed | 64 / 256 / 16 / 96 |
| epochs | 100 000 |
| GN max_iters | **8** |
| GN tol | 2e-2 |
| AE-reuse | yes |

## Files

- `experiment.py` (configure `max_iters=8` before running)
- `run.py`
- `generate_heat_data_2d.py` + `upsample_data.py`
- `checkpoint.pkl` → `../ACC/checkpoint_N256_k64_r256_h256_p16_e96_nh4_nl4_ep100000_bs8_lr0.001_wd0.002_s0.pkl`

## Reproduce

```bash
python experiment.py    # gn_max_iters=8
```

## Source

`20260423-NEURIPS/Autoresearch/Heat-2D/N256_fixed/` (commit fab812b — Run2 speedup anchor).
