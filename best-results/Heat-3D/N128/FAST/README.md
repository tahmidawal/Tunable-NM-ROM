# Heat-3D N=128 — FAST variant

| Metric | Value |
|---|---|
| val_rel_l2 | 4.75e-2 |
| speedup | **191.39×** |

## Hyperparameters

| | |
|---|---|
| k | 40 |
| CP rank | 512 |
| patch_size | 16 |
| epochs | (see N128_run2 training config) |
| GN max_iters | low (speedup tuning) |
| n_eq_samples | 16 |
| eq_mode | nnls |

## Files

- `experiment.py`
- `run.py`
- `generate_heat_data.py`
- `checkpoint_vitcp.pkl` — from `Autoresearch/Heat-3D/runs/N128_run2/`

## Reproduce

```bash
python generate_heat_data.py --N 128
python experiment.py
```

## Source

`20260423-NEURIPS/Autoresearch/Heat-3D/N128/` — slurm-503885 (191.39× / 4.75e-2).
The N128_run2 checkpoint also served slower benchmarks (7.52× at the same rel-L2); the speedup comes from solver-side tuning.
