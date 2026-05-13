# Heat-3D N=128 — ACC variant

| Metric | Value |
|---|---|
| val_rel_l2 | **4.33e-2** |
| speedup | 112.11× |

## Hyperparameters

| | |
|---|---|
| k | 40 |
| CP rank | 512 |
| patch_size | 16 |
| epochs | 30 000 (finetune from N128_run10 base) |
| GN max_iters | 12 |
| GN tol | 1e-3 |
| n_eq_samples | 16 |
| eq_mode | nnls |

## Files

- `experiment.py`
- `run.py`
- `generate_heat_data.py`
- `checkpoint_vitcp.pkl` — from `Autoresearch/Heat-3D/runs/N128_run10/`

## Reproduce

```bash
python generate_heat_data.py --N 128
python experiment.py
```

## Source

`20260423-NEURIPS/Autoresearch/Heat-3D/N128/` — slurm-609078 (112.11× / 4.33e-2).

**Known floor:** Heat-3D N=128 has a low-κ AE-reconstruction ceiling. Cases with κ < 0.03 cannot be improved by more epochs or higher k — the bottleneck is data-distribution / manifold capacity, not solver iteration count. The 4.33e-2 ACC is averaged over the standard benchmark; some low-κ cases will be significantly worse.
