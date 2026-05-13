# Poisson-3D N=64 — FAST variant

| Metric | Value |
|---|---|
| val_rel_l2 | 7.69e-3 |
| speedup | **125.14×** |

## Hyperparameters

| | |
|---|---|
| k | 16 |
| CP rank | 512 |
| GN max_iters | 11 |
| GN tol | 5e-2 |
| eq_mode | nnls |

## Files

- `experiment.py`
- `run.py`
- `generate_poisson_data.py`
- `checkpoint.pkl` — from `Autoresearch/Poisson-3D/runs/N64_run26/`

## Reproduce

```bash
python generate_poisson_data.py --N 64
python experiment.py
```

## Source

`20260423-NEURIPS/Autoresearch/Poisson-3D/N64/` — slurm-36761289 (125.14× / 7.69e-3).
2.3× faster than the older sweep headline (53.51×). Same checkpoint as `../ACC/`.
