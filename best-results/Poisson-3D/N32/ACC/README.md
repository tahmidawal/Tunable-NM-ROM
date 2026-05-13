# Poisson-3D N=32 — ACC variant

| Metric | Value |
|---|---|
| val_rel_l2 | **1.11e-2** |
| speedup | 84.20× |

## Hyperparameters

| | |
|---|---|
| k | 16 |
| CP rank | 512 |
| GN max_iters | 25 |
| eq_mode | nnls |
| min_eq_points | 2000 |
| n_eq_snaps | 200 |

## Files

- `experiment.py`
- `run.py`
- `generate_poisson_data.py`
- `checkpoint.pkl` → symlinked to `../FAST/checkpoint.pkl` (same AE, different ROM knobs)

## Reproduce

```bash
python generate_poisson_data.py --N 32
python experiment.py
```

## Source

`20260423-NEURIPS/Autoresearch/Poisson-3D/N32/` — slurm-36681125 (84.20× / 1.11e-2). Same trained AE as `../FAST/`.

**Critical EQ setting:** `min_eq_points ≥ 1000` is required. Below 1000, GN converges to a local minimum with ~19% error.
