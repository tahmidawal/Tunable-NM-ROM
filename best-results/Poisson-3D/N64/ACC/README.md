# Poisson-3D N=64 — ACC variant

| Metric | Value |
|---|---|
| val_rel_l2 | **6.62e-3** |
| speedup | 111.26× |

## Hyperparameters

| | |
|---|---|
| k | 16 |
| CP rank | 512 |
| GN max_iters | (more iters than FAST) |
| eq_mode | nnls |
| n_eq_snaps | 600 |
| seed | 0 |

## Files

- `experiment.py`
- `run.py`
- `generate_poisson_data.py`
- `checkpoint.pkl` → symlinked to `../FAST/checkpoint.pkl`

## Reproduce

```bash
python generate_poisson_data.py --N 64
python experiment.py
```

## Source

`20260423-NEURIPS/Autoresearch/Poisson-3D/N64/` — slurm-36761062 (111.26× / 6.62e-3). Seed=0 retrain with n_eq_snaps=600.
