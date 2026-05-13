# Poisson-3D N=32 — FAST variant

| Metric | Value |
|---|---|
| val_rel_l2 | 2.95e-2 | 
| speedup | **116.66×** |

## Hyperparameters

| | |
|---|---|
| k (latent dim) | 16 |
| CP rank | 512 |
| GN max_iters | low (speedup-tuned) |
| eq_mode | nnls |
| min_eq_points | ≥ 2000 |
| n_eq_snaps | 200 |

## Files

- `experiment.py`
- `run.py`
- `generate_poisson_data.py` — analytical Poisson-3D snapshots (interior nodes only for NNLS)
- `checkpoint.pkl` — from `Autoresearch/Poisson-3D/runs/N32_run2/`

## Reproduce

```bash
python generate_poisson_data.py --N 32
python experiment.py
```

## Source

`20260423-NEURIPS/Autoresearch/Poisson-3D/N32/` — slurm-36712011 (116.66× / 2.95e-2). Same trained AE as `../ACC/` — only ROM-side knobs differ.
