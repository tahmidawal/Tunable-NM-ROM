# Heat-3D N=64 — BEST (FAST = ACC)

| Metric | Value |
|---|---|
| val_rel_l2 | **1.76e-2** |
| speedup | **269.29×** |

Single Pareto-optimal point. The val_rel_l2 here is 5× better than N=32 (1.76e-2 vs 9.30e-2) and the speedup is 1.5× higher — N=64 strictly dominates N=32 on both axes.

## Hyperparameters

| | |
|---|---|
| k (latent dim) | 40 |
| CP rank | 512 |
| patch_size | 8 |
| epochs | 100 000 |
| GN max_iters | 2 |
| GN tol | 1e-2 |
| n_eq_samples | 32 |
| eq_mode | nnls |

## Files

- `experiment.py`
- `run.py`
- `generate_heat_data.py` — generates `training_data_64.pkl` (500 trajectories)
- `checkpoint_vitcp.pkl` — from `Autoresearch/Heat-3D/runs/N64_fixed_run3/`

## Reproduce

```bash
python generate_heat_data.py --N 64
python experiment.py
```

## Source

`20260423-NEURIPS/Autoresearch/Heat-3D/N64_fixed/` — slurm-609000. Checkpoint from `runs/N64_fixed_run3/`.

The N=64 → N=128 scaling: speedup decreases ~2.4× (269× → 112×) because FOM cost grows 8× and EQ cost grows slower; training shrinks from 500 → 100 trajectories due to 8× larger pickle.
