# Heat-3D N=32 — BEST (FAST = ACC)

| Metric | Value |
|---|---|
| val_rel_l2 | **9.30e-2** |
| speedup | **176.43×** |

Single Pareto-optimal point — no FAST/ACC tradeoff at this resolution.

## Hyperparameters

| | |
|---|---|
| k (latent dim) | 32 |
| CP rank | 512 |
| patch_size | 8 |
| epochs | 80 000 |
| GN max_iters | 3 |
| GN tol | 1e-3 |
| n_eq_samples | 16 |
| eq_mode | nnls |

## Files

- `experiment.py` — training + ROM driver
- `run.py`
- `generate_heat_data.py` — generates `training_data_32.pkl` (500 trajectories, log-uniform κ ∈ [0.01, 0.5])
- `checkpoint_vitcp.pkl` — from `Autoresearch/Heat-3D/runs/N32_fixed_run10/`

## Reproduce

```bash
python generate_heat_data.py --N 32
python experiment.py
```

## Source

`20260423-NEURIPS/Autoresearch/Heat-3D/N32_fixed/` — slurm-574170. Checkpoint from `runs/N32_fixed_run10/`. JIT-cache fix (kappa-runtime + fori_loop) boosted speedup 20.7× over the baseline (8.53× → 176.43×).
