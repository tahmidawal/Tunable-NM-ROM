# Poisson-3D N=128 — FAST variant

| Metric | Value |
|---|---|
| val_rel_l2 | 1.07e-2 (10-case mean) |
| speedup | **120.65×** |

The N=128 breakthrough — higher rank (768 vs 512) + fresh training unlocked speedup matching N=64 FAST while staying within-noise of the older accuracy anchor.

## Hyperparameters

| | |
|---|---|
| k | 16 |
| CP rank | **768** |
| GN max_iters | (speedup-tuned) |
| eq_mode | nnls |

## Files

- `experiment.py`
- `run.py`
- `generate_poisson_data.py`
- `checkpoint.pkl` — from `Autoresearch/Poisson-3D/runs/N128_rank768/`

## Reproduce

```bash
python generate_poisson_data.py --N 128
python experiment.py
```

## Source

`20260423-NEURIPS/Autoresearch/Poisson-3D/N128_rank768/` — slurm-710829.

**Cap warning:** Case-3 anisotropic mode peaks at 3.3% — that's a CP-decoder limit, not a training one. Eight other variants around this peak (different k/rank/seed) fail to break it. k=16/rank=768/seed=42 is a sharp local maximum.
