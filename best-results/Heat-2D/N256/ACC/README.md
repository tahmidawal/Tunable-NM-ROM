# Heat-2D N=256 — ACC variant

| Metric | Value |
|---|---|
| val_rel_l2 | **1.08e-1** |
| speedup | 47.48× |

## Hyperparameters

| | |
|---|---|
| k / CP rank | 64 / 256 |
| patch_size / embed_dim | **16** / 96 |
| n_heads / n_layers | 4 / 4 |
| epochs | 100 000 |
| batch_size | 8 |
| lr / weight_decay | 1e-3 / 2e-3 |
| GN tol | 2e-2 |
| GN max_iters | 10 |
| seed | 0 |

## Files

- `experiment.py`
- `run.py`
- `generate_heat_data_2d.py` + `upsample_data.py` — upsample N=64 → N=256
- `checkpoint_N256_k64_r256_h256_p16_e96_nh4_nl4_ep100000_bs8_lr0.001_wd0.002_s0.pkl`

## Reproduce

```bash
python generate_heat_data_2d.py --N 64
python upsample_data.py --src training_data_2d_64.pkl --dst training_data_2d_256.pkl --N 256
python experiment.py
```

## Source

`20260423-NEURIPS/Autoresearch/Heat-2D/N256_fixed/` (commit d94f2e2 — Run1 Pareto-dominates Run0).

AE-reuse + looser GN tolerance simultaneously improves speed (+91%) and accuracy (1.17e-1 → 1.08e-1).
Single-snapshot validation floor ~1.11e-1 — bigger latent (k=96, rank=384) didn't break this.
