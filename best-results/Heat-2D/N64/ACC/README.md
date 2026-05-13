# Heat-2D N=64 — ACC variant

| Metric | Value |
|---|---|
| val_rel_l2 | **5.21e-3** |
| speedup | 39.63× |
| checkpoint size | ~4 MB |

## Hyperparameters

| | |
|---|---|
| k (latent dim) | 64 |
| CP rank | 256 |
| patch_size | 8 |
| embed_dim | 96 |
| n_heads / n_layers | 4 / 4 |
| epochs | 80 000 |
| batch_size | 32 |
| lr / weight_decay | 2e-3 / 5e-4 |
| seed | 0 |
| GN max_iters | (default in `experiment.py`) |

## Files

- `experiment.py` — training + ROM driver
- `run.py` — entry point
- `generate_heat_data_2d.py` — generates `training_data_2d_64.pkl`
- `checkpoint_N64_k64_r256_h256_p8_e96_nh4_nl4_ep80000_bs32_lr0.002_wd0.0005_s0.pkl`

## Reproduce

```bash
python generate_heat_data_2d.py --N 64       # creates training_data_2d_64.pkl
python experiment.py                         # loads checkpoint, runs ROM benchmark
```

## Source

`20260423-NEURIPS/Autoresearch/Heat-2D/N64_fixed/` (commit da48498 — Run0 fix-verify fresh train).

The JIT/fori_loop port boosted speedup 7.7× over the pre-fix baseline (5.10× → 39.63×) at identical accuracy.
