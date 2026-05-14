# Mirrored ViT Encoder–Decoder Ablation — Status Report

**Date:** 2026-05-14 (started 2026-05-13 evening)
**Branch:** `mirrored-encoder-decoder`
**Commits:** see `git log mirrored-encoder-decoder ^main`

## What was built

A standalone ablation environment under `mirrored-vit-ablation/` with two pip packages (`tunable-rom-vit-heat`, `tunable-rom-vit-poisson`) — stripped-down copies of the public-release packages with the LinearCP/CP decoder swapped for a symmetric ViT decoder (MAE-style un-patchify mirror of the ViT encoder). Heat uses a plain mirror (no skip), Poisson keeps a linear skip routed through un-patchify onto the full grid (load-bearing for cold-start Gauss–Newton).

Key change beyond the decoder itself: the solver no longer precomputes CP factor rows at stencil nodes. Self-attention couples all output tokens, so per-node CP-style evaluation is impossible. Each GN iteration evaluates the decoder on the full grid and indexes at EQ stencil nodes. This intentionally exposes the speedup collapse the paper attributes to losing per-node locality.

## Files

```
mirrored-vit-ablation/
├── README.md
├── STATUS.md                    (this file)
├── run_smoke.slurm              gates the sweep
├── submit_all.sh                submits the 6 sweep configs
├── scripts/
│   ├── summarize_runs.py        parses train.log + rom.log -> table
│   └── estimate_params.py       param-count estimator (needs pyyaml)
├── heat/    tunable-rom-vit-heat package
│   ├── src/tunable_rom_vit_heat/
│   │   ├── models/{encoder,decoder,autoencoder}.py
│   │   ├── solver/nm_rom.py
│   │   ├── eq/nnls.py
│   │   └── utils/{config,training}.py
│   ├── scripts/{generate_data,train,run_rom}.py, run_experiment.slurm
│   ├── configs/heat3d_n64_{A_mirror,B_deeper,C_wider,D_longtrain,E_biglatent}.yaml
│   └── tests/test_smoke.py
└── poisson/ tunable-rom-vit-poisson package (same layout)
```

## SLURM jobs submitted

| Job ID | Name | State | Depends on | Stage |
|--------|------|-------|------------|-------|
| 776147 | mirvit_smoke | PENDING (Backfill) | — | pytest smoke gate |
| 776207 | poisson_poisson3d_n64_A_mirror | PENDING | afterok:776147 | data + train + rom |
| 776208 | poisson_poisson3d_n64_B_deeper | PENDING | afterok:776147 | data + train + rom |
| 776209 | poisson_poisson3d_n64_C_wider | PENDING | afterok:776147 | data + train + rom |
| 776210 | heat_heat3d_n64_A_mirror | PENDING | afterok:776147 | data + train + rom |
| 776211 | heat_heat3d_n64_B_deeper | PENDING | afterok:776147 | data + train + rom |
| 776212 | heat_heat3d_n64_C_wider | PENDING | afterok:776147 | data + train + rom |

Smoke job estimated start: ~2026-05-14T02:11 EDT (Backfill priority). Each training job: 8h wall-clock budget, A100, 160G RAM, expected ~1–3h actual.

If smoke fails (exit≠0), SLURM will purge all 6 dependent jobs automatically and they will not run.

## Sweep matrix — first round (30k epochs each)

| Config | Decoder shape | Total params (est.) |
|--------|---------------|---------------------|
| **poisson3d_n64_A_mirror** | mirror of encoder (d_emb=64, L=4) | ~5.3 M (skip 4.2 M) |
| **poisson3d_n64_B_deeper** | mirror, L=6 | ~5.5 M |
| **poisson3d_n64_C_wider**  | d_emb=128, num_heads=8 | ~7 M |
| **heat3d_n64_A_mirror**    | mirror of encoder (d_emb=96, L=6) | ~3.5 M |
| **heat3d_n64_B_deeper**    | mirror, L=8 | ~4.0 M |
| **heat3d_n64_C_wider**     | d_emb=192, num_heads=8 | ~7.5 M |

All comfortably fit on a single A100 (40 GB). No OOM risk.

## Baseline targets

| PDE | val rel-L2 (LinearCP/CP) | Speedup |
|-----|--------------------------|---------|
| Poisson-3D N=64 | 6.62e-3 | 111× |
| Heat-3D N=64    | 1.76e-2 | 269× |

## Round-2 configs (already written, not submitted)

If A/B/C all under-shoot rel-L2, escalate:

| Config | Change |
|--------|--------|
| **{p,h}3d_n64_D_longtrain** | 100k epochs (3.3× more), otherwise A_mirror |
| **{p,h}3d_n64_E_biglatent** | latent_dim doubled (Poisson k=32, Heat k=64) |

Submit by hand once first-round results are in: `cd mirrored-vit-ablation && sbatch <pkg>/scripts/run_experiment.slurm <config_name> all`.

## How to monitor and digest results

```bash
# job state
squeue -u $USER

# once smoke runs, check it
tail -100 mirrored-vit-ablation/runs/smoke_776147.out

# once training jobs land
python mirrored-vit-ablation/scripts/summarize_runs.py
```

The summarizer prints train best-val and ROM mean rel-L2 + median speedup per config, sorted within each PDE.

## Expected story

The most likely outcome — and the one this ablation is designed to document — is:

1. **Heat A_mirror**: probably trains fine, reconstruction rel-L2 in the same order as CP (1e-2 region). ROM solve runs, but **speedup collapses** because the decoder is O(N³ · ViT) per GN iteration instead of O(rank · stencil_size · ViT_MLP).
2. **Poisson A_mirror**: with the linear skip, cold-start GN should converge. Reconstruction may compete with LinearCP (both reach 1e-3 region given enough capacity). **Speedup collapses** for the same reason.
3. **B_deeper / C_wider**: marginal gains on rel-L2, no change in speedup.

If reconstruction matches or beats LinearCP at the cost of speedup, this is a clean Pareto comparison for the paper's architecture-choice section. If reconstruction lags despite the larger decoder, that supports the claim that CP isn't throwing away accuracy.

## Known limitations / open issues

* Cannot run smoke test on login node (compute-node-only python/jax). SLURM smoke gate is the substitute. If smoke fails, the dependent training jobs will be in `DependencyNeverSatisfied` state and need manual `scancel`.
* The solver's `jax.jacfwd(residual)` backprops through the full ViT decoder each step. JIT-compiled but VRAM scales with `n_eq × decoder_size`; A100 should handle the chosen configs but Heat's n_eq=32 helps.
* No data has been generated yet — `data/*.npz` is built by the first stage of each training job (~5 min for Heat-3D, ~30 s for Poisson-3D).

## Next steps when you wake up

1. **Check smoke**: `tail mirrored-vit-ablation/runs/smoke_776147.out` and `runs/smoke_776147.err`. Expected last line: `Heat rc=0  Poisson rc=0`.
2. **If smoke failed**: read the err log, fix, re-submit smoke, re-submit sweep with new dep job ID.
3. **If smoke passed**: watch `squeue` for the 6 sweep jobs. They'll start running once smoke succeeds.
4. **Once any training+rom is done**: `python mirrored-vit-ablation/scripts/summarize_runs.py` for a comparison table.
