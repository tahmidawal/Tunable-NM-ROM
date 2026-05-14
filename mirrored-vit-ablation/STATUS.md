# Mirrored ViT Encoder–Decoder Ablation — Status Report

**Date:** 2026-05-14 (started 2026-05-13 evening)
**Branch:** `mirrored-encoder-decoder`
**Commits:** see `git log mirrored-encoder-decoder ^main`

## Headline result so far (Poisson sweep done)

| Config | AE val rel-L2 | ROM mean rel-L2 | Speedup | vs LinearCP baseline |
|--------|---------------|-----------------|---------|----------------------|
| **Baseline (LinearCP)** | 6.62e-3 | 6.62e-3 | 111× | — |
| poisson3d_n64_A_mirror | 1.46e-2 | **1.17** | 0.00× | AE 2.2× worse; ROM diverged |
| poisson3d_n64_B_deeper | 1.75e-2 | **1.12** | 0.00× | AE 2.6× worse; ROM diverged |
| poisson3d_n64_C_wider  | **5.84e-3** | **1.25** | 0.00× | **AE matches baseline**; ROM diverged |

**What this means:** the symmetric ViT decoder *can* match LinearCP on reconstruction (C_wider beats the published 6.62e-3 at only 30k epochs vs 100k), but the **cold-start latent Gauss-Newton solve does not recover the solution** even with the linear skip — every test sample lands at rel-L2 ≈ 1.0–1.3, which is essentially noise. The 0× speedup is the expected speedup-collapse: every GN iteration runs the full ViT decoder on N³ = 262144 nodes regardless of how few EQ nodes we keep.

This is direct experimental evidence for the paper's "ViT decoder fails because per-node locality is required" claim, with one twist beyond the paper: even the *reconstruction-quality issue* alone isn't what kills the ROM — it's the **decoder Jacobian geometry at z = 0**. C_wider's autoencoder *trains better than LinearCP*, and the solver still fails. The skip restores `dU/dz|_0 ≠ 0` (its purpose) but the descent direction doesn't lead to a useful solution.

Heat sweep first attempt all failed at training-time with `CUDA_ERROR_ILLEGAL_ADDRESS` — the symmetric-ViT activations during `jax.jit(eval_loss)` over the full 2550-snapshot val set blew A100 VRAM (CP decoder's activations are tiny, ViT's are 100× bigger). Fixed by chunking the val-loss eval into `batch_size`-sized batches (commit `a32fc7c`). Re-submitted as jobs **778655–778657**; data files are cached on disk so they skip the slow datagen and go straight to training.

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

First smoke job (776147) failed instantly — incorrect module names. The Tufts cluster moved `python/3.10.4`, `cuda/12.2`, `cudnn/8.9.7-12.x` under `modtree/deprecated`, and the binary is `python3` (not `python`). Fixed in run_smoke.slurm and both per-package run_experiment.slurm; previous dependent jobs (776207–776212) were canceled and the sweep was re-submitted.

| Job ID | Name | State | Depends on | Stage |
|--------|------|-------|------------|-------|
| 776380 | mirvit_smoke | PENDING (Backfill) | — | pytest smoke gate |
| 776381 | poisson_poisson3d_n64_A_mirror | PENDING | afterok:776380 | data + train + rom |
| 776382 | poisson_poisson3d_n64_B_deeper | PENDING | afterok:776380 | data + train + rom |
| 776383 | poisson_poisson3d_n64_C_wider | PENDING | afterok:776380 | data + train + rom |
| 776384 | heat_heat3d_n64_A_mirror | PENDING | afterok:776380 | data + train + rom |
| 776385 | heat_heat3d_n64_B_deeper | PENDING | afterok:776380 | data + train + rom |
| 776386 | heat_heat3d_n64_C_wider | PENDING | afterok:776380 | data + train + rom |

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
tail -100 mirrored-vit-ablation/runs/smoke_776380.out

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

1. **Check smoke**: `tail mirrored-vit-ablation/runs/smoke_776380.out` and `runs/smoke_776380.err`. Expected last line: `Heat rc=0  Poisson rc=0`.
2. **If smoke failed**: read the err log, fix, re-submit smoke, re-submit sweep with new dep job ID.
3. **If smoke passed**: watch `squeue` for the 6 sweep jobs. They'll start running once smoke succeeds.
4. **Once any training+rom is done**: `python mirrored-vit-ablation/scripts/summarize_runs.py` for a comparison table.
