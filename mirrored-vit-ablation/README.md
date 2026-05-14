# Mirrored ViT Encoder–Decoder Ablation

Companion ablation for the Tunable-ROM paper: replaces the LinearCP / CP
decoder with a symmetric ViT decoder (mirror of the ViT encoder, MAE-style
un-patchify head), and re-runs the NM-ROM pipeline for Heat-3D N=64 and
Poisson-3D N=64. Purpose: empirical confirmation of the paper's
architecture-section claims about why CP wins.

Baseline targets (from the public package):

| PDE        | val rel-L2 | Speedup |
|------------|------------|---------|
| Poisson-3D N=64 | 6.62e-3 | 111×    |
| Heat-3D    N=64 | 1.76e-2 | 269×    |

## Layout

```
mirrored-vit-ablation/
├── heat/      tunable-rom-vit-heat package (ViT+ViT, no skip)
├── poisson/   tunable-rom-vit-poisson package (ViT+ViT, with linear skip)
├── scripts/   shared helpers (summarize_runs.py)
├── run_smoke.slurm   gates the sweep — pytest on a compute node
└── submit_all.sh     submits all 6 sweep configs
```

Each subpackage is a standalone copy of the corresponding `Tunable-ROM/{heat,poisson}` package with:

* `models/decoder.py` rewritten as `ViTDecoder` (heat) / `LinearSkipViTDecoder` (poisson).
* `models/autoencoder.py` renamed `ViTViTAutoencoder` and rewired to the new decoder.
* `solver/nm_rom.py` rewritten to evaluate the decoder on the full grid each GN iteration and index at stencil nodes (no CP-factor precomputation possible — that is the load-bearing change).
* `eq/nnls.py`'s `build_v_eq` returns only stencil indices.
* `utils/config.py` drops `rank`/`hidden_dim`, adds optional `dec_patch_size`/`dec_embed_dim`/`dec_num_heads`/`dec_num_layers` for asymmetric decoders.

The encoder, FOM, data generators, training loop, EQ NNLS solve, and CLI scripts are unchanged.

## Architectural notes

**Heat decoder.** Plain symmetric ViT: `latent -> Dense -> tokens + pos_embed -> L transformer blocks -> per-token Dense -> un-patchify`. No linear skip — Heat warm-starts each ROM step from the previous latent, so cold-start Jacobian regularity at z=0 isn't required.

**Poisson decoder.** `LinearSkipViTDecoder = unpatchify(W_dir @ z) + ViTBranch(z) + bias`. The skip is load-bearing for cold-start GN convergence: at z=0 the ViT branch's Jacobian is essentially zero (every transformer block multiplies the propagated quantity), so dU/dz|_0 has to come from the skip.

**Solver path.** Self-attention couples all output tokens, so we cannot evaluate the decoder at one stencil node without computing the full grid. Each GN iteration calls `decoder(z)` on the entire (N,)^d field and indexes at EQ stencil nodes. This is the central tradeoff: EQ correctness is preserved (the residual is still evaluated at EQ nodes with EQ weights), but the speedup vs FOM collapses because decoder cost stays O(N^d · ViT) per iteration. The actual numbers are the experimental finding.

## How to run

```bash
cd /cluster/tufts/paralab/tawal01/NMROM-Apr8/20260423-NEURIPS/Tunable-ROM/mirrored-vit-ablation

# 1. Smoke test (gates the sweep).
sbatch run_smoke.slurm
# Wait, then check runs/smoke_<jobid>.out — should end "Heat rc=0 Poisson rc=0".

# 2. Submit the full sweep (6 configs: 3 per PDE).
bash submit_all.sh

# 3. Monitor.
squeue -u $USER

# 4. Once jobs finish, summarize.
python scripts/summarize_runs.py
```

## Configs

Per PDE: A_mirror (decoder = encoder exactly), B_deeper (more decoder blocks), C_wider (larger decoder embed_dim). Training is at 30k epochs for the initial sweep (one-tenth of the baseline 100k, fast enough to iterate before committing to longer runs).

If A/B/C all under-shoot the baseline rel-L2, the next-round playbook is: (a) extend `num_epochs` to 100k on the best config, (b) bump `latent_dim`, (c) lower `peak_lr` if training diverges.

## License

MIT (per-package).
