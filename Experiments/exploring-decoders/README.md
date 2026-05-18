# Exploring Decoders — Partial-Decoding Ablation

A controlled comparison of two **implicit-neural-representation (INR) decoders**
for the parametric Poisson NM-ROM autoencoder. Both decoders take a latent
`z ∈ R^k` from the (shared) ViT encoder plus a continuous coordinate
`x ∈ [0,1]^2` and return `u(x)` — i.e. they enable decoding at *any* point
in the domain, not just at the discrete FD grid nodes the main paper uses.

## Why

The paper's `(Linear)CPDecoder` is grid-bound: its CP factors are `(R, N)`
lookup tables indexed by integer node id. To support "give me u at any point
in the domain" — relevant for sensor-placement queries, super-resolution
visualization, and adaptive sampling — the decoder must be a continuous
function of `x`.

This experiment compares two ways of giving a decoder global spatial
coupling (Poisson's Green's function has global support, so a purely local
INR would have to rediscover global coupling through depth alone):

- **Modulated SIREN** — sine-activation MLP with per-layer FiLM modulation
  from the latent. "Global" from sine activations + depth + spectral bias.
- **Cross-attention INR** — query coordinate `x` cross-attends into the
  ViT encoder's per-token features. Explicit content-addressed lookup.

## Scope

- Poisson-2D only (Phase 1). N=128, analytical-PDE data + analytical
  off-mesh ground truth. No ROM, no Gauss-Newton, no EQ.
- Metric: rel-L2 on randomly-sampled in-domain off-mesh points, swept
  across the number of query points `M ∈ {16, 64, 256, 1024, 4096, 16384}`.

## Layout

```
configs/        — one YAML per decoder
scripts/        — generate_data, train, eval_sweep_M
src/tunable_rom_decoders/
  encoder.py            — pinned ViT (matches poisson/)
  fom/poisson_analytical.py  — closed-form Poisson + parameter sampler
  decoders/
    modulated_siren.py
    cross_attn_inr.py
  autoencoder.py
  flops.py              — per-architecture per-query FLOPs counter
  training.py
  utils/config.py
tests/test_decoders.py
runs/           — SLURM logs, plots
data/           — *.npz snapshots
checkpoints/    — *.pkl trained AEs
JOURNAL.tex     — running log of the experiment
```

## Reproduce

```bash
# 1. Smoke test on a compute node.
sbatch run_smoke.slurm

# 2. Full training, both decoders, in parallel.
sbatch run_experiment.slurm poisson2d_siren
sbatch run_experiment.slurm poisson2d_xattn

# 3. M-sweep evaluation after both finish.
python -m scripts.eval_sweep_M \
    --ckpt-siren checkpoints/poisson2d_siren.pkl \
    --ckpt-xattn checkpoints/poisson2d_xattn.pkl \
    --out runs/sweep.json
```
