# Tunable-ROM

Tunable nonlinear manifold reduced-order models (NM-ROM) for parametric PDEs. This directory contains two self-contained, publishable Python packages — one per PDE family — plus a curated `best-results/` archive of the runs that produced the numbers in our paper.

```
Tunable-ROM/
├── heat/             # tunable-rom-heat package: Heat 2D + 3D
├── poisson/          # tunable-rom-poisson package: Poisson 2D + 3D
└── best-results/     # frozen scripts + checkpoints from the original sweeps
    ├── Heat-2D/      N{64,128,256} × {FAST, ACC}
    ├── Heat-3D/      N32 BEST, N64 BEST, N128 × {FAST, ACC}
    ├── Poisson-2D/   N{64,128,256} × {FAST, ACC}
    └── Poisson-3D/   N{32,64,128} × {FAST, ACC}, N256-CG × {FAST, ACC}
```

## What each subdirectory is

### `heat/` and `poisson/` — the clean packages
Self-contained, pip-installable Python packages intended for public release. Each contains:

- A library (`src/tunable_rom_{heat,poisson}/`) with the ViT encoder, decoder, FOM, EQ-NNLS hyper-reduction, and NM-ROM solver.
- YAML configs for every (N, d) resolution we report.
- CLI scripts: `generate_data`, `train`, `run_rom`.
- A pytest smoke test that exercises the full pipeline at $N = 8$.
- GitHub Actions CI that runs the smoke test on every push.
- MIT license, `.gitignore`, `pyproject.toml`.

Code design follows the architecture reports in the original Autoresearch scripts but strips κ-conditioning, manifold-regularisation, dual entry points, and other autoresearch residue. The two packages share an encoder design but differ in their decoders (Heat: plain CP; Poisson: **LinearCP** with a load-bearing linear skip for Gauss-Newton cold-start regularity) and their solver loop (Heat: time-stepped rollout; Poisson: single elliptic solve).

### `best-results/` — the frozen sweep archive
The actual scripts + checkpoints + per-cell READMEs that produced the numbers in our paper. Kept for reproducibility and provenance. Code style mirrors the original Autoresearch experiments (one `experiment.py` per cell, ~1000 lines, organic growth). Read this if you need to verify exact numbers or hyperparameter choices; read the clean packages if you want to extend or build on top.

## Headline results

### Heat

| N | d | val rel-L2 | Speedup |
|---|---|---|---|
| 64 | 2D | 5.21e-3 | 39.6× |
| 128 | 2D | 1.02e-2 | 37.8× |
| 256 | 2D | 1.08e-1 | 47.5× |
| 32 | 3D | 9.30e-2 | 176.4× |
| 64 | 3D | 1.76e-2 | 269.3× |
| 128 | 3D | 4.33e-2 | 112.1× |

### Poisson

| N | d | val rel-L2 | Speedup |
|---|---|---|---|
| 64 | 2D | 4.84e-4 | 180.2× |
| 128 | 2D | 1.08e-2 | 183.4× |
| 256 | 2D | 1.20e-1 | 149.4× |
| 32 | 3D | 1.11e-2 | 84.2× |
| 64 | 3D | 6.62e-3 | 111.3× |
| 128 | 3D | 1.05e-2 | 120.7× (FAST, rank=768) |
| 256 (CG) | 3D | 8.44e-3 | 15.3× |

## Publishing the packages

Each subdirectory under `heat/` and `poisson/` is a standalone repo. To push either to its own GitHub home:

```bash
cd heat   # or poisson
git init
git add .
git commit -m "Initial public release"
git remote add origin git@github.com:<user>/tunable-rom-heat.git
git push -u origin main
```

The smoke-test GitHub Action will run on the first push to `main`.

## License

MIT (per-package). See `heat/LICENSE` and `poisson/LICENSE`.
