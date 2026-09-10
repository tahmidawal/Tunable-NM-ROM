# Tunable NM-ROM — Heat Equation

A tunable nonlinear manifold reduced-order model (NM-ROM) for the parametric Heat equation in 2D and 3D. The model combines a ViT encoder and a CP-tensor decoder, with empirical-quadrature (EQ) hyper-reduction and a JIT-compiled latent-space Gauss-Newton solver. A single trained autoencoder supports a continuous accuracy/speedup tradeoff at inference time by varying Gauss-Newton iterations and tolerance.

## Reported results

| N | d | val rel-L2 | Speedup |
|---|---|---|---|
| 64 | 2D | 5.21e-3 | 39.6× |
| 128 | 2D | 1.02e-2 | 37.8× |
| 256 | 2D | 1.08e-1 | 47.5× |
| 32 | 3D | 9.30e-2 | 176.4× |
| 64 | 3D | 1.76e-2 | 269.3× |
| 128 | 3D | 4.33e-2 | 112.1× |

Each (N, d) pair has a config under `configs/`. Speedup is the median over a held-out parameter set; rel-L2 is the mean.

## What it solves

Linear Heat equation
$$\partial_t u(\mathbf{x}, t) = \kappa \, \Delta u(\mathbf{x}, t), \qquad u\big|_{\partial\Omega} = 0$$
on $\Omega = [0,1]^d$ with parametric diffusivity $\kappa \in [0.01, 0.5]$ (log-uniform) and Gaussian-blob initial conditions. Discretized with second-order finite differences and backward Euler ($dt = 0.005$, 50 steps).

## Architecture

- **FOM**: matrix-free implicit operator + `jax.scipy.sparse.linalg.cg`.
- **Encoder**: ViT — patchify $(N)^d$ into tokens of size $\text{patch}^d$, embed, $L$ transformer blocks, mean-pool, project to $k$-dim latent.
- **Decoder**: plain CP — small MLP maps $z \in \mathbb{R}^k$ to rank-$R$ channel weights, contracted with per-axis factor matrices $W_x, W_y[, W_z] \in \mathbb{R}^{R \times N}$.
- **EQ hyper-reduction**: NNLS selects a sparse subset of interior grid nodes. The decoder Jacobian is materialized through a precomputed $V_{eq}$ basis (never via `jax.jacfwd` on the full grid).
- **Solver**: Levenberg-Marquardt Gauss-Newton on the latent code + amplitude scale, with `jax.lax.fori_loop` over time steps and `jax.lax.while_loop` over GN iterations. The entire rollout compiles to a single XLA program.

## Install

```bash
pip install -e .
# optional: dev tools for tests
pip install -e ".[dev]"
```

JAX with GPU support is recommended for any $N \geq 64$ in 3D:

```bash
pip install --upgrade "jax[cuda12]"
```

## Reproduce a result

```bash
# 1. Generate FOM training trajectories.
python -m scripts.generate_data --config configs/heat3d_n64.yaml --out data/heat3d_n64.npz

# 2. Train the ViT-CP autoencoder.
python -m scripts.train --config configs/heat3d_n64.yaml \
    --data data/heat3d_n64.npz --out checkpoints/heat3d_n64.pkl

# 3. Build EQ and benchmark the ROM rollout.
python -m scripts.run_rom --config configs/heat3d_n64.yaml \
    --data data/heat3d_n64.npz --ckpt checkpoints/heat3d_n64.pkl
```

## Tunability — one model, a Pareto curve

At inference, the **same trained autoencoder** can be operated at different accuracy/speed points by changing two ROM-time knobs in the config:

- `gn_max_iters` — upper bound on Gauss-Newton iterations per timestep.
- `gn_rel_tol` — relative gradient-norm tolerance.

Decreasing both gives higher speedup at the cost of rel-L2. The reported numbers above use a tight setting; the matching `FAST` variants in our paper run the same checkpoint with `gn_max_iters` reduced by a factor of ~2.

## Repository layout

```
heat/
├── src/tunable_rom_heat/
│   ├── models/        ViT encoder, CP decoder, AE
│   ├── fom/           Heat FOM (Laplacian stencils, CG step, trajectory gen)
│   ├── eq/            EQ-NNLS node + weight selection
│   ├── solver/        NM-ROM Gauss-Newton solver
│   └── utils/         Training loop, config loading, checkpoint I/O
├── configs/           One YAML per (N, d) cell
├── scripts/           CLI entry points
├── tests/             Smoke tests (run by CI)
└── pyproject.toml
```

## Citation

Paper in submission. Companion repository for the Poisson equation: [`tunable-rom-poisson`](../poisson).

## License

MIT.
