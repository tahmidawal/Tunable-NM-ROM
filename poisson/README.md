# Tunable NM-ROM — Poisson Equation

A tunable nonlinear manifold reduced-order model (NM-ROM) for the parametric Poisson equation in 2D and 3D. The model combines a ViT encoder and a **LinearCP** decoder (linear skip + shallow MLP + CP tensor) with empirical-quadrature (EQ) hyper-reduction and a JIT-compiled latent-space Gauss-Newton solver. A single trained autoencoder supports a continuous accuracy/speedup tradeoff at inference time by varying Gauss-Newton iterations and tolerance.

## Reported results

| N | d | val rel-L2 | Speedup |
|---|---|---|---|
| 64 | 2D | 4.84e-4 | 180.2× |
| 128 | 2D | 1.08e-2 | 183.4× |
| 256 | 2D | 1.20e-1 | 149.4× |
| 32 | 3D | 1.11e-2 | 84.2× |
| 64 | 3D | 6.62e-3 | 111.3× |
| 128 | 3D | 1.05e-2 | 35.2× (ACC) / 120.7× (FAST, rank=768) |
| 256 (CG) | 3D | 8.44e-3 | 15.3× |

Each (N, d) pair has a config under `configs/`. Speedup is the median over a held-out parameter set; rel-L2 is the mean.

## What it solves

Negative-Laplacian Poisson with homogeneous Dirichlet boundary conditions:
$$-\Delta u(\mathbf{x}) = F(\mathbf{x}; \mu), \qquad u\big|_{\partial\Omega} = 0$$
on $\Omega = [0,1]^d$ with a parametric tensor-product source
$$F(\mathbf{x}; \mu) = A \prod_{i=1}^{d} \sin(k_i \pi x_i), \qquad k_i \in [1, 3].$$

Discretized with second-order centered finite differences (5-point in 2D, 7-point in 3D). Time-independent — one elliptic solve per parameter, no rollout.

## Architecture highlights

- **FOM**: matrix-free negative-Laplacian operator + `jax.scipy.sparse.linalg.cg`.
- **Encoder**: ViT (shared with the [Heat](../heat) repo modulo patch dimension).
- **Decoder — `LinearCPDecoder`**: three-branch decoder with a **linear skip** from latent to rank channels:

  ```
  h_lin = W_direct @ z                           (linear skip)
  h_nl  = W_rank @ swish(W2 @ swish(W1 @ z))      (shallow MLP)
  h     = h_lin + h_nl
  u(x)  = sum_r h[r] * W_x[r,ix] * W_y[r,iy] [* W_z[r,iz]] + bias
  ```

  The linear skip is **required** for Gauss-Newton convergence from a cold start `z = 0`. With a plain MLP-only decoder, $\partial u / \partial z |_{z=0} \approx 0$ in a wide neighbourhood of the origin and GN's first step has no descent direction. The skip restores a non-degenerate rank-$\min(k, R)$ decoder Jacobian at the origin.

- **EQ hyper-reduction**: NNLS on $|K u_i|$ over training snapshots, restricted to strictly **interior** nodes $[1, N-2]^d$ so the 5-/7-point stencil never touches the boundary at runtime.
- **Solver**: Levenberg-Marquardt Gauss-Newton in latent space with backtracking line search, wrapped in `jax.lax.while_loop`.

## Data generators

Two are shipped, selected by `config.data_source`:

| | `analytical` | `cg` |
|---|---|---|
| Source | closed-form continuous PDE solution | CG-discrete solution |
| Speed | trivial | tens of CG iterations |
| Valid for | $N \le 128$ | all $N$ |
| Required at | — | $N \ge 256$ |

**Use `cg` for $N \ge 256$.** The analytical and discrete solutions differ by an $O(\Delta x^2)$ consistency gap. At low resolution that gap is below the AE-reconstruction floor and benign. At $N = 256$ it becomes comparable to the ROM error budget — the AE then learns a manifold that's "wrong" relative to the FOM benchmark, and rel-L2 floors near 0.8. **Rule:** the data generator must match the test-time FOM operator.

## Install

```bash
pip install -e .
pip install -e ".[dev]"   # optional: pytest for tests
```

For GPU support:

```bash
pip install --upgrade "jax[cuda12]"
```

## Reproduce a result

```bash
# 1. Generate training data (analytical for N<=128, CG for N>=256).
python -m scripts.generate_data --config configs/poisson3d_n64.yaml \
    --out data/poisson3d_n64.npz

# 2. Train the ViT-LinearCP autoencoder.
python -m scripts.train --config configs/poisson3d_n64.yaml \
    --data data/poisson3d_n64.npz --out checkpoints/poisson3d_n64.pkl

# 3. Build EQ and benchmark NM-ROM vs FOM.
python -m scripts.run_rom --config configs/poisson3d_n64.yaml \
    --data data/poisson3d_n64.npz --ckpt checkpoints/poisson3d_n64.pkl
```

## Tunability — one model, a Pareto curve

The same trained autoencoder operates at different accuracy/speedup points by varying two ROM-time knobs:

- `gn_max_iters` — Gauss-Newton iteration cap.
- `gn_rel_tol` — relative gradient-norm tolerance.

Tighter values → lower rel-L2, lower speedup. Looser values → higher speedup, higher rel-L2. The FAST/ACC variants in our paper share a checkpoint and differ only in these knobs.

## Repository layout

```
poisson/
├── src/tunable_rom_poisson/
│   ├── models/        ViT encoder, LinearCPDecoder, AE
│   ├── fom/           Poisson FOM (Laplacian, CG, source field) + data generators
│   ├── eq/            EQ-NNLS interior-only node selection
│   ├── solver/        NM-ROM Gauss-Newton solver
│   └── utils/         Training loop, config, checkpoint I/O
├── configs/           One YAML per (N, d) cell
├── scripts/           CLI entry points
├── tests/             Smoke tests (run by CI)
└── pyproject.toml
```

## Citation

Paper in submission. Companion repository for the Heat equation: [`tunable-rom-heat`](../heat).

## License

MIT.
