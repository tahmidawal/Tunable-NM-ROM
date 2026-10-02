# Partial decoding and quadrature for mesh-independent NM-ROM solves

Companion study to the draft *Tunable non-linear manifold ROMs ... via matrix-free
Petrov–Galerkin projection* (`neurips26.pdf`).  The question: in the online
Gauss–Newton / Levenberg–Marquardt loop, can the one term that still touches every
mesh node (the tested nonlinear term `P N(G c)`) be evaluated at a cost that
depends only on the reduced sizes `(k, q, R, M)` and a quadrature count `m`, with
`m` independent of the mesh, while keeping the solver's convergence and accuracy?

The paper answers this with NNLS empirical quadrature (EQ) on mesh nodes.  This
study adds two families that need **partial decoding at arbitrary points**, which
the paper's coordinate-network bank supports natively (`u(x) = mu(x) g_phi(x)^T c`,
so any point and its gradient can be decoded without the rest of the grid):

* **Smolyak sparse grids** (nested Clenshaw–Curtis; Gauss–Legendre variants),
* **quasi-Monte Carlo**: scrambled Sobol, scrambled Halton, rank-1 Fibonacci
  lattices, and Fibonacci lattices with the tent (baker) periodising transform,

plus the reference baselines: dense mesh evaluation, uniform mesh sub-lattices
(the paper's "lattice" rule), NNLS EQ (re-implemented), tensor Gauss–Legendre,
plain Monte Carlo.  Off-mesh rules use the *continuum* form of the nonlinear term
(analytic gradients of the network) and for Burgers also the integrated-by-parts
flux form `-int grad(psi) . F(u)` that needs no gradient of `u` at all.

## PDEs (all on the unit square, homogeneous Dirichlet, `u_t + N(u) = nu Lap u + r u + f`)

| name | nonlinear term N(u) | FOM stencil | note |
|---|---|---|---|
| `burgers` | `u (u_x + u_y)` | sign-upwind, backward Euler (paper's) | flux form available |
| `allen_cahn` | `u^3 / tau`, tau=0.05 (linear `u/tau` preassembled) | pointwise | stiff; ROM amplifies per-step error, used for the hyper-reduction column only |
| `allen_cahn_mild` | same, tau=0.2 | pointwise | milder reaction |
| `hj` | `|grad u|^2 / 2` | Osher–Sethian monotone Hamiltonian | gradient nonlinearity, not a divergence |
| `bratu` | `-lam exp(u)` (steady, `-Lap u = lam e^u + f`) | pointwise | elliptic, one solve |

Every PDE shares one reduced solver: linear terms are exact in the sine test basis
(`B0 = P G`, eigenvalues `Lambda`), the nonlinear term is the only quadrature
dependent piece, and the LM step is the paper's block-damped variable-projection
step with the stationarity / tiny-step / damping-limit / budget exits.

## Layout

```
nmrom/grid.py         grid, sine tests (separable transform), Helmholtz solve (DST)
nmrom/pdes.py         PDE definitions (stencil form + continuum form + flux form)
nmrom/fom.py          Newton–BiCGStab full-order solver (JAX, matrix free)
nmrom/bank.py         RFF coordinate-network bank, variable-projection training, off-mesh eval + gradients
nmrom/head.py         latent head with linear skip, auto-decoder training, nested corrections
nmrom/quadrature.py   all rules; NNLS EQ fit with support cap
nmrom/rom.py          MeshModel (offline B0), evaluators (dense / mesh-sampled / off-mesh), LM, rollout
scripts/gen_data.py   training trajectories + evaluation references (same-grid at each mesh, refined)
scripts/train.py      bank + head + corrections -> data/<pde>_model.pkl
scripts/quad_study.py rho of every rule vs m (continuum and mesh targets), eval timing vs N
scripts/rollout_study.py  end-to-end errors, iterations, exits, ms/query for every rule, mesh, rank
scripts/report.py     figures + results/REPORT.md
scripts/timing_study.py  clean per-step timing vs mesh (run on an idle machine)
scripts/retrain_head.py  retrain the head with another latent size
scripts/decoder_chain.sh  train/ladder/rollout for alternative decoders (--bank siren|rbf|spectral|pod_cubic|pod_linear)
scripts/report_decoders.py  results/DECODERS.md
nmrom/dim3.py, scripts/*3d.py  3D Burgers extension (results/REPORT3D.md)
gpu_env.sh            environment for the ROCm GPU build
run_all.sh            the whole pipeline
```

## Metrics

* `rho`: relative error of the tested nonlinear term of a rule against a target, on
  states reached by the dense reduced solver (paper's eq. 13).  Two targets are
  reported: the **continuum** integral (tensor Gauss–Legendre 300x300 on the same
  decoded state; mesh independent) and the **dense mesh** evaluation with the FOM
  stencil at each `N`.  The gap between the two is the O(h) consistency error of
  the FOM stencil, a floor for any off-mesh rule measured against the mesh.
* end-to-end: worst relative L2 error over cases and output times against the
  same-grid FOM and against a refined reference (1024^2, dt/2), LM attempts per
  step, exit codes, ms per query and per LM attempt.
* complexity: ms per residual+Jacobian evaluation and ms per query as a function
  of `N` for dense / EQ / sparse-grid / QMC evaluation at fixed `m`.

Sizes used here (scaled down from the paper for CPU): `R = 128`, `k = 32` (`k = 16` also trained; `scripts/retrain_head.py`),
`q in {0, 64}`, `M = 4 (k + q)`, meshes 128^2 .. 1024^2, six evaluation cases.

## Running

```
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python "jax[cpu]" numpy scipy matplotlib optax
./run_all.sh burgers      # or allen_cahn, hj, bratu
```
Results and figures land in `results/`. Read `results/SUMMARY.md` first (findings and
verdict), then `results/REPORT.md` (all tables and figures) and `results/LITERATURE.md`
(prior work and novelty assessment).

## Headline findings

* Tensor Gauss–Legendre and rank-1 Fibonacci lattices reproduce the tested nonlinear term of
  the decoded state to 1e-3 .. 1e-6 with 1000–4000 points (paper's acceptance bar: 0.116),
  need no data fit and no per-mesh work, and give a per-step cost that is flat in the mesh
  (Burgers, q=64, 1024²: 4.3 ms per LM step vs 673 ms dense).
* Their end-to-end error against a refined reference equals the dense solve's at every mesh
  and is mesh-invariant, because the off-mesh solve targets the continuum problem.
* Smolyak sparse grids achieve the cost but not the accuracy: the tensor sine tests carry mixed
  high frequencies that sparse grids discard; they are 100–1000× worse than tensor Gauss at
  equal point count and degrade the LM solve.
* Scrambled Sobol/Halton converge only like 1/m; the tent transform hurts because the integrand
  already vanishes on the boundary.
* The result holds for any analytic decoder (RFF, SIREN, Gaussian-RBF, fixed sine banks give
  spectral convergence); POD modes evaluated off-mesh by cubic or bilinear interpolation lose
  2 to 4 orders of magnitude (`results/DECODERS.md`). Decoder smoothness is what matters.
* In 3D (Burgers, `results/REPORT3D.md`) a CBC-constructed rank-1 lattice beats tensor Gauss
  by 1–2 orders at equal point count (4096 lattice points ≈ 20³ Gauss), Smolyak stays above
  `rho = 1` at 15k points, and every off-mesh rule keeps a flat 3–9 ms per LM attempt at 128³
  against 561 ms dense.
* GPU: the ROCm build in `~/Venv/jax` works once `gpu_env.sh` is sourced (library path to the
  ROCm 7.12 SDK and preallocation disabled); see the comments in that file. Caveat: the ROCm
  plugin segfaults (exit 139) in some fused operations (JAX's `bicgstab` with a tensordot
  preconditioner, the bank-training step, and the reduced solve with `q > 0`); the 3D FOM
  therefore runs its BiCGStab loop in Python over jitted kernels on the GPU, and training and
  the reduced studies run on the CPU venv.
