# Student guide: off-mesh quadrature for nonlinear-manifold ROMs

This repository is a complete, self-contained study of one question: can the online
Gauss–Newton / Levenberg–Marquardt solve of a nonlinear-manifold reduced-order model (NM-ROM)
evaluate its nonlinear term with a *fixed classical quadrature rule* (tensor Gauss, rank-1
lattices, Sobol/Halton, Smolyak sparse grids) instead of a data-fitted empirical quadrature,
by decoding the coordinate-network bank only at the quadrature points? Everything you need
to reproduce, extend or challenge the conclusions is here; the generated data (~5 GB) is not,
but every script regenerates it.

## Read in this order

1. `results/SUMMARY.md` — the findings, verdict and limitations (start here).
2. `results/DECODERS.md` — the same question for six decoder architectures.
3. `results/REPORT3D.md` — the 3D Burgers extension (where lattice rules win).
4. `results/REPORT.md` — every table and figure behind the summary (long).
5. `results/LITERATURE.md` — prior work and what appears novel.
6. `README.md` — code layout and headline findings.

## Setup

```
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt     # CPU JAX; all results were produced this way
.venv/bin/python scripts/smoke_test.py burgers                    # ~10 s, checks the whole pipeline on a random model
```
A GPU is optional. `gpu_env.sh` documents the AMD/ROCm setup used here and its caveats;
on NVIDIA install `jax[cuda12]` instead and keep `XLA_PYTHON_CLIENT_PREALLOCATE=false` if
the device shares system memory.

## Reproducing the 2D study (per PDE, on a 32-core CPU)

```
./run_all.sh burgers        # data (~1 h, dominated by the 1024^2 references), train (~35 min),
                            # quadrature study (~1.5 h, mostly NNLS EQ fits), rollouts (~1 h), report
```
Cheaper variants: `scripts/quad_study.py burgers --states train --meshes 128,256 --skip_eq
--skip_time` runs the rule ladders alone in minutes once a model exists;
`scripts/rollout_study.py burgers --model_suffix _k32 --tag _k32 --rules dense,gauss_tensor,fibonacci`
runs a subset of rules. `scripts/timing_study.py` must run on an idle machine.
Other PDEs: `allen_cahn_mild`, `hj`, `bratu` (`allen_cahn` is the stiff variant; its ROM
diverges and it is kept only for the hyper-reduction column). Decoders:
`scripts/decoder_chain.sh burgers siren rbf spectral pod_cubic pod_linear` then
`scripts/report_decoders.py burgers rff siren rbf spectral pod_cubic pod_linear`.
3D: `scripts/gen_data3d.py`, `train3d.py`, `quad_study3d.py`, `rollout_study3d.py`, `report3d.py`.

## Where things are

| what | where |
|---|---|
| grid, sine tests, DST Helmholtz solve | `nmrom/grid.py` (2D), `nmrom/dim3.py` (3D) |
| PDE definitions (stencil form, continuum form, flux form) | `nmrom/pdes.py` |
| full-order Newton–BiCGStab solver | `nmrom/fom.py` |
| banks: RFF (paper), SIREN, RBF, spectral, POD+interpolation | `nmrom/bank.py` (`bank_eval` dispatches on parameter keys) |
| latent head, nested corrections | `nmrom/head.py` |
| quadrature rules, NNLS empirical quadrature | `nmrom/quadrature.py` (2D), `nmrom/dim3.py` (3D incl. CBC lattices) |
| reduced residual, LM with block damping, rollout, evaluators | `nmrom/rom.py` |
| the paper this study accompanies | ask the author for the draft (`neurips26.pdf`) |

Conventions worth knowing: tests are mesh-orthonormal sine vectors, so an off-mesh rule
reproduces the mesh-tested term as `N * sum_q w_q psi(x_q) F(x_q)` in 2D (`N^{3/2}` in 3D);
`rho` is the paper's eq. (13) relative error of the tested nonlinear term; `vs dense` in the
rollout tables is the distance of a rule's rollout from the dense-evaluation rollout of the
same model (the hyper-reduction error proper), which is the right number to compare rules
with, because the absolute errors are dominated by the scaled-down models.

## Open threads worth exploring

1. **A-priori error bound.** The observed super-algebraic convergence of lattice rules and
   tensor Gauss follows from the decoder's smoothness. Turn it into a bound: the integrand is
   `psi_ab * F(u, grad u)` with `u = mu * MLP(RFF(x))`, whose Fourier coefficients decay at a
   rate set by the RFF scale and the MLP; lattice-rule error is bounded by the sum of Fourier
   coefficients on the dual lattice (Sloan & Joe; Dick–Nuyens–Pillichshammer). Even a
   semi-rigorous bound would replace the paper's held-out certification.
2. **Worst-case states.** The worst `rho` lags the median by orders of magnitude on the
   narrowest early-time bumps. Options: an exact (dense) first step as in the paper, an
   adaptive rule that refines when the state's spectral content grows, or bounding the state's
   bandwidth from the latent code.
3. **Three dimensions at scale.** The 3D bank here is weak (R = 128 on 64 trajectories). Train
   a bank at the paper's scale on the GPU and repeat the 3D ladders; the lattice-vs-Gauss gap
   should widen with the point count.
4. **Better lattices.** Only CBC with the P₂ criterion and product weights 1 was used. Try
   weighted CBC (weights matched to the test-function frequencies), embedded lattice sequences
   (so `m` can grow without recomputing the block), and polynomial lattice rules.
5. **Test functions.** Smolyak fails because the tensor sine tests have mixed high frequencies.
   Would hyperbolic-cross or radial test functions make sparse grids viable, and what do they
   do to the projection error? This changes the ROM, not just the quadrature.
6. **Flux form in 3D and for other PDEs.** The integrated-by-parts form removes the network
   gradient (one `m x R` block instead of four in 3D); it matched the gradient form for Gauss
   and lattices in 2D but was worse for Sobol. Quantify the block-size/accuracy trade-off.
7. **Mesh-bound decoders.** Convolutional or masked autoencoders cannot be decoded off-mesh;
   the POD-plus-interpolation results show how much smoothness costs. A learned, smooth
   *lifting* of a mesh decoder (e.g. spline coefficients as the output) might recover the
   spectral rates.
8. **Randomised rules for error estimation.** Random shifts of the lattice give an unbiased
   estimator with a cheap variance estimate; a few shifted copies per time step would give an
   online error indicator at little cost.

## Caveats you should not lose sight of

* Models are scaled down (R = 128, 96 trajectories, k = 32); absolute ROM errors are far worse
  than the paper's. Compare rules with `vs dense` and the `rho` ladders, not with absolute errors.
* Against the mesh-based (upwind) target every off-mesh rule plateaus at the stencil's O(h)
  consistency gap. That is discretisation error of the full-order model, not quadrature error.
* The NNLS EQ here is a re-implementation (greedy Lawson–Hanson with a support cap), not the
  paper's code.
* Timings are CPU medians on one machine; the flat-in-N behaviour is the claim, not the ms.
