# Off-mesh quadrature for the NM-ROM nonlinear term: summary of findings

Companion study to `neurips26.pdf`. Full tables and figures: `REPORT.md` and `*_fig*.png`
in this directory; literature search: `LITERATURE.md`; code: `../nmrom`, `../scripts`.

## 1. Question and setup

The online Levenberg–Marquardt loop of the paper's NM-ROM is O(M R) per residual except for
the tested nonlinear term `P N(G c)`, which touches every mesh node (O(n R)) unless it is
hyper-reduced. The paper hyper-reduces it with NNLS empirical quadrature (EQ) on mesh nodes
fitted to stored states, which has to be certified statistically on held-out reached states
and, at 4096², passed five held-out draws but failed the confirmation draw.

This study asks whether a *fixed, state-independent, mesh-independent* quadrature rule on the
continuum can replace EQ. The paper's bank is a coordinate network, so it can be decoded, with
analytic gradients, at any point of the domain ("partial decoding"). The tested nonlinear term
then becomes `N * sum_q w_q psi_ab(x_q) F(u, grad u)(x_q)` with `(x_q, w_q)` from a classical
rule and `u(x_q) = mu(x_q) g(x_q)^T c` read from a cached `m x R` block; one residual costs
O(m R + m M) and one Jacobian O(m R (k+q) + m M (k+q)), with no dependence on the mesh.

Rules compared: tensor Gauss–Legendre; Smolyak sparse grids (nested Clenshaw–Curtis, and two
Gauss–Legendre growth rules); scrambled Sobol; scrambled Halton; rank-1 Fibonacci lattices
with a random shift, with and without the tent (baker) periodising transform; plain Monte
Carlo; and, on the mesh, the dense evaluation, uniform sub-lattices (the paper's "lattice"
rule) and a re-implementation of the paper's NNLS EQ (greedy Lawson–Hanson support growth with
exact non-negative refits and a support cap). For Burgers, off-mesh rules were also run in the
integrated-by-parts flux form `-int grad(psi) . F(u)`, which needs no gradient of `u`.

PDEs (unit square, homogeneous Dirichlet): Burgers `u_t + u(u_x+u_y) = nu Lap u` (the paper's
sign-upwind stencil), Allen–Cahn `u_t = eps Lap u + (u - u^3)/tau` in a stiff (`tau = 0.05`)
and a milder (`tau = 0.2`) variant, viscous Hamilton–Jacobi `u_t + |grad u|^2/2 = nu Lap u`
(Osher–Sethian monotone stencil), and the steady Bratu problem `-Lap u = lam e^u + f`.
Model sizes are scaled down from the paper for CPU: bank `R = 128`, head `k = 32`
(`k = 16` was also trained), corrections `q in {0, 64}`, tests `M = 4 (k + q)`, meshes 128²
to 1024², six evaluation cases per PDE, refined reference at 1024² with half the time step.

## 2. Quadrature error of the tested nonlinear term

`rho` is the paper's eq. (13): the relative error of the tested nonlinear term of a rule against
a target, worst case over 96 states. Two targets are used. The *continuum* target is a
300×300 tensor Gauss rule on the same decoded state (200×200 agrees with it to 1e-14, and the
gradient and flux forms agree to 1e-14, which also validates the analytic network gradients).
The *mesh* target is the dense evaluation with the FOM stencil at each `N`.

Worst `rho` against the continuum target at the accurate test count (`M = 320`), Burgers:

| rule | 1024 points | ~2300–2600 points | ~4100–5100 points |
|---|---|---|---|
| tensor Gauss–Legendre | 7.8e-2 (32²) | 4.5e-4 (48²) | 3.9e-6 (64²) |
| Fibonacci lattice | 1.0e-1 (987) | 1.3e-2 (2584) | 3.2e-3 (4181) |
| Fibonacci + tent | 4.3e-1 (987) | 4.9e-2 (2584) | 2.9e-2 (4181) |
| scrambled Sobol | 2.3e-1 | 1.1e-1 (2048) | 4.7e-2 (4096) |
| scrambled Halton | 2.9e-1 | 1.3e-1 (2048) | 8.9e-2 (4096) |
| Smolyak Clenshaw–Curtis | 1.8 (1025) | 6.0e-1 (2305) | 7.6e-2 (5121) |
| Smolyak Gauss (exp. growth) | 2.1 (1573) | 6.7e-1 (3881) | 9.2e-2 (9261) |
| plain Monte Carlo | 8.2e-1 | 7.9e-1 (2048) | 4.3e-1 (4096) |
| paper's acceptance bars | 0.116 (primary), 0.06 (tight) | | |

Median `rho` is far smaller than the worst case for the Gauss and lattice rules (Fibonacci
1597 points: median 3e-5, worst 4e-2); the worst cases are the narrowest early-time bumps,
the same states the paper found hardest for EQ. Allen–Cahn and Bratu integrands are easier
(Fibonacci 4181 points: worst 6e-6 and 2e-4); Hamilton–Jacobi (`|grad u|^2` of narrow bumps)
is the hardest (Gauss 48²: 1.7e-2; Fibonacci 4181: 1.5e-1 worst, 6e-5 median). The
reached-state check (states reached by the dense reduced solver, the paper's protocol) gives
the same picture as the projected training snapshots (`REPORT.md`, per PDE).

Conclusions from the ladders:

* **Tensor Gauss–Legendre and the plain Fibonacci lattice converge super-algebraically.** The
  decoded integrand is smooth and vanishes on the boundary with the Dirichlet factor, so the
  trapezoidal-type lattice rule behaves like the exponentially convergent trapezoidal rule of a
  periodic integrand; the tent transform, which helps non-periodic integrands, only adds a kink
  here and hurts by one to two orders.
* **Smolyak sparse grids fail for this problem class.** The test functions are tensor products
  `sin(a pi x) sin(b pi y)` with both frequencies up to ~25; those mixed high-frequency
  components are exactly the terms a sparse grid discards. Every Smolyak variant is 100 to 1000
  times worse than a tensor Gauss rule at equal point count and needs about 5000 points to pass
  the paper's primary bar.
* **Scrambled Sobol and Halton converge like 1/m.** They pass the bar at 4096 points but remain
  four orders behind tensor Gauss at the same count. The flux form is worse for them (the
  derivative moves onto the test function and raises the integrand's variation).
* **Against the mesh target every off-mesh rule plateaus at the FOM stencil's own O(h)
  consistency gap**: 0.25, 0.12, 0.065 at 128², 256², 512² for Burgers (first-order upwind);
  5e-5 for Allen–Cahn (pointwise term, no gap). Only the mesh-fitted EQ rule goes below it. That
  gap is the full-order discretisation error, not a quadrature error; the end-to-end errors
  against the refined reference (Section 3) show it does not penalise the off-mesh rules.
* **The paper's EQ, re-implemented**, reaches worst `rho` of 6e-2 (Burgers), 5e-3 (Allen–Cahn),
  7e-2 (HJ) and 8e-4 (Bratu) with 1024 nodes against the mesh target, i.e. comparable to a 32²
  Gauss rule against the continuum, but each rule needs a per-mesh NNLS fit (3 to 6 minutes at
  1024 nodes, longer at 2048) and inherits the fit's data dependence.

## 3. End-to-end reduced solves

For every rule, mesh and correction rank the reduced solver was run on the six evaluation
cases with the same frozen model. `REPORT.md` lists worst error against the same-grid FOM,
worst error against the refined reference, the worst distance of the rule's rollout from the
dense rollout of the same model (`vs dense`, the hyper-reduction error proper), LM attempts
per step, exit codes and timings.

Burgers, worst error against the refined reference (%) and distance from the dense rollout (%):

| N | q | dense | EQ 2048 | Gauss 32² | Fibonacci 4181 | Sobol 4096 | Smolyak CC 8 |
|---|---|---|---|---|---|---|---|
| 128 | 0 | 22.37 | 22.37 (0.004) | 22.24 (4.2) | 22.24 (4.1) | 22.28 (4.0) | 22.48 (9.7) |
| 256 | 0 | 22.31 | 22.31 (0.008) | 22.28 (2.2) | 22.28 (2.2) | 22.32 (2.0) | 22.53 (8.5) |
| 512 | 0 | 22.29 | 22.29 (0.015) | 22.28 (1.1) | 22.28 (1.1) | 22.33 (1.0) | 22.53 (7.9) |
| 1024 | 0 | 19.16* | – | 22.28 (0.57) | 22.28 (0.57) | 22.33 (0.45) | 22.53 (7.7) |
| 128 | 64 | 14.05 | 14.05 (0.005) | 13.81 (4.7) | 13.80 (4.7) | 13.85 (4.5) | 21.59 (18.3) |
| 256 | 64 | 13.90 | 13.90 (0.006) | 13.88 (2.5) | 13.87 (2.5) | 13.92 (2.3) | 21.61 (18.4) |
| 512 | 64 | 13.86 | 13.86 (0.008) | 13.88 (1.3) | 13.88 (1.3) | 13.92 (1.1) | 21.61 (18.3) |
| 1024 | 64 | – | – | 13.88 | 13.87 | 13.92 | 21.61 |

`*` dense at 1024² was run on two cases only (cost). All solves exit on the stationarity test
with 2.2 attempts per step for every rule except Smolyak.

Reading the table:

* **Gauss, Fibonacci and Sobol reproduce the dense solve's accuracy at every mesh and rank.**
  Their error against the refined reference is the same as the dense solve's to within 0.1
  points, and it is *identical across meshes* (22.28 / 13.88 at every `N`), because the off-mesh
  solve targets the continuum problem and never sees the mesh online. Their distance from the
  upwind dense rollout halves with each mesh doubling (4.2, 2.2, 1.1, 0.57 %), which is the
  stencil's O(h) error, not theirs.
* **EQ tracks the dense rollout to 0.01 %** by construction (it is fitted to the stencil), at
  the price of a per-mesh fit.
* **Smolyak degrades the solve**: 8 to 18 % from the dense rollout, worst error 21.6 % instead
  of 13.9 % at `q = 64`, and on Hamilton–Jacobi it causes budget and damping-limit exits.
* Hamilton–Jacobi and Bratu behave the same way (`REPORT.md`): off-mesh rules give
  mesh-invariant refined errors (HJ 19.9 % at `q = 0`, 12.3 % at `q = 64`; Bratu 7.16 % and
  1.15 %), EQ tracks dense to 0.5 %, and on Bratu every rule is within 0.02 % of dense.
* The stiff Allen–Cahn variant (`tau = 0.05`) is not a usable end-to-end test bed: the reduced
  dynamics amplify the per-step error (2.5 % per step, matching the manifold's fit error) by
  the linear growth rate `1/tau = 20` to above 100 % over the horizon, with exact evaluation.
  Its hyper-reduction column is still clean (Fibonacci 4181: 0.02 % from dense at `q = 0`, EQ
  2048: 0.05 %, Gauss 32²: 0.6 %, Smolyak: 33 % and divergence at `q = 64`).
* The milder Allen–Cahn variant (`tau = 0.2`, bank floor 0.15 % median, head fit 0.3 %) is a
  stable test bed and gives the same ranking at every mesh from 128² to 1024²: the reduced
  solve's worst refined error is 19.4 % at `q = 0` and 21.0 % at `q = 64` (one narrow-bump case
  outside the head's reach; the median case is far better) for dense, EQ, Gauss and Fibonacci
  alike, and the hyper-reduction errors are Fibonacci 4181: 0.00–0.03 %, Gauss 32²: 0.01–0.3 %,
  EQ 2048: 0.05–0.12 %, Sobol 4096: 0.3–1.3 %, Smolyak CC 8: 0.8 % at `q = 0` but 49–53 % at
  `q = 64` (worst error 57 % instead of 21 %). Per-step cost at 1024²: dense 105 ms, Gauss 32²
  1.3 ms, Fibonacci 4181 1.8 ms.

Absolute error levels are set by the scaled-down model, not by the quadrature: the bank floor
on the narrowest Burgers bumps is 4 to 11 % at `t = 0` and the head's reach on held-out cases
is the next limit (a diagnostic that fits the manifold to each FOM state shows the solver
lands within 2 to 3 % of the FOM step, equal to the fit error). The comparison between rules
does not depend on this.

## 4. Cost

Measured ms per residual-plus-Jacobian evaluation and per LM attempt are in `REPORT.md`
(quadrature-study timing tables, taken under heavy machine load) and in the clean timing run
(`*_timing.json`, Section 5). In the rollouts, the cost per LM attempt of the dense evaluation
grows with the mesh (Burgers `q = 64`: 26, 27, 105 ms at 128², 256², 512²; 145 ms at 1024²
for `q = 0`) while Gauss 32² stays at 4 to 14 ms and EQ 2048 at 5 to 19 ms at every mesh, with
the same 2.2 attempts per step. The off-mesh evaluators never touch the mesh online: their only
per-mesh objects are the offline tested bank `B0 = P G` and the initial-state projection.

## 5. Clean timing run

Medians of repeated jitted calls on an otherwise idle 32-core CPU (`scripts/timing_study.py`,
full tables and `*_fig6_clean_timing.png` in `REPORT.md`). Burgers, ms per LM time step
(one jitted while-loop, 3 attempts), `k = 32`:

| rule | m | q | 128² | 256² | 512² | 1024² |
|---|---|---|---|---|---|---|
| dense (mesh) | n | 0 | 8.2 | 21.3 | 83.2 | 333.6 |
| dense (mesh) | n | 64 | 13.8 | 43.5 | 168.2 | 673.4 |
| EQ (paper's) | 2048 | 0 | 4.5 | 3.6 | 3.7 | – |
| EQ (paper's) | 2048 | 64 | 7.8 | 7.4 | 7.3 | – |
| Gauss 32² | 1024 | 0 | 1.4 | 1.5 | 1.3 | 1.5 |
| Gauss 32² | 1024 | 64 | 4.5 | 4.4 | 4.3 | 4.3 |
| Gauss 48² | 2304 | 64 | 7.7 | 7.8 | 7.4 | 7.5 |
| Fibonacci | 4181 | 64 | 10.2 | 10.6 | 10.7 | 10.7 |
| Sobol | 4096 | 64 | 11.8 | 12.3 | 12.6 | 12.4 |
| Smolyak CC 8 | 1025 | 64 | 4.5 | 4.4 | 4.4 | 4.7 |

Hamilton–Jacobi gives the same picture (`q = 64`: dense 18, 56, 220, 868 ms; Gauss 32² 6.2 ms
and Fibonacci 4181 12 to 13 ms at every mesh). The dense cost grows linearly with `n` (×4 per
mesh doubling, Jacobian dominated: 175 ms at 1024², `q = 64`). Every sampled rule is flat in `N` and linear in `m` (about 1 ms per 1000
points per step at `q = 64`, on top of a ~3 ms floor for the head Jacobian, the `B0` products
and the (k+q)-dimensional normal solve). EQ at 2048 nodes costs the same as Gauss 48² at 2304
points, i.e. the cost is set by `m`, not by whether the points are on or off the mesh. At 1024²
and `q = 64` the off-mesh rules are 90 to 155 times cheaper per step than dense evaluation.

## 6. Verdict on the two proposals

* **Quasi-Monte Carlo: yes, with the right point set.** A rank-1 Fibonacci lattice (2D) reaches
  the accuracy of the paper's certified EQ rules with a comparable number of points, needs no
  fit, no per-mesh recomputation and no held-out certification, and gives the ideal
  mesh-independent cost. Sobol and Halton also achieve the cost and pass the paper's bar at
  4096 points, but converge only like 1/m. Tensor Gauss–Legendre, the obvious 2D baseline, is
  the best rule of all here and should be the default in two dimensions; lattice rules are
  what carries over to three dimensions, where tensor Gauss becomes expensive.
* **Smolyak sparse grids: no.** They achieve the cost but not the accuracy, for a structural
  reason (mixed high frequencies of tensor sine tests) that will not go away with tuning; they
  also destabilise the LM solve. A sparse grid could only be competitive with test functions of
  low mixed order (e.g. hierarchical or radial tests), which would change the projection.
* The **flux form** for Burgers matches the gradient form for Gauss and lattice rules and
  removes the need for network gradients at quadrature points (one `m x R` block instead of
  three), but is worse for Sobol/Halton.

## 7. Novelty (from `LITERATURE.md`)

Evaluating an implicit-neural-representation decoder off-mesh for a reduced solve is
established (CROM, ICLR 2023; neural stress fields, 2023; Weder–Schwerdtner–Peherstorfer,
2025), and the mesh-independent cost claim has been made there. What was not found in the
literature is (i) replacing data-fitted hyper-reduction (ECSW/EQP/ECM/greedy cubature, CROM's
residual-driven sampler) with a fixed classical quadrature rule on the continuum for the tested
term of an LSPG NM-ROM, with the error controlled a priori by the integrand's smoothness, and
(ii) the characterisation of which rules work for spectral tensor-product tests, in particular
the sparse-grid failure and the super-algebraic behaviour of lattice rules and tensor Gauss.
The closest cousins are good-lattice training for PINNs (Matsubara & Yaguchi, AAAI 2025) and
VPINN quadrature analysis (Berrone–Canuto–Pintore, 2022), both for full-order training losses.

## 8. Alternate decoder architectures (Burgers 2D, `DECODERS.md`, `burgers_fig7_decoders_rho.png`)

The off-mesh rules only need a decoder that can be evaluated (with gradient) at arbitrary
points. Six banks were trained or built with the same head (`k = 32`) and corrections, and
run through the same ladders and rollouts with Gauss, Fibonacci, Sobol and Smolyak:

| decoder | floor med / worst % | Gauss 48² worst / med rho | Gauss 64² | Fibonacci 4181 | Fibonacci 6765 |
|---|---|---|---|---|---|
| RFF coordinate MLP (paper) | 0.24 / 6.7 | 4.5e-4 / 4.6e-6 | 3.9e-6 | 3.2e-3 / 2.1e-6 | 6.3e-4 |
| SIREN coordinate MLP | 0.21 / 8.4 | 2.4e-4 / 1.4e-7 | 1.5e-7 | 1.3e-3 / 2.3e-6 | 2.9e-4 |
| Gaussian RBF bank (fixed centres) | 0.28 / 8.8 | 1.3e-5 / 2.3e-8 | 6.4e-13 | 5.6e-4 / 2.3e-6 | 1.3e-4 |
| fixed sine bank (no training) | 0.24 / 17.2 | 2.3e-8 / 1.8e-11 | 2.0e-14 | 3.0e-5 / 2.1e-6 | 8.3e-6 |
| POD modes + cubic interpolation (C¹) | 0.21 / 4.3 | 6.3e-2 / 1.7e-4 | 1.7e-2 | 2.5e-2 / 7.2e-5 | 5.6e-3 |
| POD modes + bilinear interpolation (C⁰) | 0.21 / 4.3 | 4.9e-2 / 1.1e-2 | 4.7e-2 | 4.3e-2 / 3.9e-3 | 2.8e-2 |

* **The result is not specific to the paper's random-Fourier-feature bank.** Every analytic
  decoder (RFF, SIREN, RBF, sines) gives spectral convergence with tensor Gauss and
  super-algebraic convergence with the Fibonacci lattice; the smoother the decoder (RBF and
  sines are entire functions of low effective bandwidth) the faster. The fixed sine bank is
  integrated exactly by Gauss 48² (the integrand is a trigonometric polynomial).
* **Decoder smoothness is the controlling property.** A POD basis evaluated off-mesh by
  interpolation, the natural way to give a mesh-based decoder (POD, convolutional or masked
  autoencoders) partial decoding, converges only algebraically: cubic interpolation stalls near
  1e-4 median and 1e-2 worst, bilinear near 1e-2, and the 200² and 300² Gauss references
  themselves disagree by 1e-3 and 2.5e-2. Two to four orders of magnitude are lost relative
  to the analytic banks at equal point count, and Sobol-level accuracy is all one gets.
* **End to end, every decoder reproduces its own dense rollout with Gauss 32² and Fibonacci
  4181 to within the O(h) stencil gap** (2.1 to 2.6 % at 256², 1.0 to 1.4 % at 512²) and the
  same refined-reference error as dense (e.g. SIREN 13.74 vs 13.77 %, RBF 13.32 vs 13.34 %,
  sines 8.95 vs 8.96 %, POD-cubic 14.50 vs 14.52 % at 512², `q = 64`), including the
  interpolated POD banks, because a 1e-2 quadrature error is still below the model error.
  Smolyak level 8 damages every decoder (5 to 20 % from dense at `q = 64`).
* Representation quality is similar across decoders (median floor 0.21 to 0.28 %); POD is the
  optimal mesh basis (worst floor 4.3 %), the fixed sine bank has the worst tail (17 %) but,
  with corrections, the best end-to-end error here (8.95 % at `q = 64`), since its correction
  directions span the residual well. Mesh-bound decoders that cannot be evaluated off-mesh at
  all (convolutional or masked autoencoders) can only use the mesh-sampled rules (EQ, lattice).

## 9. Three dimensions: Burgers 3D (`REPORT3D.md`, `nmrom/dim3.py`)

Burgers `u_t + u (u_x+u_y+u_z) = nu Lap u` on the unit cube, sign-upwind stencil, 64
training trajectories at 64³, six evaluation cases at 64³ and 128³ (refined reference 128³,
dt/2), bank `R = 128` (3D floor 2.6 % median, 40 % worst on 4000 training steps, so absolute
ROM errors are large and only the comparison between rules is meaningful), head `k = 32`,
`q in {0, 64}`, `M = 4 (k + q)` tensor sine tests `(a, b, c)`. Rules: tensor Gauss p³, Smolyak
Clenshaw–Curtis (d = 3 combination formula), scrambled Sobol, Monte Carlo, and rank-1 lattices
constructed by component-by-component (CBC) search with the P₂ criterion (the 3D analogue of
the Fibonacci lattice; e.g. n = 4096: z = (1, 1557, 1741), n = 32768: z = (1, 12031, 7247)),
plus Korobov lattices and tent variants.

Worst / median `rho` against a converged 64³ Gauss reference (48³ agrees to 2e-7), `M = 384`:

| rule | ~4k points | ~14–16k points | 32768 points |
|---|---|---|---|
| CBC rank-1 lattice | 8.3e-3 / 1.2e-3 | 8.9e-5 / 3.2e-6 (16384) | 1.4e-5 / 3.8e-7 |
| Korobov lattice | 2.6e-2 / 3.1e-3 (4093) | 9.6e-5 / 3.8e-6 (16381) | – |
| tensor Gauss–Legendre | 1.4e-1 / 1.2e-2 (16³) | 1.8e-3 / 5.7e-4 (24³) | 7.8e-5 / 2.2e-5 (32³) |
| scrambled Sobol | 2.5e-1 / 1.0e-1 | 7.0e-2 / 2.6e-2 (16384) | 2.9e-2 / 1.1e-2 |
| CBC lattice + tent | 2.5e-1 / 9.6e-2 | 3.8e-2 / 6.9e-3 | 3.1e-3 / 4.4e-4 |
| Smolyak Clenshaw–Curtis | 1.6e+1 / 2.3 (2559) | 2.4 / 2.9e-1 (15103) | – |
| plain Monte Carlo | 6.3e-1 / 3.2e-1 | 2.8e-1 / 1.7e-1 | 1.9e-1 / 1.2e-1 |

* **In 3D the rank-1 lattice is the best rule, by one to two orders over tensor Gauss at equal
  point count**: 4096 lattice points match roughly a 20³ Gauss rule, 16384 match 32³. This is
  the regime the 2D study could not show (there tensor Gauss won) and it is the reason the
  QMC proposal matters: lattice cost is one point per node in any dimension.
* **Smolyak sparse grids fail more badly than in 2D**: still `rho > 1` at 15,000 points, and
  in the rollouts they diverge (73 to 570 % from the dense solve). Sobol keeps its `1/m` rate
  (2.9e-2 at 32768 points), the tent transform hurts as in 2D, and Monte Carlo is useless.
* The paper-style NNLS EQ fitted on 8192 candidate nodes reaches only `rho = 0.6` at 2048
  nodes against the mesh target in 3D: with 16 fit states and 384 tests the fit is far from
  interpolating, so the data-fitted route is much weaker in 3D at this support.
* **End to end**, Gauss 16³, Sobol 4096 and the CBC lattice (4096 and 32768 points) give the
  same refined-reference error as the dense solve to two decimals at 64³ (`q = 0`: 51.18 %;
  `q = 64`: 41.49 vs 41.45 %), are mesh-invariant at 128³ (51.18 / 41.49 %), and sit 7 to 8 %
  from the dense rollout at 64³ and 1.1 to 1.5 % at 128³, i.e. at the 3D upwind stencil's O(h)
  gap (0.27 and 0.14 in `rho`), with 2.05 to 2.07 LM attempts per step and stationary exits
  throughout. Smolyak levels 7 and 8 give 75 to 576 % error and budget exits.
* **Cost is flat in the mesh and linear in `m`**: one residual-plus-Jacobian at `q = 64` costs
  157 ms dense at 64³ and 390 ms at 128³, against 2.8 ms (Gauss 16³ / 4096 points), 8 ms
  (24³), 3 ms (Sobol or lattice 4096) and 34 ms (32768 points) at both meshes; per LM attempt
  in the rollouts at 128³, `q = 64`: dense 561 ms, lattice 4096 9.3 ms, EQ 2048 7.2 ms.
* Caveats: dense at 128³ was run on two cases only; the "Kuo" embedded-lattice vector the
  agent first tried was unverified and poor (kept in the JSON as `kuo_lattice`, not used in
  the tables); the 3D FOM ran on the GPU, everything else on the CPU because the ROCm build
  segfaults in some fused JAX operations (see README).

## 10. Limitations of this study

* CPU-only, scaled-down models (`R = 128`, 96 training trajectories); absolute errors are
  larger than the paper's. Timings are single-machine CPU medians.
* The 3D study uses a weak bank (R = 128 on 64 trajectories, 2.6 % median floor) and small
  meshes (64³, 128³); it establishes the ranking of the rules and the flat cost, not
  competitive absolute accuracy.
* The refined reference uses the same first-order stencils at 1024² with half the time step; the
  FOM's own error against it is 1 to 5 %, which bounds how finely the end-to-end comparison can
  resolve differences between rules. The `vs dense` column and the `rho` ladders are the
  precise comparisons.
* EQ was re-implemented (greedy Lawson–Hanson with a support cap) rather than taken from the
  paper's code; it reproduces the qualitative behaviour but not the paper's exact rules.
