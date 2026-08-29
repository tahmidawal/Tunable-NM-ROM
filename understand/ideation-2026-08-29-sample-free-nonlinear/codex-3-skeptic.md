# Candidates

## 1. Keep sampling

**Idea.** Retain the exact-linear residual and evaluate only advection at NNLS nodes.

**Exactness.** Approximate relative to the full-grid oracle, but already adequate: NNLS-32 matches the oracle rollout in 1D, while the decoder—not quadrature—limits practical 2D configurations.

**Online cost.** Roughly 0.07M flops for 1D `m=32`, and 1.8–3.5M for 2D `m=256`, including Jacobian tangents. The optimized path needs about two post-head reductions. Tables are approximately 41 KiB in 1D and 0.8 MiB in 2D.

**Offline cost.** NNLS fitting and certification; learned nodes only help when `m` is starved.

**What could go wrong.** Accuracy deteriorates rapidly below the established budget; node sets transfer imperfectly. Conversely, sampling supports arbitrary pointwise fluxes, limiters, spatial coefficients, and future PDEs without redesign.

**Smallest decisive experiment.** This is the incumbent control. Any sample-free arm must match the oracle’s per-trajectory rollout accuracy, not merely beat NNLS-16, and must reduce measured end-to-end time or eliminate enough setup complexity to justify specialization.

## 2. Fixed-backward dense tensor

**Idea.** Precompute the symmetric part of

`T[i,j,k] = sum_x Phi[x,i] G[x,j] (D- G)[x,k]`

and evaluate `q(h)=h^T T h`. In 2D, combine the x- and y-advection tensors.

**Exactness.** Algebraically exact only inside the all-positive stencil region. It is not exact for the implemented sign-upwind oracle on decoded undershoots or at zero, where JAX selects the forward branch. An existing scratchpad CPU check found 6.9% negative decoded points, minimum `u=-1.2e-2`, and nonlinear-projection errors up to `1.6e-4` on training states. At `z+0.05N(0,I)`, the maximum was `1.3e-2`; Jacobian relative error reached `2.7e-2`, although cosine remained at least `0.9996`. The actual trust radius is smaller, but every rejected LM candidate still matters.

A positivity clamp or softplus field would destroy `u=Gh`. Nonnegative banks plus nonnegative head coefficients preserve separability, but constitute a restrictive retraining experiment. Maintaining exact sign regions would require evaluating all signs on the grid, defeating the purpose.

**Online cost.** Analytic residual plus Jacobian costs about 0.1M flops in 1D and 0.7–0.8M in 2D. Storage is 256 KiB for `32^3`, 2 MiB for `64^3`, and 4.5 MiB for `64×96^2`; packed symmetry halves this but introduces gathers.

“One contraction equals one kernel” is optimistic. The optimized sampled path folds the head’s last layer into its node matrices. The quadratic tensor cannot do that cheaply: stock code likely needs a separate head output, `Qh`, and final contraction—two or three reductions. In 1D it reads six times more table data than sampling while doing comparable arithmetic, so parity or regression is more plausible than speedup. In 2D it saves flops but reads roughly 2–5 MiB versus 0.8 MiB; only compiled fusion counts and paired timings can decide.

**Offline cost.** `nMR²` accumulation: 134M MACs at 1D `n=4096`, and 275B MACs at 2D `N=1024,R=M=64`. A naive `n×R²` temporary is about 32 GiB in the latter case, so construction must be blocked.

The learned bank may be poorly scaled. Oscillatory `Phi`, derivatives, and redundant bank columns can make individual tensor entries cancellation-dominated. In f64, build two different chunk/reduction orders and measure the contraction condition

`sum_jk |T_ijk h_j h_k| / |q_i|`.

Algebraic exactness does not imply bitwise equality to the full-grid summation.

**What could go wrong.** Off-manifold sign switching changes LM basins; f64 cancellation corrupts small modes; the extra launch erases the flop saving; flux-form `D(u²/2)` silently changes the current non-conservative FOM.

**Smallest decisive experiment.** On one existing N=256 checkpoint, record every accepted and rejected oracle LM candidate. Compare tensor/oracle `q`, `J`, and `J^T r`, signs, and two independently accumulated tensors. To claim exactness, require `≤1e-12` at every candidate. To accept it as an approximation, require identical stop reasons, per-trajectory rollout-error differences `≤1e-9` absolute, and end-to-end time no worse than NNLS-32. Otherwise retain sampling.

CP/Tucker compression should wait. The dense 2 MiB table is already small; compression adds approximation and usually a reduction. Head-PCA Tucker is the best option only if profiling shows tensor traffic matters: project `h` to `r≈K–2K`, fold that projection into the head, and gate values, gradients, and rollouts against dense `T`.

## 3. Choose a polynomial FOM, then use a structured bank

**Idea.** Make sample-freedom a discretization property. A conservative centered or skew-symmetric Burgers flux is quadratic for every sign. Global Lax–Friedrichs/Rusanov with trajectory-constant `alpha >= max|u0|` is quadratic plus linear diffusion; the latter collapses through `Lambda B h`.

A fixed sine/Chebyshev bank then offers analytic product rules. A nonnegative local B-spline bank with positive coefficients is another route for positive-only families.

**Exactness.** Exact relative to the new FOM. It is not equivalent to the incumbent truth: constant-alpha Rusanov is more diffusive, while centered/skew changes the O(dx) upwind viscosity. All data and decoder training must be regenerated.

**Online cost.** Dense current-bank tensor cost as above. A genuinely product-closed spectral core can approach `O(MR)` or sparse-convolution cost, but small sparse kernels may launch more slowly than a dense GEMV.

**Offline cost.** New truth data and stage-1 training. Current backward differencing is not sparse in a Dirichlet sine basis because it produces half-grid-shifted cosines. The present boundary-masked neural bank is not product-closed either.

**What could go wrong.** A fixed bank may need larger `R`; positive coefficients increase nonnegative rank; modified-FOM accuracy may worsen despite perfect ROM residuals.

**Smallest decisive experiment.** Compare incumbent, centered/skew, and Rusanov FOMs against a refined-grid reference before retraining. Proceed only if the polynomial scheme is at least as accurate. Then require a fixed-bank decoder to match the `3.7e-3` reconstruction floor and its tensor ROM to equal its own oracle within `1e-9`.

## 4. Lifting

**Idea.** Write sign-upwind as `u Dc u - (dx/2)|u|D2u`, introduce `v=|u|`, and decode `v` with another bank/head. Other nonlinear functions receive analogous lifted fields.

**Exactness.** A learned `v(z)` is approximate. Enforcing `v=|Gh|` exactly is again a pointwise non-polynomial constraint. Classical exact lifting usually adds intrusive evolution or closure equations.

**Cost.** One additional head and `M×Rv×R` tensor per lift: roughly another 1–3 kernels and up to doubled training/storage.

**What could go wrong.** Moving zero-crossing kinks, inconsistent lifted states off-manifold, and an auxiliary decoder floor becoming the residual floor.

**Smallest decisive experiment.** Use a deliberately sign-changing family. Require value and gradient fidelity plus rollout error within 2× the oracle. Do not test it on the current positive family, where the fixed tensor already approximates the correction as negligible.

## 5. Learned point-free projection

**Idea.** Learn `q(h)=Phi^T N(Gh)` directly, preferably as an analytic polynomial baseline plus a small learned correction.

**Exactness.** Learned approximation. AD provides the surrogate’s exact derivative, not the teacher’s derivative.

**Cost.** A small MLP is cheap in flops but adds approximately 2–5 GEMM/activation kernels per residual/Jacobian evaluation. Offline labeling requires full-grid values and Jacobians at training codes, trust-region perturbations, and actual rejected LM candidates.

**What could go wrong.** Off-manifold extrapolation and slightly wrong gradients can redirect LM despite excellent value fit.

**Smallest decisive experiment.** Train with value, Jacobian, and `J^T r` losses; hold out whole rollouts and wider-than-trust-region perturbations. Require gradient cosine `≥0.999`, identical stop reasons, and oracle rollout parity. Use a distance tripwire with sampled fallback; without fallback it is not trustworthy, while with fallback it is not universally sample-free.

# Ranking

1. **Keep sampling** as the production default: already accurate, smaller tables, and vastly more general.
2. **Fixed-backward tensor** as a narrow positive-family experiment, not yet an exact method.
3. **Polynomial FOM plus structured bank** if sample-freedom is a primary design requirement for a new dataset.
4. **Polynomial baseline plus learned correction** for genuinely non-polynomial PDEs.
5. **Lifting**, only where the lifted variables have a defensible exact closure.
6. **CP/Tucker compression**, only after dense-tensor profiling proves a problem.

# What I would do first

Run the smallest possible sign/conditioning audit on the existing N=256 checkpoint: every oracle LM candidate, two tensor accumulation orders, and direct comparisons of `q`, `J`, and `J^T r`. Then perform one tensor-versus-oracle-versus-NNLS-32 rollout and paired compiled timing.

My default decision is “keep sampling” unless the tensor simultaneously preserves every LM outcome and is no slower. In 1D it cannot repair the fundamental result that the optimized ROM remains slower and much less accurate than the tridiagonal FOM; in 2D, where sampling already does not bind at the practical budget, a tensor is more likely to simplify a paper narrative than improve the method.