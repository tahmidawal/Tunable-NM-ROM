# Sample-free nonlinear residual — angle B: STRUCTURED BANKS

Numbers below marked [CPU] come from a CPU-only diagnostic I ran on the committed N=256, K=8,
R=32 checkpoint (`runs/b1dqf/b1ds_n256/out/sep_b1d_scale_n256.pkl`; 64 regenerated train
trajectories, 8 test, 2048 decoded training states; script
`scratchpad/ideation/bank_reexpress{,2}.py`). No training, no GPU. Everything else is estimate.

## Two facts that reframe the question

1. **The bank is not the bottleneck; the K=8 head is.** Projecting the test truth onto
   span(G) (free h in R^32) gives 2.65e-4 [CPU]; the decoder floor is 3.7e-3. Worse for the
   learned bank: the first **32 discrete sines** give a 3.4e-5 floor and POD-32 gives 1.4e-5
   [CPU]. The learned bank is a *worse* 32-dim linear space than plain sines on this family.
   Fronts/Gibbs do not bite here (nu >= 0.01, n=256); the sine-P floor only plateaus at ~6e-6
   for P >= 48. Any fixed basis (learned or trig) is a linear span, so moving-front Kolmogorov
   decay hurts both equally; only x-dependent nonlinearity (FiLM/coord nets) escapes it, and
   the separable ansatz gave that up by design. **Nothing is lost going to a trig bank for
   this family; POD-K (K=8: 1.1e-2) is beaten by the nonlinear head, not by the bank.**
2. **Sparsity is not a cost lever at these sizes.** T is 32^3 = 262 kB (1D) or 64^3 = 2 MB
   (2D f64). h^T T h is two tiny matvecs (reshape T to (MR,R)@h, then (M,R)@h) — 2 kernels,
   Jacobian via `jax.linearize` reuses them. A sparse/banded core would *add* kernels. The
   value of a product-closed basis is **analyticity / grid-freedom / resolution-independence /
   trivial B**, not flops.

## Candidates

### C0. Coordinator's tensor on the frozen learned bank (D^- fixed)
- **Exactness** [CPU]: h^T T h reproduces grid Phi^T(u D^- u) to 8e-15. Against the true
  sign-upwind oracle on *decoded* fields: mean 2.3e-6, median 2.3e-7, max 1.6e-4 relative in
  the nonlinear term — 6.9% of decoded points are < 0 (min -1.2e-2) but N = u u_x is
  quadratically small there. On FOM truth fields: exactly 0. Compare: NNLS-32's fit error is
  5.5e-2 and it *already matches the oracle rollout* (6.24e-3 vs 6.16e-3). So C0 is ~300x
  closer to the oracle than the rule that already suffices. Class: exact for u > 0, 1e-4-level
  for this data; not exact for sign-changing data.
- **Sign switch handling**: (a) accept (measured negligible); (b) for sign-changing data
  choose the FOM to be polynomial: global Lax-Friedrichs/Rusanov flux with constant alpha is
  `Dc(u^2/2) - alpha*dx/2 * (D+ - D-)u` = quadratic + linear, exactly tabulable; a smooth
  sign is *not* tabulable. Caveat measured [CPU]: on the *current* upwind truth, centered
  (2.8e-2), flux-D^- (1.6e-2) and Rusanov (2.8e-2) tensors all differ systematically from the
  oracle term — the upwind numerical viscosity u dx/2 ~ 3e-3 is 30% of nu at nu=0.01, where
  the nonlinear term is 4x the viscous term. Those are "consistent with a modified FOM" and
  need a rollout test; I would not swap the FOM unless data with sign changes is the goal.
- **Online**: 2 matvec kernels (+2 for the tangent), replaces the feature-gather/where/
  weighted-projection chain of the sampled rule. Expect a kernel-count *win* vs m=32 nodes.
- **Offline**: one einsum, 8M MAC in 1D; 2D at R=64, n=65k: 17 GMAC (~1 s); at R=256: 275 GMAC
  (seconds). Symmetry: only the u-factor/derivative-factor asymmetry prevents symmetrizing;
  contract as h^T T h anyway (no saving needed).
- **Breaks**: sign-changing families; non-polynomial N (other PDEs); nothing else.
- **Decisive experiment**: in `make_full_rw`, replace `upwind(G_int@hz)` projection with
  `einsum('ijk,j,k->i', T, hz, hz)`; 8 test trajectories at N=256. Must reproduce the oracle
  arm's rollout error 6.16e-3 to <1e-5 absolute and beat its `roll_ms` (47.6 ms ref /
  ~15 ms optimized). One sub-minute job.

### C1. Tucker re-expression of the frozen bank in a sine basis (post-hoc, no retraining)
- **Idea**: G = S_P C, C = S_P^T G (one DST, P coefficients per column). Then
  T = C^T-contracted analytic core: T[i,j,k] = sum_ab C[a,j] C[b,k] T_sine[i,a,b], with
  T_sine[i,a,b] = sum_x phi_i s_a D^- s_b — a closed form (product-to-sum + geometric sums;
  n enters only as a parameter). B = Phi^T G becomes **C[:M,:]** exactly (Phi is the first M
  sines). The whole ROM is then grid-free: u = S_P C h(z) is only needed at decode.
- **Exactness** [CPU]: nonlinear-term error vs oracle: P=24 1.5e-3, P=32 5.2e-4, **P=48
  4.5e-6 (the sign-switch floor)**; bank columns worst 1.8e-3 at P=48, 5e-4 at P=64; decoded
  test fields deviate from the learned decode by 1.6e-4 (P=48), 8.9e-5 (P=64) — all >> below
  the 3.7e-3 floor. Cheb (bc-factored) and cubic B-splines need ~48-64 dof for the same
  (Cheb-48: 6.7e-3 worst col; B-spline 46 dof: 3.7e-3) — sines win because they satisfy the
  Dirichlet BC and *are* the test modes.
- **Sparsity honesty** [CPU]: with the FOM's D^- stencil the sine core is **dense** (50% of
  entries > 1e-8; D^- of a sine is a half-cell-shifted cosine, not sine-sparse under
  Dirichlet). Only the centered stencil gives the classic i = |a +- b| structure (exactly 2
  nonzeros per (a,b): 2512 of 73728 at P=48) — and that is the modified-FOM class (2.8e-2
  off). Use the dense analytic D^- core; at P=48 it is 590 kB. Fine.
- **Online**: same as C0 (contract C h first — one extra (P,R) matvec — or pre-contract into
  a dense M x R x R table, identical cost to C0). **Offline**: a DST + one contraction.
- **Breaks**: P too small (P=16: 1.2e-2 nonlinear error); families with thinner fronts push P
  up (~L/front width).
- **Decisive experiment**: same job as C0 with G := S_48 C and B := C[:32]. Must match C0's
  rollout error to ~1e-4 absolute (expected ~1e-5).

### C2. Pure sine bank + nonlinear head (retrain stage 1 with a fixed bank, C = I)
- **Idea**: since sine-32 is a better linear space than the learned bank, drop the RFF bank
  MLP entirely: u = S_P h(z), P = 32 (or 48 with the head widened). Then **B = [I_M | 0]** and
  the residual is r(z) = W[ h_{1:M}(z) - h_prev + dt( h^T T_sine h + nu Lambda h_{1:M}(z) ) ]:
  a Fourier-Galerkin Burgers solver constrained to a K-dim manifold, with T_sine analytic and
  n-parametric. Stage 1 becomes "fit a nonlinear map from z to Fourier coefficients". Modes
  above M enter only through the nonlinear term (same as today's bank).
- **Exactness**: same class as C0 (exact for u > 0 with D^- core). Residual is then
  *identical in structure* at every n; only T_sine's n-parameter changes — the 1D scale
  ladder becomes a re-tabulation, not a retrain of the bank.
- **Online**: fewer kernels than C0 (no B matvec; h is the projection). **Offline**: one
  40k-step training (~30 s A100 today; cheaper with a fixed bank).
- **What is lost / could go wrong**: the head may train worse without a co-adapted bank (the
  orthogonality regulariser is free here; sines are orthogonal). If held recon lands above
  ~4e-3 the co-adaptation mattered and C1 is the fallback. Gibbs only if nu << 0.01.
- **Decisive experiment**: train K=8, P=32 sine bank, same data/seed/steps; **held recon must
  be <= 3.7e-3**; then rollout with the analytic T must match the oracle arm 6.16e-3. One
  ~1-minute job. This is the experiment that decides whether the learned bank survives.

### C3. Spline bank (local products, banded tensors)
- **Idea**: cubic B-splines; products are local so T[i,j,k] is banded in (j,k) (|j-k| <= 3,
  7 diagonals): M x nb x 7 = 10k entries at nb = 46; exact integration per knot interval by
  Gauss quadrature (analytic, no grid). Knot positions are a *sample-free analog of learned
  node positions* (stage 2 could move knots toward fronts).
- **Expressiveness** [CPU]: 46 dof -> decoded-test error 2.1e-4 (30 dof: 2.3e-4 on truth,
  1.4e-2 worst bank column). Comparable to sines, slightly worse per dof.
- **Cost**: online identical (banded 10k vs dense 32k is one kernel either way; banded gather
  is slower on GPU). Offline trivial.
- **Breaks**: with sine test modes T is dense in i anyway; the bandedness pays only if test
  functions are also splines, which changes the weak form (FEM-Galerkin, modified-FOM
  class) — do not do that. Splines are the right basis only if the FOM were FEM or for
  non-Dirichlet BCs / local refinement.
- **Decisive experiment**: same as C1 with the spline re-expression; not worth running
  before C1/C2 unless knot-adaptivity is the goal.

### C4. Hybrid (learned bank + trig closure)
Concatenating [S_P | G] gives cross tables that still need the grid offline (fine) and
buys nothing measurable: the sine block alone already exceeds the learned block. The only
useful "hybrid" is C1's Tucker form (learned factors C, analytic core), or training C
directly (P > R, e.g. P = 64, R = 32) — the "bank projected onto a product-closed subspace".
Skip as a separate candidate.

### C5. 2D (brief)
Re-express the 2D bank in the product sine basis (2D DST, P = 32x32 = 1024 coefficients per
column, C: 1024 x 64). u u_x + u u_y then has a **Kronecker core**: T2D = Tx (x) Sy + Sx (x) Ty
from 1D cores, memory 2(Mx Px^2 + My Py^2) instead of M P^2. Pre-contracting with C gives
back a dense M x R x R table (2 MB at R=64, 34 MB at R=256) — still one kernel, so the
Kronecker form matters only if the pre-contraction is skipped. Offline: 17 GMAC. Decisive
experiment: must match the 2D oracle arm, and beat the m=256 rule's kernel count.

### Other routes (not my angle; one line each)
Lift-and-learn: unnecessary for Burgers (already quadratic); relevant only for other PDEs.
Learned surrogate h -> Phi^T N: replaces a 300x-more-accurate exact table with a fit whose
Jacobian is untrusted; dominated by C0 here.

## Ranking
1. **C0** — exact to 1.6e-4 worst / 2e-6 mean on this data, zero new training, fewer
   kernels; the measured sign-switch effect is 300x smaller than a rule that already matches
   the oracle.
2. **C2** — same exactness, retires the bank MLP, B = I, n-parametric analytic core;
   one training job to decide.
3. **C1** — the safe grid-free version of C0 if C2's head does not train (P = 48).
4. **C5** — 2D, after C0 lands.
5. **C3 / C4** — only for FEM FOMs, non-Dirichlet BCs, or knot adaptivity.
Modified-stencil (Rusanov) tensors only if sign-changing data becomes a target, and then by
changing the FOM, not by approximating the upwind one (2.8e-2 systematic).

## What I would do first
C0 today: a ~40-line change to `make_full_rw` (build T from `G_int` and `upwind`'s D^-
branch, contract), 8 trajectories at N=256, check rollout error == 6.16e-3 (oracle) and time
it. In the same job run C2's stage-1 training with G := S_32 (one flag in
`train_autodecoder_1d`: fixed bank, skip `g`); if held recon <= 3.7e-3, the bank MLP is
gone, B is the identity, and the Burgers ROM is quadrature-free in the same sense the
Poisson one already is.
