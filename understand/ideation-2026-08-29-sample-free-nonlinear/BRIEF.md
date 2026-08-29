# Ideation brief: can the nonlinear (Burgers) residual be made sample-free?

You are one of several independent agents asked to ideate on this question. Think hard,
be concrete, be honest about what is exact vs approximate, and give cost estimates.
Do NOT run any GPU jobs. Reading code is fine (paths below). Write your answer as a
structured markdown report (headings: Candidates / For each: idea, exactness, online
cost, offline cost, what could go wrong, smallest decisive experiment / Ranking /
What I would do first). Keep it under ~1500 words. No LaTeX needed; plain math is fine.

## The setup (already established; do not re-derive)

Reduced-order model for PDEs. The field on the grid is produced by a *separable decoder*

    u(x; z) = bc(x) * < g(x), h(z) >,   i.e.   u = G h(z)

- G in R^{n x R}: the "bank" — R spatial shapes evaluated on the n grid points
  (random Fourier features -> small MLP -> R outputs, times a boundary mask). Trained in
  stage 1, then FROZEN and cached as a table. R = 32 (1D Burgers), 64-96 (2D Poisson).
- h(z) in R^R: the "head" — small MLP + linear skip from the latent z in R^K (K = 8 or
  16) to R mixing coefficients. Also frozen after stage 1.
- The ROM solves for z each time step (damped Levenberg-Marquardt) by driving a weak
  residual r(z) in R^M to zero, M = 32 or 64 sine test modes Phi (exact eigenvectors of
  the discrete Laplacian: Lap Phi = Phi Lambda).

Key consequence used everywhere: any LINEAR term in the residual collapses through one
precomputed matrix B = Phi^T G (M x R): Phi^T u = B h(z). No grid work online. For
Poisson (fully linear) the whole residual is r(z) = W * [Lambda B h(z) - Phi^T f] and the
ROM is exactly "quadrature-free" — verified to 1e-13 against the full-grid residual, and
fastest of all paths.

For 1D viscous Burgers  u_t + u u_x = nu u_xx  (backward Euler, dt = 0.005, 50 steps) the
residual is

    r(z) = W * [ B (h(z) - h(z_prev)) + dt * ( Phi^T N(u) + nu Lambda B h(z) ) ],
    N(u) = u * u_x  with the FOM's SIGN-UPWIND stencil:
       u_x[i] = (u[i]-u[i-1])/dx if u[i] > 0 else (u[i+1]-u[i])/dx

Only Phi^T N(u) is currently SAMPLED: evaluated at m nodes (m = 32 tight / 16 starved /
128 generous) with NNLS weights, optionally with learned node positions (stage 2).
The full-grid "oracle" (N summed over every grid point) is the best any sampled rule can
do; learned 32-node sets land on it; sampling error is the only residual approximation.

Measured facts (single seed, A100, f64): decoder floor ~3.7e-3 relative L2; oracle
rollout ~4.5-6e-3; NNLS-32 ~5-6e-3; NNLS-16 ~2.2e-2; learned-16 ~1e-2. Online cost is
launch/kernel-count bound (~15 ms latent solve per 50-step trajectory after
optimization, flat in n from 128 to 4096). FOM (tridiagonal Newton) ~8-9 ms at these
sizes. The 2D Burgers analog uses the same structure (sign-upwind on both axes, K=8-16,
R=64, M=64, m=256).

Data facts you may use: the 1D training/test initial conditions are POSITIVE Gaussian
blobs a*exp(-(x-c)^2/(2w^2)), a in [0.5,1.5], with zero Dirichlet walls; nu in
[0.01, 0.1]. Viscous Burgers with nonnegative data and zero walls keeps u >= 0 (max
principle), and the monotone backward-Euler upwind FOM does too. The decoded ROM field
may have tiny undershoots below 0.

## The coordinator's own candidate (critique it, improve it, or beat it)

Because u = G h is linear in h, a QUADRATIC nonlinearity is a fixed 3-index table:

    Phi^T ( (G h) * (D G h) ) = sum_{j,k} h_j h_k T[:, j, k],   T[i,j,k] = sum_x Phi[x,i] G[x,j] (D G)[x,k]

T is M x R x R = 32^3 = 32768 numbers in 1D (64^3 = 262k in 2D), precomputed ONCE on
the full grid from the frozen bank. Online: one contraction h^T T h (M R^2 flops, one
kernel) + Jacobian 2 T h (dh/dz). This is the classic POD-Galerkin quadratic-tensor
trick, applied to the bank rather than to a POD basis, with the nonlinear head
untouched. If the sign-upwind stencil were a FIXED backward difference D^- (true wherever
u > 0, i.e. essentially everywhere in this data), T is EXACT — it reproduces the
full-grid oracle, no sampling, nothing to fit, nothing to train.

Open issues with it: (1) sign switching where the decoded field undershoots 0; (2)
general data with sign changes (|u| is not polynomial); (3) non-polynomial
nonlinearities in other PDEs; (4) 2D cost / memory / kernel count vs the 256-node
sampled rule; (5) whether T should be built from the FOM's flux form (u^2/2)_x instead.

## What we want from you

1. Assess the tensor candidate honestly: when exact, when not, how to handle the sign
   switch (e.g. positivity guarantee; smooth sign; flux splitting with a polynomial
   numerical flux such as global Lax-Friedrichs / Rusanov with constant alpha, which is
   quadratic + linear and hence tabulable; choosing the FOM discretization itself to be
   polynomial), and what it costs in 1D and 2D.
2. Propose OTHER sample-free routes for nonlinear terms in this separable-decoder
   setting. Think about: bank bases closed under products (Fourier/trig banks where
   products of modes are sums of modes, sparse analytic tensors); lifting /
   polynomialization of non-polynomial nonlinearities (Kramer-Willcox lift & learn
   style); a learned surrogate that maps h -> Phi^T N(G h) directly (a "learned
   projection" with no points — how to keep its gradient trustworthy, training cost,
   failure modes); exploiting that the head is small so the residual is a function of
   R=32 numbers; low-rank / CP / Tucker compression of T for 2D; symmetric-tensor
   savings; anything else you can justify.
3. For each route: exactness class (exact / consistent with a modified FOM / learned
   approximation), online cost (flops AND kernel count — we are kernel-bound), offline
   cost, memory, what breaks it, and the SMALLEST decisive experiment (which existing
   arm it must match, e.g. "must equal the oracle arm's rollout error to 1e-9").
4. Rank them and say what you would do first and why.

Code (read-only, optional): the 1D testbed is
/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-27-b1d-poissonqf/experiments/separable-decoder/
(b1d_common.py: grid, upwind_adv_field_1d, fom_residual_int, tridiag_jac; sep_b1d_scale.py:
the residual / rollout; b1d_fast_common.py: the optimized rollout).
The 2D exlin residual lives in sep_burgers_exlin.py in the same directory.
