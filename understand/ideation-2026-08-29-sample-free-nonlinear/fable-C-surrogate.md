# Ideation C: learned, point-free surrogates for Phi^T N(G h) (and where they sit vs the tensor)

Angle: the residual needs only M numbers q(h) = Phi^T N(G h) as a function of R = 32 numbers.
Everything below is about *what* to learn, *from which input*, and *how to keep dr/dz honest*.
Two tiny CPU-only checks were run on the committed N=256 checkpoint
(`runs/b1dqf/b1ds_n256/out/sep_b1d_scale_n256.pkl`, single seed, residual level only, no
rollout; scripts `tensor_check_cpu.py`, `regress_check_cpu.py` beside this file). Those numbers
are labelled **[cpu-check]** and are provisional until an arm runs on the cluster.

## Headline finding (changes the ranking of everything else)

**[cpu-check]** The fixed backward-difference tensor T (coordinator's candidate, D^- with ghost
zero at the left wall) vs the true sign-upwind projection Phi^T N(G h):

| states | rel err median | p95 | max | frac(u<0) pts | neg-mass rel (max) |
|---|---|---|---|---|---|
| training codes | 1.4e-7 | 1.4e-5 | 6.0e-5 | 6.9% | 5.3e-3 |
| z + 0.05 N(0,1) | 1.8e-7 | 2.4e-3 | 1.3e-2 | 11% | 0.21 |
| z + 0.20 N(0,1) | 9.6e-6 | 5.3e-2 | 0.12 | 13% | 0.99 |

Jacobian dq/dz (tensor vs oracle, z+0.05): rel err median 1.3e-6, max 2.7e-2, **cosine min 0.9996**.
Restricted to points with u>0 the mismatch is 8e-15 (exact, as claimed); all error is the
undershoot points, and it is second-order small there (u_- times a one-sided difference of a
tiny field). Compare NNLS-32: held-out b = 6.3%, c1_cos 0.91; learned-32: 4.3%, 0.96.

The LM trust region is 0.017 in z (`trust_delta` in the JSON), i.e. LM iterates live inside
the sigma~0.02 column, where the tensor is ~1e-7. So on this data the tensor is not "a
candidate": it is the oracle to 5 decimal places with no sampling, no fitting, no training. **Any
learned surrogate must therefore be judged against 1e-7, not against 6%.** That kills
"learn q(h) from scratch" and reframes learning as *a correction for the non-polynomial part*.

## Candidates

### C1. Exact tensor (the coordinator's; assessed, not mine)
- Exactness: exact wherever u>0 (measured 8e-15); error = sum over undershoot points, O(|u_-| |u_xx| dx). Head untouched, bank frozen: T[i,j,k] = sum_x Phi[x,i] G[x,j] (D^- G)[x,k]. Flux form (u^2/2)_x gives a different T (D^- applied to G_j G_k); the FOM here is the non-conservative u u_x form, so build T to match the FOM that made the data; whichever is used, the gate is the same.
- Online: h^T T h = M R^2 = 32k flops, 1 fusion (reshape T to (M R, R) matmul + row-reduce), Jacobian 2 T_sym h via `jax.linearize` in the same pass. Fewer kernels than the sampled rule (G3 einsum + where + Phi_q^T: 3-4 fusions). Store only the symmetric part T_sym (M x R(R+1)/2 = 32 x 528). 2D: M R^2 = 64^3 = 262k flops, 2 MB f64; combine u u_x + u u_y into ONE tensor (both quadratic in h) so the kernel count does not grow with the number of advection terms. No CP/Tucker needed: at 2 MB the cost is kernels, not bytes (PROFILE.md: r+J is 23 fusions at 1.6 flop/byte).
- Offline: n M R^2 flops: 8e6 (1D), 2.7e11 (2D at 1M points) = seconds on an A100. Build it with the same `upwind_adv_field_1d` projection applied to basis pairs so gate-consistency is trivial.
- Breaks on: sign-changing data (|u_-| = O(1) makes the error O(1)); non-polynomial N in other PDEs.
- Decisive experiment E1: add arm `tensor` to `sep_b1d_scale.py` (`r_w` with T, same LM). Must match the `oracle` arm: held-out b <= 1e-5, c1_cos >= 0.999, rollout error 6.165e-3 (N=256) to <= 1e-4 relative, roll_ms <= base_tight's. Existing checkpoint, no training, ~10 min.

### C2. Quadratic regression on the bank (OpInf-on-the-bank) — "learned tensor"
- Idea: least squares W (528 x M) on monomials h_j h_k against full-grid teacher targets; recovers T when the FOM is quadratic and D is unknown (black-box FOM).
- **[cpu-check]** 4000 training codes, f64 lstsq: on-manifold test median 3.0e-6 / p95 1.1e-5 / max 3e-4; at z+0.02: median 4.8e-6, p95 7.4e-4, max 7.7e-3. Fit from h-space perturbations is *worse* on-manifold (2.5e-4) — h never leaves head(z) during LM, so perturb z, not h. The LS solution absorbs the sign-switch part into the coefficients: 20x worse than the analytic tensor on-manifold, with fat tails.
- Exactness: learned. Online: identical to C1. Offline: ~2000 teacher evaluations + one lstsq, seconds.
- Use only when D is not available analytically. Decisive experiment E2: same arm as E1 with W; must match oracle rollout to 1e-3.

### C3. Residual-learned correction on top of T (the principled hybrid) — my main proposal
Write q(h) = h^T T h + c(h), c = Phi^T [N_upwind(Gh) - (Gh)(D^- G h)], and learn only c.
- Why this form: c is identically zero on u>=0 states, so the surrogate's *scale* is the undershoot mass; an untrained c=0 already gives 1e-7. The polynomial Jacobian is exact; the learned Jacobian error is multiplied by |c|. This is exactly the project's "exact linear terms + learn the rest" lesson (exlin), one polynomial degree up.
- Three parametrisations, cheapest first:
  (a) **Separable correction bank** (project-native): train a second bank G_c (n x R_c, R_c ~ 8) and head h_c(z) on the grid so that G_c h_c(z) ~ N_upwind(u) - u D^- u (the correction *field*); precompute B_c = Phi^T G_c. Online: c = B_c h_c(z), one small MLP + one matvec (2-3 fusions); Jacobian through h_c by the same linearize pass. The whole residual stays in "linear collapse" form: r = W[B(h-h_prev) + dt(h^T T h + B_c h_c(z) + nu Lambda B h)].
  (b) **Rectified rank-P model**: c(h) = sum_p phi_p relu(-a_p . h)(b_p . h), free vectors a_p, b_p in R^R, phi_p in R^M. Mirrors the exact structure (a_p . h ~ u at a virtual point, b_p . h ~ (D^+ - D^-)u there): a learned sampling rule whose "nodes" are free rows instead of g(x_p) — the point-free generalisation of the stage-2 learned nodes. 3 fusions, P(2R+M) params.
  (c) Generic MLP 32-64-64-32: ~6k params, 3-5 fusions. Fallback only.
- Training (reuse the existing teacher(): full-grid Phi^T N and its jacfwd): loss = value term + JAC_REL x gradient-matching term (L_jac, same normalisation as `term_jac`), on training codes + z-perturbations with sigma >= trust_delta (0.017) and a few 0.05-0.2 outliers, f64, Adam, minutes. Anti-collusion: decoder frozen, teacher is the grid oracle, the surrogate never sees rollout error, and certification is held-out b / c1_cos / rollout-vs-FOM (never the fit residual — the EQ-ladder lesson).
- Jacobian trust: Sobolev term is mandatory (a small-valued c can have a large-gradient fit); certify c1_cos >= 0.999 on held-out states (the Poisson NNLS failure with cos down to -0.65 was invisible in b and only visible in c1); optionally clip: the LM can use J_poly only (exact) and c only in the value — a Gauss-Newton with an inexact Jacobian still converges when |dc/dz| << |dq_poly/dz|, which is true by construction on this data.
- Generalisation off-manifold: the solver visits head(z) for z within 0.017 of the previous step's code; the danger is 50-step drift out of the training cloud. Guard: train on z-perturbations up to 0.2, and add a "distance-to-cloud" tripwire (min ||z - Z_tr|| > 3 sigma -> log; do not fall back silently).
- Cost: +3-5 fusions per r+J on ~23 today = +15-20% per LM iteration ~ +2-3 ms on the 15 ms rollout. **On this data it buys nothing** (tensor already 1e-7), so it should not be in the 1D default arm; it is the tool for sign-changing data and for other PDEs.
- Failure modes: kinked target (relu-type) smoothed by an MLP — fine at the 1e-3-of-total level the rollout needs; f32 would floor at 1e-4 (use f64); drift; a bank for the correction needs the correction to be reasonably low-rank on the grid (check singular values of the correction-field snapshot matrix first; if it is not low-rank, use (b)).
- Decisive experiment E3 (only after E1, and only once a sign-changing family exists, e.g. a in [-1.5, 1.5] or opposite-sign double blobs): (i) show the plain tensor arm FAILS there (rollout error >> oracle), (ii) train (a) or (b), (iii) the corrected arm must match the oracle arm's rollout to 1e-3 relative with c1_cos >= 0.999, at <= +20% roll_ms. If (i) does not fail, there is nothing to learn.

### C4. From-scratch surrogate q(h) or q(z) (MLP, no tensor) — reject
Reaches 1e-3..1e-4 at best (beats NNLS-16's 2e-2, roughly ties NNLS-32 after 50 steps of error growth), 3-4 orders worse than C1, more kernels than C1, and its Jacobian is entirely learned. Input from z (K=8) is even worse: it welds the surrogate to the head and makes the Jacobian a black box. No experiment warranted.

### C5. Modified-FOM route: global Lax-Friedrichs / Rusanov with constant alpha (brief; covers the sign switch exactly)
F_{i+1/2} = (F_i + F_{i+1})/2 - alpha/2 (u_{i+1} - u_i) with F = u^2/2: the flux difference is central-difference quadratic (tabulable, T_central) plus alpha dx/2 x discrete Laplacian, which collapses through the EXISTING Lambda B h. alpha = max|u0| per trajectory is valid for all t by the max principle. Exact for any sign of u, zero learning, same kernel count as C1. Class: consistent with a modified (more uniformly dissipative) FOM — the data must be regenerated with it; numerical viscosity alpha dx/2 ~ 3e-3 at N=256 is comparable to upwind's |u| dx/2 and to the smallest nu. Decisive: regenerate with the LF FOM, tensor arm must equal the LF oracle to 1e-9 (it is exactly polynomial now). Best "exact for general data" option if the FOM is ours to choose.

### C6. Others, briefly
- Product-closed (sine) banks: sparse analytic tensor (i = |j +- k|), but the bank becomes the first-32-sines subspace, giving up the adapted 32-dim subspace the trained bank provides (the CP-decoder ceiling is the subspace, so this costs accuracy) to shrink a 32k-entry table that already costs one kernel. Reject.
- Lift-and-learn for non-polynomial PDEs: in this setting the lifted variable w = f(u) has no decoder, so "lifting" degenerates into C3(a) — a nonlinearity bank G_f, head h_f, projection B_f h_f(z). That is the right general-PDE route and is already covered by C3(a).
- Positivity-projected decoder (clip u at 0 before N): exact-izes T on this data but clip is non-polynomial — no gain over T, which is already 1e-7 because the undershoot is tiny.

## Ranking

1. **C1 tensor** — exact where u>0, measured 1e-7 on the real checkpoint, fewer kernels than sampling, no training. Run E1 now.
2. **C5 Lax-Friedrichs FOM** — if sign-changing data is on the roadmap: exact, no learning, same cost; requires owning the FOM.
3. **C3 residual-learned correction (a)/(b)** — the only learned object worth building; needed only for sign-changing data or non-polynomial PDEs; must reuse L_jac and the c1_cos gate.
4. **C2 OpInf quadratic regression** — black-box FOMs only; 20x worse than C1 with tails.
5. **C4 from-scratch surrogate** — never.

## What I would do first

E1 today: `tensor` arm in `sep_b1d_scale.py` from the cached N=256 checkpoint (then N=2048 via `CKPT_CACHE`), gated against `oracle` (b <= 1e-5, c1_cos >= 0.999, rollout 6.165e-3 to 1e-4 rel, roll_ms <= base_tight). Then the 2D analogue in `sep_burgers_exlin.py` with x and y advection folded into one 64^3 tensor, against the m=256 NNLS arm. Only if a sign-changing family is adopted do I train C3(a) — and I would first try C5, which removes the need to learn anything.
