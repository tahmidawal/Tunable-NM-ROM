# What the EQ fidelity ladder taught us

This note explains the 2026-08-25 empirical-quadrature diagnostic. The central result is that the quadrature samples the state and its Jacobian well, but distorts the small weak residual enough to rotate the optimization gradient. Most of that residual error is in linear terms that the separable decoder can evaluate exactly.

## A. Setup: the weak residual that the latent solver sees

At one backward-Euler step, the decoder represents the new state by \(u(z)\), while \(u^n=u(z^n)\) is the previous ROM state. Let \(\phi_p\) be column \(p\) of \(\Phi\), where \(\Phi\) contains the \(M\) lowest discrete sine modes on the interior grid and every column has unit Euclidean norm. If \(\lambda_p\) is the discrete-Laplacian eigenvalue of that mode, the full-grid weak residual is

\[
R_{f,p}(z)=w_{t,p}\left[\phi_p^T\bigl(u(z)-u^n\bigr)+\Delta t\,\nu\lambda_p\phi_p^T u(z)+\Delta t\,\phi_p^T N(u(z))\right],
\qquad
w_{t,p}=\bigl(1+\Delta t\,\nu\lambda_p\bigr)^{-\alpha}.
\]

In words, the four pieces are the new-state mass term, the negative previous-state term, diffusion represented through the sine eigenvalue, and nonlinear advection; \(w_{t,p}\) prevents high modes from dominating merely because their Laplacian eigenvalues are large.

The incumbent advection operator is not a continuum derivative. It is \(N(u)=u(u_x+u_y)\) evaluated with the FOM's sign-upwind five-point stencil. Preserving that stencil matters: changing it would change the discrete model rather than merely accelerate its evaluation.

The latent state is chosen by Levenberg–Marquardt (LM):

\[
z^*=\arg\min_z \frac12\lVert R(z)\rVert_2^2,
\qquad
\bigl(J^TJ+\lambda_{\mathrm{LM}}D\bigr)\,\delta z=-J^TR.
\]

This means LM seeks the decoder state whose projected PDE imbalance is smallest, using damping to control a Gauss–Newton step when the local quadratic model is unreliable.

At \(N=1024\), an exact projection sums over roughly one million interior points. Online, it is replaced by an empirical quadrature with \(m\approx4M\) nodes:

\[
\phi_p^T v\;\approx\;\sum_{j=1}^{m} w_j\,\phi_p(x_j)v(x_j),
\qquad w_j\ge 0.
\]

This means the solver retains \(M\) weak equations but evaluates them from a small, positively weighted subset of the grid.

The weights used today are not uniform. For each EQ set, 64 training codes are selected. For every decoded snapshot, the fit stacks rows for both \(\Phi^Tu\) and \(\Phi^TN(u)\); the targets are their exact full-interior-grid projections. A capped Lawson–Hanson active-set NNLS solve chooses nonnegative weights and stops when its support reaches \(m\). The measured decoders have \(R=512\) spatial features and \(K=16\) or \(32\) latent variables; the measured rules are the control \(M/m=64/256\) and fine \(M/m=256/1024\) sets.

## B. The fidelity ladder: from fields to solver steps

Write \(s\) for sampled quadrature and \(f\) for exact full-grid projection. Rung (a) compares decoder-versus-truth reconstruction error on the nodes with reconstruction error on the whole grid:

\[
e_s=\sqrt{\frac{\sum_jw_j\lvert u(x_j;z)-u_{\rm true}(x_j)\rvert^2}{\sum_jw_j\lvert u_{\rm true}(x_j)\rvert^2}},
\qquad
e_f=\frac{\lVert u(z)-u_{\rm true}\rVert_2}{\lVert u_{\rm true}\rVert_2},
\qquad (a)=e_s/e_f.
\]

This asks whether the selected nodes are representative places at which to measure state error.

For rung (b), split \(R=R^{\rm lin}+R^{\rm adv}\), with mass, previous state, and diffusion in the linear part. The driver reports

\[
(b)=\frac{\lVert R_s-R_f\rVert_2}{\lVert R_f\rVert_2},
\qquad
b_{\rm lin}=\frac{\lVert R_s^{\rm lin}-R_f^{\rm lin}\rVert_2}{\lVert R_f\rVert_2},
\qquad
b_{\rm adv}=\frac{\lVert R_s^{\rm adv}-R_f^{\rm adv}\rVert_2}{\lVert R_f\rVert_2}.
\]

This asks whether the weak integral is right and identifies which physical term supplies its error; the two component norms need not add because their error vectors can cancel.

Let \(g=J^TR\). Rung (c1) is

\[
c_1=\frac{\lVert g_s-g_f\rVert_2}{\lVert g_f\rVert_2},
\qquad
\cos(g_s,g_f)=\frac{g_s^Tg_f}{{\lVert g_s\rVert_2\lVert g_f\rVert_2}},
\qquad
c_{1,\rm abs}=\frac{\lVert g_s-g_f\rVert_2}{\lVert J_f\rVert_F\lVert R_f\rVert_2}.
\]

This asks whether sampled and full objectives push the latent variables in the same direction, while the absolute normalization remains interpretable when the true gradient is near zero.

With \(H=J^TJ\), rung (c2) is

\[
c_2=\frac{\lVert H_s-H_f\rVert_F}{\lVert H_f\rVert_F}.
\]

This asks whether the sampled Jacobian supplies the correct Gauss–Newton curvature.

Finally, at the sampled solver's own current \(\lambda_{\rm LM}\), the driver forms

\[
\delta z_q=-\bigl(H_q+\lambda_{\rm LM}D_q\bigr)^{-1}g_q,
\qquad
c_3=\frac{\lVert\delta z_s-\delta z_f\rVert_2}{\lVert\delta z_f\rVert_2},
\qquad
\cos(\delta z_s,\delta z_f)=\frac{\delta z_s^T\delta z_f}{\lVert\delta z_s\rVert_2\lVert\delta z_f\rVert_2},
\quad q\in\{s,f\}.
\]

This asks the operational question: at the same latent state and damping value, would exact integration make LM move somewhere else?

A modest residual error can become a large gradient error because \(J^TR\) mixes every residual component through the tangent directions. Near a stationary point, \(\lVert g_f\rVert\) is deliberately small, so even a stable absolute discrepancy can produce a huge relative \(c_1\); that is why \(c_{1,\rm abs}\) is essential.

The ladder probes three state families. **Solver-path** states are every LM iterate from real sampled-operator rollouts, using the ROM's own previous latent. They are “off-manifold” in the practical sense that they are away from the distribution of encoded truth and training snapshots, although every decoded iterate still lies in the decoder's image. **Oracle** states are full-grid least-squares codes of truth states, with oracle previous codes. **Training snapshots** are 16 fitted training codes paired with their preceding snapshot. These distributions separate interpolation quality from the states the nonlinear solver actually visits.

## C. Findings F1–F7, and what they mean

**F1 — NNLS rel fit is not residual certification.** The control-path residual errors are directly in the 30–50% range: T-L1 gives (b) \(3.3\mathrm{e}{-01}\) for `sep_hfit_dense_mid_N256 / ctrl` and \(5.3\mathrm{e}{-01}\) for `sep_burgers_r3_N1024_K16_R512 / ctrl`. In the latter row, NNLS rel fit is only \(5.4\mathrm{e}{-03}\); in the corresponding `M256` row, rel fit and (b) are \(5.9\mathrm{e}{-04}\) and \(6.7\mathrm{e}{-02}\). As a worked example, \(5.3\mathrm{e}{-01}/5.4\mathrm{e}{-03}\approx98\): a half-percent fit statistic coexists with a residual error about two orders of magnitude larger.

There is no contradiction. Rel fit measures \(\Phi^Tu\) and \(\Phi^TN(u)\) on large, smooth snapshot fields. The time-step residual is a small difference: \(u-u^n\), diffusion, and advection nearly cancel at a solution. The same absolute projection error is therefore a much larger fraction of \(R_f\). At the oracle solution this is sharper: T-L3, the same decoder's `ctrl` and `M256` rows have (b) \(8.9\mathrm{e}{-01}\) and \(8.0\mathrm{e}{-02}\).

**F2 — the largest error is removable algebra.** T-L1, row `sep_burgers_r3_N1024_K16_R512 / M256`: (b) is \(6.7\mathrm{e}{-02}\), with `lin` \(7.1\mathrm{e}{-02}\) and `adv` \(2.6\mathrm{e}{-02}\). Physically, the sampled mass and diffusion projections are more wrong than the genuinely nonlinear flux projection, even though only the latter needs quadrature.

**F3 — \(J\) is sampled well; \(R\) corrupts the gradient and step.** T-L2, row `sep_burgers_r3_N1024_K16_R512 / ctrl`: (c2) is \(4.7\mathrm{e}{-03}\), but the gradient cosine is \(0.34\) and the step cosine is \(0.44\). In its `M256` row, (c2) is \(2.5\mathrm{e}{-04}\), while those cosines are \(0.79\) and \(0.67\). Curvature is already accurate; the solver is being steered mainly by residual error.

**F4 — the sampled optimum is displaced.** T-L3, row `sep_burgers_r4_N1024_K32_R512_h512x3 / ctrl`: at the oracle code, (c1) is \(2.2\mathrm{e}{+01}\), its cosine is \(0.38\), and the step discrepancy is \(2.7\mathrm{e}{+00}\). The `M256` row improves these to \(1.6\mathrm{e}{+00}\), \(0.82\), and \(2.4\mathrm{e}{-01}\). Thus a code that is good for the full weak objective is not stationary for the sampled objective, so sampled LM is driven toward another minimizer.

**F5 — better decoders encounter a quadrature floor.** In T-L4 `ctrl`, full-grid reconstruction error improves from \(3.7\mathrm{e}{-02}\) for `sep_burgers_r3_N1024_K16_R512` to \(2.2\mathrm{e}{-02}\) for the K=32 decoder. Yet T-L3 `ctrl` shows (c1) worsening from \(9.5\mathrm{e}{+00}\) to \(2.2\mathrm{e}{+01}\), while \(c_{1,\rm abs}\) remains of the same order, \(1.8\mathrm{e}{-01}\) versus \(2.9\mathrm{e}{-01}\). As the true residual and gradient shrink, a similar quadrature scale occupies more of the remaining error budget.

**F6 — the points themselves are representative.** Across T-L4, the node/full reconstruction ratio ranges from \(0.95\) (`sep_hfit_dense_mid_N256 / ctrl`) to \(0.99\) (`sep_burgers_r3_N1024_K16_R512 / M256`). The issue is not that EQ picked pathological locations for observing \(u\); it is cancellation inside the projected residual.

**F7 — refinement does not cure this.** T-L1 gives (b) \(4.3\mathrm{e}{-01}\) at N=256 and \(5.3\mathrm{e}{-01}\) at N=1024 for the K=16 control rows; the fine rows give \(5.6\mathrm{e}{-02}\) and \(6.7\mathrm{e}{-02}\). These are comparable changes, not mesh-driven convergence, so the fitting target must change rather than waiting for resolution to help.

There is one structural caveat. At \(t=0\), the oracle construction sets \(u^n=u\), so the mass difference vanishes and \(R_f\) is not the same small cancellation as later steps. T-L3, `sep_burgers_r3_N1024_K16_R512 / ctrl`, shows (b) \(5.5\mathrm{e}{-02}\) at \(t=0\) versus \(1.2\mathrm{e}{+00}\) at \(t\ge5\); compare like time buckets.

## D. The key derivation: make every linear term exact

For the separable decoder, include the boundary factor in the spatial bank \(G\in\mathbb{R}^{n_i^2\times R}\):

\[
u(z)=G\,h(z),
\qquad G_{xi}=bc(x)g_i(x).
\]

This means all dependence on the latent code is confined to the \(R\)-vector \(h(z)\).

Precompute \(A=\Phi^TG\in\mathbb{R}^{M\times R}\). Then

\[
\Phi^Tu(z)=\Phi^TGh(z)=A\,h(z)
\]

This identity is exact on the full interior grid, so it has no quadrature error.

Consequently the entire linear residual is

\[
R_f^{\rm lin}(z)=W_t\left[A\bigl(h(z)-h(z^n)\bigr)+\Delta t\,\nu\Lambda A h(z)\right],
\qquad W_t=\operatorname{diag}(w_t),\quad\Lambda=\operatorname{diag}(\lambda_p).
\]

This means mass, previous-state, and Laplacian terms require only one precomputed matrix; for the fine rule it is \(256\times512\), and the current-state online work is one dense matrix-vector product whose previous-state result can be cached.

Advection does not factor this way. At node \(j\), the implemented stencil is

\[
N_j(u)=c_j\left[\operatorname{where}\!\left(c_j>0,\frac{c_j-u_{x-}}{\Delta x},\frac{u_{x+}-c_j}{\Delta x}\right)+
\operatorname{where}\!\left(c_j>0,\frac{c_j-u_{y-}}{\Delta x},\frac{u_{y+}-c_j}{\Delta x}\right)\right].
\]

This means that even though every stencil value is linear in \(h(z)\), their product is quadratic within a fixed-sign region, and the selected one-sided difference changes when the sign of \(c_j\) changes.

The exact-linear change therefore leaves only \(\Phi^TN(u)\) sampled. Its added dense \(256\times512\) product is one small kernel. The existing N=1024 profile fits about \(0.14\) ms per LM iteration and shows the code is limited by kernel dispatch count, not arithmetic or bandwidth, so replacing sampled linear projections with one cached product should preserve the online cost scale.

## E. Refit advection weights with a convex teacher problem

With nodes fixed and a frozen decoder, define the sampled advection projection by

\[
q_{s,p}(w;z)=\sum_{j=1}^{m}w_j\phi_p(x_j)N(u(z))(x_j).
\]

This means \(q_s\), the residual \(R_s\), and the residual Jacobian \(J_s=\partial R_s/\partial z\) are affine in \(w\), because differentiation with respect to \(z\) passes through the weighted sum.

Across selected states, residual and Jacobian teacher rows can therefore be stacked as

\[
\min_{w\ge0}\lVert\mathcal A w-b\rVert_2^2.
\]

This is an NNLS problem with the same fixed nodes and nonnegativity constraint, but with rows aimed at the quantities the solver consumes rather than at large snapshot fields.

There is an important algebraic qualification to the design report's “gradient-target NNLS” shorthand. The literal \(g_s(w)=J_s(w)^TR_s(w)\) is quadratic in \(w\), so directly minimizing \(\lVert g_s(w)-g_f\rVert^2\) is not linear least squares. A genuinely convex same-target construction instead freezes the full-grid teacher in one factor—for example, stack \(J_f^TR_s(w)\approx J_f^TR_f\) and \(J_s(w)^TR_f\approx J_f^TR_f\), together with residual/Jacobian rows—then certifies the resulting weights using the actual rung (c1). That distinction keeps the refit convex without pretending that a product of two weight-dependent sums is linear.

The new rows should be advection residual and gradient-aware teacher rows, drawn from off-snapshot solver iterates as well as oracle/training states and from several viscosities. The state distribution matters: for `sep_burgers_r3_N1024_K16_R512`, the gradient cosine is \(0.34\) off-manifold in T-L2 `ctrl` but \(0.50\) on-manifold in T-L3 `ctrl`; for `M256` it is \(0.79\) versus \(0.94\). Node count is unchanged within each comparison, so the gap points to the fitting target and state distribution. Meanwhile (c2) remains small, showing that adding nodes merely to improve \(J\) is not the first lever.

## F. Ranked fixes and honest caveats

1. **Compute the linear terms exactly.** Precompute \(\Phi^TG\), remove their quadrature error, and remeasure the ladder.
2. **Refit an advection-only, gradient-aware NNLS rule.** Use residual/Jacobian teacher rows from real solver iterates and several \(\nu\), then gate the actual \(J_s^TR_s\) with (c1).
3. **Learn node positions or a weight network only if advection still binds.** Nothing measured yet justifies paying the optimization and validation complexity of moving nodes or joint training.

Four caveats prevent overclaiming. First, `lin` and `adv` were separated in \(R\), not in \(J^TR\), so removing the linear residual discrepancy need not reduce gradient error by the same factor. Second, gate 0 currently proves identity to the incumbent *sampled* weak residual; after the change it must be redefined against the full-grid weak residual. Third, the \(t=0\) oracle rows are structurally lower because \(u^n=u\). Fourth, `N_TEST=4`: T-L5 rollout cells are diagnostic means of four trajectories, including one hard trajectory, rather than the headline accuracy protocol. For example, T-L5, row `sep_burgers_r4_N1024_K32_R512_h512x3`, reports `ctrl / M256` rollout errors \(3.0\mathrm{e}{-02}/1.4\mathrm{e}{-02}\); those values should not be promoted into the main accuracy table.

## G. What to tell the advisor

The weights were never uniform: they were nonnegative, NNLS-fitted weights trained to reproduce full-grid \(\Phi^Tu\) and \(\Phi^TN(u)\) projections on 64 decoded training snapshots.

The “gradient is the holy grail” point is confirmed. State error on the nodes is representative and \(J^TJ\) is accurate, yet residual cancellation makes \(J^TR\) rotate enough to change the LM step.

The data do not yet motivate joint training of \((g,h,w)\). The first fix is exact algebra for the linear terms; the second is a frozen-decoder, advection-only convex refit with gradient-aware teacher rows. Learned nodes or joint training become justified only if the remeasured advection rung remains binding.
