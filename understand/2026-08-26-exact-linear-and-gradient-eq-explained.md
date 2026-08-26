# What exact linear terms and learned quadrature taught us

This note explains the three empirical-quadrature experiments completed overnight on 2026-08-26. The short version is that exact algebra gave a free and reliable improvement, better-targeted quadrature helped only when the node budget was coarse, and learned node positions confirmed both the opportunity and the limit: at the fine budget, quadrature is not the error that binds.

## A. Exact linear terms: stop approximating what the decoder already gives us exactly

The separable decoder has the pointwise form

\[
u(x;z)=bc(x)\langle g(x),h(z)\rangle.
\]

After evaluating the spatial track \(bc(x)g(x)\) on the interior grid, this becomes

\[
u(z)=G_{\mathrm{int}}h(z),
\]

where the columns of \(G_{\mathrm{int}}\) are the spatial feature bank, including the boundary factor, and \(h(z)\) supplies the feature coefficients for the current latent state. The crucial separation is that \(G_{\mathrm{int}}\) depends on space but not on \(z\), while all latent dependence is confined to the small vector \(h(z)\).

Let \(\Phi\) contain the weak test modes. We can precompute

\[
A=\Phi^T G_{\mathrm{int}}.
\]

Then every projected state term follows from the exact identity

\[
\Phi^T u(z)=\Phi^T G_{\mathrm{int}}h(z)=Ah(z).
\]

The new online weak residual is therefore

\[
r_w(z)=w_t\odot\left[A\bigl(h(z)-h(z^n)\bigr)
+\Delta t\left(\Phi_q^T N(u)|_{\mathrm{nodes}}
+\nu\lambda\odot Ah(z)\right)\right].
\]

Here \(z^n\) is the previous latent state, \(w_t\) is the weak-mode scaling, \(\lambda\) contains the discrete-Laplacian eigenvalues, and \(\Phi_q^T N(u)|_{\mathrm{nodes}}\) is the one remaining sampled term. The mass increment and previous state are linear in \(u\), and the sine test modes diagonalize the discrete Laplacian, so the precomputed matrix \(A\) also makes the diffusion projection exact. The sign-upwind advection \(N(u)\) is nonlinear and remains on the empirical-quadrature nodes.

This is why separability matters. With a decoder that coupled all spatial outputs through \(z\), there would be no fixed spatial matrix to project once and reuse. Here, the full-grid work is moved offline into \(A\). Online, \(Ah(z)\) is only an \(M\times R\) matrix-vector product, with \(M\le256\) and \(R=512\). “Free” does not mean that the matrix-vector product performs no arithmetic; it means that it replaces sampled linear projections without changing the measured online cost. The paired timings in stage 1 confirm that point.

Once the linear terms were removed from quadrature, the same node budget was refit using advection rows only. Every empirical-quadrature degree of freedom was now assigned to the only term that still needed it.

## B. The gate rule had to change with the mathematics

The old gate 0 certified that the complete sampled residual was bit-identical to the incumbent `make_weak_ops` residual. That is no longer the right claim. The new residual deliberately replaces the incumbent's sampled linear projection with an exact full-grid calculation, so equality of the complete old and new residuals would mean the intended change had not happened.

Gate 0 still checks the incumbent-form operators built on the same node set, preserving the code-identity check for that machinery. Two new gates certify the actual boundary introduced by the change:

- **Gate L** compares the exact-linear part against the full-grid linear part. Its acceptance threshold is \(\le1\mathrm{e}{-12}\), and it measured \(\le3.4\mathrm{e}{-15}\) everywhere, including \(N=1024\).

- **Gate A** compares the sampled advection part against the incumbent advection evaluation on the same nodes. Its acceptance threshold is \(\le1\mathrm{e}{-12}\), and it measured \(\le5.5\mathrm{e}{-14}\), mostly exactly zero.

The logic is simple: gate L proves that the newly exact part really is exact, while gate A proves that the nonlinear part retained the established sign-upwind discretization. Together they certify the new residual more meaningfully than asking it to reproduce the old approximation.

## C. Stage 1: a zero-cost accuracy gain, exactly where quadrature had been binding

On the matched-accuracy speed protocol, the exact-linear residual improved the rollout at both measured resolutions. At \(N=1024\), error moved from \(9.69\mathrm{e}{-3}\) to \(7.98\mathrm{e}{-3}\), a \(-18\%\) change. At \(N=256\), it moved from \(8.96\mathrm{e}{-3}\) to \(8.02\mathrm{e}{-3}\), a \(-11\%\) change.

The paired cost did not meaningfully move. The reported speedup changed from \(1.90\times\) to \(1.87\times\) at \(N=1024\), and from \(0.37\times\) to \(0.36\times\) at \(N=256\), within run-to-run noise. These are paired, within-run comparisons; the result is that the accuracy improvement did not require a new online-cost tradeoff.

The fidelity ladder shows why accuracy improved. The linear share of the residual-projection error, \(b_{\mathrm{lin}}\), collapsed to approximately \(1\mathrm{e}{-13}\), which is numerical zero here. On the fine set, total residual error improved by \(1.9\text{--}3.3\times\) along the solver path and by up to \(4.8\times\) at the oracle states. The oracle gradient cosines became \(0.90\text{--}0.99\) on all four checkpoints. This is the predicted effect from the previous fidelity-ladder note: a large source of gradient corruption was removable algebra, not an unavoidable failure of the nonlinear flux quadrature.

The more important result is the coarse/fine asymmetry. At the coarse budget \(m=256\), every checkpoint's rollout error improved by \(7\text{--}20\%\); for example, the `r4a6` rollout moved from \(3.03\mathrm{e}{-2}\) to \(2.47\mathrm{e}{-2}\). At the fine budget \(m=1024\), the rollout was unchanged to three digits on all four checkpoints, even though the ladder metrics improved.

That is not a contradiction. At the coarse budget, quadrature error was large enough to steer the latent solve, so removing its linear component improved the trajectory. At the fine budget, quadrature was already below the error that controlled the rollout. A better integral can make the residual and gradient more faithful without changing the final state when something else has become the floor.

## D. Stage 2: teach the quadrature what the solver consumes

Stage 2 held the node budget fixed and compared four ways to build the nonnegative least-squares row system. Every set used the exact-linear residual online:

- `inc` is the incumbent two-block fit, using state and advection projections at training codes.

- `adv` removes the now-unnecessary state rows and fits advection projections at training codes.

- `path` evaluates the advection rows at off-manifold LM iterates from training-trajectory rollouts, so the fit sees states resembling those visited by the nonlinear solver.

- `grad` adds gradient-teacher rows to the `path` system.

The gradient construction needs one careful piece of algebra. At fixed node locations, the sampled advection projection is

\[
q_{s,p}(w;z)=\sum_j w_j\phi_p(x_j)N(u(z))(x_j).
\]

Thus \(q_s\), the sampled residual \(R_s(w)\), and its Jacobian \(J_s(w)\) are each affine in the quadrature weights \(w\). But the actual least-squares gradient

\[
g_s(w)=J_s(w)^T R_s(w)
\]

is quadratic in \(w\), because both factors depend on the weights. Putting that literal expression into the row system would no longer be linear NNLS.

The convex teacher therefore freezes the full-grid Jacobian \(J_f\) and fits a target such as

\[
J_f^T R_s(w)\approx J_f^T R_f.
\]

Now only \(R_s(w)\) depends on \(w\), so the teacher row remains linear in the weights. The full nonlinear quantity \(J_s^T R_s\) is still checked afterward by the fidelity ladder; freezing \(J_f\) is a construction device, not a relaxation of the certification target.

At \(N=1024\) and \(m=256\), moving from training codes to solver-path iterates was the main fidelity improvement: `path` was worth approximately \(2.5\times\) on held-out residual error, and the held-out gradient cosine moved from approximately \(0.46\) to approximately \(0.90\). The `grad` rows added another approximately \(1.6\times\) improvement in held-out gradient error, reaching a cosine of \(0.95\).

That ordering reached the coarse-budget rollout on the \(N=1024\), \(K=32\) checkpoint:

\[
\texttt{inc}\;2.61\mathrm{e}{-2}
\;\to\;\texttt{adv}\;2.47\mathrm{e}{-2}
\;\to\;\texttt{path}\;2.19\mathrm{e}{-2}
\;\to\;\texttt{grad}\;2.07\mathrm{e}{-2}.
\]

That is a \(-21\%\) change for `grad` relative to `inc`.

The same method produced a clear **negative transfer** on the \(N=256\) `dense_mid` checkpoint. At \(m=256\), the rollout sequence was `inc` \(1.52\mathrm{e}{-2}\), `adv` \(1.51\mathrm{e}{-2}\), `path` \(1.71\mathrm{e}{-2}\), and `grad` \(1.71\mathrm{e}{-2}\). The supposedly more solver-aware systems were \(+13\%\) worse. Their test-path residual error degraded from \(0.21\) to \(0.39\) even while their held-out fit-side metrics improved. This is a negative result, not a tie: on that checkpoint, `path` and `grad` overfit the state distribution used to build the rows.

The broader generalisation gap is visible even in the successful \(N=1024\) coarse arm. Held-out residual error improved from \(0.39\) to \(0.15\), while test-solver-path error improved only from \(0.39\) to \(0.34\). Training-trajectory iterates do not fully cover the states reached on unseen trajectories, so a better fit to the chosen teacher distribution does not guarantee an equally large rollout gain.

At the fine budget, all four row systems tied on rollout at both resolutions: \(1.433\text{--}1.440\mathrm{e}{-2}\) and \(1.102\text{--}1.106\mathrm{e}{-2}\). This is the same asymmetry as stage 1. If quadrature does not bind, changing its targets cannot improve the rollout.

## E. Stage 3: learn continuous node locations, but keep the teacher frozen

Stage 3 asked the advisor's question in its strongest direct form: do not merely refit weights on a fixed grid subset; allow the node positions themselves to move continuously.

The optimization used variable projection. The outer variables were the node positions, represented through a sigmoid so they stayed inside the spatial box and regularized by a minimum-separation penalty. For every proposed node set, the inner problem re-solved the nonnegative weights by NNLS. This removes the weights from the outer search: the node optimizer is always judged using the best inner weights available at its current positions.

The decoder and full-grid teacher remained frozen. The loss targeted the residual rung and the frozen-\(J_f\) gradient rung over the same fit states as stage 2. The sign-upwind `where` switch is only piecewise smooth in node position, so its subgradient was used as-is.

Gate C checked that the continuous-node machinery, when initialized at grid nodes, reproduced the grid operators. Its threshold was \(\le1\mathrm{e}{-12}\), and it held on every run. This isolates any later change to learned node movement rather than a mismatch between the grid and continuous implementations.

At the coarse budget, learned nodes beat the `grad` convex baseline on the required certification metrics and on rollout for both checkpoints. For \(N=1024\), held-out residual error moved from \(0.146\) to \(0.071\), a \(-51\%\) change; the step cosine moved from \(0.906\) to \(0.985\); test-path residual error moved from \(0.338\) to \(0.262\); oracle residual error moved from \(0.77\) to \(0.53\); and rollout error moved from \(2.067\mathrm{e}{-2}\) to \(2.005\mathrm{e}{-2}\). The nodes moved approximately \(7\) grid spacings on average. For \(N=256\), held-out residual error moved from \(0.051\) to \(0.017\), and rollout moved from \(1.706\mathrm{e}{-2}\) to \(1.591\mathrm{e}{-2}\).

There is an important qualification at \(N=256\). The plain `adv` set still had the best rollout, \(1.505\mathrm{e}{-2}\). Learned nodes beat the `grad` baseline and therefore met the stated success bar, but they recovered only part of stage 2's negative transfer. They did not make the same-target family the best choice on that checkpoint.

At the fine budget, node learning was a **clean reportable negative on both checkpoints**. The initial `grad` loss was already three orders of magnitude below the coarse arms' loss. The optimizer wandered or overfit; the learned sets certified worse than both baselines, and the rollouts tied. On the \(N=1024\) fine arm, the final refit loss was worse than its initialization. This is exactly the negative one should expect when the optimized component is no longer the binding error: extra flexibility gives the optimizer room to fit the teacher states without giving the rollout anything useful.

## F. Bottom line: should the model learn the quadrature?

The answer is “sometimes, but it is not the main bottleneck.” Learned continuous nodes can beat every convex weight fit on every fidelity rung when the node budget is coarse. At \(N=1024\), their rollout gain was \(-3\%\) relative to `grad` and \(-19\%\) relative to `adv`. That proves node locations are a real optimization lever, not merely a speculative one.

It also establishes the limit. The gain is small relative to the added machinery, depends on the checkpoint, and is bounded by the mismatch between fit-side iterates and unseen solver paths. At the fine budgets used for the headline results, learned nodes are a negative and all quadrature refits leave the rollout essentially unchanged.

The robust default is therefore stage 1: compute every linear term exactly and fit the remaining node budget to advection. It is algebraically justified, passed the new gates, improved rollout accuracy where quadrature mattered, and did not change paired online cost. Solver-path and gradient-teacher rows are optional checkpoint-specific refinements at coarse budgets. Continuous node learning is a still more situational refinement, not the next universal architecture change.

What remains binding at the fine budget is \(h\)'s generalisation. In

\[
u(z)=G_{\mathrm{int}}h(z),
\]

the quadrature controls how faithfully the PDE objective is integrated, while \(h\) controls which feature coefficients the decoder produces at unseen latent states. Once integration is already accurate enough, improving it cannot repair a coefficient map that does not generalize well enough to the states visited by new trajectories. The next accuracy question is therefore about generalising \(h\), not about giving quadrature still more freedom.

## G. Sources

- [2026-08-25 EQ fidelity ladder explainer](2026-08-25-eq-fidelity-ladder-explained.md) — notation, prior state of knowledge, and the exact-linear and frozen-teacher derivations.

- [2026-08-26 exact-linear terms and gradient-EQ report](../reports/2026-08-26-exact-linear-terms-and-gradient-eq.md) — findings F1–F8 and generated tables T-X1–T-X5; the only source of numerical claims in this note.

- [2026-08-26 overnight notes](2026-08-26-overnight-notes.md) — experiment chronology and implementation context only.
