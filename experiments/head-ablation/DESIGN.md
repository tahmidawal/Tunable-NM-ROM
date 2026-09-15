# Head ablation at matched latent dimension — predeclared design

One question: **at one frozen spatial bank, does the nonlinear coefficient map earn
its place?** This answers reviewer 5mgh's demand for a linear POD-Galerkin/DEIM
baseline with the same knobs, and the follow-up objection that the rebuttal's wins
came from a frozen POD basis plus the solver rather than from the manifold.

## The common contract

Every arm writes the reduced state as

$$u(z) = B\,h(z),\qquad B\in\mathbb R^{n\times D},\quad h:\mathbb R^{K}\to\mathbb R^{D},$$

and solves the **same** overdetermined weak residual for $z$. With $\Phi\in\mathbb R^{n\times M}$
the $M$ lowest discrete sine test modes ($\Phi^\top\Phi=I$), $\lambda$ their Laplacian
eigenvalues, $A=\Phi^\top B$, backward Euler at step $\Delta t$ and the FOM's own
sign-upwind advection $\mathcal N$,

$$r_w(z)=\frac{A h(z)-p+\Delta t\,\big(\Phi^\top\mathcal N(Bh(z))+\nu\,\lambda\odot A h(z)\big)}{1+\Delta t\,\nu\lambda},
\qquad p = A h(z^{\rm prev}).$$

The linear terms are exact for every arm. The advection term uses either the retained
nonnegative empirical-quadrature (EQ) rule at $m=4M$ points, refit from **that arm's own
decoder-output advection snapshots**, or the exact dense grid sum.

Shared across arms: test modes $M=4K$, $\Delta t=0.005$, the fixed 48×48 Gauss
state-fitting initializer with nearest-candidate multistart, the stationarity-aware
Levenberg–Marquardt stopping rule (`ic_budget` 400, `step_budget` 180, normalized
gradient tolerance $10^{-6}$), and the output contract: one supplied dense initial
field on the GPU to six dense output fields at $t=0,0.05,\dots,0.25$.

## The arms

| arm | $B$ | $h$ | $K$ |
| --- | --- | --- | --- |
| `a_neural` | frozen separable bank $G$ | retained MLP head $h_\theta$ | 16 |
| `b_linear_dec` | $G$ | $c + Wz$ fitted to $h_\theta(Z_{\rm train})$ | 16 |
| `b_linear_truth` | $G$ | $c + Wz$ fitted to $G^{\dagger}U_{\rm train}$ | 16 |
| `c_quad_dec` | $G$ | $c + Wz + Q\,\mathrm{vech}(zz^\top)$, same $W$ as `b_linear_dec` | 16 |
| `d_freebank` | $G$ | identity on $\mathbb R^{R}$ | 512 |
| `e_pod{k}` | classical POD $V_k$ of $U_{\rm train}$ | identity on $\mathbb R^{k}$ | 8…128 |

$Z_{\rm train}$ are the checkpoint's own recorded training codes; $U_{\rm train}$ are
FOM trajectories regenerated on the cluster at the working mesh from the incumbent
recorded training draw (`params_draw(seed=0)`), disjoint from every evaluation case.

**Why decoder-output snapshots for `b_linear_dec` / `c_quad_dec`.** They are exactly
the set arm (a) can represent, so the linear and quadratic maps are asked to cover the
neural head's *own image*. If a linear map covers it, the nonlinearity buys nothing.
`b_linear_truth` is the complementary practitioner's baseline fitted to truth.

**Why the coefficient metric is the field metric.** With $G=Q_GR_G$ thin-QR,
$\|G\delta\|_2=\|R_G\delta\|_2$, so POD of $R_G$-whitened coefficients is exactly
field-metric POD inside the bank, and a plain least-squares regression of coefficients
on $z$ is already field-metric optimal.

## Necessary, recorded deviations

1. `d_freebank` solves $R=512$ coefficients, so the weak system needs $M>R$; it uses
   $M=1024=2R$ and the exact dense advection, because an NNLS rule with $m=4M$ points
   is not constructible inside the job budget. Its cost is therefore grid-bound — which
   is itself the "how much solver work does compression save" reading.
2. EQ is fitted at the matched rank ($K=16$, $M=64$, $m=256$) for every $K=16$ arm and
   for `e_pod16`. The higher POD rungs run with exact dense advection, which *favours*
   the POD baseline, so the reported "POD rank that matches the neural head" is a
   conservative lower bound on the rank a hyper-reduced POD ROM would need.
3. The incumbent unpivoted Gauss–Jordan step solve unrolls one graph level per unknown;
   arms above 64 unknowns use a pivoted dense solve instead. That is a more accurate
   step, not a weaker one.
4. POD modes have no continuum representation, so they are evaluated at the Gauss
   initializer points by the same aligned bilinear interpolation that the supplied input
   field already receives in every arm.

## Reported per arm

Bank projection error, best-found reconstruction error, online rollout error against the
campaign's own refined reference ($L=4096$, $\Delta t=3.125\times10^{-4}$), LM iteration
counts, stopping status, and same-job complete-query GPU/host cost with burn-in and all
repetitions retained. Development cases only; the final cohort stays sealed.

## Gates before any verdict

- `smoke_arms.py` part 1 must reproduce the consolidated saved Burgers case through the
  *generic* arm machinery to $10^{-8}$ relative, and agree with the incumbent
  `accuracy_paths.make_rom` to $10^{-12}$.
- The job's `a_neural_eq` errors must reproduce the campaign's recorded
  `frozen_stationary` errors on the same cases and meshes.
