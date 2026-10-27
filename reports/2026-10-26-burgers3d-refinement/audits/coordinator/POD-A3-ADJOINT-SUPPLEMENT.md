# POD A3: independent saved-step adjoint audit

Independent reviewer freshly reconstructed one saved step with CPU NumPy; no new GPU solve or parameter change. Earlier rounded probe values had an undocumented normalization and are retired in favor of this explicit fresh output.

Validation case 0, state indices [0, 1], step 0.01, spacing 0.015625, viscosity 0.015720525393965363; first 512 columns of the pinned POD basis.

With zero-Dirichlet central differences, write $D_\Sigma=D_x+D_y+D_z$ and
$F(u)=u\odot D_\Sigma u-\nu\Delta_hu$. For $u_j=\Phi c_j$ and $\tau=\Delta t/2$,

$$
r_F=u_1-u_0+\tau[F(u_1)+F(u_0)],
$$

$$
g_F=\Phi^\top\!\left[r_F+\tau\{(D_\Sigma u_1)\odot r_F-D_\Sigma(u_1\odot r_F)-\nu\Delta_hr_F\}\right].
$$

The direct stencil row is

$$
J_i=\Phi_i+\tau\{(D_\Sigma u_1)_i\Phi_i+u_{1,i}(D_\Sigma\Phi)_i-\nu(\Delta_h\Phi)_i\},
\qquad g_E=\sum_{i\in\mathcal I}w_iJ_i^\top r_{F,i}.
$$

The full gradient uses the discrete adjoint above. The sampled gradient uses direct
selected rows. This restricts the state to the trial span while retaining the full
residual; it does not first Galerkin-project that residual.

The Jacobian Frobenius norm was accumulated in 32-column blocks. The common full-metric denominator is $\|J_F\|_F\|r_F\|=6.12292257414319$. Target-norm-relative columns instead divide by the corresponding saved target norm.

| Compared quantity | Absolute difference norm | Full-metric-normalized difference | Saved target norm | Target-norm-relative difference |
| --- | --- | --- | --- | --- |
| Full gradient | 1.1057735220123976e-14 | 1.8059570550214515e-15 | 0.00804449693302751 | 1.3745713762069206e-12 |
| Sampled gradient | 9.73278015642404e-14 | 1.58956446673442e-14 | 2.8706470389396333e-7 | 3.390448224529602e-7 |
| Residual norm | 3.0531133177191805e-16 | not used | 0.2431965661212582 | 1.2554097150355706e-15 |
| Jacobian Frobenius norm | 0.0 | not used | 25.17684633380179 | 0.0 |

Fresh printed output of the independent read-only reconstruction. Last-bit differences from earlier CPU summation are expected. Near-zero sampled-gradient target-relative error must not be described as the full-metric-normalized error.

Result SHA256 `b255bebbed01038d67b7ccc9e97f1de8d13c170510a7a03f9d099ae4852baff6`; array SHA256 `5ddab4c2d7271d210f9e7cf2d4dd603209d2b0ec0ca81cac7ea6c309c4c3fb24`. Basis/support hashes and exact fresh output are in [the immutable review record](../../data/POD-A3-ADJOINT-REVIEW.json).

## Glossary

- Adjoint: transpose of the residual derivative applied to a vector.
- Trial span: fields represented by the frozen basis times coefficients.
- Full metric: the full Jacobian Frobenius norm multiplied by the full residual norm.
- Target-relative: difference divided by the corresponding saved target norm.
