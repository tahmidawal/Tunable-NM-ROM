# What the matched mechanism experiment measures

This is a method note for the registered experiment. Numerical claims come only from the retained run JSONs and their generated summaries.

At a reached head state, the full-span and head models use the same coefficient vector $a=h(z)$, previous state, weak rows, row weights and frame increment. Their physical tangents include the three free translation variables:

$$
B_s=\left[G,-\partial_1(Ga),-\partial_2(Ga),-\partial_3(Ga)\right],
\qquad
C_h=\begin{bmatrix}J_h(z)&0\\0&I_3\end{bmatrix},
\qquad B_h=B_sC_h.
$$

Translation of all tangent columns by the same current frame is an isometry on the periodic grid, so their Gram matrix can be evaluated in the centered frame. The discrete physical inner product uses the uniform cell-volume weight. Let the retained eigenpairs of the full-span physical Gram matrix be

$$
B_s^T W B_s=V\Lambda V^T,
\qquad W_s=V_+\Lambda_+^{-1/2}.
$$

The implementation retains eigenvalues above the declared relative and absolute cutoffs, saves the entire spectrum and reports excluded directions. For the tested full-rank states, $B_sW_s$ is an orthonormal physical tangent basis. The weak-residual Jacobian in that basis is $A_s=J_sW_s$.

The head tangent, represented in those orthonormal physical coordinates, is

$$
E_h=\Lambda_+^{1/2}V_+^TC_h,\qquad Q_h=\operatorname{orth}(E_h),
\qquad A_h=A_sQ_h.
$$

A separate rank check is applied to $E_h$, and any head mass lost through the span cutoff is saved. The inequality $\sigma_{\min}(A_h)\geq\sigma_{\min}(A_s)$ follows algebraically from restricting an operator to a subspace. Observing that inequality is therefore not a new empirical mechanism result. The magnitudes, the actual weakly observed directions and the trajectories are the diagnostics of interest.

For the least-visible unit span direction $v_{\min}$, the reported outside-head physical-norm fraction is

$$
\left\|(I-Q_hQ_h^T)v_{\min}\right\|_2.
$$

For an actual coefficient/frame change $\Delta\theta_s$ from the reached head solution to a stationary full-span solution sharing the same previous state, its tangent-linearized physical coordinates are $\xi=\Lambda_+^{1/2}V_+^T\Delta\theta_s$. The corresponding outside-head norm fraction is

$$
\frac{\left\|(I-Q_hQ_h^T)\xi\right\|_2}{\|\xi\|_2}.
$$

These quantities are fractions of a physical norm. They are not percentages of directions, energy, or variance. The finite departure projection is tangent-linearized; it is not an exact distance to a curved manifold.

The common-previous-state test supplies the direct physical evidence: initialize the full-span solve from the reached head coefficients and frame increment, retain its local convergence record, and compare the returned weak objective and physical error against the same reference field. A lower weak residual paired with a larger physical error demonstrates a limitation of that particular discretized objective. It does not establish a uniform error bound or a certified global minimum for either model.

The separate multistart distance fits are best-found upper bounds on distance to the head image in a fixed reached frame. Even a stationary local fit does not certify a global minimum. All fit statuses and initial candidates remain available so that unsuccessful fits cannot silently become separation claims.

## Glossary

A physical tangent is the first-order change in the velocity field caused by changing solved coordinates. A frame is the translation of the periodic field. A Gram matrix records physical inner products between tangent directions. Whitening rescales independent tangent directions to unit physical norm. A weak row is one scalar test of the PDE residual. A singular value measures how strongly the linearized residual detects a normalized physical direction. A cutoff excludes numerically unresolved directions instead of dividing by their nearly zero norms. Stationarity is a small local least-squares gradient. A best-found upper bound is the smallest returned fit distance from the declared starts, with rounding ties resolved by the registered convergence rule; it is not a global certificate.
