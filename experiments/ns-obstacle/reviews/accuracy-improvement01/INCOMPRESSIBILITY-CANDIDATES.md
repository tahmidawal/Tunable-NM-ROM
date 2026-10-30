# Possible incompressibility improvements

This note distinguishes an active supervised trace-error experiment from prospective structural designs. The preceding warm refinement improved actual PDE accuracy but still failed the divergence target. Trace-error training is now running as job6651600; its actual PDE accuracy has not yet been evaluated.

## First controlled candidate: divergence regularization

Keep the existing vector-valued neural bank, rank, features, teacher, mode weights, and velocity/gradient fitting terms. Add one prospectively fixed term

$$
\lambda\sum_r \omega_r
\frac{\|\operatorname{tr}(G_r-G_{\mathrm{teacher},r})\|_w^2}
{\|G_{\mathrm{teacher},r}\|_w^2},
$$

where $G_r$ is the predicted mode gradient with respect to dimensionless coordinates. This version weights one component of the existing supervised gradient error and preserves the FE teacher target. Record predicted absolute divergence and teacher divergence separately. An absolute-divergence penalty was initially considered, but would impose a different target because the teacher modes need not be pointwise solenoidal. The released candidate fixes $\lambda=1$; this weight is not established as optimal. Select checkpoints only by the frozen training objective, preserve the previous objective separately, and evaluate actual flow/force/conservation errors afterward. This derivative-fitting term is not a pointwise PDE-residual solver or a new collocation rule. The online weak least-squares formulation remains unchanged.

## Structural candidate: streamfunction bank

Let $\xi=x/D$ and $\psi=\Psi/(UD)$. Define a neural velocity mode as

$$
\widehat u_r=(\partial_{\xi_y}\psi_r,-\partial_{\xi_x}\psi_r).
$$

For a network evaluated at physical coordinates, the returned dimensionless velocity is $D(\psi_y,-\psi_x)$ and its physical-coordinate gradient is

$$
D\begin{pmatrix}\psi_{xy}&\psi_{yy}\\-\psi_{xx}&-\psi_{xy}\end{pmatrix}.
$$

The existing ROM builder multiplies physical gradients by $D$, producing a dimensionless gradient with the required $D^2$ Hessian factor. Training this gradient requires spatial Hessians and their parameter derivatives; cost and memory have not been measured.

Let $A$ be the product of bounded distance factors for the inlet and upper/lower walls; let $C$ be the bounded obstacle-distance factor. Use

$$
B=AC,\qquad h=\frac{A^2}{A^2+C^2},\qquad
\psi_r=B^2f_r+c_rh.
$$

The cylinder is separated from the outer prescribed boundaries, so $A$ and $C$ do not vanish simultaneously on the closed fluid domain. On the outer prescribed boundary, $\psi_r=0$ and its gradient vanishes; on the cylinder, $\psi_r=c_r$ and its gradient vanishes. Thus the velocity satisfies homogeneous no-slip while retaining the independent obstacle streamfunction constant. Omitting $c_rh$ would restrict this topological freedom. The outlet remains unconstrained. Actual retained curved-FE boundary points still require trace verification.

## Capacity candidate: additive spatial correction

An alternative that preserves the current four online velocity coefficients is

$$
\widehat\phi_r^{\mathrm{new}}(x)
=\widehat\phi_r^{\mathrm{parent}}(x)+B(x)f_{\theta,r}(x).
$$

Freeze the parent and initialize the new network's final weights and biases to zero, with nonzero random hidden weights. This preserves the parent values and gradients initially. Hidden-layer gradients are zero on the first update, but output-layer updates can unlock subsequent hidden learning. Zeroing every layer would obstruct this mechanism.

The same boundary envelope preserves homogeneous velocity traces. Product-rule spatial derivatives are required. Higher Fourier frequencies or cylinder-local features could help unresolved spatial variation, but scalar loss histories do not establish that they are necessary. Frequencies and feature widths need explicit bounded choices: gradient fitting amplifies high frequencies and narrow features, and the existing integration rule must still resolve them. This is a separate capacity hypothesis, not yet implemented or released.

The sum requires a new explicit serialized evaluator, with independently hashed frozen-parent and correction parameters; it must not be loaded silently as an ordinary single `SpatialBank`. Check initial parent equality and value/physical-gradient consistency, including boundary points. Assemble the same four combined columns in the actual weak ROM and retain the same force/conservation diagnostics. Extra network capacity does not remove teacher truncation, lift or supremizer limitations, and it increases evaluation cost. Source, memory and archive bounds must include both networks before an execution decision.

## Important full-ROM limitation

The deployed velocity also includes the FE lift and FE supremizers:

$$
\widehat u=\widehat u_{\mathrm{lift}}+
\sum_r a_r\widehat\phi_r+\sum_j b_j\widehat s_j.
$$

An analytic curl makes the neural modes divergence-free, but leaves

$$
\nabla_\xi\cdot\widehat u=
\nabla_\xi\cdot\widehat u_{\mathrm{lift}}+
\sum_jb_j\nabla_\xi\cdot\widehat s_j.
$$

The current inf-sup gate uses pressure coupling to trial divergence. Removing supremizers while retaining the present mixed formulation would remove that coupling for an exactly solenoidal neural bank. Do not silently remove them or claim the complete ROM velocity is exactly divergence-free. First quantify lift/neural/supremizer contributions, including cross terms because squared norms are not additive.

The FE POD teachers need not be pointwise divergence-free. For any exactly solenoidal prediction in two dimensions,

$$
\|G_{\mathrm{pred}}-G_{\mathrm{teacher}}\|_w^2
\geq \tfrac12\|\operatorname{tr}(G_{\mathrm{teacher}})\|_w^2.
$$

This is a limitation of matching those teacher gradients, not evidence of an implementation bug. An exactly divergence-free full velocity would additionally require a compatible lift and a separately reviewed pressure/stability formulation.

## Next candidate: local cylinder traction supervision

The accepted saved-force decomposition locates the trace-trained bank's lift discrepancy primarily in viscous traction. This motivates an additional supervised derivative target from the existing TRAIN FE-mode boundary arrays, without modifying the weak online PDE objective or using development truth in training. The trace-trained parent retains the best measured velocity accuracy; energy reweighting improved force and divergence but worsened velocity and pressure. Neither passes all physical checks.

For dimensionless mode gradient $G_r$, define $\tau_r=(G_r+G_r^T)n_{\mathrm{body}}$, with body normal opposite the stored fluid normal. A candidate normalized loss is

$$
L_\tau=\frac{\sum_r\sigma_r^2\int_{\Gamma_c}\|\tau_r-\tau_{T,r}\|^2\,d\hat s}{\sum_r\sigma_r^2\int_{\Gamma_c}\|\tau_{T,r}\|^2\,d\hat s}.
$$

Require a positive total denominator; individual zero-energy modes need no artificial floor. The boundary measure is scaled by cylinder diameter. The coefficient for physical viscous force is applied later by the existing force evaluator. Local traction supervision controls a stronger quantity than two integrated force moments, which can hide cancellation; it does not guarantee full pressure-plus-viscous force accuracy. Parent, weight, replay and resource settings require an exact reviewed execution packet. No fit is released by this design note.

## Separate capacity option: independent residual heads

Independent correction heads can remove a shared trainable hidden representation, but shared optimizer steps and tiny loss weights still create competition. Energy-model tail-mode regression alone does not establish a capacity bottleneck. Preserve the same online coefficients and boundary envelope; retain original teacher normalization. If frozen-parent values and gradients are cached at TRAIN volume and cylinder points, record exact order, units and hashes, then require a final uncached serialized full-rule objective and gradient replay. Parameter counts alone do not establish runtime or memory cost. This is a separate change from traction supervision.

## Evaluation and terminology

All candidates retain actual reduced solves, separate best-fit diagnostics, full-rule force/conservation checks, and closed final validation until the recipe is frozen. A **mode** is one learned spatial velocity pattern; the online solver chooses its coefficient. A **supremizer** is an additional FE velocity pattern used to stabilize pressure coupling. The **lift** imposes the prescribed nonhomogeneous boundary data. A **streamfunction** is a scalar whose rotated gradient gives a two-dimensional divergence-free velocity. **Inf-sup** denotes the pressure–velocity stability check. **Full rule** means the original complete numerical integration rule rather than fitted reduced quadrature. These candidate designs were independently assessed from the local implementation and mathematical identities; no new training or scientific data evaluation was used for this note.
