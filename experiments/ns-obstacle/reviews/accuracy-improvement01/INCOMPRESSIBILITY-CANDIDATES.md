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

## Evaluation and terminology

All candidates retain actual reduced solves, separate best-fit diagnostics, full-rule force/conservation checks, and closed final validation until the recipe is frozen. A **mode** is one learned spatial velocity pattern; the online solver chooses its coefficient. A **supremizer** is an additional FE velocity pattern used to stabilize pressure coupling. The **lift** imposes the prescribed nonhomogeneous boundary data. A **streamfunction** is a scalar whose rotated gradient gives a two-dimensional divergence-free velocity. **Inf-sup** denotes the pressure–velocity stability check. **Full rule** means the original complete numerical integration rule rather than fitted reduced quadrature. These candidate designs were independently assessed from the local implementation and mathematical identities; no new training or scientific data evaluation was used for this note.
