# Explicit quadratic head for the Burgers architecture comparison

This document specifies the quadratic implementation and its component checks.
It makes no PDE accuracy, rollout, or speed claim; the shared campaign protocol
is in `B3D-ARCH-DESIGN.md`.

## Head and derivatives

The model consumes latent coordinates only. The learned spatial bank stays fixed,
and the head emits its orthonormal coefficient coordinates. Define the frozen
normalizations $s$ and $o$, and normalized code $x_i=z_i/s_i$. The trainable
parameters are an affine matrix $A$, bias $b$, and one coefficient vector $B_{ij}$
for each lexicographically ordered pair $i\le j$:

$$
q(z)=b+Az+o\sum_{i\le j}B_{ij}x_i x_j.
$$

Mixed products occur once without a factor of two. The affine term uses the
unscaled latent code. Its initialization is the common anchor and center, with
all quadratic weights zero. Initialization therefore has no random dependence;
the two campaign seeds vary the optimizer's minibatch stream, not initial weights
or data. Input/output scales remain outside the optimizer.

The Jacobian and constant Hessian provide independent derivative references:

$$
J_{c\ell}=A_{c\ell}+\frac{o}{s_\ell}
\sum_{i\le j}B_{ij,c}(\delta_{\ell i}x_j+\delta_{\ell j}x_i),
$$

$$
H_{c\ell\ell}=\frac{2o B_{\ell\ell,c}}{s_\ell^2},
\qquad
H_{c\ell m}=\frac{o B_{\min(\ell,m),\max(\ell,m),c}}
{s_\ell s_m}\quad(\ell\ne m).
$$

The parameter count is $R[1+K+K(K+1)/2]$. The shared campaign fixes $K=32$
and $R=128$; the actual count is recorded by the component test and common
training evaluator. The width-192 MLP is a close capacity control, so comparing
only against width 128 cannot isolate a structural advantage. Equal update
counts are not equal computational cost.

## Component verification and limits

`test_b3d_arch_quadratic.py` checks nonzero random polynomial weights using a
separately assembled symmetric coefficient tensor. It compares JAX and an
independent NumPy term-by-term evaluator for vectors and batches, checks a
serialized checkpoint, verifies the analytic Jacobian and constant Hessian,
and checks directional and mixed finite differences with nonuniform scales.
Isolated square and cross terms expose diagonal-versus-mixed factor errors.
Further controls verify the exact common initialization, deterministic initial
parameters across optimizer seeds, valid scales, and a short synthetic
common-trainer loss reduction. Machine-readable component results live in
`runs/b3d_arch_quadratic/component_tests.json`; they are not a Burgers result.

The quadratic head does not constrain the trained affine map or protect latent
rank after training. Polynomial values and derivatives can grow outside the
training-code range. A constant Hessian does not guarantee useful tangent
directions or accurate dynamics. All scientific screening, local-fit convergence
checks, negative controls and promotion gates remain those of the common
protocol. No wave conclusion follows from a Burgers screen.

## Glossary

- **K / R:** number of latent coordinates / fixed spatial features.
- **Head:** map from a latent code to coefficients of the fixed spatial bank.
- **Orthonormal coordinates:** coordinates preserving the bank's field norm.
- **Affine map:** a linear map plus a bias.
- **Anchor:** common initial independent coefficient directions from the neural skip.
- **Quadratic / mixed term:** a square or product of two latent variables.
- **Jacobian / Hessian:** first / second derivatives with respect to the latent code.
- **Normalization:** fixed input and output scales derived from training data.
- **Optimizer repeat:** a run changing minibatch randomness, not the PDE data seed.
- **Capacity control:** comparison model with a similar count of trainable parameters.
- **Tangent / rank:** available infinitesimal field motions / independent directions.
- **Component check:** a numerical implementation test, not scientific evidence of accuracy.
- **Promotion gate:** required check before proceeding to a later experiment stage.
