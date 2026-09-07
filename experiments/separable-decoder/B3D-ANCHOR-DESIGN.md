# Burgers 3D protected-anchor head

This implements the protected-head arm of [the common architecture protocol](B3D-ARCH-DESIGN.md).
Component validation has passed on the local GPU; no architecture training or
validation-cohort cluster result is claimed here. Independent implementation review
and the cluster screen remain pending.

## Construction

The common evaluator factors the existing learned bank as $G=QR$ and supplies
an orthonormal anchor $U$ derived from its trained linear skip. The new head emits
orthonormal-bank coordinates

$$
q(z)=b+Uz+P f_\theta(z),\qquad P=I-UU^\top,
$$

where $b$ is trainable and $U$ is frozen. The residual $f_\theta$ uses the same
two-hidden-layer width-128 SiLU network as the matched control. It scales its
input by the training-code RMS and its output by the centered training-coefficient
RMS, exactly as the control does. The affine path remains in physical QR
coordinates. The output layer starts at zero, so all models begin with the same
linear field predictions and assigned training codes.

The production implementation applies $P$ through two narrow matrix products;
the independent NumPy path forms $P$ densely and evaluates its own network.
Only the bias and residual weights enter the trainable parameter tree. The
initializer rejects invalid shapes, nonfinite values, nonpositive normalization
scales and a nonorthonormal anchor. No PDE-family descriptors enter this head.

## What the constraint guarantees

Because $U^\top U=I$ and $U^\top P=0$,

$$
U^\top(q(z)-b)=z,\qquad
J_q=U+P J_f,\qquad
J_q^\top J_q=I+J_f^\top P J_f\succeq I.
$$

The nonlinear correction cannot cancel a latent direction. Distances also obey

$$
\|q(z_1)-q(z_2)\|^2
=\|z_1-z_2\|^2+\|P(f_\theta(z_1)-f_\theta(z_2))\|^2.
$$

These statements hold at fixed model parameters, up to numerical roundoff.
They do not bound the largest singular value or guarantee good conditioning.
The manifold is a single-valued graph over this particular anchor, so it cannot
fold over those anchor coordinates. A negative result would concern this chosen
anchor and training protocol. A positive result would still need physical
tangent and rollout checks; full latent rank is not a wave-accuracy guarantee.
The common controls already retained full measured latent rank.

## Validation record and next step

`test_b3d_arch_anchor.py` passed its GPU smoke through `jaxrun`, using the project
environment, f64 and highest matmul precision. The component checks cover:

- Exact control initialization at the campaign dimensions and exclusion of the
  anchor from trainable parameters.
- Independent NumPy/vector/batch parity with a nonzero residual; coordinate
  recovery and distance bounds; an omitted-projection mutation that fails recovery.
- Independent NumPy central differences for the full Jacobian and Hessian under
  nonuniform latent scales, plus projected derivative identities and Gram bounds.
- A short common-trainer fit that updates the trainable bias while preserving the
  anchor and recovery identity.
- Rejection of malformed normalization and anchor inputs.

Reproduce the bounded component check from this directory with:

```bash
source /etc/profile.d/jax-mem.sh
JAX_DEFAULT_MATMUL_PRECISION=highest timeout 55s jaxrun /home/tahmid/Dev/.venv/bin/python -u test_b3d_arch_anchor.py
```

The coordinator reviews the implementation before the common validation-only
cluster screen. Frozen-bank dimensions, training membership, optimizer seeds,
training budget, multistart fitting and acceptance gates remain those in the
common protocol. Cluster namespace: `b3d_anchor_20260906`. No cluster job has been
submitted by this implementation session.

## Glossary

- **Bank:** fixed learned spatial features used to reconstruct a field.
- **QR coordinates:** orthonormal coordinates for that same bank, without replacing it.
- **Anchor:** fixed independent directions inherited from the trained linear skip.
- **Head:** the map from latent coordinates to bank coefficients.
- **Residual:** here, the nonlinear correction added to the affine head.
- **Projection:** removal of a vector's components along the anchor.
- **Graph / fold:** a single-valued map over anchor coordinates, versus multiple
  field values sharing those coordinates.
- **Jacobian / Hessian:** first and second derivatives with respect to latent coordinates.
- **Gram matrix:** the Jacobian transpose times the Jacobian.
- **Singular value / rank:** directional stretching and the number of independent directions.
- **RMS:** root mean square, used here for fixed training-derived normalization.
- **Tangent / rollout:** locally available field changes, and online evolution through time.
- **Mutation control:** an intentionally incorrect formula used to verify that a test detects the error.
