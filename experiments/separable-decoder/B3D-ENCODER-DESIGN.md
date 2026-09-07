# Shared offline encoder for the Burgers architecture comparison

This implements the encoder arm of `B3D-ARCH-DESIGN.md` in the approved
`exp/2026-09-06-b3d-encoder` worktree, based on `dd77383`. Scientific accuracy
results are recorded separately in `B3D-ENCODER-NOTES.md`; component checks
validate implementation only.

## Architecture and normalization

Let $a=Q^\top u$ be the exact orthonormal-bank coefficients of a solution
snapshot, $b$ the mean training coefficients, and $U$ the common neural-skip
anchor. Define the shared encoder by

$$
e_\theta(a)=U^\top(a-b)+
F_\theta\!\left(\frac{a-b}{s_a}\right)\odot s_z.
$$

This equation uses column vectors; the implementation stores batches in rows. The scalar
$s_a$ is the centered training-coefficient RMS; the vector $s_z$ contains the
RMS of each initial latent coordinate. Both are frozen training-only statistics.
The encoder correction is a two-hidden-layer width-128 SiLU MLP. Its final
layer is initially zero, so initial codes exactly equal the common anchor
projection. A folded random key initializes the encoder; the decoder receives
the original unchanged key.

The decoder is exactly the width-128 control, including its trainable affine
map, two-hidden-layer residual, output normalization, initial parameters and
independent NumPy evaluator. The environment setting `HEAD_WIDTH` must be 128;
the encoder module rejects an incompatible control width. The encoder changes
how training codes are assigned, without enlarging the decoder function class.

Training uses the common global relative field-MSE objective and optimizer.
Only training solution coefficients enter the encoder. There are no physical
family descriptors, additional losses or individually optimized snapshot codes.
The full projection and encoder are offline: primary validation fitting retains
zero plus the codes of the common training-state IDs. It receives no
solution-dependent encoder initialization. Direct encoder reconstruction is a
separate reported diagnostic.

## Component validation and remaining gates

`test_b3d_arch_encoder.py` checks exact decoder parameter equality with the
control and exact initial code projection; nonlinear vector/batch and NumPy
parity; finite-difference derivatives of both networks; reconstruction gradients
through every layer; a brief joint-training decrease with frozen-statistic
integrity; and rejection of an intentionally incorrect encoder initialization.
The local component suite passed in a sub-minute GPU smoke with f64 and highest
matrix precision. Its raw log is retained at
`runs/b3d_arch_encoder/component-test.log`. These synthetic checks do not measure
the Burgers validation error or establish an architectural improvement.

The coordinator reviewed the implementation before cluster submission.
Scientific comparisons retain the common two optimizer repeats, cohort,
multistart budgets, reconstruction and tangent diagnostics, and all inherited
pilot gates. No wave accuracy or rollout claim follows from these component
checks. The full predeclared cluster screen is complete; both repeats fail
the preliminary representation gates. Raw results and generated notes are
retained, and coordinator independent result review remains pending. No pilot,
rollout or wave experiment was promoted.

## Glossary

- **Bank / QR coefficients:** fixed learned spatial patterns and exact orthogonal
  coordinates for reconstructing fields from their span.
- **Anchor:** the common independent directions derived from the trained linear
  skip, used to initialize latent coordinates.
- **Encoder / decoder / latent code:** the snapshot-to-code network, the
  code-to-coefficient network, and its small vector of adjustable inputs.
- **RMS:** root mean square, used here only as a fixed training normalization.
- **SiLU / MLP:** a smooth activation and a stack of dense neural-network layers.
- **Global relative field-MSE:** total squared field reconstruction error divided
  by total squared field magnitude.
- **Multistart fit:** latent optimization from several predeclared guesses.
- **Tangent diagnostic:** error in representing physical field changes using
  decoder derivatives.
- **Pilot gates:** existing numerical and accuracy requirements before promotion.
