# Smooth mixture head for the Burgers 3D architecture screen

This arm implements the user-approved smooth mixture within the shared
`B3D-ARCH-DESIGN.md` protocol. Scientific performance is unmeasured; component
verification is recorded separately and does not establish improved PDE accuracy.

## Architecture and initialization

The learned spatial bank remains fixed. The head emits its exact orthonormal
QR coefficients, using the same latent dimension and common training-only
normalization as the MLP control. With $x=z/s_z$, the head is

$$
q(z)=b+Az+s_q\sum_{e=1}^{2}w_e(x)f_e(x),\qquad
w(x)=\operatorname{softmax}(xW_g+b_g).
$$

Here each $f_e$ is an independently initialized SiLU network with two hidden
layers of width $128$. Both last layers start at zero. The affine parameters
start at the shared center and anchor and remain trainable, so every initial
output is exactly the common affine map. The small nonzero gate weights start
with standard deviation $0.1/\sqrt{K}$ and zero biases; gating remains smooth
and fully differentiated. Distinct hidden features and initial routing break
the expert symmetry once their output layers update. No balancing loss, labels,
hard routing, temperature schedule, or physical-parameter inputs are added.

The evaluator controls data membership, training objective, initialization,
optimizer schedule, repeated seeds, and validation fitting. Parameter counts
come from the actual trainable tree. The wider MLP is a capacity comparison,
not an exactly parameter-matched model or a matched-compute control.

## Derivative checks

The mixture derivative includes changes in the routing weights:

$$
Dq=A+s_q\sum_e\left(w_e Df_e+f_e Dw_e\right).
$$

The functions above include the latent normalization, so the chain rule applies
to both terms. The tests use nonzero expert outputs and nonconstant routing;
an initially zero nonlinear output would conceal omitted routing derivatives.
Independent NumPy values supply central finite-difference Jacobians, Jacobian
vector products, and Hessians. A deliberate mutation stops differentiation of
the routing weights while preserving function values; both first and second
derivative comparisons must detect it. Additional checks cover single/batch
agreement, saturated but numerically stable softmax, common affine
initialization, and output-symmetry breaking under the shared training routine.

## Descriptive routing checks

The validation artifact retains every fitted state's routing weights, entropy,
maximum weight, and expert-output disagreement, plus mean soft usage, hard
assignment counts, maximum-weight quantiles, and saturation counts above
$0.95$ and $0.99$. Entropy uses natural logarithms. Hard assignments are
descriptive; the model itself always uses the smooth weights.

Two independent flags describe the fitted validation cohort. Routing collapse
means some expert receives mean soft usage below $0.05$. Identical-expert
collapse means the RMS disagreement, divided by the fixed training output
scale $s_q\sqrt{R}$, is at most $10^{-6}$. These declared thresholds summarize
behavior and are not acceptance gates or claims about every possible latent
state. Both flags are possible independently; initially zero experts have
identical outputs even with healthy routing. No auxiliary objective changes
are made based on these diagnostics in this bounded comparison.

## Validation record and remaining work

The component run passed through the repository's local GPU guard, in double
precision with highest matrix-multiplication precision. The retained command
output is `runs/b3d_arch/mixture/component.log`. Root review is required before
the two-seed cluster screen. No cluster training, held-out scientific result,
pilot promotion, wave test, or rollout is claimed by this implementation record.

## Glossary

- **Head:** the function converting latent variables to spatial coefficients.
- **Latent variables / codes:** the adjustable coordinates describing a state.
- **Spatial bank:** the fixed learned shapes used to reconstruct the field.
- **QR coefficients:** coordinates in an orthonormal basis for that same bank.
- **Anchor / affine map:** common independent initial directions and their
  linear map plus a constant offset.
- **Expert:** one nonlinear residual network contributing to the head.
- **SiLU:** the smooth activation multiplying its input by its logistic sigmoid.
- **Routing / softmax:** smoothly varying positive expert weights that sum to one.
- **Gate:** the latent-dependent function producing those weights.
- **Jacobian / Hessian / JVP:** first derivatives, second derivatives, and the
  first derivative applied to a chosen direction.
- **Saturation:** weights close to exclusive selection of one expert.
- **Entropy / nats:** uncertainty of the routing weights, measured with natural
  logarithms; equal weights maximize it and exclusive weights minimize it.
- **Soft usage / hard assignment:** average probability mass for an expert,
  and a descriptive count of where it has the largest weight.
- **RMS / quantile:** root mean square and a specified percentile of values.
- **Collapse:** the explicitly thresholded routing or expert-output behavior
  defined above, restricted to the evaluated codes.
- **Capacity / matched compute:** model parameter count, and equal computational
  effort; equal update counts alone do not establish the latter.
- **Component test / mutation:** a focused implementation check, and a deliberate
  code defect used to confirm that a check can detect the intended failure.
- **Held-out / pilot:** data excluded from training, and the existing acceptance
  checks required before any online rollout stage.
