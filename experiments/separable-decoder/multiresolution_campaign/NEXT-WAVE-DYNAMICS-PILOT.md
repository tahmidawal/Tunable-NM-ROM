# Fresh-wave compression and dynamics controls

This is the next bounded development experiment after the frozen-network wave
pilot. It specifies work to run in the already approved wave worktree; it contains
no new numerical findings or final-cohort result.

The user chose continued development with the existing four separate worktrees.
Use `worktrees/2026-09-07-mr-wave2d`, its current branch and
`/cluster/tufts/paralab/tawal01/mr_wave2d_20260907/`, with a new attempt directory.
The frozen learned spatial banks and MLP checkpoint remain those of the fresh
verified lineage. Do not import a discarded wave implementation or checkpoint.

## Question and controls

The previous comparison used a nonlinear head with sixteen displacement
coordinates and a linear control with sixty-four. That comparison cannot isolate
the effect of nonlinear evolution from different compression levels. The next
pilot needs a linear and nonlinear model with equal state dimension, plus a
linear dimension ladder to expose the cost of compression.

Use the same two predetermined validation cases per boundary, unchanged physical
family, horizon, observation times, meshes and physical references as the first
multiresolution pilot. The final cohort stays closed. The prospective configuration
is recorded in JSON before submission, including all arm names and search choices.

| Arm | Configuration coordinates | Phase-state coordinates | Purpose |
|---|---:|---:|---|
| Frozen MLP | 16 | 32 | Original nonlinear control, unchanged weights |
| Common affine | 16 | 32 | Exact original training-coefficient PCA initialization, including its offset |
| Training linear ladder | 32 | 64 | Additional compression level within the same learned bank |
| Full learned bank | 64 | 128 | Uncompressed linear dynamics in the learned spatial span |

All linear features must remain combinations of the frozen coordinate network's
outputs. The saved common initialization supplies the affine control without
fitting validation data. For additional dimensions, regenerate the original
training coefficient snapshots from their recorded seed, using only training
displacements and the original centering/scaling convention. Record covariance,
singular values, basis matrices, data hashes and all normalization choices.
Verify that the regenerated sixteen-dimensional projector agrees with the saved
training projector, allowing orthogonal changes of coordinates. If it does not,
retain the saved common affine arm and identify the new training ladder as a
separate construction; do not relabel mismatched data as identical.

The full-bank arm is a classical linear Galerkin diagnostic. It is not an
equal-dimensional nonlinear comparator. The nonlinear weak system retains many
more test functions than latent coordinates. No strong-form collocation arm is
introduced.

## Linear evolution and independent controls

In the mass-orthonormal bank, let $a$ and $b$ be displacement and velocity
coefficients. Let $K$ and $D$ be the positive stiffness and boundary-damping
matrices, including physical speed scaling. For an affine decoder
$a=a_c+Vq$, use the actual reduced mass matrix $M_V=V^TV$ unless $V$ has been
orthonormalized by an exactly checked change of coordinates. The equation is

$$
M_V\ddot q + V^TDV\dot q + V^TKVq = -V^TKa_c.
$$

The constant force from the affine offset is essential. Preserve it through
homogeneous augmentation when using a matrix exponential. Initial displacement
and velocity are fitted from the supplied physical fields, never from reference
trajectories or Gaussian descriptors. Compute the propagation for the supplied
wave speed inside the measured query, or charge every speed-dependent setup
operation explicitly. Cached mesh-only operators are separate offline setup.

Check the affine propagation against an independent small-system exponential or
high-accuracy solve, including a nonzero offset and absorbing damping. Verify the
zero-nonlinearity version of the current manifold equation agrees with this
affine equation. Keep the existing nonlinear time-step pair and rank checks.
This comparison must not hide an unresolved time integration error.

## Representation and force compatibility

Separate initial-field fitting, best-recorded snapshot fitting, velocity tangent
fitting, and autonomous trajectory errors. Use the original multiple starts with
paired fitting budgets, record gradients and rank, and report every nonstationary
fit. A local best-recorded fit is not a global nonlinear optimum.

At a fitted latent state $z$, let $J$ be the head Jacobian and $w$ the latent
velocity fitted to the reference physical velocity. The manifold acceleration
obeys

$$
J\ddot z = -K h(z)-D Jw-h''(z)[w,w].
$$

With $Q$ an orthonormal basis for the range of $J$, save the normal component

$$
r_\perp=(I-QQ^T)\bigl(-K h(z)-D Jw-h''(z)[w,w]\bigr).
$$

This is a force-compatibility diagnostic in the weak bank coordinates. It is not
a physical trajectory error or proof of the cause of accumulated phase error.
Report its absolute norm and an explicitly recorded fixed physical scale; avoid
dividing silently by a vanishing instantaneous force. Use the corresponding
zero-curvature expression for the affine controls. Keep bank projection error
separate from the additional head and tangent errors.

If fitting every observation is too expensive, predeclare a subset of times for
these diagnostics and label every summary as restricted to those times. Complete
rollout physical metrics still use every requested observation.

## Timing, outputs and decision

Measure all complete queries against the same efficient reflective DST and
absorbing FOM controls within the same GPU job. Include full host initial fields,
projection/fitting, speed-dependent preparation, evolution, dense output and
host transfers. Preserve every repetition, warm and burn the GPU before each
timed block, and score the output of that exact invocation. Diagnostics that see
truth never provide an online initializer.

Save sufficient bank tables, coefficient states, physical reference fields and
representative full outputs to reconstruct common-grid and finer-grid metrics
independently. Record hashes for repeated outputs. A display grid is not an
accuracy observation grid. Displacement uses initial displacement L2 scaling;
velocity and energy-state error use initial phase energy. Preserve absorber
current-relative errors, absolute errors and vanishing flags.

Initially run one bounded GPU pilot with a two-hour ceiling after a local smoke
under a minute. Review accuracy, cost, outlier counts, force diagnostics and
reference uncertainty together. A linear equal-dimensional failure points to a
compression limitation in that linear family; it does not establish that every
nonlinear manifold of that dimension must fail. A nonlinear trajectory failure
despite good snapshot and tangent fits motivates dynamics-aware training or a
different phase-state decoder. Neither conclusion should be selected in advance.

## Plain-language glossary

- **Bank / head / frozen:** learned spatial features / coefficient map from a
  reduced state / unchanged network weights.
- **Configuration / phase state:** displacement coordinates / displacement and
  velocity coordinates together.
- **PCA / projector / affine:** principal directions of training coefficient
  variation / map into a linear span / linear map plus a fixed offset.
- **Galerkin / weak coordinates:** equation projected onto spatial test functions /
  the resulting finite set of averaged equations.
- **Mass / stiffness / damping:** matrices defining physical kinetic energy /
  spatial restoring force / energy loss through the boundary.
- **Tangent / curvature / normal force:** locally allowed displacement directions /
  acceleration due to bending of the decoder / force outside the tangent span.
- **Stationary / rank / snapshot:** locally small fitting gradient / number of
  independent decoder directions / a field at one observation time.
- **Matrix exponential / augmentation:** exact propagation of a finite linear
  system up to numerical arithmetic / adding a constant coordinate to represent
  an affine forcing term.
- **Complete query / outlier / final cohort:** all supplied-input to requested-output
  work / case missing the stated requirement / unopened independent confirmation
  data.
