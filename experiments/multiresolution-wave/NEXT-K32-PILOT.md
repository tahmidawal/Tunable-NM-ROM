# Larger nonlinear wave head with matched linear and time-step controls

This is the approved bounded follow-up to the audited `dynamics02` result. It contains
no new training endpoint or numerical finding. `k32-proposal.json` records the
concrete configuration, job limits and accounting choices before implementation and submission.

Use the existing approved wave worktree/branch and a fresh `k32heads03` directory
in its existing cluster namespace. Keep all four experiment worktrees separate.
No neural spatial bank is replaced, broadened or retrained, and no final cohort
is opened.

## Accuracy comparison

Train the original reconstruction-only MLP head with configuration dimension
$k=32$ and phase dimension $2k=64$ on each existing fresh-wave learned bank of rank
$R=64$. Use the original training seed, count, mesh, time horizon, observation
interval and FOM CFL; regenerate those trajectories from seed on the cluster.
The retained original coefficient arrays are lineage checks only. Compare
regenerated original field hashes, scales, PCA center and principal projectors
against the audited artifacts before assigning that lineage.

Initialize both larger heads from the same training-only affine map
$a=a_c+Vz$ with standardized PCA scores, extending the original initialization.
Train both recorded optimizer seeds with the unchanged head width, number of
updates, batch size, learning rates, code penalty and output scaling convention.
The objective remains

$$
\mathcal{L}=\operatorname{mean}_i
\frac{\|h(z_i)-a_i\|_2^2}{\|u_i(0)\|_{L^2}^2}
+\lambda\operatorname{mean}(z_i^2).
$$

There is no velocity, tangent, force, curvature, energy, rollout or validation
loss. Preserve both fixed training endpoints; do not select a stopping step or
training repeat by validation results. Save the full head/code checkpoints,
optimizer/data seeds, training histories, actual offline training cost and frozen
spatial-parameter equality checks.

The timed controls are the unchanged existing MLP16, the two new MLP32 endpoints,
and the affine32 map that exactly initialized those endpoints. All maps use the
same learned spatial bank. The affine propagation retains its nonzero offset
force and actual reduced mass. The previously evaluated full64 control remains
available as prior development evidence; it is not retimed in this bounded panel.

The weak-bank equations remain fixed at 64. Thus the new nonlinear system has
64 equations for 32 configuration unknowns, versus the incumbent ratio of four.
This remains overdetermined and never uses the collapsed square nonlinear
objective. The changed overdetermination ratio is explicit; widening the bank or
introducing an additional test-space construction would be a separate experiment.

## Separate time-step and complete-query cost controls

Keep the frozen MLP16 at its primary step with three repetitions. Apply the three
declared timed steps to each new MLP32 endpoint, also with three repetitions.
Each larger head additionally receives one finer accuracy-only query per case.
That query charges the complete input/output path and preserves its timing, but
its latency is explicitly ineligible for speed comparisons or winner selection;
it may include uncached compilation. Compare each adjacent time-step pair using
the same initial displacement/phase-energy scales. Keep failed steps, nonstationary
initial fits and time-refinement failures visible. Smaller error from additional
coordinates does not itself establish reduced query cost.

Use the same two development inputs per boundary on both existing meshes. Every
query takes full host displacement and velocity plus wave speed and returns both
dense host fields at all requested times. Charge full input projection, all eight
initial-fit starts, speed-dependent work, evolution, dense decoding and transfer.
Mesh-only bank/operator construction remains offline. Alternate method order,
warm all shapes, burn the GPU before each timed invocation and retain three raw
repetitions. Include the same-job requested-mesh exact discrete DST and absorbing
RK4 controls. Coarser FOM/interpolation envelopes remain outside this panel; no
raw linear-control ratio is promoted to a paper cost-to-tolerance claim.

Keep snapshot/tangent/normal-force/zero-field diagnostics on the same declared
subset of observation times and the same paired fitting budgets. Each endpoint
uses its own trained code library with the same fixed index-selection rule.
Truth-only fitted states never seed an online query. Record all fitting endpoints,
gradients, stationarity and Jacobian ranks. Always retain absorbing absolute and
current-relative errors, current truth norms and vanishing flags; small fixed-scale
error does not pass an unstated late-time accuracy requirement.

## Bounded execution and audit

The proposed panel comprises the invocation counts in `k32-proposal.json`. The
runtime estimate uses the measured prior timings with a conservative larger-head
factor, plus allowance for seed regeneration, training, compilation, diagnostics
and saving. It is one GPU job with a two-hour hard wall limit, not an open training
sweep. Do not submit additional repeats or change acceptance if a run fails.

Before submission, run small GPU controls and one actual-checkpoint integration
smoke, each under the repository's local minute limit. Check dimension-generic
fitting, zero-nonlinearity affine acceleration, output/velocity decoding, time
sampling, frozen bank equality and original training-objective correspondence.
Run the batch with GPU preflight, f64/highest precision, source/checkpoint hashes,
its own directory, recorded queue checks and disk check. Regenerate the independent
physical reference checks as in the previous pilot.

Save common-grid outputs for each first repetition, full-grid reference fields,
all bank tables and all nonlinear coefficient/latent states. Hash repeated full
outputs. Preserve the complete checked split archive and its tested restore helper
for large extracted arrays. Repeat the independent NumPy field, checkpoint,
geometry, fitting and affine audits; verify all source and output checksums before
deleting only this exact remote attempt.

## Plain-language glossary

- **Bank / head / frozen:** learned spatial functions / map from reduced coordinates
  to their coefficients / unchanged weights.
- **Configuration / phase / rank:** displacement coordinates / displacement plus
  velocity coordinates / number of independent spatial features.
- **PCA / affine / standardized scores:** principal directions of training
  coefficients / linear map plus a fixed offset / training coordinates divided
  by their standard deviations.
- **Weak equations / overdetermined:** projected spatial equations / more equations
  than unknown coordinates.
- **Endpoint / code / seed:** fixed trained model / a training snapshot's fitted
  coordinate / recorded random generator state.
- **Snapshot / tangent / curvature / normal force:** one field in time / locally
  representable velocities / decoder bending / force outside the tangent span.
- **Stationary / budget / rank failure:** locally converged fit / allowed iterations
  or wall time / loss of independent local directions.
- **CFL / DST / RK4:** time-step-to-grid ratio / sine-transform propagation /
  fourth-order time integration.
- **Complete query / repetition / offline:** supplied input to requested output /
  repeated timing of a fixed case / setup outside individual queries.
- **Current-relative / vanishing / final cohort:** normalized by the current truth /
  small relative to its initial scale / unopened independent confirmation inputs.
- **Lineage / hash / archive:** data and model origin / content fingerprint /
  complete preserved run output.
