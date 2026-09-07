# Fresh wave benchmark for the separable decoder

Revised after the user instructed: “Discard any old wave experiments. I don't think they
were correct necessarily.” This supersedes the inherited-wave proposal at `b460667`.
It is a fresh-benchmark design direction, not an executed experiment or numerical result.

## Evidence reset

All earlier wave experiments, favorable and unfavorable, are excluded from the evidence
for this project direction. Their code, data, trained banks, checkpoints, purported
verification, thresholds and causal explanations are not a trusted starting point.
Archive files and append-only chronology remain for traceability. Exclusion does not
assert that every old number has been proved wrong; their correctness is not assumed.

In particular, withdraw the previous proposal to use an old POD wave bank, old wave
solvers/integrators, old parameter tables, old gates, or old supervised absorbing-wave
success as a reference. No historical wave result establishes whether this architecture
works or fails. No legacy wave module is imported or copied into the new implementation.
The completed Burgers architecture results retain their existing scope and status.

## Question and retained architecture

Test whether the separable learned spatial bank with an MLP or quadratic coefficient
head can reconstruct and evolve unseen wave states, for both absorbing and reflective
boundaries. Retrain on freshly generated wave data. Gaussian descriptors and time are
not decoder inputs; physical wave speed belongs in the PDE operator.

The first implementation is two-dimensional to establish the full numerical path;
three-dimensional waves remain a subsequent extension. The intended interior equation is

$$u_{tt}=c^2\Delta u.$$

Use separate boundary configurations: fixed-wall reflection, and a newly specified
absorbing condition. State precisely which physical reflection is tested and the
absorber's approximation. A periodic traveling wave can serve as an additional
propagation control; it is not a replacement for either requested boundary case.

Use the neural separable representation from the beginning:

$$u(x;z)=G_\theta(x)h_\phi(z).$$

Train a fresh spatial coordinate network for each boundary configuration, then freeze
its learned bank for the matched head comparison. Keep the bank identical across head
arms within that configuration. Independently measure the best unrestricted bank
reconstruction and velocity projection errors before blaming a head. Weighted QR may
change coordinates but must preserve the learned span. Fresh POD models are permitted
only as explicit comparison baselines, never as the spatial bank of the claimed neural
method. No old wave bank or weights are reused.

## First task: establish trustworthy wave truth

Before training a decoder, independently derive and implement a full-order wave solver
and its boundary operators. Declare the spatial stencil/weak discretization, quadrature
weights, time update and energy balance in a new implementation specification.

Verify the implementation with independently calculated references:

- Analytic standing modes for reflective boundaries, including nonzero initial velocity.
- Analytic traveling modes for propagation, before testing wall interactions.
- A manufactured solution with known displacement and velocity for spatial and temporal
  convergence; report refinement errors and observed order for each boundary treatment.
- A pulse reaching a reflecting wall: compare its travel time, reflected sign and shape
  against the chosen physical boundary and an independent reference.
- A pulse reaching the absorber: measure outgoing energy and remaining reflection against
  a separately refined/larger-domain reference. Energy decay alone cannot validate an
  absorber that also damages the interior wave.
- An independent small-grid matrix implementation, plus controlled mutations of the
  stiffness sign, boundary terms and first-step velocity, to demonstrate sensitivity.

Validate displacement and velocity together. Use energy balance as an additional check,
not a substitute for state or phase accuracy. Derive numerical tolerances from the
reference error and refinement study before inspecting any trained-model results.
No old wave pass/fail threshold or verification label is inherited.

## Fresh data and controlled learning

Generate new training, validation and final-test trajectories from distinct recorded
seeds. Retain the user's requested smooth/localized data style with varying positions,
widths, amplitudes and wave speeds. Construct initial data compatible with the chosen
boundaries; do not inherit an old hard wall truncation. Include separate nonzero-velocity
controls. Record initial compatibility defects, where applicable.

Finalize and commit the parameter ranges, membership, mesh, capacity, normalization,
optimizer schedule and acceptance criteria after the truth checks and before any head
comparison. Keep the final test closed during model development. No old wave trajectory
or cache is part of these datasets.

The planned controlled comparison remains:

| Arm | Head | Training objective |
|---|---|---|
| mlp | SiLU MLP with affine skip | Displacement reconstruction |
| quadratic | Affine plus unique quadratic latent products | Displacement reconstruction |
| mlp_velocity | Same MLP | Reconstruction plus velocity consistency |
| quadratic_velocity | Same quadratic | Reconstruction plus velocity consistency |

The source MLP and quadratic architectures come from the Burgers campaign. Their
initial decoded fields, training bank, data, normalization, optimization schedule and
repeat seeds must match across arms. Test their outputs, Jacobians, Hessians and
checkpoint reload independently at wave dimensions. Record parameter counts and all
source hashes. Neither parameter count nor equal updates imply equal compute.

For a displacement manifold, represent compatible velocity by

$$v(x;z,\dot z)=G_\theta(x)J_h(z)\dot z.$$

This provides the velocity-training target. Derive the reduced phase-space dynamics
and time integrator afresh from the weak wave equations, including the induced mass,
stiffness and boundary damping. Keep the physical velocity definition explicit for
any alternative integrator; do not hide a mismatch behind energy agreement.

## Establishing whether it works

Separate three questions: can the learned bank represent the fields; can the head
represent the fields and required velocity directions; and can the reduced solver
evolve them accurately from initial data?

Persist per-start latent-fitting objectives, gradients, stopping reasons and doubled
budget checks. Report reconstruction and tangent error on unseen trajectories, with
means, medians, worst cases, outlier counts and failed fits. Use trajectory normalization
and physical scales to avoid dividing by a field that passes through zero.

Run actual time evolution through initial propagation and multiple reflections, and
through wave exit in the absorbing configuration. Save displacement, physical velocity,
energy balance, phase/arrival error, boundary reflection and solver completion for each
trajectory. The energy of an error state is distinct from the difference between two
solution energies. Assess absorbing residual displacement/mean separately where the
energy does not control it. Do not align phases after the fact to improve acceptance.

Define the time-step refinement ladder before evaluating the head, preserve failures,
and use a predetermined completion-based rule for reporting. Compare against newly
computed full-order and POD references on the same data. Freeze accuracy requirements
before neural training; neither the old wave gates nor the Burgers percentages supply
a wave acceptance standard. Speed claims and mesh/3D extensions require their own
successful accuracy and verification checks.

## Execution status and ownership

Only this proposal has been revised. No new wave code, data, worktree or cluster job has
been created. The previous base/name question was not answered by the user's evidence
correction; the old proposal to inherit wave infrastructure is withdrawn.

Use the approved repair tree for preparation and the canonical lab log for status.
A new implementation worktree must be proposed from the current non-wave architecture
line and confirmed under `AGENTS.md`. Its scientific entry points and dependency
manifest must exclude all legacy wave modules and artifacts, even if git ancestry
contains historical files. Retain only audited generic architecture/optimization and
job-management utilities, and test their use independently in the new benchmark.

Real numerical work runs on the cluster GPU partition in f64 with highest matmul
precision and the mandatory backend preflight. Regenerate data from the new recorded
seeds; use one immutable directory per job, verify source/checkpoint/result checksums,
then delete the exact remote job directory. No new merge is authorized.

## Glossary

- **Full-order solver / truth:** the resolved numerical PDE calculation used as a reference, itself subject to verification.
- **Head / learned bank:** the latent coefficient function and the trained spatial coordinate network.
- **POD:** a fresh linear projection baseline derived from training snapshots.
- **Displacement / velocity / phase:** the wave field, its time derivative and its oscillation timing.
- **Reflective / absorbing boundary:** a wall that returns waves, or a boundary intended to let outgoing waves leave.
- **Manufactured solution:** a chosen exact field with the forcing and boundary data needed to satisfy the PDE.
- **Refinement / convergence:** reducing spatial/time steps and checking that error decreases as predicted.
- **Weak equations:** the PDE tested against smooth spatial functions, rather than minimized pointwise.
- **Tangent / Jacobian:** locally available changes in the decoded field and their derivative map.
- **Bank floor:** the smallest reconstruction error available in the learned spatial span without the latent-head restriction.
- **Compatibility:** agreement of initial data with the boundary conditions and required initial derivatives.
- **Training / validation / final test:** fitting cases, development comparisons on unseen cases, and a later untouched evaluation.
- **Mutation / outlier / completion:** a deliberate implementation error used to test a check, a case outside a declared tolerance, and successful completion of every required solve.
