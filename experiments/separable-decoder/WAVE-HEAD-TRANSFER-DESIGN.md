# Wave head transfer: absorbing and reflective boundaries

Proposed experiment for the user's request to test the new decoder architectures on waves.
This document contains a protocol, not new wave results. New-worktree creation and cluster
execution await confirmation of the exact base and name required by `AGENTS.md`.

## Question and scope

Does the quadratic coefficient head improve unseen wave reconstruction and reduced time
integration compared with an otherwise matched MLP? Does training its tangent directions
with physical wave velocities improve either architecture? The first experiment is two
spatial dimensions, using the existing independently checked wave solvers. A successful
result does not establish three-dimensional wave performance.

Two configurations use the same interior wave equation and parameter draws:

$$u_{tt}=c^2\Delta u,\qquad (x,y)\in[0,1]^2.$$

Reflective boundaries impose $u=0$ on every wall. Absorbing boundaries retain the existing
first-order Engquist–Majda condition $u_t+c\partial_nu=0$ and its exact discrete damping
closure; these boundaries are approximate absorbers, not perfectly reflection-free.
Train separate models for each configuration. Reuse the architectures; retrain their
weights on wave data. No Burgers-trained weights are assumed to transfer unchanged.

The inherited initial data are single Gaussian displacement pulses with zero initial
velocity and varied center, width, amplitude and wave speed. Preserve their existing
wall treatment for the controlled comparison, and report the known boundary-compatibility
defect separately. A smooth wall-compatible datum and the existing traveling-pulse
operator checks are numerical controls, not substitutes for the declared data family.
Gaussian descriptors and time are never head or encoder inputs. Wave speed remains a
physical coefficient of the PDE solver.

## Explicit bank distinction

The inherited `wav2d_bank.py` builds a POD spatial bank. It is not the learned spatial
network used by the Burgers separable decoder. The initial wave experiment deliberately
freezes that existing wave bank to isolate the coefficient head; name every output
`bank_kind=pod_diagnostic`. POD construction uses training snapshots only. This stage
cannot establish a fully neural separable wave NM-ROM.

A fully neural confirmation requires training a wave coordinate network, freezing its
learned spatial bank, and repeating the successful head/control comparison. Preserve
Dirichlet walls for reflective data but allow nonzero boundary values for absorbing
data. Recompute the bank projection floor, velocity floor and reduced operators;
weighted QR may change coordinates but must preserve the learned span. This is a
separate gated stage: no learned-bank success or mesh-independent speed claim follows
from the initial POD-bank experiment.

## Fixed first-stage comparison

Proposed grid and capacity are $N=64$, $K=8$, $R=64$ and $M=64$ smooth test modes,
matching the existing wave free-code capacity. These are wave settings, not the Burgers
capacity. The new wave MLP must be retrained under the same initialization and schedule
as quadratic; historical wave numbers are context, not a matched baseline.

| Arm | Head | Training objective |
|---|---|---|
| mlp | Matched SiLU MLP with affine skip | Displacement reconstruction |
| quadratic | Affine plus all unique quadratic latent products | Displacement reconstruction |
| mlp_velocity | Same MLP | Reconstruction plus velocity consistency |
| quadratic_velocity | Same quadratic | Reconstruction plus velocity consistency |

Use the dimension-independent modules from the repair MLP and the completed quadratic
branch, with source hashes and the port diff recorded. First instantiate them at wave
sizes and verify the interface; do not import the Burgers data/training driver, which
hard-codes its grid and cohorts. Every arm initially decodes the same linear projection
onto the first training-bank coordinates; nonlinear outputs start at zero. Use the same
training-only center/scales, batch stream and optimizer repeats (200 and 201), with
40,000 updates, batch 2048 and peak learning rate $3\times10^{-4}$ under the common
warmup/cosine schedule. Report parameter counts. Equal updates are not equal compute.

Retain the original seed-0 draw of 576 parameter rows: train on the first 512 trajectories
and use the remaining 64 for validation, with all 51 stored times through $T=1$.
Do not call `sample_params` with a different table length: its vectorized draws would
change the cohort. The old seed-1 test cohort has already been inspected historically;
it may be used for implementation regression but cannot be relabeled an untouched test.
Reserve a separate seed-2 cohort for a later locked final evaluation; do not generate or
open it during this architecture comparison.

## Wave state and loss

A wave state contains both displacement and velocity. With frozen bank $G$ and head
$h$, the induced phase-space state is

$$u(z)=G h(z),\qquad v(z,\dot z)=G J_h(z)\dot z.$$

This tangent lift describes the manifold state and the variational solver. The
LSPG/Newmark arm retains its own CN-consistent dynamic velocity; report its discrepancy
from a tangent lift separately rather than silently replacing its state definition.

The velocity term asks whether the decoder can move in the required physical direction:

$$\mathcal L=\mathcal L_u+\lambda_v\mathcal L_v,\qquad
\mathcal L_v\propto\|GJ_h(z_s)\dot z_s-v_s\|_M^2.$$

Use the inherited trajectory-normalized displacement and velocity coefficient losses,
with each loss scaled by its own training energy and $\lambda_v=0$ or $1$ as declared
above. Free per-snapshot latent velocities are offline training variables in the velocity
arms; the online solver evolves its own reduced state. Keep physical $v=u_t$ distinct
from latent $\dot z$. Account for the out-of-bank residual in reported field errors.
Do not divide by instantaneous displacement or velocity norms near zero. Use trajectory
RMS normalization and initial total energy, and report the absorbing constant-mode
remainder separately because energy alone does not control that mode.

## Required implementation and verification

- Introduce a generic head adapter exposing `h(z)`, `hj(z)`, dimensions and checkpoint
  provenance to both wave time integrators. The existing `HeadNP` and save/load paths
  assume MLP parameters; a name change alone is insufficient. Preserve old-model behavior.
- Generalize training, latent fitting, negative controls and metadata without changing the
  wave PDE, bank, cohorts or numerical thresholds. The historical `HEADS`/`K_OF` dictionaries
  and oracle functions cannot accept these modules unchanged.
- Independently check outputs and first/second derivatives at nonzero nonlinear weights,
  mass-weighted bank reconstruction, checkpoint reload, zero-velocity states and both
  boundary closures. Validate actual solver dispatch and the first-step velocity mapping.
- Reuse the independent sparse-stencil and linear-head controls. Every reduced mass,
  stiffness, damping and weak test table must match its full-grid projected action.
  Wave operators are linear, so these tables are exact; no sampled strong-form residual
  or empirical quadrature approximation is needed in this experiment.
- Extend long-horizon truth persistence to include physical velocity. The inherited
  long-horizon driver retains displacement and energy only, which is insufficient for
  a displacement/velocity comparison. Preserve the initial velocity projection defect
  and the nonzero-velocity traveling-pulse control.
- Use the corrected current completion/negative-control logic, including recorded earlier
  wave amendments, and make the acceptance aggregator explicit. Do not silently recompute
  historical results under new thresholds.

## Measurements and decision

First report train/validation reconstruction, bank floors, physical-velocity tangent
residuals, Jacobian rank/conditioning and latent-fit stationarity. Use the same multistart
selection and initial guesses for matched architectures, with training-only starts,
400/800 fit budgets and saved per-start arrays. Declare eight total starts: zero plus
seven recorded training-state codes, using identical state IDs across arms. The old
oracle's `n_starts` parameter adds another code-pool start; do not inherit that ambiguity.
Save each start's objective, gradient and stop reason. Keep initial, propagation and boundary
interaction behavior visible. Report means, medians, worst cases and counts above declared
thresholds; a single aggregate cannot establish a reflective pass.

Then run actual validation rollouts on a predeclared 16-trajectory subset of the validation
cohort (its first 16 rows), through $T=1$ and $4T=4$, with both existing solvers: weak
LSPG/Newmark and variational Verlet. Use the inherited refinement ladder of 8, 20 and 40
substeps per stored interval; choose the finest fully completed rung by completion,
never by lowest validation error. Preserve all incomplete trajectories as failures.
Numerically verified heads may receive these bounded mechanism rollouts even if their
representation gate fails, but those are diagnostic runs, not promotion to a cost study.

Report displacement and velocity errors, phase/arrival discrepancies at fixed probes,
reflection behavior, and energy histories separately. Also report the energy norm of
the state error: agreement of solution energies alone does not imply solution accuracy.
Measure reflective phase drift in displacement/velocity quadratures of a predeclared
set of excited modes, rejecting near-zero modal amplitude as undefined. Record zero-state
representability and latent excursions; quadratic extrapolation can grow rapidly.
Do not phase-align predictions
before computing acceptance errors. Apply the inherited wave representation and rollout
requirements and every negative control: comparison to the matched POD-K rollout and
its own fitted representation floor at both horizons, energy behavior for reflective
cases, and pre-exit/remaining-field behavior for absorbing cases. The Burgers percentage
thresholds are not wave thresholds. A passing proxy for tangent quality cannot replace
a passing rollout: the previous wave campaign already exposed that limitation.

Advance a head only after reconstruction, solver convergence, physical-state rollout,
refinement and negative controls all pass. A boundary-specific pass is reported only for
that boundary condition. Recheck a successful first-stage comparison at $N=128$ before
any mesh-transfer claim. The learned-bank confirmation has its own bank-floor gate and
uses the same model-independent acceptance criteria. Cost studies remain gated on
accuracy; timings from separate jobs or GPU types are never compared.

## Worktree and job proposal

Use the current `exp/2026-09-06-burgers3d-repair` line, which contains the reviewed wave
solvers and current architecture audit tooling. The exact proposed base commit is recorded
in the canonical lab-log closing entry after this design is committed. Port quadratic
from `exp/2026-09-06-b3d-quadratic` at
`4dbe77cbb3a6269ee3894d264bbc4769799b1500`, preserving its tested unique-product convention.
This is not approval to merge any existing experiment branch.

Proposed single experiment worktree: `worktrees/2026-09-06-wave-head-transfer`, branch
`exp/2026-09-06-wave-head-transfer`, owned by its implementation subagent. Root continues
coordination in the approved repair tree. Proposed cluster namespace:
`/cluster/tufts/paralab/tawal01/wave_head_transfer_20260906/`. Boundary configurations and
stages get distinct immutable job directories. Confirm this base/worktree/namespace
before creating them. No cluster submission has been made for this proposal.

All numerical work is GPU/f64/highest, with the mandatory backend preflight. Local
component checks use `jaxrun` under one minute; real runs use the cluster GPU partition,
regenerated seed data and checksummed source/checkpoint/result provenance. Check the
queue around each submission, preserve all attempted jobs, and delete exact remote
job directories only after verified pulls.

## Glossary

- **Head / bank:** the latent-to-coefficient function and its spatial feature functions.
- **POD:** a linear spatial basis extracted from training snapshots; different from a trained coordinate network.
- **Displacement / velocity / phase space:** the wave field, its time derivative, and the state containing both.
- **Tangent / Jacobian:** field changes obtainable by locally changing the reduced coordinates.
- **Velocity consistency:** training those changes to include observed wave velocities.
- **N / K / R / M:** grid nodes per axis, latent dimension, spatial-bank size and weak test-mode count.
- **Mass norm / energy:** quadrature-weighted field size, and the wave's kinetic-plus-gradient energy.
- **Energy norm of the error / modal phase:** kinetic-plus-gradient size of the difference between states, and oscillation timing measured in fixed spatial modes.
- **Projection floor / oracle:** error from the unrestricted bank, and the best locally fitted latent reconstruction found.
- **Training / validation / final test:** fitting data, development comparisons on unseen trajectories, and a later untouched evaluation.
- **Boundary compatibility:** whether initial data satisfy the boundary relation and its required startup behavior.
- **LSPG / Newmark / Verlet:** projected residual minimization and the two inherited time-stepping constructions.
- **Refinement rung / completion:** a declared time-step setting and successful solves for every evaluated trajectory.
- **Negative control:** an intentionally broken or untrained alternative that a validation check must detect.
- **Promotion / cost study:** advancement after all required checks pass, and subsequent measurement of solve cost.
