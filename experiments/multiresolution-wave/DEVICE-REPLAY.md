# Fresh waves with device-resident full-field queries

This is the September 10 follow-up to the historical Burgers/Poisson replay. It
prepares new measurements and contains no new scientific result. All wave
mathematics, families and checkpoints come from the independently verified work
after the September 6 evidence reset.

Use the already audited meshes and two development cases from `heads32-config.json`.
The primary nonlinear model is the frozen first-seed larger head; the original
first-seed smaller head is a control. Both use their original fitting budgets,
eight initial starts, fixed learned spatial bank and a predeclared primary step.
Each receives one finer, accuracy-only query. Neither training seed nor time step
is selected using this run. The final cohort remains unopened.

The input displacement and physical velocity fields and scalar speed are ready on
the GPU before timing. The timer includes both full-field projections, all cold
fitting, speed-dependent stiffness/damping, nonlinear evolution and both complete
decoded fields at every observation time. It ends only after both output fields
are ready on the GPU. Host copies occur afterward and are recorded separately.
Mesh-only neural-bank evaluation, coordinate conversion, assembly and compilation
are offline. They are reported separately and cannot enter the reduced-solve cost.

The new measurement wrapper compiles the existing cold-fit/projection expression
as one kernel. It does not change the fit objective, budget, starting codes or
model. The timed query retains explicit synchronization between initialization,
evolution and output so its component costs come from that same invocation.
This execution change and the removed host-transfer charge distinguish this
measurement from the archived host-to-host wave experiment. The existing fields
and measured host costs remain archived under their original contract.

The reflective FOM is the verified exact discrete sine-transform propagator. The
absorbing FOM is the verified RK4 discretization with its existing primary CFL.
Both return the same two fields and observation times as the ROM. Absorbing
accuracy uses a finer same-grid RK4 reference, whose temporal refinement is
displayed explicitly. The finer reference is not the timed FOM baseline. No
pre-reset wave solver, old checkpoint, substituted boundary model, continuum
claim or coarse-grid FOM envelope enters this comparison.

Every actual timed result is scored before being discarded. Save full first
timed outputs, complete per-repetition output hashes and all raw timings. Reuse
one artifact for later repetitions only when both full-field hashes are
identical; otherwise stop rather than silently assuming deterministic output.
Retain failed rollouts, nonstationary initial fits, rank checks, reference
refinement and ROM step-refinement failures. Report initial-normalized and
current-relative displacement, velocity and phase-energy errors, including
vanishing-reference flags. A timing ratio is a comparison against the named
same-grid FOM, and requires the adjacent accuracy and failure counts.

One job runs both boundaries and meshes sequentially on one GPU, after the
Burgers/Poisson jobs complete and the coordinator reviews the timer. It uses the
existing wave branch/namespace, f64, highest matrix precision, backend preflight,
burn-in before each timed invocation and checksum collection. New fields and
references regenerate from the declared seed. Only frozen head/bank weights and
the fitted initializer maps are uploaded; no trajectories or training snapshots
are staged. `cluster_device.py` stages and collects the private attempt.

## Glossary

- **FOM / ROM:** full spatial solve / reduced-coordinate solve with field reconstruction.
- **Device-resident:** an input or output already stored in GPU memory at the timing boundary.
- **Cold fit:** fitting reduced coordinates to the supplied initial fields without a prior trajectory state.
- **CFL:** the dimensionless quantity setting the explicit FOM time step relative to mesh spacing and wave speed.
- **DST / RK4:** discrete sine transform / fourth-order Runge–Kutta integration.
- **Initial-normalized error:** error divided by the declared initial reference scale.
- **Current-relative error:** error divided by the reference norm at that observation time.
- **Vanishing reference:** a reference norm below the declared fraction of its initial scale.
- **Phase energy:** the joint displacement-gradient and physical-velocity norm.
- **Refinement:** comparison with a smaller time step using the same spatial discretization.
- **Stationary fit:** an initial fit meeting the retained gradient and conditioning criteria.
- **Checkpoint / initializer map:** frozen learned weights / frozen linear map used to propose one initial coordinate guess.
- **Median / outlier:** the middle sorted value / a repetition or trajectory exceeding its explicitly declared threshold.
