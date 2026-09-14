# Burgers matched-operator pilot

This branch prepares a Gaussian continuum-family pilot. Reference calibration is
in progress; no new matched-data accuracy or speed result is established.

`data.py` freezes independent per-case seeds and records the six-setting spatial
and temporal refinement anchor before permitting bulk generation. Eight distinct
calibration inputs, decreasing refinement differences, and a conservative empirical
margin at each requested mesh are necessary. A one-case runtime profile cannot
unlock data generation. Failed reference margins remain failures.

The fine reference uses the analytic initial Gaussian on its own fine grid, while
a deployed model receives only nodal initial values and viscosity. Generation
descriptors are offline metadata. Thus these targets define a smooth Gaussian
continuum-family pilot; they do not establish arbitrary sampled-field operator
accuracy or equal fine-grid initial-information access. All methods compared online
receive the same sampled field and viscosity. The original head also has unmatched
training history and supports preliminary diagnosis only.

`diagnose.py` compares evolved reference fields with free bank projection,
best-found multi-start nonlinear fitting, and the actual stationary weak rollout
on each identical case. These errors are not additive. It also interleaves complete
ROM and tolerance/coarse-grid FOM invocations on one GPU with stored fields,
repetition times, convergence evidence and error from each invocation. Oracle fits
are excluded from deployment timings. Full matched-data training and efficient
neural-operator comparisons follow the reference/resource decision.

The initial calibration allocation has an eight-hour cap, checkpoints after every
solve, profiles its first case before continuing the independent case bank, and
exits after the useful calibration backlog. It does not hold an idle GPU or silently
start a training campaign.

## Glossary

- **Interval:** one grid cell; a mesh with L intervals has L+1 nodes per axis.
- **Reference:** the recorded fine numerical solution, not an exact continuum solution.
- **Empirical margin:** summed observed space/time differences plus candidate-to-anchor error.
- **Bank:** fixed learned spatial features with freely chosen coefficients.
- **Head:** nonlinear map from a small latent state to bank coefficients.
- **Stationarity:** a small normalized optimization gradient, separate from physical accuracy.
- **FOM / ROM:** full-order numerical solver / reduced-order numerical solver.
- **Oracle fit:** an offline fit using the requested answer; never an online prediction.
