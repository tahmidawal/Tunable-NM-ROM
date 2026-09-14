# Burgers matched-operator pilot

This branch runs a Gaussian continuum-family pilot. The original reference gate
failed and is preserved; the focused time refinement passed its independent
calibration. Shared data generation is running. The inherited ROM is slower and
less accurate than an efficient FOM in the new same-job diagnostic; no advantage
over a neural operator or efficient FOM is established.

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

The lane has an eight-hour total GPU cap. Its workers checkpoint after every
case and exit when useful tasks finish. `worker.py` runs focused reference
refinement, gates diagnosis and bulk generation, then generates the training and
validation splits only if the independent gate and runtime forecast pass.

`checks/refinement02-reference-audit.json` verifies the refined fields, stopping
records, original cached provenance links and gate arithmetic.
`checks/refinement02-diagnosis-audit.json` independently checks all saved query
fields, errors, stopping evidence and repetition coverage. Its quadrature section
compares the advection component on saved output states; it is not a trajectory
bound or a causal explanation. `eq_ablation.py` is prepared follow-up source only,
with no quadrature-ablation GPU result yet.

`cluster/collect_when_done.py` monitors only owned job 3702709. It submits no GPU
work. On completion it verifies the archive, audits references and all generated
cases, retains full compressed raw arrays in Git, then removes the exact completed
job directory. Inspect `checks/refinement02-collection-status.json` for its actual
phase and failure status; the existence of this script does not imply collection
has completed. It leaves a verified, cluster-generated `pilot-data01` cache in the
Burgers namespace for downstream FNO copying. Original indices remain unchanged;
`RELOCATION.json` explicitly maps the old calibration path. The cache requires
cleanup after the downstream owner copies and verifies it. Extracted local `runs/`
copies are convenient views; the durable archives are under `artifacts/`.

The native ROM diagnostic retains its fitted initial output. A future deployable
panel must return the supplied initial field exactly for every method, still
charging internal initial fitting and evolution, and retain compression error as
a separate diagnostic. That wrapper change must be labelled rather than replacing
the archived native metrics.

## Glossary

- **Interval:** one grid cell; a mesh with L intervals has L+1 nodes per axis.
- **Reference:** the recorded fine numerical solution, not an exact continuum solution.
- **Empirical margin:** summed observed space/time differences plus candidate-to-anchor error.
- **Bank:** fixed learned spatial features with freely chosen coefficients.
- **Head:** nonlinear map from a small latent state to bank coefficients.
- **Stationarity:** a small normalized optimization gradient, separate from physical accuracy.
- **FOM / ROM:** full-order numerical solver / reduced-order numerical solver.
- **Oracle fit:** an offline fit using the requested answer; never an online prediction.
