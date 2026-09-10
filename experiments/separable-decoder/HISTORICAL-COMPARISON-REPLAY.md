# Replaying the earlier separable-ROM comparison

This protocol implements the user's September 10 request to compare Burgers and Poisson “the old way.” It specifies a new reproduction study; archived measurements are available in the canonical historical cost audit and are not new results from this study.

## Selected comparison

Restore the earlier separable decoder configurations and their original device-resident query boundary. The supplied field starts on the GPU; the timed query includes initialization or source projection, the reduced solve, and reconstruction of the requested full fields on the GPU. Transfers to and from host memory are outside this primary timer. Offline training, operator construction and compilation remain separately identified.

Compare against the original same-grid FOM algorithms, with their original tolerance selection. The newer host-to-host cost-to-accuracy envelope remains a separate archived experiment. Its coarse-grid selection, different checkpoints and different output contract are not substituted into this historical reproduction. The older ViT/CP architecture and discarded wave evidence are outside this study.

## Panels and ownership

The Burgers owner uses the existing `2026-09-07-mr-burgers2d` worktree and approved cluster namespace. Restore the original mesh-specific small banks, sampled-field initializer, tensor advection, sampled/full controls, timestep, trajectory cohort, and dense-sine-preconditioned tolerance-terminated FOM. Retain the historical tensor's sign-dependent fidelity checks: a polynomial backward-upwind tensor is not unconditionally equal to the sign-dependent upwind operator.

The Poisson owner uses the existing `2026-09-07-mr-poisson2d` worktree and approved cluster namespace. Restore the frozen historical checkpoints and exact preassembled weak operator. Time the original iterative CG comparator in the same allocation and retain the direct spectral comparator under its own name. The older quadrature-free driver did not itself time a FOM; adding same-job baselines extends that driver and must be documented as such.

Each owner copies historical source and checkpoint bytes into an isolated replay cell, records hashes, and leaves archived worktrees unchanged. Each PDE's mesh ladder runs sequentially in one GPU allocation so its scaling comparison is on one device. Existing branches remain separate; no new worktree or merge is part of this request.

## Evidence required

Preserve original numerical algorithms and report every instrumentation change. Use float64, highest matrix precision, a verified GPU backend, regenerated seeded inputs, GPU burn-in, retained raw timing repetitions and accuracy from timed invocations. A source commit alone does not establish staged provenance; retain content manifests and checkpoint hashes. Collect result checksums before deleting the exact remote attempt directory.

Report reduced-solve time separately from complete device-query time. Preserve the historical error norm and aggregate, and additionally expose per-case errors, medians, worst cases and timing outliers. Label cohort-based FOM tolerance selection as development evidence. Do not infer universal FOM superiority from a win against one named solver, or attribute differences from archived jobs to algorithm changes when hardware and software also differ.

## Glossary

- **Separable decoder:** a learned spatial bank combined with coefficients produced by a nonlinear latent head.
- **FOM:** full-order model, which solves for the discretized field directly.
- **ROM:** reduced-order model, which solves for reduced coordinates and reconstructs the field.
- **Device-resident query:** a computation whose supplied input and returned output remain in GPU memory at the timing boundary.
- **CG:** conjugate gradient, the historical iterative Poisson solver.
- **Tensor advection:** a preassembled quadratic expression for the selected Burgers advection discretization.
- **Weak operator:** the PDE operator projected onto smooth test functions.
- **Checkpoint:** saved trained model weights and their configuration.
- **Cohort:** the specified collection of test inputs.
- **Timing outlier:** a repetition exceeding the report's explicitly stated threshold, retained in the raw data.
