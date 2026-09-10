# Modified CP and empirical quadrature pilot

This is the executable September 10 architecture comparison. Numerical results
remain provisional until full validation and the frozen evaluation phase finish.

## Fixed scientific configuration

Train independently on Burgers2D, reflective wave2D and absorbing wave2D. The
common source in this tree is the canonical implementation; the separate wave
owner copies it with matching content hashes. Training uses 256 intervals and
frozen models transfer to 512. Burgers uses $k=16$, $R=64$; joint wave uses $k=32$,
two state outputs and $R=64$. CP and modCP share 6,000 initial CP updates, followed
by 10,000 updates each. FiLM uses 16,000 updates. Adam uses independent cosine
schedules from $10^{-3}$ to $10^{-4}$ per stage; the latent code rate is three
times the parameter rate. Batch size is 64 snapshots and 256 spatial points.

Each PDE trains from 64 trajectories and validates/evaluates on separate 16-case
cohorts. Seeds and generated configurations are recorded by `campaign.py`.
The modCP coordinate branch has width 64 with latent FiLM before SiLU and zero
output initialization. The full FiLM INR has two width-128 layers. No physical
initial-condition descriptor or time is a decoder input. Three separate weights
per PDE are compared under the same reconstruction, weak solve and timing rules.

## Boundary and quadrature refinements

Masked CP table endpoints never receive a training gradient. Refined points must
not interpolate those arbitrary parameters. All Dirichlet arms therefore clip
evaluation coordinates to the training interior and multiply the entire output
(including bias) by a linear boundary-strip envelope. This preserves every
training-grid prediction and gives continuous boundary-aware transfer. Absorbing
boundaries remain unconstrained. Coordinate-only stems/interpolated factors are
cached; every latent-dependent operation remains online.

The weak test dimension is $M=4k$. Quadrature rules use $m=4M$ or $8M$ selected
nodes, positive active weights and explicitly zero-weight padding. Constant-volume,
decoder-output mass and FOM-exact sign-dependent Burgers upwind integrands train
the rule. Candidate pools are deterministic grid subsets; targets always use the
complete grid. Diffusion is transferred onto sine test functions. Final latent
steps are audited against full-grid weak residuals after timing.

## Solver and measurement contract

Burgers uses backward Euler, weak damped Gauss–Newton and a corrected full-order
Newton/BiCGStab solve with FFT Helmholtz preconditioning. The ROM sweep uses
iteration caps 10/30, normalized tangent-gradient tolerances $10^{-4}/10^{-6}$,
the two quadrature rules and steps $0.005/0.0025/0.00125$. Full-order nonlinear
tolerances $10^{-2}/10^{-4}/10^{-6}$ are swept with relative Krylov tolerance 0.1.
Every stopping reason and actual full-order residual is retained. Budget or small
step exits are not described as converged; a zero tangent is a separate failure.

Every configuration runs on all 16 validation cases once. A predeclared case-0
proxy supplies two discarded warmups and seven recorded selection timings. The
coordinator approved this bounded validation-timing refinement before scientific
launch; it retains the complete accuracy cohort and solver grid. Final evaluation
retains two warmups and seven timed repetitions on every one of its 16 cases.
Initial fields and reconstructed output trajectories remain on the GPU; the
charged region includes sampled initialization, evolution and dense decoding.
Setup, training, compilation and host transfers are excluded. Every measured
region is preceded by GPU burn-in. Time and errors come from the same invocation;
raw repetition arrays, per-run provenance and output hashes are retained. Complete
fields/truth are saved for the first deterministic repetition per configuration.

The full validation grid completes before target configurations are frozen. The
evaluation seed is not drawn until the freeze artifact exists. An unattained
1%/5% target is explicit; the best-validation-error diagnostic is then retained
without claiming target success. Nested references provide empirical uncertainty,
not a rigorous continuum error bound.

## Execution and result files

`cluster/submit.sh <attempt> <smoke|train|all>` stages committed code directly to
the approved namespace, verifies content hashes, checks the queue around submit,
and runs the mandatory GPU preflight. Scientific defaults are never reduced by
the smoke profile. The campaign saves resumable optimizer states, independently
generated seeded data, quadrature weights, invocation JSONL and `handoff.json`.
The latter supports the coordinator's independent field audit and generated report.

## Glossary

CP: sum of products of one-dimensional spatial factors. ModCP: CP with latent
modulation inside its spatial branches. FiLM: feature-wise affine modulation
before an activation. INR: a network mapping coordinates and latent states to
field values. EQ: positive fitted empirical quadrature. $k$: latent dimension.
$R$: CP rank. $M$: weak test modes. $m$: selected quadrature nodes, including
explicitly zero-weight padding. NNLS: nonnegative least squares. FOM: full-order
discrete solver. Validation: data used to choose configurations. Evaluation:
untouched data used after that choice. Stationarity: small normalized residual
gradient, which does not alone certify an accurate PDE trajectory. Query: the
complete supplied-field-to-output-trajectory computation.
