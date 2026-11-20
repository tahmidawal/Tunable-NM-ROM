# B8 actual trained CROM online qualification

Prospective preparation only; no submission. After boundary training finishes,
retain and verify its completed selected checkpoint, full result/data/source
provenance and dependency manifest. Run a new owned15-minute single-GPU fixture
within the shared two-job cap, from backed-up source:
`qualify_crom_trained_online.py --out OWNED_OUTPUT --model-job b8cromboundary`.
A running/partial training attempt is explicitly rejected.

Use the pinned native Decoder plus existing crom_loader and scalar2d adapters.
Verify retained checkpoint/result/data/port hashes, source fingerprint, exact
128primary+32expanded training and44primary+8expanded development cohort order,
training-only normalization, original source records and native dependencies.
Verify the trained boundary wrapper hash against the training record. No final
fields and no development solution fields are loaded. Normalization is checked
against the matched training array; probe codes are zero plus encoder codes for
four training initial fields (pulse, smooth, ridge, two-pulse).

All numerical fixture operations execute on the allocated GPU in Float64 with
TF32 disabled. Compare the trained decoder's full analytical input Jacobian
against automatic differentiation and per-coordinate central differences,
covering every latent variable and both physical coordinates. Thresholds:
max-difference/(1+max-reference)<=1e-9 for analytic/AD and<=1e-6 for field finite
differences with1e-6perturbations.

At the true fine spacing1/16384 and64fixed sample centers, check the scalar2d
RHS code Jacobian in three seeded unit directions. Retain all finite-difference
attempts at1e-3,3e-4,1e-4. Each direction requires at least two same-upwind-branch
attempts with max-difference/(1+max-reference)<=5e-4. This threshold accounts for
fine-h second-difference cancellation; it does not certify PDE accuracy. A sign
switch is reported and cannot count as a passed derivative comparison. Retain
failed cases rather than selecting a convenient code or perturbation.

Check all four boundaries and first/second interior strips at L4096,8192,16384.
Boundary values and code Jacobians must be<=1e-12. Compare the first-strip normal
secant with the analytical boundary normal derivative (scaled error<=0.02),
and retain strip values/scaling. These checks reject discontinuous endpoint-only
clamping without comparing to solution truth. The envelope imposes the same
continuous physical Dirichlet condition as the neural tree.

A tiny L32grid is used solely to compare deduplicated adapter RHS assembly with
the identical dense algebraic stencil on the same trained field, and to verify
explicit-target refresh ignores stale q_sample. No time step or scientific PDE
trajectory is evolved; L32must never replace the fine physics. Identity tolerances
are1e-12. Preserve codes, every derivative attempt and all failures.

Classify this as a method-native CROM interface plus a continuous-boundary
scalar2D new-PDE port, not an unchanged reproduction or a common weak-form tree
comparison. Passing demonstrates trained implementation checks only. Full
trajectory-level representation qualification, independent current-time
physical scoring, strict projection convergence and actual4096²output remain
required before scientific PDE admission. A native solver return alone is not
convergence; no comparator accuracy/runtime or tree superiority claim follows.
