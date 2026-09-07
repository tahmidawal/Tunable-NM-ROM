# Independent architecture and result review

The coordinator read every model and its component tests before authorizing the
scientific jobs, then tested every actual head through the inherited decoder's
pilot adapter with independent NumPy field, Jacobian and Hessian references.
The coordinator's campaign auditor checks the completed result artifacts across
all isolated worktrees. This review accepts bounded representation findings;
no global-optimum, rollout or cross-PDE result is inferred.

The protected-head author independently reviewed the encoder and mixture code
and numerical outputs. That reviewer rechecked hashes, reconstructed fitted and
direct-encoder errors with NumPy, finite-differenced the worst-state Jacobians,
and recomputed tangent and invariant-stationarity diagnostics. The worst cases
are converged. The encoder's reconstruction benefit is inconsistent across the
optimizer repeats. The mixture improves training and tangent means while its
validation mean and worst error worsen; the declared collapse flags are false.
Both are accepted as bounded negative screen results.

The encoder author independently reviewed the quadratic code and numerical
outputs. That reviewer rechecked data/bank/schedule/start identities, all input
and output hashes, matching common evaluator source, GPU and precision flags,
and polynomial output, Jacobian, constant Hessian, tangent and objective-curvature
calculations. The quadratic gain against matched MLP controls is accepted as a
bounded improvement. Effective mean and worst-error gates still fail, and its
worst states have converged. Deterministic initial parameters make its repeats
optimizer/minibatch repeats only. No global-minimum claim is supported.

The coordinator independently reviewed the protected-head implementation and
results. The frozen-anchor geometry guarantee is implemented correctly and
verified by its nonzero-residual component tests and actual-head integration
checks. State and tangent accuracy worsen relative to the matched control;
rank protection alone does not explain or resolve the observed error gap.
The author did not certify this arm as its own independent reviewer.

The peer review identified useful additional artifact checks: explicit ordered
optimizer-seed and fit-budget identities, derived-parameter rounding bounds and
archive hash, and the unopened final-test flag. These were added to the
coordinator auditor and pass for every completed numerical job. No scientific
result was retracted. The earlier pending cancellations are recorded separately
from completed numerical jobs.

Exact numerical comparisons are generated in B3D-ARCH-NOTES.md from raw JSONs;
source hashes, checkpoints, peer worker audits, and full state arrays remain in
their experiment branches. The canonical lab log records branch and job IDs.
