# Proposed bounded Poisson follow-up

The first frozen-checkpoint development result and its provisional status are in
`runs/pilot01/FINDINGS.md`. This proposal is not a submission: root requested a
pause in Poisson GPU work until heat starts and explicitly reactivates this owner.

Keep the existing worktree, namespace, checkpoint, exact source draws and mesh
conventions. Do not add a new final cohort or train the networks. Request one GPU
job with at most two hours and use a new unique attempt directory.

First locate the error limitation on the development cohort. Rebuild the exact
weak matrix for requested test-space dimensions 64, 128 and 256, recording actual
retained sine shells. Evaluate the existing tolerance ladder on both meshes and
retain every source. On each mesh also compute unrestricted full-bank least-square
projection and a diagnostic full-field head fit. The latter uses exact QR
coordinates of the frozen bank so its iteration cost remains reduced while the
objective equals full-field least squares, including the orthogonal residual in
reported field error. Start from the training mean and a fixed small set of
recorded training codes; compare bounded optimizer budgets, gradients and failures.
These oracle fits use reference fields only for diagnosis, never as deployed
initialization or an online accuracy-selection rule.

Separately fuse the unchanged source projection, cold mean-latent initialization,
trust-LM solve and full field readout into one compiled query. Pass large bank and
projection arrays explicitly and include full host input/output and all requested
solver counters in the measured invocation. Verify field, latent, residual and
counter parity against the original segmented implementation before timing. The
same job compares both implementations and efficient same-grid/coarse-grid DST
with balanced repeated timing and GPU burn-in. Retain a separately measured
component diagnostic rather than forcing stage synchronizations inside every
production fused query.

Use results to distinguish missed test modes, frozen-bank error, nonlinear-head
representation and dispatch overhead. If the bank/head approximation remains the
limiter, propose a separately labeled training study from that evidence. If FOM
remains faster at qualified accuracy, preserve that negative result; do not expand
the resolution ladder merely to seek a favorable point.

## Plain-language glossary

- **Test-space dimension:** count of smooth functions used to average the PDE.
- **Bank projection / head fit:** best unrestricted spatial-bank approximation /
  approximation constrained to the frozen nonlinear coefficient map.
- **QR coordinates:** an orthonormal coordinate system preserving the same field
  and least-square objective while making fitting cheaper.
- **Oracle:** diagnostic that sees the reference answer and cannot serve as a
  deployable initializer or solver-selection rule.
- **Fused / segmented query:** one compiled pipeline / separately dispatched
  stages; both must implement the same algorithm and return the same output.
- **Parity / counters / dispatch:** numerical agreement / solver work statistics /
  host instruction overhead for launching accelerator work.
- **Qualified accuracy:** every declared development source meets the target with
  reference uncertainty included and the required solver-validity checks.
