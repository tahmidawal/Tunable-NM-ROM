# Independent Codex ROM/POD infrastructure review

Read-only physics_reviewer inspected b8e50edbf on2026-10-27; no numerical code run.

One blocker before measured comparison: timing.py46–50 hashes outputs but retains their
array references. A callable reusing FE/NumPy buffers can overwrite earlier fields while
old hashes/timings remain, breaking same-invocation evidence. Copy/freeze the entire output
and accounting snapshot before hashing/storing. Regression must reuse and overwrite one
buffer and mutable work-metadata dictionary across repeated calls.

No further blocking algebra finding: analytic bank gradients/product envelopes, QR/SVD
ordering, tested quadratic convection and symmetrized Jacobian, pressure/continuity signs,
row/coefficient whitening, H1/L2 inf-sup orientation, constant pressure mode, output-derived
quadrature moments and training-only PCHIP are consistent. Training/runtime/FE integration
and actual physical/stability gates remain open.

Scoped verdict: feasible synthetic infrastructure for FE integration after retention fix,
not permission for training or accuracy/cost claims. Fitted quadrature currently contracts
to the same online tensor shape as the full rule; fewer integration points can reduce
assembly, but do not by themselves reduce the subsequent tensor-solve cost.
