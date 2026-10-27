# Independent scalar-return compatibility reviews

The numerical reviewer found no blocker in 2D `9d431a0ccdf3f9477370e032a95d3f0c1a0b4323`:
the helper accepts finite numeric scalars or one-element returns, rejects other shapes,
and records each distinct representation. CF algebra, thresholds, scientific configs and
resource caps are unchanged. All input hashes, syntax and scalar-contract tests pass.
Original failed tuple length/content were not saved; the new runtime will reject any
non-scalar representation. One control retry only; no reference admission.

The physics reviewer found no blocker in 3D `7a5a6e4277aa3c532c4b0d19c20c3cf118f1f18b`:
strict extraction covers pressure probes and new direct/wrapped divergence and scalar-H1
point controls. Prior 3D divergence checks were integral-only, so the exact failing 2D
conversion was not present there. Science/resource limits and dependency pins are unchanged.
Its fresh candidate still requires 2D G1 success and explicit release before execution.

The coordinator independently verified every staged source hash for both candidates. The
2D allocation is recorded in the ledger. The 3D source remains unexecuted in g1-smoke05;
INPUTS.sha256 has SHA `72d964fc372c0ba4af96444747eb1377ccf806cefa3ef2dfb2cd8fe7d539fd53`.
