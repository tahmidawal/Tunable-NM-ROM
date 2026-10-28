# Independent normal diagnostic result review — 2026-10-28

Read-only numerical reviewer: **diagnostic evidence accepted** for 2D job 6610978,
executed source `09a660f7b47926f3ef6dc8dea16f2d1848947dc4`. Full G1 remains failed.
No PDE solve, reference, training or method-comparison result is admitted.

The reviewer verified the complete archive, every extracted-file checksum and every
executed input against the frozen commit. Archive SHA256:
`9ef375747ec3c50dae53a8a5165e5ba7c940254b697012b097d1e300ee0ae177`.
Scheduler evidence and GPU/f64/highest preflight agree; stderr is empty. Controlling
result SHA256 is `7ba0d6b7695a2482e78db2fcbe9c9ab3d8abe9b32790456ea1078c161bb7413b`.

All sampled obstacle raw normals point toward the uniquely adjacent fluid element.
Every positive-normal offset lies in the fluid and every negative-normal offset lies
outside; mesh and analytic classification agree. Outer boundaries have the opposite,
expected direction. All saved volume weights and determinants are positive, none of
the sampled points lies inside the analytic circle, and area agrees with the intended
channel-minus-circle region within its declared geometric threshold.

Ordinary boundary integration reproduces the raw mapped moments. The obstacle radial
moment is positive and the whole-boundary position-normal moment fails the divergence
theorem. The separate adjacent-volume facet linear form yields a negative obstacle
radial moment and satisfies the identity to roundoff. Its nested status is completed,
and the independently summed all-element-boundary control gives the same value.
The constant P0 test also passes. Numerical tables are generated from these saved
JSON values by `reporting/summarize_controls.py`, rather than copied into this review.

This establishes a boundary-transformation orientation convention differing from the
outward-fluid facet normal **on the regenerated mesh**. It does not reconstruct the
missing mesh or vectors from the original failed run. Normal closure alone would have
missed this sign error. Preserve both original failures and all uncorrected diagnostic
samples before implementing a fluid-orientation correction.

Evidence lives in the 2D lane's `artifacts/g1-normal01/results/geometry/`, with the full
extraction in `runs/g1-normal01/results/geometry/`. Boundary-sample SHA256:
`ec15d054c4fd5691c49b0b9673f17cb50046e5dcdee26ad1883e63da6c113963`;
volume-array SHA256: `35c437f47e1aa65aba0e2bcd01a185f3c90170e56fc0f92eeffba51d0418ae43`;
facet-control SHA256: `26b86b47a10c1e5a2f5d824407e2263ddb598a96f9c1623176688e7d45fe5a66`.

No files were edited or native FE operations/jobs run by the reviewer.
