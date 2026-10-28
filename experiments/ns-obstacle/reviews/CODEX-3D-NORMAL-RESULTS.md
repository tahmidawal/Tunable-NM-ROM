# Independent 3D normal diagnostic audit — 2026-10-28

Read-only reviewer: **accept the retained diagnostic; hold every 3D normal repair, flow
retry and reference admission**. Job 6611259 completed the measurements but failed its
geometric controls. Archive SHA256
`2bba51cec28384cbf07d6a32d267dd0b8556d121db15d9b80e158d9d3e002eb9`, all 774 files,
and all eight executed inputs were independently checked against
`6db733531ab0a551baa4fa816f0f386927f7d8f8`. Runtime precision and scheduler evidence
agree. Durations remain diagnostic; effective concurrency is unmeasured.

Recomputed exterior-facet and all-element-boundary identities agree with the saved failures.
The wall area matches ordinary boundary integration, while its normal and position-normal
moments differ. The all-element sum also differs from the exterior sum, requiring an
internal-facet cancellation or integration-path check. P0 constant-one normalization passes.
Generated numerical values are in `generated/controls/controls.md`, with source hashes.

Raw sphere normals point radially outward from the sphere and toward adjacent fluid
centroids. Ordinary boundary integration reproduces them. However, offset `mesh.Contains`
checks disagree with analytic membership on the sphere and even planar ports. They are
not valid one-sided fluid-region witnesses here. Their failure does not identify its cause.
Positive signed Jacobians at the saved interior points cannot exclude inversion at facet
points or elsewhere in the volume.

Pinned `intrule.cpp` shows cofactor-based normals multiplied by the sign of the volume
Jacobian determinant while surface measure stays positive. Negative facet determinants
could therefore change normal moments without changing area. **None were measured in
this run.** Curved-map folding is a hypothesis, not a diagnosed cause. Arithmetic reversal
of only the ordinary sphere sign makes its global identity much closer, but this is an
explanatory calculation rather than an admitted repair or mesh certificate.

## Minimal useful follow-up

Record signed volume Jacobians at the actual native exterior-facet and all-element-boundary
quadrature points, with element/facet IDs, reference/physical points, weights, measures and
normals. Split moments by determinant sign; compare native normals with the cofactor formula
and matched ordinary boundary maps. Retain per-element P0 contributions and paired internal
facets, plus the higher-order volume-rule Jacobians. Preserve the existing geometry and
resource caps. Curved reload parity is required before claiming identity with the saved mesh;
otherwise explicitly record regeneration.

Any diagnostic sign manipulation is explanatory only. It cannot certify an inverted map or
authorize production flow. The reviewer made no edits and ran no native FE operation or job.
