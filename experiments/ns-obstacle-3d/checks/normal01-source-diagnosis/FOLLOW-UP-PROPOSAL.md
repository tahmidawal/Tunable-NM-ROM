# Proposed geometry-only determinant attribution

This is an unexecuted proposal, requiring an exact source review and coordinator
release. No follow-up is staged. It adds measurements to the unchanged coarse
geometry and retains the original failed controls. There is no PDE solve,
normal repair, new mesh size, new curvature degree or gate-threshold change.

Use the retained `mesh.vol` as an immutable, hashed input and load its stored
curved mapping without calling `Curve` or regenerating geometry. First record
whether the complete curved representation survives loading, and repeat the
saved mapped-point, Jacobian, weight and aggregate controls. Persist all parity
differences and provenance. Do not claim exact identity with the prior in-memory
mesh. If curvature or topology cannot be recovered, stop with evidence; no
automatic remesh or reconstruction fallback is proposed. Interpretation of a
loaded-mesh result must remain limited to that serialized representation unless
the original observations are reproduced. The earlier smoke mesh remains lost.

The minimum measurements are:

1. Repeat the existing ordinary boundary, exterior volume-facet and
   all-element-boundary controls on that one loaded mesh, retaining every
   per-element P0 linear-form vector, not only its sum. Save the constant-test
   coefficients and error. Keep all original normal signs.
2. Form the signed determinant of `specialcf.JacobianMatrix(3)` by an explicit
   scalar cofactor expansion. Inside the *same* native facet forms, accumulate
   area, normal components and position-normal moments restricted to negative,
   zero and positive determinants. Also retain diagnostic near-zero bins using
   a predeclared dimensionless determinant divided by the product of Jacobian
   column norms; the near-zero cutoff is `64*float64_epsilon`, purely an
   attribution bin and never a replacement validity threshold. Save raw values
   so no conclusion depends only on binning. Record the native sign-split P0
   vectors for both exterior and all-element-boundary paths.
3. Enumerate each path's exact triangle rule and port the pinned tetrahedron
   face mapping, including its distinct vertex permutation. Save element,
   global face and local face IDs, global vertex IDs, triangle and tetrahedron
   reference coordinates, physical coordinates, signed full Jacobian, raw
   reference weight, cofactor normal, surface measure and signed determinant.
   Preserve transformation owners during point access and copy array data
   immediately. The trace must reproduce each native per-element vector and
   its sign-split components. A mismatch is a retained control failure, not a
   reason to substitute the trace silently for native behavior.
4. For every wall face, match the corresponding ordinary boundary triangle by
   global vertex IDs and map the *same barycentric points* into its local
   coordinates. Compare physical locations, tangents, surface measures and
   normal directions pointwise. Relate each discrepancy to its signed volume
   determinant and its contribution to wall area and normal/position moments.
   Record positive, negative and near-zero contributions separately.
5. Pair both sides of each interior face. Report physical-point agreement,
   opposing normals and weighted contribution cancellation. Keep both the
   native default-permutation sums and a common-point comparison; the latter
   is a diagnostic control, not a replacement integration result. This is
   necessary because exterior facets alone cannot explain the observed
   difference between the exterior and all-element-boundary identities.
6. At the existing higher-order volume rule, save signed determinants and
   retain separate reference-rule sums of determinant and absolute determinant.
   The matching native controls are the physical integrals of the determinant's
   sign and of one, respectively; integrating the determinant itself against
   physical volume would multiply by the volume scaling twice. This
   checks whether the positive scalar volume masks orientation changes missed
   by the coarser saved interior rule. Preserve both reference and mapped
   weights and distinguish negative quadrature weights from negative Jacobians.

Keep the current order and geometry for this minimum check. A later fixed-mesh
order ladder or geometric refinement is a separate proposal with predeclared
orders/configurations and review; neither is an automatic response to failure.
Ordinary-boundary order stabilization alone is insufficient to dismiss the
possibility of underintegration in a path involving a determinant sign function.

Use the existing bounded allocation envelope: one H200 GPU, sixteen reserved
CPUs, four FE threads, 64 GiB and twenty minutes, with GPU/f64/highest preflight,
hardware identifiers, immutable manifests and charged elapsed allocation time.
No matrix or solver is needed. Retain the existing element cap; derive a combined
point cap from that cap and the exact volume/face rule sizes before staging.
Stream compressed point arrays in bounded chunks rather than large JSON lists.
Any implementation must declare its output-byte cap and memory/process timeout
before review, and fail while retaining partial evidence rather than discard
samples. Collect, checksum, commit and mirror all unique output before removing
remote duplicates.

This check can distinguish a measured determinant-sign contribution from a
point/permutation/map inconsistency. It cannot establish global injectivity
between quadrature points, certify a physically valid mesh, or authorize a 3D
normal correction or subsequent flow solve.

## Terms

P0 is constant per element. A sign-split moment records separate contributions
from points with different determinant signs. A near-zero bin marks numerical
proximity to a singular map without changing an acceptance rule. Barycentric
coordinates locate a triangle point by its vertex weights. A parity check asks
whether two representations reproduce the same observations. Global injectivity
means that distinct reference points do not map to the same physical point.
