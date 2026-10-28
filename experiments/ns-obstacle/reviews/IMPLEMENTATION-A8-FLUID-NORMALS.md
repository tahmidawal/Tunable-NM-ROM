# Implementation amendment A8 — fluid-oriented boundary normals, 2026-10-28

Status: repair implementation authorized; **execution remains conditional on exact source,
configuration and independent Codex/Claude review**. This implements the already-declared
outward-fluid normal convention. It changes no physical geometry, PDE, pressure gauge,
force definition, population or scientific acceptance threshold.

The original code assumed that a raw boundary-transformation normal was outward from
the fluid. The retained 2D normal01 diagnostic disproves that assumption on the regenerated
circle mesh: raw and ordinary boundary-integration normals point into the fluid at the
obstacle, while independently assembled volume-facet normals point outward and satisfy
the divergence theorem. The initial 2D failed mesh is missing; its exact vector cannot be
recovered. The initial 3D sphere check records a similarly directed raw normal, but its
ordinary/facet relationship requires its own bounded diagnostic before a 3D repair release.

For ordinary boundary evaluations, derive orientation from the uniquely adjacent fluid
element, preserving the actual curved normal and every raw observation. Require a consistent
orientation factor within each boundary label; ambiguous or inconsistent evidence must fail.
Verify the resulting field against the analytic side/radial direction and the independent
volume-facet moments. Do not substitute the exact analytic radial normal for the curved
mesh normal. Do not modify native volume-facet normals or apply the factor twice.

Apply the same verified convention to pre-solve boundary checks, boundary loads, fluxes,
forces, balances and exported boundary data wherever they use ordinary boundary evaluation.
The exact source review must enumerate these call sites and distinguish wrapped derivatives
from normal orientation. Reloaded consumers require the same explicit convention and checks;
the accepted source-only A6 adapter does not yet qualify the new boundary interface.

The first 2D retry is isolated from the unexecuted G2 producer changes, based on the retained
G1 solver with only this normal repair and the separately reviewed thread-setup correction.
One repair retry for this distinct ordinary implementation error is allowed after review,
under the existing bounded G1 resources. Passing it does not admit reference data. Preserve
all failed source/results and the uncorrected diagnostic; do not rewrite an earlier status.

The empty-channel result remains a valid scoped control. No obstacle flow solve previously
completed, so no obstacle-force or reference claim is being salvaged or reinterpreted.
