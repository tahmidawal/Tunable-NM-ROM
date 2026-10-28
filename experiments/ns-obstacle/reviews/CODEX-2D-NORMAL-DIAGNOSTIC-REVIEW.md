# Independent 2D normal diagnostic review — 2026-10-28

Read-only numerical reviewer: **READY for the bounded geometry-only diagnostic** at
`09a660f7b`, conditional on exact staging and coordinator release. All eight candidate
inputs match their hashes, sizes and commit bytes. Manifest SHA256:
`87cc82472de0a8bd128aecbf268dec30ecd0a5a8d034f46137ced3b17f5fd0b5`.
The manufactured configuration is byte-identical to the retained smoke06 input. Python
parsing and shell syntax checks pass. No source, job or native FE operation was changed
or run by the reviewer.

The P0 constant test is evaluated in every volume element; contraction of its coefficient
vector with the assembled linear form integrates the requested expression without assuming
the coefficient values themselves are all one. The pinned source confirms that
`ds(label, skeleton=True)` on a linear form uses the labelled boundary's adjacent volume
element and computes its facet normal. This differs from ordinary boundary `Integrate`.
`dx(element_boundary=True)` separately tests cancellation over all element boundaries.

Raw mapped moments, ordinary integration and volume-facet integration remain separate.
Radial signs, adjacent-fluid dot products, offset containment, volume and the identity
$\int_{\partial\Omega}x\cdot n\,ds=2|\Omega|$ can distinguish orientation conventions
from an incorrect fluid domain. Closure alone is not treated as a sign test.

The fresh mesh is saved before probing. Raw samples and ordinary moments are saved before
the extra facet API; its exceptions are recorded separately. Geometry and signs remain
unchanged. The failed run's missing mesh identity is not asserted. The batch preserves
the reviewed resource ceilings and precision/private-dependency controls; only vector
assembly occurs, with no factorization, PDE solve or training. Thread count is set before
the TaskManager. The explicit H200 submission override must be recorded.

At result review, inspect the nested facet status: an outer diagnostic completion is not
a passed facet check, full G1 or reference admission. Fresh-mesh values do not recover
the missing original failing vectors. All earlier failures remain retained.

Source evidence is preserved in the 2D lane's
`artifacts/g1-smoke06/source-evidence/ngsolve-comp__linearform.cpp` and
`ngsolve-fem__symbolicintegrator.cpp`, with hashes and upstream source references.
