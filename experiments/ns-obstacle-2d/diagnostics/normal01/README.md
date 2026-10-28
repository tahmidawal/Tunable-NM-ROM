# Unchanged-geometry normal diagnostic

This is an unexecuted diagnostic candidate for the obstacle-normal failure in
job 6567998. It does not solve a flow case, repair a sign, or admit reference data.
The original failed obstacle mesh and normal samples were not saved, so a new
mesh cannot establish their exact values retrospectively.

The geometry recipe and configuration bytes are copied from the failed
manufactured case. `AddCircle` retains `leftdomain=0, rightdomain=1`; rectangle,
circle, mesh size and curvature order remain unchanged. The diagnostic saves the
new mesh before inspecting normals. It never changes boundary orientation or
reverses an observed normal.

The independent observations are:

- Every mapped boundary point, raw normal, tangent Jacobian, positive weight and
  signed radial cosine.
- The sole topologically adjacent volume element and a mapped point in that
  element; signed dot products toward that point.
- Volume point location on both sides of each boundary point at the fixed offset
  distances in `diagnostic.json`, with analytic fluid membership recorded too.
- Per-boundary ordinary `Integrate` moments and sums of the raw mapped samples,
  including normal integrals and position dotted with normal.
- Fluid volume and the divergence-theorem identity
  $\int_{\partial\Omega} x\cdot n\,ds=2|\Omega|$, plus separate closed outer and
  obstacle normal sums. A zero closed normal sum cannot determine its sign.
- A separate volume-facet control using a constant, verified P0 test in a
  `LinearForm` with `ds(label, skeleton=True)`, and an all-element-boundary
  control with `dx(element_boundary=True)`. The form vector is contracted with
  the coefficients of the constant-one function; no equation is solved.

These observations distinguish a wrong geometric region from a boundary normal
convention and detect disagreement between raw point evaluation and ordinary
boundary integration. The retained pinned sources suggest the latter use the
same boundary transformations, but runtime evidence must decide. The diagnostic
does not use `BoundaryFromVolumeCF` because it evaluates geometry and normals,
not a derivative of a finite-element field.

The [official DG tutorial](https://ngsolve.org/ngsolve/docs/i-tutorials/unit-2.8-DG/DG.html)
distinguishes boundary facets from ordinary boundary transformations. In pinned
v6.2.2608 source, `Integral::T_Integrate` in `comp/postproc.cpp` does not dispatch
`ds(skeleton=True)` to a volume-facet integrator. The diagnostic therefore uses
the `LinearForm` path: `MakeLinearFormIntegrator` selects
`SymbolicFacetLinearFormIntegrator`, and `comp/linearform.cpp` selects the
adjacent volume element before calling `CalcFacetVector`. This is an API-path
check, not an alteration of the production method. Raw samples and ordinary
integration moments are saved before that extra control; errors in the extra
control are retained in its own file.

No scientific threshold is weakened. A completed diagnostic only means the
measurements were saved; the failed G1 result stays failed. A repair requires an
explicit interpretation, independent review, and a newly released job.

The batch uses the previously validated pinned private-prefix installation,
GPU/f64/highest preflight and one fresh directory. It reserves one H200, sixteen
CPUs, 64 GiB and twenty minutes in `gpu`. Mesh, volume-sample and boundary-sample
caps are in `diagnostic.json`; there is no matrix assembly, factorization,
network training or PDE solve. This is a CPU geometry diagnostic in a GPU
allocation, charged to the campaign budget. No timing comparison is permitted.

Staging takes only `diagnose.py`, `diagnostic.json`, `manufactured-smoke.json`,
`geometry.sh` and `REPAIR.json` from this directory, and the lane's unchanged
`dependencies.py`, `DEPENDENCIES.json` and `scalars.py`. Freeze these eight inputs
in a manifest plus `SOURCE_COMMIT.txt` before any submission. The current G2
`fom.py`, `reference_export.py`, and `shared_lift.py` are not staged or imported.

Static validation: Python compilation and `bash -n` only. Native NGSolve remains
unexecuted for this candidate. Coordinator release is required before submission.

## Glossary

G1: small runtime and analytic controls before reference qualification.
Mapped point: a point transformed from a reference element to the physical mesh.
Normal: a unit vector perpendicular to a boundary; its sign selects a side.
Jacobian: derivative of the element coordinate map.
Adjacent volume: the fluid element sharing the boundary's topological edge.
Closure: the integral of a normal over a closed boundary, which should be zero.
Private prefix: a job-local dependency installation that does not modify the shared environment.
