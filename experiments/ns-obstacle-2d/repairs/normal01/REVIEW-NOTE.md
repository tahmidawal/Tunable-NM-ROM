# Verified fluid-outward boundary orientation repair

This is an unexecuted G1 source candidate. It preserves the existing geometry,
finite-element spaces, equations, physical acceptance thresholds and exact
control-configuration bytes. `fom.py` and `preflight.py` start from the executed
smoke06 source, not the later unexecuted G2 producer.

The new `oriented_normals.py` retains NGSolve's curved boundary normal and
infers its orientation factor from the unique topologically adjacent fluid
triangle's mapped interior point. It rejects nonunit vectors, degenerate or
nearly tangential witnesses, and mixed signs under one boundary label. It never
selects a sign from a boundary's name or substitutes an analytic radial normal.

The candidate then checks the inferred normal against independent information:
outer-wall directions, the unchanged obstacle radial-sign requirement, closed
component sums, and ordinary boundary moments versus the actual volume-facet
`LinearForm` path. The divergence theorem checks the total oriented boundary.
Closure is only a supplementary check; it cannot select the obstacle's sign.
Raw samples are saved before inference, and failed certificates are retained.

Only ordinary boundary transformations receive the `BoundaryCF` sign factors.
The volume-facet control uses native normals without those factors, preventing
double correction. The public helper returns one ordinary-boundary coefficient
function, used by the FOM's manufactured outlet load, force integration, mass
flux, energy and momentum balances, and boundary quadrature exports. Exports
retain raw normals and their orientation factors beside the oriented normals.
The presolve mapped checks receive that same verified coefficient function.

The helper is deliberately limited to the declared fixed triangular
channel/circle meshes. Mapped interior-point direction can become ambiguous on
a distorted cell; ambiguity stops the helper. Constant sign across a label is
verified at the diagnostic quadrature points, not asserted for arbitrary
geometry. Independent analytic and volume-facet checks constrain the inference.
The native helper and `BoundaryCF` combination still require the reviewed run.

The helper's extra vector assemblies are reference/control setup costs. They
are not comparative query timings and must not be charged repeatedly only to a
future FOM benchmark when a validated common mesh can reuse its certificate.

All old G1 thresholds remain active, including the existing manufactured trace
comparison against its analytic target. This repair does not silently replace
that metric with a comparison against the prescribed FE trace; the separate G2
producer design addresses that distinction, and is not included here.

`G1-DIFF.patch` records the narrow changes to the old FOM/preflight. The other
source inputs are byte-identical to smoke06 except the private-prefix attempt
binding. `INPUT-CANDIDATE.json` freezes the eighteen proposed inputs. No job has
been staged or released. Local checks are syntax checks and saved-array replay;
the replay is not native repair execution or a physical result.

## Glossary

G1: runtime and small analytic controls before reference qualification.
G2: the separate reference/data-production gate.
FOM: the full-order finite-element flow solver.
BoundaryCF: a coefficient function that assigns values by boundary label.
Volume facet: a face integrated using the adjacent volume element's orientation.
P0: an elementwise constant finite-element space, used here only as an integration test.
Certificate: persisted observations and checks supporting the orientation choice.
