# Unresolved curved-volume facet identity

The retained geometry diagnostic completed, but its independent fluid-boundary
identities and the existing geometry-volume gate failed. The source evidence
below motivates measurements; it does not establish folded cells or admit a
normal repair. The generated measurements are in `NUMBERS.generated.md`, with
full provenance and per-boundary observations in `../g1-normal01.ARRAY-AUDIT.json`.

The ordinary boundary path and the volume-facet path agree on boundary area.
They differ in wall normal moments and reverse the sphere orientation. Even the
sum over every element boundary differs from the exterior-facet sum. Ordinary
boundary moments stabilize at the retained higher orders, but that observation
does not establish adequate quadrature for either volume-facet path. Their face
rules use different vertex permutations, and no facet-order ladder was run.

In pinned `ngsolve-intrule.cpp`, `MappedIntegrationRule` constructs its mapped
points, calls the batched Jacobian transformation, and then computes facet
normals when the integration points carry a facet marker (lines 465–488).
`ComputeNormalsAndMeasure` then uses (lines 522–535)

$$c=\operatorname{cof}(J)n_{\mathrm{ref}},\qquad
  n=\operatorname{sign}(\det J)\frac{c}{\|c\|},\qquad
  dS\text{ scaling}=\|c\|.$$

The implementation sends a zero determinant through the negative branch.
The determinant and measure are distinct fields in `ngsolve-fem__intrule.hpp`;
changing the measure does not by itself overwrite the Jacobian or determinant.
Consequently, a determinant sign change at a curved facet point could change
the normal while preserving positive surface measure. This matches the *kind*
of discrepancy observed, but no determinant at those facet points was retained.
The positive saved interior samples cannot rule out a sign change on a facet or
between those samples. Other mapping, point-permutation, cofactor or scalar versus
batched evaluation differences also remain possible.

The exterior boundary linear form is dispatched through the adjacent volume
element by the retained `diagnostics/normal01/source-evidence/ngsolve-comp__linearform.cpp`
(lines 442–481). Its facet integrator constructs `Facet2ElementTrafo` with global
vertex numbers, hence sorted triangle vertices, in the retained
`ngsolve-fem__symbolicintegrator.cpp` (lines 3790–3838). The all-element-boundary
linear form instead uses the topology's default face ordering (lines 1035–1100).
Their ordinary triangle rule has order `2*P0_order + bonus_intorder`. The exact
triangle reference points are consequently not interchangeable under an assumed
arbitrary face permutation. The mapping and sorting are specified in
`ngsolve-fem__intrule.hpp` (lines 916–1120).

The saved interior samples use the tetrahedron rule selected at the configured
sample order; the higher-order scalar volume uses a different rule. The source
rule counts and the runtime sample counts are distinguished in the array audit.
The aggregate native reference points and weights were not exported. Native
forms therefore need direct determinant coefficient-function controls, alongside
an independently reconstructed point trace, before scalar mapped-point samples
can be treated as evidence about their path.

The binding in retained `ngsolve-fem__python_fem.cpp` (lines 502–505, 553–567)
exposes `specialcf.JacobianMatrix(3)`. Evaluating its determinant inside the same
linear forms can measure the native map at their actual points. A direct scalar
trace must be checked against those assembled controls. Plain
`Integrate(cf*ds(skeleton=True))` is not a replacement: the pinned postprocessing
implementation retained with the executed diagnostic follows the boundary
transformation path in that case.

The offset point-locator observations disagree with analytic membership on
some sphere and port offsets. They are preserved in the generated table and
must not be used as clean fluid-side or mesh-validity certificates. Topological
fluid adjacency is a separate recorded witness. The earlier failed smoke mesh
was never saved, so this later diagnostic does not establish its exact identity.
The later saved mesh contains curved-element data, but reload parity remains
untested; its file alone does not certify the earlier in-memory mapping.

## Terms

A cofactor is the signed-area transformation derived from a Jacobian matrix.
The Jacobian is the derivative of the reference-to-physical element map.
Orientation is the sign of its determinant. A facet rule is a list of reference
triangle points and weights. A face permutation changes how those points are
embedded into a tetrahedron. A coefficient function is a native expression
evaluated at an integrator's points. A hypothesis is an explanation requiring
new measurements; it is not an accepted result.
