# Pipe–sphere normal diagnostic candidate

This source-only candidate follows the first measured 3D normal failure. It is
not staged or released. It regenerates the same CSG pipe-minus-sphere recipe,
mesh sizes and curvature as the failed control, saves the new mesh before
orientation observations, and keeps every raw sign. The failed run did not save
its mesh, so equality of the regenerated mesh is not asserted.

The diagnostic records mapped coordinates, normals, face Jacobians, quadrature
weights and topologically adjacent fluid tetrahedra. It checks point location on
both sides of each boundary using fixed offsets and records analytic membership
separately. These observations identify which side contains fluid without
assuming that a geometry-face normal points outward from that fluid.

It compares three paths on the unchanged geometry:

- Sums of raw mapped boundary samples.
- Ordinary `Integrate` on boundary elements.
- A constant test in a discontinuous scalar space contracted with an assembled
  `LinearForm` using `ds(label, skeleton=True)`. The constant is verified at
  volume-element centers. The same procedure with `dx(element_boundary=True)`
  gives an additional sum over every fluid-element boundary.

The third path is adapted from the independently reviewed 2D diagnostic at
commit `09a660f7b`. The pinned library source dispatches this linear form through
the adjacent volume element. Plain `Integrate(cf*ds(skeleton=True))` follows a
different path in this version and is not substituted as independent evidence.
The official [DG documentation](https://ngsolve.org/ngsolve/docs/i-tutorials/unit-2.8-DG/DG.html)
describes exterior facet integration; the developer's
[normal-orientation discussion](https://forum.ngsolve.org/t/contact-normal-vs-specialfc-normal/3117)
is a source-level lead, not proof of the current CSG result.

The aggregate controls include boundary measures, normal closure, signed sphere
radial cosines, and the divergence theorem

$$\int_{\partial\Omega} x\cdot n\,ds=3|\Omega|.$$

Fluid volume for that identity comes from a separate high-order integral.
Mapped volume samples establish locations and Jacobian observations; any signed
library quadrature weights are counted and retained. They are not a fitted
positive rule or an admitted ROM integration rule. Closure alone cannot fix a
normal's sign. Saving `mesh.vol` also does not establish curved reload parity.

Raw samples and ordinary integration results are saved before the additional
facet control. If that control raises, its exception and partial results are
retained separately. A completed diagnostic admits no PDE result and changes no
scientific threshold. No sign correction, nonlinear solve, matrix factorization,
training or reference admission occurs.

The candidate retains the reviewed private-prefix installer and the exact
failed-run configuration bytes. Its batch keeps one GPU, sixteen reserved CPUs,
four requested FE threads, 64 GiB and twenty minutes, with the same GPU/f64/highest
preflight and bounded process timeouts. Additional element/point caps appear in
`diagnostic.json`. This is CPU geometry work inside a charged GPU allocation;
there is no performance comparison.

Before any staging, independently review this source and its configuration.
The prospective staging set is `diagnose.py`, `diagnostic.json`, the frozen
`g1-smoke06.json`, `geometry.sbatch`, and unchanged lane `install_deps.py`,
`DEPENDENCIES.discovery.json`, `requirements.lock`. Bind them and
`SOURCE_COMMIT.txt` in an input manifest in a fresh `g1-normal01` directory.
No job is released by this README.

## Terms

CSG means constructing a fluid region by intersections and subtraction of solids.
A mapped point is a reference-element point transformed to physical coordinates.
A face Jacobian is the derivative of that transformation on a boundary.
The adjacent tetrahedron is the fluid element sharing the boundary triangle.
A constant P0 test is a piecewise constant function set to one on every element.
Normal closure is the integral of a normal over a closed surface. A diagnostic
records behavior; reference admission requires the separate scientific gates.
