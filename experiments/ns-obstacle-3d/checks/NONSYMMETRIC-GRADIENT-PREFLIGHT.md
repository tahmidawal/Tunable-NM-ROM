# Nonsymmetric affine gradient check — candidate g1-smoke06

This focused check follows the independent Claude/Codex adapter review recommendation. It does not respond to an observed gradient error. Candidate `g1-smoke05` remains staged and unsubmitted, with its exact source and manifest retained. Candidate 06 has not been staged, installed or executed.

The identity fixture cannot distinguish a gradient from its transpose. The added independent grid function uses

$$u=(y,z,x),\qquad G_{ij}=\partial_j u_i,
\qquad G=\begin{pmatrix}0&1&0\\0&0&1\\1&0&0\end{pmatrix}.$$

The expected matrix is nonsymmetric and is written explicitly in the preflight. The same function is evaluated at the existing volume point using `Grad`, and at the existing boundary point using `BoundaryFromVolumeCF(Grad(...))`. Both matrices must be finite and their Frobenius error must satisfy the existing preflight absolute tolerance. Each result retains the expected matrix, evaluated matrix, error and row/column convention. Transformations stay alive through all point evaluations.

The cyclic field is a separate grid function. The identity-gradient fixture, its per-face traction and power checks, all scalar-return checks, and the actual control/physical equations remain unchanged. No threshold, numerical configuration field, mesh, package pin or resource guard changed. Python/shell parsing and exact configuration comparisons passed; runtime confirmation remains gated on 2D G1 and explicit release.

`g1-smoke06.SOURCE.diff` is the precise solver-source diff against receipt commit `7a28965ad7a74d8c454191778246a481aaa02a0f`, whose solver source is the reviewed/staged `7a5a6e4277aa3c532c4b0d19c20c3cf118f1f18b` source. New candidate config/batch files and a fresh dependency-prefix binding identify the prospective `g1-smoke06` directory. The previous binding is retained in `g1-smoke05.DEPENDENCY-REPAIR.json` as well as the unchanged staged package and Git history.

## Glossary

**Affine field:** a function linear in coordinates plus an optional constant. **Gradient convention:** whether rows or columns correspond to velocity components. **Transpose:** swapping matrix rows and columns. **Frobenius error:** square root of the sum of squared matrix-entry errors. **Boundary-from-volume evaluation:** evaluating the fluid-side volume derivative at a boundary point.
