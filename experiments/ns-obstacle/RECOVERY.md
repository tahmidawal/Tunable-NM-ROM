# Recovered evidence and implementation inventory

Status: discovery only; no new obstacle-flow numerical claims. Source hashes and lane heads
are saved in `state/recovery-sources.json`.

The latest completed Burgers refinement, not the earlier broad qualification headline,
is the controlling comparison evidence: all tested admitted bank settings have faster
jointly accuracy-matching spectral comparators. Tuned unsampled POD remains admitted.
Its failed fitted supports do not establish either physical-error loss or bank advantage.
The improvement campaign's holed geometry measured a loss, while its L-shaped result had
unresolved reference cases. Continuous POD invalidated the claimed comparative multi-mesh
cost advantage. Geometry/spectral investigations include hypothetical cost scenarios and
must not be promoted to measured obstacle-flow evidence.

Periodic NS2D and NS3D lanes demonstrate useful implementation patterns and record both
successes and failures. NS2D's nonlinear head did not establish the intended advantage over
POD; an older solved-versus-manifold summary compared mismatched statistics and was corrected
in the canonical log. NS3D's co-moving frozen curl bank had qualified narrow-setting outputs,
but its representation was less accurate than mesh POD and its cross-mesh floor gate failed.
Those results do not validate stationary-wall geometry, inlet-driven balance or force recovery.

| Component | Reuse decision | Missing work / risk |
|---|---|---|
| f64/JAX setup, seeded provenance, arrays and timing ledgers | reuse pattern with fresh source hashes | retain backend preflight and distinguish CPU FE kernels |
| coordinate MLP differentiation and QR/SVD ordering | adapt algorithm, not trained weights | train on admitted obstacle snapshots; preserve physical volume inner product |
| periodic curl/Leray NS3D solver and co-moving shifts | source study only | wrong boundaries and pressure/output contract |
| periodic NS2D vorticity/FFT inverse | source study only | no inlet/no-slip/outlet pressure solution |
| Burgers envelope, sine tests and box quadrature | no direct reuse | curved domain, weak mixed system, boundary integration |
| learned-bank weak least squares | adapt with fixed nondimensional row metric | mixed pressure stability, Jacobian and conditioning checks |
| POD basis construction | adapt weighted construction | identical lifting and pressure/supremizer support; exact tensor control |
| fitted positive quadrature | method only | fit on training output moments; independent returned-state tests |
| old force/pressure reference | absent | new body-fitted high-order reference required |
| publication/export tooling | reuse formatting only | new generated values and fresh review; preserve original presentation |

First compute is mostly new reference verification. Body-fitted NGSolve/Netgen is preferred
to writing a new incompressible FE framework. Pinned wheels can live outside the shared venv,
but compatibility is not yet verified. Full 3D sparse factorization memory is the largest
resource risk; the declared smoke and memory stops precede training.

Discovery drafts are recommendations, not competing active designs. The central DESIGN.md
and its dated amendments become authoritative after review. In particular the draft 3D
lane's longer domain and alternative numerical thresholds are not silently combined with
central settings. Source benchmark values are external anchors; central gates remain
prospective until their independent review.

## Glossary

Admitted: passes the declared numerical checks in its stated scope. FOM: full PDE solver.
POD: weighted linear basis from snapshots. MLP: coordinate neural network. Curl/Leray:
periodic constructions enforcing divergence constraints. Lifting: boundary-data field.
Supremizer: pressure-compatible velocity enrichment. Quadrature: numerical integration.
Weak least squares: minimize tested PDE residuals. Reference: checked approximate solution,
not exact truth. Code-only mirror: GitHub backup excluding large archive data.
