# Independent Codex A6 implementation review — 2026-10-28

Read-only numerical reviewer: **READY for a separately released bounded native integration
control**, with no blocking scientific or numerical source defect found. G2, native FE
integration, training, reduced stability and physical comparisons remain unpassed.

The exact reviewed ROM commit is `d1b4eb952abe0e454d06f3406a5cd26eab09e770`.
All 46 implementation-manifest entries match their hashes and commit bytes. Manifest SHA256
is `0ffc31e170487ce3f934a6e775fad19ddd88fbd221520749c9f2db540c23a4e9`; adapter digest
is `585136ba94515cb3e0c64ac4c18c1d8dde0bf04ffb50bc6c5675b146f242d146`.
The reviewer compared source with frozen A6, its reconciliation and the earlier Claude
findings, then reran both array-only suites with the prescribed interpreter, one BLAS/OpenMP
thread and disabled bytecode writes. Both passed, NGSolve was not imported, and the worktree
remained clean. No files were edited or jobs launched by the reviewer.

## Findings resolved in source

- `fe_pod_a6.py` uses the original sampled spectrum for rank decisions, pivoted mass-metric
  CGS2 with saved permutation/two passes, backward/Gram checks, then `Q @ U_R`. It avoids
  transport of snapshot combinations divided by small singular values.
- Centered pressure retains the uncentered scale, protects the normalized constant during
  both passes, and keeps that constant as the separately solved first mode. Actual evaluated
  pressure Gram matrices normalize supremizer and reference inf-sup calculations consistently.
- Full sampled/small-factor spectra, cluster cuts, pivot ties and registered-prefix residual
  comparisons implement A6. Zero velocity stops; zero centered pressure keeps only the constant.
  Aggregate POD optimality is distinguished from per-case consistency.
- `fe_adapter.py` expands scalar velocity eigenmodes through checked component maps, including
  local numbering, essential masks, sampled field/gradient embedding, diagonal metric blocks
  and cross blocks. Completeness of the scalar eigenspectrum is explicitly unverified.
- Nonsymmetric gradient controls use a sufficient auxiliary degree. Normal checks distinguish
  analytic orientation from closed-component closure. The existing raw-normal uncertainty
  will reject inputs rather than silently enter the adapter. Dimensionless outputs and
  physical-coordinate sidecars are explicit.
- Shared-lift conversion is consistent: pressure-per-physical-viscosity scales by $D/U$ and
  multiplies dimensionless viscosity $\nu/(UD)$, producing $p/U^2$. The natural-outlet pressure
  stays unshifted. Velocity scales by $1/U$.

The passing near-cutoff synthetic Gram values are saved in the lane's
`FE-A6-SYNTHETIC.json`; they are algebra controls, not obstacle-flow measurements.
Rejection fixtures cover zero velocity, cluster cuts, flipped obstacle normals, broken
closure, transposed gradients and component cross-coupling. The old quotient and initial
pressure-centering failures remain preserved.

## Remaining execution gate

Pinned-library controls must still verify curved-mesh serialization, component views/maps,
sampled-versus-assembled metrics, FE eigen/supremizer operations, shared-lift provenance and
resources. The full adapter requires genuine G2 training admission. This review does not
authorize substituting G1 data or inventing an admission to exercise it. A separate low-level
native fixture may test implementation primitives without claiming production data admission.

## Terms

CGS2: Gram–Schmidt with a second orthogonalization pass. POD: a weighted low-dimensional
snapshot basis. Gram matrix: inner products of basis functions. Supremizer: a velocity
enrichment paired with pressure stability. G2: independently admitted reference data.
