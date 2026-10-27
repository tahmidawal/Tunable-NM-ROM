# Reconciliation of independent design reviews

Status: design consultation completed; only implementation is released. No numerical gate
or result is accepted. Two independent Codex reviewers and Claude reviewed without reading
one another's opinions. Raw consultant response is retained; recommendations below were
checked rather than automatically adopted.

## Accepted changes

- Executable force convention is minus physical stress times fluid-outward obstacle normal;
  a known wall-shear sign and a manufactured curved-surface integral pin the sign.
- Gradient/symmetric stress are two discretizations of the same continuum obstacle force;
  their difference is a trace/divergence diagnostic. Weak reaction pairs with gradient stress.
- Global flux conservation is an algebra/quadrature diagnostic for Taylor–Hood with constants;
  local divergence and its refinement remain separate.
- Nontrivial manufactured tests must exercise the actual curved obstacle with nonzero
  prescribed velocity and consistent nonzero natural traction/forcing. This control has
  different boundary data from the physical no-slip problem and is labelled accordingly.
- Geometry order accompanies velocity order in primary h/p settings; independent geometry-
  only refinement checks prevent conflating geometry with field discretization.
- Tiny 3D factorization has an explicit conservative DOF/memory guard and cannot precede
  reviewed 2D smoke. Larger levels need measured memory projections; unresolved finest
  resolution stops reference acceptance, not silently changes the reference tolerance.
- A training-only interpolation competitor and conventional exact mixed POD-Galerkin are
  required alongside shared weak-LS POD and exact tensors. The parameter family is one-
  dimensional/interior; all claims are interpolation-only.
- Bank development representation is an early feasibility gate before online reduced solves.
  Exact input normalization, test enumeration and output-derived quadrature moments are
  frozen in code before fitting. Both methods retain equal opportunities and full outputs.

## Qualified or not adopted as assertions

Claude's Haberman–Sayre suggestion is a research lead, not a ready analytic-grade reference:
we have not established the formula, its flow/speed convention or error applicability to the
finite pipe. No unverified series or factor-of-two explanation is used. Barber–Emerson stays
auxiliary and source precision is not a reference-uncertainty bound.

Axisymmetry of the low-Re centered sphere is a plausible property to test with symmetry
and stability diagnostics, not a theorem inferred from the consultant. Full 3D fields remain
required. An independently qualified axisymmetric cross-check is desirable; until available,
reference uncertainty is conditional on h/p/MMS controls and cost claims are restricted to
the tested three-component FOM ladder, never best-possible sphere solving.

Exact quadratic reduced tensors have the same formal dimensions at equal retained ranks,
but numerical conditioning, accepted ranks and nonlinear iterations can differ. Therefore
Claude's assertion that cost differences can ONLY be reconstruction is too strong; report
actual matched-output work and timings without presuming either bank advantage or equality.

The proposal already specifies two hidden layers of width64; Claude's one-layer description
is a misreading. Do not change the fixed architecture because of it. Expected POD or bank
error guesses are not measured facts and do not enter the results.

Fitted quadrature uses decoded output/gradient/product moments and prescribed tests, never
assembled residual snapshots; boundary integration is separate. This respects the repository
instruction. Fitting net residual/tensor-error snapshots as a substitute is not authorized.
Actual operator and returned-solution checks remain independent acceptance tests.

No universal h-order band is assumed for force superconvergence on adaptive curved meshes.
Before reference acceptance the executable per-output empirical-order diagnostic and noise
floor must be frozen; nonmonotone or insufficiently resolved sequences stay unresolved.
Adding an arbitrary order after seeing results is forbidden.
