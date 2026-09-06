# Burgers 3D: diagnose and improve the corrected validation pilot

Proposed amendment to `B3D-DESIGN.md`, prepared 2026-09-06. This document contains
an experimental protocol, not new numerical results. The original corrected
checkpoints and acceptance criteria remain the reference.

## Scope and provenance

Approved base: `exp/2026-09-04-separable-tensor-consolidated`, commit
`da479125b13aaa3e15b5cae33f72709705129838`. Work is isolated in
`exp/2026-09-06-burgers3d-repair`; cluster namespace:
`/cluster/tufts/paralab/tawal01/b3d_repair_20260906/`.

Start with the corrected scalar Burgers checkpoint at 33 nodes per axis,
$K=32$, $R=128$, $M=256$. The initial-condition family, boundary conditions,
sign-dependent upwind operator, viscosity distribution, time step, seed,
training membership and validation membership are unchanged. Physical family
descriptors are never inputs to the decoder head. No test table is generated,
staged or opened during diagnosis or training selection.

The initial reproduction uses the unchanged consolidated driver, which already
persists the validation rows omitted by the historical run artifact. Its
parameter table and all solution snapshots are regenerated on the cluster.
The checkpoint and staged sources are content-hashed; the original checkpoint
is never overwritten. A failed pilot is a valid recorded outcome and cannot
promote a rollout.

## Questions and exact decomposition

The decoder is $u=G h(z)$. An economy QR factorization $G=QR$ gives the exact
field reconstruction identity

$$
\|G h(z)-u\|_2^2
=\|R h(z)-Q^\top u\|_2^2+\|(I-QQ^\top)u\|_2^2.
$$

Check the numerical rank of $R$ with a small SVD. Rank deficiency is reported
and handled by the retained numerical range, with the discarded singular
values and cutoff recorded; it must not silently be hidden by a Gram ridge.
Use the same unweighted interior-grid norm as the original D4 gate. The
constant cell-volume weight cancels in relative errors on the uniform mesh.

For every validation state, record:

1. Unrestricted bank least-squares relative error: the spatial-span limit.
2. Legacy multistart latent-fit error, normalized gradient, attempts, and reason.
3. A more tightly converged latent-fit diagnostic using the exact coefficient
   objective, together with the same quantities and directly checked field error.
4. Snapshot index, trajectory index, blob count, and whether it is an initial state.

The exact normalized gradient is

$$
\eta=\frac{\|J^\top r\|_2}{\|J\|_F\|r\|_2},\qquad
r=G h(z)-u,\quad J=G\,D h(z).
$$

The constant orthogonal residual must be included in $\|r\|_2$. A small change
in the objective alone does not certify stationarity. Compare the independent
field and coefficient calculations on representative and difficult states.
Keep per-start outputs; report the lowest-error candidate's stationarity,
even if another, worse candidate is stationary. Do not select a worse solution
merely to make a gradient gate pass.

## Solver diagnostic

Reproduce the legacy ridge-Gram/LM result first. Then compare exact QR-space
fits using the original deterministic eight starts, with at most 400 and 800
attempts as necessary. Carry out budget checks for all nonstationary or
high-error states, plus the original four-state subset for historical parity.
Use double precision and a well-conditioned damped least-squares step; if a
normal-equation step loses accuracy, compare an augmented QR/SVD solve. Explicitly
report damping saturation, tiny rejected steps, and remaining nonstationarity.
Do not claim a global representation optimum from local convergence.

This diagnostic can improve the truth-assisted representation measurement;
it is not an online speed or rollout improvement. Any later online solver change
requires its own correctness and paired-cost checks.

## Decision before training changes

- If the unrestricted bank cannot support the unchanged target, report this
  before a head-only attempt. A bank-capacity/training amendment would need its
  own stated control, not a silent change to this comparison.
- If the bank is adequate and improved latent optimization passes the existing
  representation gates, validate the resulting fit procedure in the pilot and
  proceed to the rollout stage.
- If the bank is adequate but the head still misses the gates, the first candidate
  is frozen-bank head/code refinement in exact coefficient space. Keep $K$, $R$,
  the data split and head architecture fixed initially. Compare the incumbent
  with continued training on training-only states; preserve the original global
  relative reconstruction objective for the first control. Additional loss or
  architecture changes must be separately labeled and selected on validation.
- If the bounded refinement does not pass, publish the negative result and the
  unresolved representation gap. Do not open test data or a cost ladder.

## Acceptance and downstream work

### First bounded training control

After the bank/head diagnostic, run one warm head-and-code refinement for
60,000 Adam steps, learning rate $3\times10^{-4}$ with the existing warmup/cosine
schedule, batch size 4096, and optimizer seed 200. Use the same 8192 training
states and global relative field-MSE objective. Retain the existing two hidden
layers of width 128, the linear skip, and all spatial-bank parameters. Disable
weight decay, code jitter, EMA selection and code-polishing interventions for
this first control. Save the resulting checkpoint before validation; evaluate
the declared 400/800-attempt diagnostic with zero plus seven deterministically
selected updated training codes. Training never receives validation targets.
Record both training reconstruction and validation errors, and numerically
verify the output-coordinate conversion and bitwise-frozen bank.

All original D3/D4 criteria remain, including D4 mean relative error at most
$0.05$, worst at most $0.15$, oracle/POD-$K$ ratio at most $0.5$, exact normalized
gradient at most $10^{-6}$, pool/full checks and budget stability. The original
control cases must still fail as intended. Training membership includes every
component of the frozen bank and head, not only the newly trained component.

Report mean, median, 95th percentile and worst error overall and separately
for initial states, later states, and blob counts. Persist all per-state arrays.
The original test cohort remains closed until selection. Claims beyond this
cohort or training seed require fresh confirmation, not repeated selection on it.

After promotion, compare the full-grid weak, exact-linear sampled-advection,
and tensor paths on the same decoder. The tensor is a fixed-backward-stencil
operator: sign-upwind parity must be checked against decoded undershoots, with
the existing controls and thresholds. All classical comparisons use the cheaper
eligible rung of the two existing Newton/defect ladders, with cost and error
from the same invocation, GPU burn-in, paired orders and raw repetitions.
33-node rollouts precede the 65-node stage; 129 nodes remain behind the existing
memory and runtime preflight. The existing D4 and rollout gates apply at every
resolution. Jobs run sequentially in separate directories.

## Review status

The baseline reproduction follows the existing audited design and unchanged
source. This amendment is pending review. Independent review-agent permission
has been requested; no independent approval is implied by this document.

### Provenance amendment after the first diagnostic attempt

Exact byte equality between an archived parameter table and a regenerated table
is not portable for the derived viscosity exponentials and reference-grid peak
normalizers. The first diagnostic stopped at this check; its failed artifact is
retained. The raw random draws and membership must remain bit-identical. Only
the derived positive quantities $\nu$ and $s^\star$ may differ by at most eight
float64 machine epsilons in relative value. Compare against an archived table
whose content hash matches the checkpoint's recorded hash, and persist both
hashes and per-field discrepancies. Negative controls perturb a raw parameter
and a derived parameter beyond the permitted rounding bound. The archived table
is provenance metadata; solution data and the table used to generate it are
still regenerated on the cluster. This changes a provenance compatibility check,
not the representation or rollout acceptance thresholds.

## Glossary

- **K:** number of latent unknowns solved online.
- **R:** number of spatial features output by the coordinate network.
- **M:** number of smooth modes used to test the PDE residual.
- **Bank:** the matrix of spatial features evaluated on the interior grid.
- **QR / SVD:** matrix factorizations used here to diagnose the represented
  spatial range and evaluate its error accurately; neither replaces the learned bank.
- **Head:** the neural map from latent variables to bank coefficients.
- **Oracle / latent fit:** a truth-assisted reconstruction diagnostic, unavailable
  as an online predictor because it uses the target solution.
- **Stationarity / normalized gradient:** a check that local optimization has
  approached a stationary point; it does not certify a global minimum.
- **D3 / D4:** the inherited operator-rank and held-out representation checks.
- **POD-K:** the best rank-K training-snapshot projection used as a comparator.
- **Held out:** excluded from every model-fitting component and used only for
  the declared validation or final-test purpose.
- **Promotion:** allowing the next experimental stage after all required checks.
- **Rung:** one fixed classical solver/tolerance configuration in a comparison.
- **Weak residual:** the PDE residual tested against smooth spatial modes.
- **Tensor:** a stored quadratic reduced operator for a specified discrete stencil.
