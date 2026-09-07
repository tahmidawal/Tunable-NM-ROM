# Burgers 3D: controlled decoder architecture comparison

Protocol prepared 2026-09-06 for the user-requested Codex subagent campaign.
This is a design, not a claim of improved accuracy. The user confirmed the
proposed implementation after the explicit base/name question. The four
worktrees `2026-09-06-b3d-anchor`, `2026-09-06-b3d-quadratic`,
`2026-09-06-b3d-encoder`, and `2026-09-06-b3d-mixture` branch from repaired
base `dd77383`, which includes the common evaluator and completed controls.
Each is assigned to a separate Codex subagent and cluster namespace. The
coordinator writes only in the existing repair worktree, apart from canonical
lab-log updates. No merge is authorized.

## Scope and fixed quantities

Keep the current learned spatial bank, $N=33$, $K=32$, $R=128$, training-family
seed, PDE, sign-upwind discretization and data membership fixed. Use the saved
refined checkpoint as the source of the bank and trained linear skip. Regenerate
all solution data from seed in each cluster job. No Gaussian descriptors enter
any model. Training uses exactly the original 8192 states from trajectories
0 through 511; validation is trajectories 512 through 575 at steps 0, 10, 25,
and 50. Final-test data remain closed during architecture selection.

The first campaign is a Burgers representation and tangent screen. Any wave
follow-up needs its own unchanged bank, phase-space metrics and rollout checks;
Burgers success alone does not establish reflective-wave accuracy.

## Exact coordinates and common initialization

Use the existing economy QR factorization $G=QR$. A model module produces
$q(z)$ in orthonormal-bank coordinates, with $h(z)=R^{-1}q(z)$. The loss identity is

$$
\|G h(z)-u\|^2=\|q(z)-Q^\top u\|^2+\|(I-QQ^\top)u\|^2.
$$

Compute the anchor $U$ by QR of the incumbent neural linear skip in these
coordinates, with deterministic column signs. It is not a POD of training
solutions. Let $b$ be the mean training coefficient vector and initialize

$$
z_i=U^\top(a_i-b),\qquad q_0(z)=b+Uz,\qquad a_i=Q^\top u_i.
$$

All candidates and controls must reproduce these same initial outputs and
assigned training codes, checked numerically and saved. Nonlinear inputs are
scaled by training-code RMS values; nonlinear outputs by the centered training
coefficient RMS. The affine $Uz$ path uses physical QR coordinates unchanged.
This is a new common initialization, distinct from historical warm refinement.

## Arms and controls

- **Protected head:** fixed $U$, trainable bias and a smooth two-hidden-layer
  width-128 residual, projected with $I-UU^\top$. Its Jacobian cannot lose latent
  rank, but the manifold is a graph over this specific anchor. Failure does not
  establish failure of every possible anchor; good rank does not certify dynamics.
- **Quadratic head:** trainable affine map and exactly one product $z_i z_j$ for
  each $i\le j$, with declared input/output normalization. Start its quadratic
  weights at zero. Independently check output, Jacobian and constant Hessian.
- **Shared coefficient encoder:** the width-128 control decoder plus an offline
  shared encoder of solution coefficients. Its initial encoder is the exact
  anchor projection, followed by an initially zero nonlinear correction. Optimize
  encoder and decoder jointly using training reconstruction. Direct encoder
  reconstruction is reported separately from the primary multistart fit.
- **Smooth mixture:** two independently initialized width-128 residual experts
  with zero output layers, a latent-only soft gate, and an affine path. Include
  routing derivatives. Save weights, saturation, entropy, expert disagreement and
  usage. An expert/routing collapse is an outcome to report.
- **MLP controls:** the incumbent function class, with trainable affine path
  and two hidden layers, widths 128 and 192. The latter supplies an approximate
  parameter-count control for quadratic/mixture arms. Counts are generated from
  the actual model; equal steps are not equal compute or proof of a structural gain.

All arms retain the same global relative field-MSE training objective. Use Adam,
60000 updates, learning rate $3\times10^{-4}$ with the existing warmup/cosine
schedule, batch 4096, and optimizer seeds 200 and 201. Match batch-key streams.
No code jitter, weight decay, auxiliary balancing loss, velocity loss or EMA
selection in this first comparison. Persist training curves and all checkpoints
before validation. Two optimizer repeats do not constitute independent data seeds.

## Common evaluator and acceptance

Model modules expose `NAME`, `MODE`, `init(key, shared)`, `apply(p, frozen, z)`
and independent `apply_np`. Encoder mode additionally exposes `encode(p,frozen,a)`.
Only `p` and, for free-code arms, training codes enter optimizer updates. Modules
never receive validation arrays through the training interface.

Use zero plus codes for the same seven deterministic training-state IDs as
primary starts, with 400 and 800 attempts. Numerical codes may differ between
architectures. Retain the minimum-error start even when it fails optimality.
Save every start's error, gradient, attempts and stop reason. Save partial budget
outputs before subsequent checks; failures must remain visible in the comparison.

Retain the inherited normalized-gradient gate. Additionally report the
coordinate-invariant projected residual

$$
\eta_T=\frac{\|P_{\mathrm{range}(J_q)}(q-a)\|}{\sqrt{\|q-a\|^2+\|(I-QQ^\top)u\|^2}}.
$$

Use an SVD with declared relative rank cutoff $10^{-10}$, no ridge. Report all
singular values and numerical ranks. Local fits are not global minima. Ambiguous
solver failures require an augmented QR/SVD check before blaming representation.

For tangent quality use the physical truth-state velocity

$$
v=\nu L u-N_{\mathrm{upwind}}(u),\qquad
e_T^2=\frac{\|(I-P_{\mathrm{range}(J_q)})Q^\top v\|^2+
\|(I-QQ^\top)v\|^2}{\|v\|^2}.
$$

Check this RHS against the backward-Euler snapshot identity at positive times.
This is a diagnostic, not a strong-form online objective. Report velocity norms,
absolute residuals and bank-only velocity floors; relative errors for zero
velocities are undefined and reported as null. Compare with the common MLP;
this screen does not claim a POD velocity comparator. Keep initial/later and
blob-count groups. Tangent quality alone does not certify curvature or rollouts.

The existing representation gates remain: mean at most 0.05, worst at most 0.15,
oracle/POD-K at most 0.5, normalized gradient at most $10^{-6}$, budget stability
and pool/full consistency. The actual inherited pilot must evaluate each new head
through its true function and independent NumPy path; no MLP distillation is
permitted to disguise an unsupported architecture. Rollout/cost promotion requires
all inherited gates and negative controls. Report medians and outlier counts.

`b3d_arch_pilot.py` loads a generic architecture checkpoint, verifies its model
source hash, and maps both JAX and independent NumPy outputs back through
$R^{-1}$. It is restricted to validation-only `PILOT=1, TRAIN=0`. The inherited
rollout kernel export assumes the old MLP checkpoint schema; a passing candidate
needs an explicit export/rollout extension before promotion. Adapter smoke tests
check nonzero nonlinear fields and their derivatives against the orthonormal
coordinate reference.
The adapter strengthens the inherited pilot summary by requiring every named
gate, every negative control, and mode stability before declaring a pass. The
inherited summary alone omits the D4 negative-control flag; its original value
is retained separately as `inherited_pilot_passed` for auditability.

## Execution and review

The user explicitly requested subagent Codex sessions. Each implementation agent
gets its own approved worktree and cluster namespace; common evaluator work stays
in the repair tree. With four slots including the coordinator, at most three
workers run simultaneously. The fourth implementation starts when a slot frees.
The coordinator reviews code, watches jobs through completion, verifies pulls and
produces a generated comparison. No merge occurs without the user's choice.

All real jobs use Tufts GPU/f64/highest precision, unique directories, explicit
backend preflight, code/metadata hashes, regenerated data and checked output pulls.
Delete each exact remote job directory after verification. Local numerical tests
are capped below one minute through `jaxrun`. Do not compare wall times across
jobs; training elapsed times are provenance, not online speed claims.

Independent Codex protocol and initial evaluator review is conditionally approved
with initialization, cohort, seed-uniqueness and component-test safeguards. Each
architecture implementation still needs its own code and mathematical review.

## Glossary

- **N, K, R:** grid nodes per axis, latent unknowns, and learned spatial features.
- **Bank:** spatial features evaluated on the interior grid; fixed in this screen.
- **QR:** orthogonal coordinates for that same learned bank.
- **Anchor:** fixed independent coefficient directions used for initialization and
  for the protected head's graph constraint.
- **Head / encoder / code:** coefficient-generating network, snapshot-to-latent
  network, and one snapshot's latent coordinates.
- **Graph:** a single-valued correction over the anchor coordinates.
- **Jacobian / Hessian:** first and second derivatives with respect to latent variables.
- **Tangent:** field changes available through the decoder's Jacobian.
- **Invariant stationarity:** residual component along available decoder directions,
  normalized by the full reconstruction residual.
- **SVD / rank cutoff:** singular-value factorization and its declared numerical
  threshold for deciding which directions are independent.
- **Oracle / multistart:** truth-assisted latent fitting from several guesses;
  it is unavailable as an online prediction method.
- **POD-K:** the inherited training-snapshot rank-K linear projection comparator.
- **Global relative field MSE:** total squared reconstruction error divided by
  total squared field magnitude on training states.
- **Optimizer seed / data seed:** randomness in fitting, and in generating PDE cases.
- **Weak objective:** PDE residual tested against smooth modes in the online solver.
- **Routing / saturation / collapse:** mixture weights, nearly exclusive selection,
  and failure to use distinct experts effectively.
- **Promotion:** permission within the protocol to run the next scientific stage
  only after all existing checks pass.
