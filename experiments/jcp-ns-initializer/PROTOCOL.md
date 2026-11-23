# Navier–Stokes initializer repair and untouched confirmation

This is a prospective development protocol. No new confirmation outcomes exist.
The original November 19 evidence and source remain unchanged in their directories.

## Development v1

Use all exposed original-development cases (seed `202609202`, first 16) and all
exposed November 19 cases (seed `2026111901`, 16). Both are development data for
this repair. Evaluate learned and POD heads, each at normalized stationarity
tolerances $10^{-7}$, $10^{-9}$ and $10^{-11}$. Repeat the physical-input paths:
phase uses `F.initial`, query uses the additional spectral projection returned
at time zero by the full solver. This is 384 selected-fit checks, with no omitted
case and no new training. POD is regenerated using the original pinned recipe.

The original objective is
$$f(z)=\tfrac12\|H(z)-a_0\|_2^2,$$
with the same target, four nearest training-code seeds, model and f64 arithmetic.
Both inherited seed selectors must agree with the shared four seed identities.
Original host phase LM and original compiled query LM are distinct, unchanged
controls in the same allocation. The compiled control is a standalone replay
of the legacy LM core, not a claim of bitwise replay of whole-query compilation.
Their original stop reasons and a common
fresh evaluation are retained separately; historical cross-job gradient
differences are not treated as exact reproduction.

For each start, preserve the compiled baseline endpoint, then retry at most
40 LM iterations using augmented QR instead of normal equations. Save that
stage separately. If needed, perform at most 12 SVD Gauss–Newton polishing
iterations, each with at most 8 trials $\delta/2^j$, $j=0,\ldots,7$. Use the
scaled-Jacobian cutoff $\max(m,k)\epsilon_{64}\sigma_{\max}$ and the step cap
$10^{-6}(1+\|z\|_2)$. Save every singular value, cutoff, retained rank,
proposal, code, acceptance reason and stage termination. Truncation is never
a stationarity certificate.

The original step acceptance and selector are unchanged: the only roundoff
allowance is $1024\epsilon_{64}$ times objective scale, requiring at least a
factor-two gradient reduction and the original small-gradient/small-step
conditions. The per-start final endpoint also cannot exceed its original
baseline objective plus that single original band; repeated small increases
cannot accumulate. The final selected objective must also stay inside the
same band of the original raw minimum. Stationary alternatives outside the
original selector band cannot replace a nonstationary minimum.

Every final residual and analytic Jacobian is freshly evaluated from the
returned code and original target. Qualification requires the original
normalized gradient
$$\max_j\frac{|J_j^T r|}{\max(\|J_j\|_2,10^{-30})\max(\|r\|_2,10^{-30})}
\leq \texttt{gtol}.$$
The returned and original endpoint residuals/Jacobians are retained in small
arrays for an independent arithmetic audit. Same-evaluator costs, stable
residual-identity objective differences, old/new physical-field differences,
all control outcomes and all unsuccessful repair attempts are retained.
Selected fits must pass; every unselected local start need not converge.
External initializer qualification also checks the returned objective against
the minimum of all four original host-phase endpoints under that same fresh
evaluator and unchanged band. Actual reconstructed physical initial fields,
not merely coefficient distances, must agree across all three tolerance pairs
within $10^{-6}$ of the original physical initial-field norm. Old/new initial
field agreement uses that same bound. Solver pass and full initializer
qualification are reported separately; failure of these external checks is
retained even if the selected gradient passes.

The first registered implementation changes linear algebra only. If it fails,
retain its results and register any arithmetic amendment separately before
running it. Never widen a gate, discard a case or train a new model to obtain
qualification. Host orchestration and all fit work are accounted explicitly;
no timing or speed claim is made.

## Planned untouched confirmation, not yet opened

Reserve seed `2026112301`, 32 original-family cases at $N=64$, horizon $0.2$.
Before generating new inputs/outcomes, freeze and independently review the
source, protocol and configuration. Check exact and rounded physical tuples
against known training/development/exposed cohorts, including training seed
`202609201` (512), development seed `202609202` (64), and November 19 seed
`2026111901` (16). Historical sealed cohorts remain closed.

Use the frozen learned setting $M=584$, $\Delta t=0.01$ for head/common-span/
native-span trajectories. Keep original rollout, reference and refinement
checks. A prospectively fixed new reference contract uses $\Delta t=0.000125$
and the $0.0000625$ refinement on every case, with unchanged absolute and
fractional gates. The old November 19 reference and qualifications remain
unchanged. Check learned and POD initial fits independently at all three declared
tolerances and both input paths. Initializer success cannot qualify a failed
rollout or reference. No classical performance campaign or new speed claim is
part of this work; prior cost failures remain explicitly unresolved.

## Development v2: fixed-anchor arithmetic amendment

This prospective version retains the complete v1 result and its unsuccessful
QR/SVD proposals. V2 combines both fixed legacy implementations and stable
objective arithmetic; no ablation is claimed to isolate their effects.

Run both original phase and compiled-core solves for each of the same four
starting codes: four seed identities, eight control solves. Freshly evaluate
their ordinary-direct objectives on the same GPU. The smallest chooses a
single fixed anchor for that fit and tolerance; ties use original seed order,
then phase before compiled core. This rule is tolerance-dependent and is never
changed or re-anchored during polishing or endpoint comparison.

Evaluate the same residual independently from each trial code using
$$r(z)=H(z_*)-a_0+\Delta H(z_*,z-z_*).$$
This is a fresh evaluation, not an accumulated residual. For each SiLU layer,
with $s=\sigma(a)$, compute the exact real identities
$$\Delta\sigma=\frac{s(1-s)\operatorname{expm1}(d)}{1+s\operatorname{expm1}(d)},
\qquad \Delta\operatorname{SiLU}=d\sigma(a+d)+a\Delta\sigma.$$
Use these local formulas only for $|d|\leq0.5$; substitute a bounded zero in
the unused exponential branch and use ordinary SiLU subtraction outside this
range. This prevents overflow or small-denominator saturation in unused
branches. Propagate increments through both existing affine/SiLU layers and
the unchanged linear skip. No parameters or model architecture change.

Reevaluate all eight original endpoints under this one anchored objective.
For each start, the original selector chooses between its two endpoints; use
the minimum of both as its unchanged endpoint-quality bound. The global
returned bound uses the minimum over all eight controls. QR/SVD budgets and
acceptance bands remain those of v1. Apply the original selector to all final
start endpoints; there is no optional-stage rollback and no substitution of
a stationary point outside the unchanged band.

Both anchored and ordinary-direct residuals, analytic Jacobians, costs and
normalized gradients are retained for every original and returned endpoint.
Final qualification requires **both** f64 gradient tests at the same original
tolerance, actual physical old/new and tolerance-pair agreement, and objective
quality. Save ordinary-direct objective failures as separate flags; they are
never silently relabeled as passing the original f64 arithmetic.

Check the stable increment against direct values at fixed perturbation scales
$10^{-2},10^{-4},10^{-6},10^{-8},10^{-10},10^{-12}$, with scale-relative parity
at most $10^{-12}$. The smallest-step first-order remainder must be at most
$10^{-7}$, the $10^{-6}$ remainder at most $10^{-3}$, and central analytic-J
action agreement at step $10^{-5}$ at most $10^{-6}$. Run these fixed checks
at every exposed fit anchor and on both models in the fixture. Retain all
values for all eight coordinate directions plus one fixed sine direction;
they are evaluation-consistency checks, not loosened solver gates.
Saturation fixtures cover preactivations from $-1000$ to $1000$ and both local
and nonlocal increments, including both sides of the $\pm0.5$ branch cutoff.
Record both layer preactivation ranges for every control and returned code;
require finite absolute values below 600 for safe independent host exponentials.

An independent **cluster-host arithmetic audit** evaluates the original
network directly at all eight control endpoints and the returned endpoint,
using NumPy long double with at least 63 explicit mantissa bits. Record actual
precision and objective strings. This is separate from the primary f64 GPU
method and physical references. If the ordinary-direct objective guard fails,
or the long-double objective difference is within $0.1$ original band of its
threshold, evaluate all those endpoints again at fixed 50-digit Decimal
precision. `Decimal.from_float` converts every stored f64 parameter, code and
target exactly. Save all objective strings and differences. The unchanged
$1024\epsilon_{64}$ band applies to the independently evaluated original
objective; long-double and Decimal objective differences must agree within
$0.01$ band when Decimal is triggered. Both independent objective guards must
pass. A direct f64 ranking failure is attributed to evaluation rounding only
when this independent evidence supports it.

V2 runs the entire declared exposed matrix again. No v1 result is overwritten,
no historical sealed cohort is read, and no new-case outcome is opened until
all development qualification and independent review are complete.

## Execution and records

Use only the cluster GPU partition, a unique job directory, f64/highest
precision and mandatory GPU preflight. Development jobs have a 20-minute cap;
the planned fresh trajectory job has a 30-minute cap. Record actual scheduler
usage. Commit and mirror every attempted implementation before running it.
Pull outputs with hashes before cleanup. No scientific arrays are generated
on the local workstation beyond sub-minute implementation fixtures.

## Glossary

- **Selected fit:** the returned result after fitting the same four starting codes.
- **Stationarity:** the original normalized residual-gradient stopping test.
- **Objective band:** the fixed allowance for floating-point objective comparisons.
- **QR/SVD:** factorizations that solve least-squares steps without forming normal equations.
- **POD:** the fixed snapshot-based comparison basis and its existing trained head.
- **Phase/query input:** respectively the physical initial field and its extra solver projection.
- **Exposed development:** cases whose prior outcomes have already been inspected.
- **Untouched confirmation:** a cohort withheld until source and checks are frozen.
