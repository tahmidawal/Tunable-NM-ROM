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
checks. Check learned and POD initial fits independently at all three declared
tolerances and both input paths. Initializer success cannot qualify a failed
rollout or reference. No classical performance campaign or new speed claim is
part of this work; prior cost failures remain explicitly unresolved.

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
