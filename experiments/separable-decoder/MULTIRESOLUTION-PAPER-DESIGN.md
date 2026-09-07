# Four-PDE multiresolution study for the NM-ROM paper

Proposed protocol dated 2026-09-07; no new experimental results are reported here.
Architecture choice, experiment bases/names and compute budget await the user's
answers. The canonical repository-root `LAB-LOG.md` remains the project record.

## Question and scope

Measure the accuracy–cost tradeoff of the method on scalar two-dimensional waves
(absorbing and fixed-wall reflective), Burgers, Poisson and heat as the mesh changes.
The primary deliverable is the cheapest validated configuration meeting each
accuracy target, together with failures and uncertainty. Publication readiness
requires a supported methodological contribution as well as these experiments;
more PDE names alone do not establish novelty or guarantee acceptance.

Default proposal: the continuous-coordinate separable decoder is the main method;
the older ViT + CP method is a corrected comparator. This is explicitly pending
because the canonical chronology records interest in both architectures. Do not
substitute the older method silently or inherit its headline numbers as verified.

Keep the current localized data style and exclude Gaussian-descriptor inputs to
the head. Preserve each PDE's physical parameters and initial/source-field draws
across resolutions. The initial wave family is the fresh verified single-bump
family. Multiple-bump wave data would be a separately labeled extension requiring
new reference checks; do not silently expand this campaign's data distribution.

## Two distinct experiments

**Frozen-weight transfer.** Choose a training mesh using validation, freeze both
spatial-network and coefficient-head weights, then solve on unseen meshes. Keep
latent, bank and test-space dimensions fixed in this comparison. Evaluate the
coordinate network on the new mesh and rebuild discrete operators. Refit any
mesh-dependent quadrature from decoder outputs, with its cost disclosed. New-grid
truth may be used for evaluation, never to tune the frozen head or select weights.
Displaying the same solution at more pixels is not a transfer experiment.

**Per-resolution accuracy–cost optimization.** Allow separate training and a
bounded search over latent dimension, bank size, head size, test modes, solver
tolerance and applicable quadrature budgets. Report trained parameters, storage,
iterations and offline cost beside each point. This answers the best attainable
tradeoff within a declared search budget; it does not establish frozen-weight
transfer. Include the fixed-capacity study so capacity growth cannot masquerade
as flat complexity.

Proposed mesh ladder: 64, 128, 256, 512 and 1024 **intervals per axis** on the fixed
physical domain. These are proposed settings, not measured coverage. Record total
nodes, interior unknowns and boundary unknowns explicitly; older drivers use
different meanings of `N`. Start with a representative resolved mesh and an
adjacent transfer mesh per PDE. Expand the ladder only after reference checks and
pilot costs are known. A coarse mesh can be a legitimate inaccurate test case;
it cannot serve as fine reference truth for the same comparison.

Proposed accuracy targets are relative errors 0.1, 0.05, 0.01 and 0.001. These are
plotting/selection targets, not promises that any model reaches them. Require
reference uncertainty comfortably below the target (proposed budget: one tenth
of the target); otherwise label that target unresolved. Preserve historical gates
and report new criteria separately rather than revising old verdicts.

## Reference and baseline work

| PDE | Starting point | Required comparison and first check |
|---|---|---|
| Waves | Fresh wave branch and its frozen mathematical source | Re-establish spatial/time accuracy on the mesh ladder. Use independent standing-mode checks for fixed walls and boundary/energy/refinement checks for the absorber. Compare compressed heads with matching-dimensional affine/POD models and the full learned bank. Current compressed heads miss the full target. |
| Burgers | Consolidated separable/exact-linear implementation | Use tolerance-adaptive FOM solves with the same upwind discretization. Keep the sampled weak nonlinear operator as the general path. Audit tensor-versus-upwind differences on actual decoded states before claiming exactness; decoded negative values can invalidate the positive-field tensor identity. |
| Poisson | Consolidated quadrature-free exact weak operator | Generate discretely consistent truth. Include the fast direct sine-transform solver for the current constant-coefficient rectangular problem, plus a properly preconditioned iterative solver where useful. Unpreconditioned CG alone is not the primary speed baseline. |
| Heat | Separable bank infrastructure plus an explicitly verified heat port | Reproduce analytic eigenmode decay and actual state advancement, verify space/time refinement, and compare with efficient transform-based propagation/implicit solves for the declared operator. Public frozen-rollout code is not a behavioral reference. |

Heat's corrected public branch and the frozen archived heat implementation are
read-only reference material, not interchangeable algorithms or automatic sources
of accepted speed numbers. The separable heat port must have its own verification.
The old wave evidence reset remains in force.

Include same-bank affine heads and fresh POD at matched latent/state dimensions,
then include the unrestricted bank as a larger-capacity diagnostic. Record head
parameter counts; equal update counts are not equal compute. If the older CP
method is included, correct its rollout/data inconsistencies, distinguish its
residual from the new weak residual, and label any interpolated axis-factor mesh
transfer separately from newly trained CP factors. Avoid a comparison that changes
the bank, residual and optimizer together without controlled ablations.

## Accuracy accounting

Separate unrestricted bank projection, best recorded multistart reconstruction,
tangent quality, reduced-operator approximation and autonomous evolution. A
stationary local fit is not a proven global optimum. For every resolution record
both ROM-versus-same-grid-FOM discrepancy and error against a common independently
refined physical reference; the former alone can hide discretization error.
Define the common observation grid/norm and validate any restriction or
interpolation used in cross-mesh comparisons.

For evolutionary PDEs, report time-maximum errors per trajectory and then cohort
means, medians, worst cases and threshold-failure counts. Record time-integrated
errors separately if useful. Hold final time and output times fixed across grids;
choose time steps by accuracy/stability rather than comparing different horizons.
For Poisson, aggregate per source field. Preserve initial-state normalization for
comparison to existing wave results, and additionally report current-field relative
and absolute errors. Heat and absorbing waves need explicit vanishing-amplitude
flags. Wave displacement, velocity, energy-state error, phase and mean-field checks
remain separate. Neither a small weak residual nor a small energy-balance defect
certifies small full-state error.

Use validation for configuration selection, then freeze choices before evaluating
new independent final cohorts. Do not consume the existing sealed wave cohort
during pilots. Plan independent data/training repeats for confirmation rather than
reporting only optimizer repeats on one split; the number and budget will be fixed
after the pilot and before confirmation. Retain failed and nonstationary cases.

## Cost accounting

Persist timing repetitions, physical errors, convergence status and configuration
from the same solver invocations. Interleave ROM/FOM timing within one job on one
GPU, with GPU clock burn-in, compilation warmup and completed-device synchronization.
JAX dispatch is asynchronous, so timers must wait for results; compilation and
transfers must be accounted for explicitly. See the official
[JAX benchmarking guidance](https://docs.jax.dev/en/latest/benchmarking.html).

Report separately:

- Offline training, reference-data generation, compilation, operator assembly,
  quadrature fitting and any work recurring on a new mesh or new physical parameter.
- Reduced-solver time, including its actual iteration/timestep counts.
- Complete query time: input handling, source projection/initial latent fitting,
  evolution/solve and the requested physical output, including required transfers.
- Memory and storage for parameters, banks, operators and reconstructed fields.
- Single-query latency and separately labeled batched throughput, with the same
  batching/output contract offered to classical baselines.

Dense field output necessarily grows with requested output size; a flat latent
solver does not imply flat total query time. If a reduced-output use case is also
measured, label it explicitly and give all baselines the same output request.
Calculate amortization only where the ROM has positive per-query savings and show
the number of queries needed to repay incremental offline cost.

For each target, compare with the cheapest validated classical solver meeting the
same physical accuracy and output requirement. Include both same-mesh comparisons
and a cost-to-accuracy envelope allowing the classical solver to choose an adequate
coarser mesh and produce the requested output. Do not force it to over-resolve or
over-solve. Declare unattained targets instead of choosing easy cases retrospectively.

Measure a resolution scaling exponent only from resolutions timed within one job
on the same physical GPU. If that is impractical, publish per-job paired speedup
ratios and hardware-free operation/iteration counts; do not fit cross-job raw times.

## Stages and concrete artifacts

1. Agree on the architecture, study scope, worktree bases/names and pilot budget.
   Create a shared immutable protocol/config and common result schema.
2. In isolated PDE trees, reproduce references and baselines, then run a small
   pilot. For waves, include the matched affine/nonlinear dimension comparison;
   for heat, establish the new separable port's correctness before timing claims.
3. Use pilot reconstruction-versus-dynamics diagnostics to choose bounded
   improvements. Prioritize verified convergence/kernel changes and exact linear
   operators; increase head/bank capacity only when measurements support it.
   Measure any dynamics-aware training as its own ablation, including offline cost.
4. Freeze the protocol, run the resolution studies and independent confirmation,
   and obtain cross-agent numerical/provenance review before accepting results.
5. Generate accuracy-versus-resolution, query-time-versus-resolution,
   cost-versus-accuracy, memory, offline amortization and frozen-transfer figures.
   Save representative and failure-case still sequences for evolving PDEs. Generate
   every results table from JSON and collect a reproducible report and artifact manifest.

No accuracy or speedup value is promised. A PDE where the classical or affine
baseline wins remains a result and informs which contribution the paper can defend.
Do not launch the full combinatorial search before pilot cost/accuracy is known.

## Proposed ownership and bases — approval required before creation

Root coordinates in the already existing `2026-09-06-burgers3d-repair` tree.
The following are proposed new trees under the repository's `worktrees/` directory;
each has branch `exp/<directory-name>`, its own named subagent and its own namespace
under `/cluster/tufts/paralab/tawal01/`.

| Directory | Base branch and pinned commit | Subagent | Cluster namespace |
|---|---|---|---|
| `2026-09-07-mr-wave2d` | `exp/2026-09-06-wave-head-transfer` at `906cbe6f6f9fbe9376c05d2dccb3a4c1c7d7d66e` | `mr_wave2d` | `mr_wave2d_20260907` |
| `2026-09-07-mr-burgers2d` | `exp/2026-09-04-separable-tensor-consolidated` at `da479125b13aaa3e15b5cae33f72709705129838` | `mr_burgers2d` | `mr_burgers2d_20260907` |
| `2026-09-07-mr-poisson2d` | Same consolidated branch/commit | `mr_poisson2d` | `mr_poisson2d_20260907` |
| `2026-09-07-mr-heat2d` | Same consolidated branch/commit; verified heat port required | `mr_heat2d` | `mr_heat2d_20260907` |

Wave's base owns the fresh implementation and checkpoints. The consolidated base
owns the current non-wave separable, exact-linear and tensor work; starting on main
would lose those corrections. For heat, read the correction branch
`fix/heat-rollout-warm-start` at `292c8d9bc316d0ec77d162015d883144910cc99f` and the
archived solver without merging either automatically. If the user chooses the older
CP pipeline as primary, revise this base proposal before creating the worktrees.

Schedule at most three child agents concurrently alongside root; queue the fourth
PDE owner as a slot becomes available. Never share a job directory or silently reuse
an old campaign namespace. All real work runs on cluster GPUs in f64/highest precision
with backend preflight, recorded provenance, checksummed collection and exact
remote-directory cleanup. Follow the repository cancellation helper rules.
Ask whether to merge completed experiment branches when they finish.

## Decisions requested

- Main architecture: current continuous-coordinate separable decoder, older ViT +
  CP, or both as equal primary methods. Recommendation: current decoder with CP baseline.
- Both frozen-weight transfer and per-resolution optimization, or only one.
  Recommendation: both, reported separately.
- Paper deadline and total compute budget. Recommendation: a capped pilot before
  committing to the full campaign, using measured costs to set its size.
- Approval of the concrete worktree/base/subagent/namespace table above.

The repository AGENTS.md explicitly requires: "Propose the names and get
confirmation before creating any of them" and "Starting new work: ask where to
branch from." This approval is for new parallel experiment ownership, not for
read-only inspection or preparation of this reviewable protocol.

## Plain-language glossary

- **PDE / FOM / ROM:** partial differential equation / full reference model /
  reduced model. **NM-ROM:** reduced model whose possible states form a learned
  nonlinear set.
- **Resolution / intervals / unknowns:** spatial grid fineness / cells along an
  axis / field values actually solved for. These counts are not interchangeable.
- **Bank / head / latent dimension:** spatial functions / map producing their
  coefficients / number of compressed coordinates. **Affine:** linear plus offset.
- **ViT / CP / POD:** vision-transformer encoder / products of learned axis factors /
  a data-derived linear reduced basis.
- **Frozen transfer / per-resolution optimization:** unchanged network weights on
  new meshes / separately selecting or training a model for each mesh.
- **Weak residual / test modes / quadrature:** PDE equations averaged against
  smooth functions / those functions / weighted spatial sampling for the averages.
- **Reconstruction / tangent / rollout:** fitting a snapshot / directions of
  motion representable by the decoder derivative / evolving without repeated truth fits.
- **Reference uncertainty / same-grid discrepancy:** uncertainty in the trusted
  comparison solution / difference from the discrete full model on the ROM mesh.
- **Accuracy–cost envelope:** least measured cost meeting each declared error target.
  **Ablation:** controlled change to one part of a method to assess its effect.
- **Latency / throughput / amortization:** time for one query / queries completed
  per unit time / how repeated savings offset setup cost.
- **Median / worst / failure count:** middle measured result / largest error /
  number of cases missing the declared conditions. Repetitions and independent
  data seeds measure different kinds of variability.
- **Namespace / commit / manifest:** isolated cluster directory prefix / saved
  code revision / file listing used to verify provenance and content hashes.
