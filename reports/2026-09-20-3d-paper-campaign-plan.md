# Three-dimensional NM-ROM paper campaign: launch contract

This is an experiment plan, not a results report. The user approved continuation with the proposed checkout choices and requested an overnight campaign within the next seven to eight hours. The execution window starts at September 20, 04:07 UTC, with an eight-hour target of 12:07 UTC; all scientific results remain to be measured and audited.

## Scope and ownership

The requested campaign covers scalar Burgers, heat, Poisson and incompressible Navier–Stokes in three spatial dimensions, using the current learned spatial bank, nonlinear coefficient head and correction-rank method. Each PDE has a dedicated agent, worktree and cluster namespace. The coordinator reviews designs, prioritizes tuning, checks resource use and assembles independently checked paper tables. Worker turns are staggered within the available agent concurrency; submitted GPU jobs may run concurrently.

The following bases preserve the relevant corrected implementations. The user's instruction to continue approves the proposed bases, worktree names and namespaces. No further checkout confirmation is needed for these lanes.

| PDE | Proposed worktree and branch suffix | Starting branch | Pinned commit | Cluster namespace |
| --- | --- | --- | --- | --- |
| Burgers 3D | `2026-09-20-paper-b3d` | `exp/2026-09-17-b-panel` | `25434a27bfc8857d81859784250b0bfd9a78adfd` | `paper_b3d_20260920` |
| Heat 3D | `2026-09-20-paper-h3d` | `exp/2026-09-13-nmrom-consolidated` | `02ff0f1f18db37d8589b28e90c3a93dae5bc5a88` | `paper_h3d_20260920` |
| Poisson 3D | `2026-09-20-paper-p3d` | `exp/2026-09-17-p-linear` | `a366980a46ce79741f7c1e12f576887678efcc6d` | `paper_p3d_20260920` |
| Navier–Stokes 3D | `2026-09-20-paper-ns3d` | `exp/2026-09-17-ns2d` | `2d70f36a02ef5109f3e406e513dcf89a513fb271` | `paper_ns3d_20260920` |

Worktrees live under the repository's `worktrees/`; branches are `exp/<suffix>`; remote roots are `/cluster/tufts/paralab/tawal01/<namespace>/`. Each job receives a unique attempt directory. No existing experiment tree, frozen archive or final cohort is overwritten. No merge or publication is included in this launch.

Burgers selectively imports three-dimensional FOM/data/representation primitives from `exp/2026-09-06-burgers3d-repair` at `303bb6f6e582f876d9e4034da141f394b17f1dda`, preserving the current correction solver and comparison machinery. Heat's consolidated base has the same corrected heat implementation as `exp/2026-09-07-mr-heat2d` at `5974d3e0df3f2e0906059564a72511de375ea2a5`. Operator infrastructure comes from `exp/2026-09-17-no-second` at `ea812685e7386e6af631a646e996c5428794f560`; its dimension-specific code and internal dtypes must be audited during the port. Navier–Stokes inherits infrastructure only: a scalar two-dimensional vorticity equation is not a three-dimensional flow solver.

## Deliverables and experiment order

Every lane produces a verified reference, a common dataset, trained current-method checkpoints, linear-ROM and full-order controls, operator comparisons, retained training/timing arrays, and generated paper tables. A missing or failed stage is reported explicitly rather than filled with historical values.

1. **Reference and data:** fix the PDE, boundary conditions, physical parameter family, requested output fields/times, mesh convention, error normalization and disjoint train/validation/final cohorts before tuning. Check manufactured solutions or an independent implementation and spatial/temporal refinement. The final cohort stays unopened until all comparison configurations freeze.
2. **Current method:** measure unrestricted bank error, best-found neural reconstruction, solved error and actual trajectory error separately. Screen capacity and training settings, then evaluate a nested correction ladder from a frozen checkpoint. Keep the weak test count fixed in the principal rank-control experiment; report any scheduled-test ladder separately.
3. **Baselines:** include the unrestricted learned bank, POD models at matched and larger dimensions, tuned full-order solves and applicable fast/coarse solvers. Train FNO and U-Net first; implement DeepONet and Transolver comparisons as the remaining operator arms. All receive the same physical input information and target outputs. Preserve the full operator backlog if an arm cannot finish by the freeze.
4. **Matched panels:** time complete queries, including initial fitting and requested dense readout, in one allocation on one GPU. Record offline/training/setup costs separately. Accuracy and cost must come from the same invocation, with GPU burn-in, synchronization, balanced method order and retained repetition arrays.
5. **Confirmation and paper:** freeze validation-selected settings, repeat finalist training with independent initialization seeds, evaluate the untouched final cases, independently recompute representative fields/metrics, and generate the manuscript tables from retained JSONs. Keep median, worst-case, failure and outlier statistics visible.

Each lane writes and commits its detailed `DESIGN.md` before its first job. Numerical tolerances and accuracy criteria are set using the reference error budget before model comparisons are observed. Updates after observations are documented as exploratory amendments; a new confirmatory evaluation uses untouched cases.

## Targeted tuning

Tuning is diagnostic and bounded. If the spatial bank limits accuracy, adjust its rank, coordinate-network capacity or coverage. If the head underfits training states, adjust its latent dimension, width/depth, optimizer and loss weighting. If training fits but validation fails, examine coverage and generalization. If best-found representations are accurate but trajectories fail, investigate initialization, weak-space coverage, time stepping and nonlinear convergence. Expand compute only when learning curves or the error decomposition justify it.

The retained Burgers head's failed reconstruction screen remains a failed historical result. Corrected models with nonzero correction rank are new candidates and can be evaluated under the new design without relabelling the old head as a pass. Enrich the bank if its error floor prevents the declared target. At a full-bank endpoint, use the appropriate free-coefficient control and avoid redundant latent-plus-full-bank parameterizations.

Heat and Poisson retain linear PDE residuals in three dimensions, so direct linear-bank solves remain mandatory controls. A neural model losing to that control is informative evidence; the method's implementation should still be tuned carefully. For Navier–Stokes, verification must establish divergence control and genuinely three-dimensional evolution before training is treated as a scientific experiment.

Use float64 throughout, including neural-operator internals, with `JAX_DEFAULT_MATMUL_PRECISION=highest`. Stream large bank evaluations and projections; avoid dense full-grid decoder Jacobians. Do not silently substitute POD for the learned bank, supply family-generator parameters to only one model class, or alter test cases to improve a headline.

## Implementation handoff from the preparation agents

- **Burgers:** start with the retained bank/head as an immutable diagnostic control and implement the current correction decoder, dense weak residual and actual evolution. Reuse `b3d_common.py`, `b3d_fom_gates.py` and the repair diagnostics; adapt current `cheap-corrections`, `q-ridge` and `b-panel` infrastructure. Keep sign-dependent upwinding: the old fixed-backward-stencil quadratic tensor is not interchangeable when decoded fields undershoot. Use the later `b-seeds` training recipe and initial-state coverage lessons for targeted retraining.
- **Heat:** extend `mr-heat2d/heat_core.py`, `verify_heat.py`, `accuracy_training.py`, `linear_paths.py` and the correction elimination primitives already present in the consolidated base. Carry the previous **decoded augmented state** into each step. Make nodes versus intervals explicit when importing Burgers grid helpers; update volume normalization, boundary factors and restriction/interpolation on every axis. Include an explicit regression check for the historical frozen rollout.
- **Poisson:** extend `p-bank-head` training and `p-linear/plin_core.py`, retaining its corrected orthonormal-projector oracle and direct QR full-bank endpoint. Coordinate features, sine modes, boundary masks, DST indexing and array reshapes all need dimension-aware changes. The bank's last hidden width must support its requested rank. Historical retracted oracle columns are not comparison inputs.
- **Navier–Stokes:** implement a fresh periodic velocity formulation with dealiased spectral advection, a divergence-free projection and an efficient explicit/IMEX full-order control. Reuse the existing head-fitting, correction-direction, audit and timing infrastructure. The learned bank and smooth vector test functions must satisfy the chosen discrete divergence constraint. An implicit nonlinear ROM may be compared against efficient explicit/IMEX FOMs at matched physical accuracy; do not slow the FOM merely to make the algebraic residual nonlinear. Verify interactions with nonzero projected advection and dependence on every spatial coordinate; an extruded planar flow or a decaying mode alone is insufficient verification. Primary outputs are velocity trajectories, with initial-norm error, kinetic-energy and divergence diagnostics; pressure is an additional output only if its reconstruction and cost are shared by every comparator.

The operator wrappers need genuine ports. Existing endpoint-inclusive Dirichlet grids and scalar output masks are unsuitable for periodic vector flow; convolution/patching dimensions and all internal arithmetic dtypes must be checked. Use moderate meshes for the first verified panel, then a second-resolution confirmation chosen from measured refinement and memory behavior. A larger grid alone does not establish physical accuracy.

## Proposed budget and schedule

The proposed campaign ceiling is **256 GPU-hours**, with at most **four simultaneous single-GPU jobs** across these lanes. This is a requested resource limit, not a runtime estimate or a guarantee of completion. The first tranche is at most **64 GPU-hours** total; initial jobs request at most **four hours** each. Reallocate within the accepted total after reviewing throughput, validation progress and remaining work. The shared account's other users take precedence over any assumed unused capacity; only campaign-owned jobs are managed.

The user's subsequent instruction replaces the initial multi-day experiment schedule with an **eight-hour overnight target**. Prioritize implementation and verification immediately; use measured throughput for the training/tuning phase; reserve the closing portion for held-out comparison, independent audits and generated tables. Aim to freeze candidate selection by hour six, use hour seven for final comparisons, and finish reporting by hour eight. These are scheduling targets, not permission to skip verification or label incomplete training as converged. At the stated concurrency, the overnight window can consume at most **32 GPU-hours**, within the previously proposed total ceiling. The conference cutoff discrepancy does not block this overnight target.

First complete the requested three-dimensional panels as far as verified implementations and measured throughput permit. Then direct remaining time to weak or inadequately supported results still used by the manuscript, based on an explicit claim audit. Any additional experiment must stay in its owner's approved isolated checkout and must not overwrite an existing campaign's results. The main coordinator owns the canonical log and reports; workers own their experiment trees.

At every completed job, the owner checks numerical/precision/provenance gates, collects checksummed artifacts, removes only the verified remote attempt directory, regenerates its summary, and reports its proposed next experiment to the coordinator. The coordinator reviews the campaign at least once per working day while active. Cluster jobs continue when the conversation is idle; ongoing autonomous monitoring must be provided by an explicitly established execution mechanism, not assumed from this document.

The largest schedule risk is implementing and independently verifying Navier–Stokes 3D from a two-dimensional infrastructure base. Operator ports and three-dimensional training throughput are also unmeasured. The target is complete, well-tuned comparison panels; no numerical win or completion of every arm is promised before those measurements.

## Launch decision

The user instructed the coordinator to continue after the proposed bases, namespaces and compute limits were presented. These choices are accepted for the four lanes. Use the existing cluster; external GPU rental is unnecessary while it can run the campaign.

The [AGENTS.md](../AGENTS.md) requirements to propose names and ask where to branch from have been satisfied. No checkout hold remains. Preserve the requirement to ask about merges when the worktrees finish.

## Glossary

- **NM-ROM:** a reduced numerical model whose states lie on a learned nonlinear manifold.
- **Bank / head:** learned spatial functions / the neural network producing their coefficients from a small latent state.
- **Latent dimension / bank rank:** the number of head inputs / the number of spatial functions available to represent a field.
- **Correction rank:** the number of additional linear coefficient directions solved alongside the latent state; the proposed accuracy-control variable.
- **Weak test count:** the number of smooth test functions against which the PDE residual is projected.
- **FOM:** the full-order numerical solver operating on the physical mesh.
- **POD / linear ROM:** a basis derived from training snapshots / a reduced model with unrestricted linear coefficients.
- **FNO / U-Net / DeepONet / Transolver:** Fourier, convolutional, branch–trunk and transformer model families used as learned solution-map baselines.
- **Checkpoint:** saved trained weights and the configuration needed to reproduce them.
- **Train / validation / final cohorts:** cases used to fit parameters / select settings / evaluate frozen selections independently.
- **Best-found reconstruction:** the smallest representation error found by the recorded optimization procedure, not a proof of a global optimum.
- **Manufactured solution:** a known analytic field with a constructed forcing, used to test a numerical solver.
- **Refinement / error floor:** comparison with finer spatial or temporal resolution / error that cannot be removed by adjusting a later stage alone.
- **GPU-hour / tranche:** one GPU allocated for one hour / a bounded initial portion of the compute budget.
- **Worktree / branch / commit / namespace:** an isolated checkout / its version-control line / a pinned source revision / its dedicated cluster output root.
- **AoE:** Anywhere on Earth, a UTC−12 deadline convention.
- **Proposed:** not executed or accepted as a result; a choice still subject to the launch contract.
