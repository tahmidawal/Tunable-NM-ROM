# Claude Code continuation handoff: accuracy, speed and fixed-checkpoint tuning

This is a reproducible handoff snapshot of retained development evidence and unfinished work; it introduces no new GPU result. The canonical root `LAB-LOG.md` remains authoritative, especially when the active Burgers collector advances after this snapshot.

## Read and preserve before editing

Read [LAB-LOG.md](</home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/LAB-LOG.md>) first, then [AGENTS.md](</home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/AGENTS.md>). The live handoff observations are in [worktrees/2026-09-14-no-audit/experiments/neural-operator-audit/checks/claude-handoff-live-snapshot.json](</home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-14-no-audit/experiments/neural-operator-audit/checks/claude-handoff-live-snapshot.json>); the adjacent [manifest](</home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/reports/2026-09-14-claude-code-continuation-handoff.json>) pins every JSON used here. Regeneration reads retained evidence only:

```bash
/home/tahmid/Dev/.venv/bin/python reports/generate-claude-code-continuation-handoff.py --snapshot /home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-14-no-audit/experiments/neural-operator-audit/checks/claude-handoff-live-snapshot.json --check
```

Omit `--check` only when deliberately rebuilding this handoff and its manifest from an updated snapshot.

The user's objective has two linked parts: better physical accuracy and complete-query speed against tuned FNO/other efficient neural operators and efficient full-order solvers; and classic ViT + CP fixed-checkpoint tunability. Keep decoder weights, latent dimension and CP/bank rank identical within a tuning curve. Change Gauss–Newton iteration cap, stopping tolerance, and selection among offline-fitted EQ rules where implemented. The preceding nested-capacity recommendation was rejected as an answer to this request. Training separate ranks is a capacity experiment, not this inference-time tuning mechanism. No useful competitive joint operating point is established yet.

The scientific worktrees were approved from the corrected consolidated baseline. Continue in an existing approved tree after its current owner finishes; do not branch scientific work from frozen `main`, whose heat rollout is known broken. Preserve every pre-existing modified and untracked main file. Do not reset, stash, stage, commit or clean those files incidentally. Do not merge any experiment branch without the pending user merge decision. Final campaign cohorts remain sealed. The existing frozen archives remain untouched.

## Coordinator snapshot

Observed at `2026-09-15T01:07:08.617128+00:00`. These observations expire; recheck before acting. The date in the filename is the local session date. A collector can legitimately advance a branch after capture. Full command outputs and process identity are retained in the linked snapshot. Its archival commit is recorded in the canonical closing entry; the table records observed heads before that snapshot commit and the subsequent documentation commit, avoiding a self-referential hash.

| Checkout | Absolute path | Branch | Observed HEAD |
| --- | --- | --- | --- |
| main | `/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude` | `main` | `19403cf242067bde82c4d38ffbb0970d14699550` |
| consolidated | `/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-13-nmrom-consolidated` | `exp/2026-09-13-nmrom-consolidated` | `02ff0f1f18db37d8589b28e90c3a93dae5bc5a88` |
| poisson | `/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-14-no-poisson` | `exp/2026-09-14-no-poisson` | `02fac7a161333b8013c92df0f9de439e887ec475` |
| burgers | `/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-14-no-burgers` | `exp/2026-09-14-no-burgers` | `21e12e979005468cce7247ae49cd7d703aa3a423` |
| audit | `/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-14-no-audit` | `exp/2026-09-14-no-audit` | `fd9b5eb8dd26f4f8d16fff86bcce0b1e81d4b447` |

Pre-existing main status to preserve (captured before this handoff):

```text
M LAB-LOG.md
 M understand/2026-09-02-poisson-rom-training-to-prediction.pdf
 M understand/2026-09-02-poisson-rom-training-to-prediction.tex
?? reports/2026-09-04-results-deck-burgers-poisson.aux
?? reports/2026-09-04-results-deck-burgers-poisson.fdb_latexmk
?? reports/2026-09-04-results-deck-burgers-poisson.fls
?? reports/2026-09-04-results-deck-burgers-poisson.nav
?? reports/2026-09-04-results-deck-burgers-poisson.snm
?? reports/2026-09-04-results-deck-burgers-poisson.toc
?? reports/2026-09-10-quadrature-free-paper-strategy.md
?? reports/2026-09-11-accuracy-aggregates.json
?? reports/2026-09-11-accuracy-aggregates.md
?? reports/generate_accuracy_aggregates.py
```

Completed allocation accounting at snapshot time; durations come from Slurm, not model query timings.

| Job ID | Name | State | Elapsed seconds | Node |
| --- | --- | --- | --- | --- |
| 3701221 | ctol_nob_calibration01 | COMPLETED | 1575 | pax007 |
| 3701314 | ctol_nop_pilot01 | COMPLETED | 208 | pax106 |
| 3702464 | ctol_noa_fno_p01 | COMPLETED | 3496 | pax106 |
| 3702473 | ctol_nop_matched01 | COMPLETED | 529 | pax007 |
| 3702709 | ctol_nob_refinement02 | COMPLETED | 22852 | pax007 |

Original resource allowance, remaining rather than renewed:

| Lane | Authorized seconds | Consumed seconds | Remaining seconds | Remaining minutes |
| --- | --- | --- | --- | --- |
| burgers | 28800 | 24427 | 4373 | 72.883 |
| poisson | 28800 | 737 | 28063 | 467.717 |
| audit | 28800 | 3496 | 25304 | 421.733 |

All listed GPU jobs are completed at this snapshot. These are unused portions of the original per-lane allocation budgets, not reserved GPUs or authorization for fresh eight-hour blocks. Recheck accounting and available resources before submitting.

Collector refreshed at `2026-09-15T01:10:35.837609+00:00`.

| Job ID | Current PID | Phase | Repair source commit |
| --- | --- | --- | --- |
| 3702709 | 3432725 | collecting | 629efba81ac42c93067b0d388ae1481337e05117 |

Burgers computation is complete in the scheduler record. Remote dataset indices at snapshot time:

| Split | Complete | Recorded cases | Expected cases |
| --- | --- | --- | --- |
| train | True | 128 | 128 |
| validation | True | 32 | 32 |

Collection, independent audit, raw archive retention, cache preparation and deletion are separate phases; follow the current collector record rather than inferring these from completed computation.

## Active Burgers collection: finish safely before implementation

The repaired collector is PID `3432725`, observed in phase `collecting` for job `3702709`. Its command is `/home/tahmid/Dev/.venv/bin/python /home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-14-no-burgers/experiments/neural-operator-burgers/cluster/collect_when_done.py`. Its current log is [worktrees/2026-09-14-no-burgers/experiments/neural-operator-burgers/runs/refinement02/collection-monitor-restarted.log](</home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-14-no-burgers/experiments/neural-operator-burgers/runs/refinement02/collection-monitor-restarted.log>); durable current phase is [worktrees/2026-09-14-no-burgers/experiments/neural-operator-burgers/checks/refinement02-collection-status.json](</home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-14-no-burgers/experiments/neural-operator-burgers/checks/refinement02-collection-status.json>). Its bounded lifetime is recorded in the original launch evidence; an old PID or a `monitoring` file alone does not prove it is still alive. Inspect command identity, process state, scheduler state, logs and local lock before any takeover.

The old monitor retried an invalid completed-job queue lookup; the narrow repair queries the account queue and reaches accounting. Preserve `checks/collection-monitor-repair.json`, `checks/collection-monitor-restart.json`, the original monitor log and `runs/refinement02/collection-monitor-restarted.log`; use the restart log for current progress. This recovery submitted no GPU job.

**Do not edit the Burgers scientific source or collector dependencies while collection is pending.** `audit_diagnosis.py` compares archived diagnosis against current source hashes, including engines, accuracy paths and data provenance. Source edits could make valid evidence fail collection. The existing monitor owns its `runs/refinement02/collection-monitor.lock` single-instance lock, full raw archive/checksum verification, saved-field/reference/dataset audits, Git retention, exact remote cleanup, and locked canonical log append. Do not start a duplicate collector or a second writer. Let it complete; verify successful status, archive manifest/parts, retained audit results and exact job-directory removal before assuming completion.

The collector's planned cluster-generated dataset handoff cache is `/cluster/tufts/paralab/tawal01/no_burgers_20260914/pilot-data01`. Verify its copied index/case/sidecar hashes and calibration-provenance relocation before use by FNO. A partial training index or ready cache is not proof that validation and full raw collection finished. Do not delete the cache as if it were the completed job directory. If the monitor fails, preserve all evidence and diagnose its actual failed phase; do not rerun dataset generation, restart a job, or delete remote files blindly. Resume/recovery must fit remaining campaign resources and use exact owned paths.

The original failed calibration remains evidence, alongside the refined passing empirical reference. Do not silently replace its protocol or present refinement as a continuum-error theorem. No Burgers FNO training, learned step correction, trajectory retraining, or complete classic tuning sweep is claimed complete in this handoff.

## Retained Poisson common-data evidence

Single-seed development screen against the same empirical physical-reference candidates. Shared train/validation case and target hashes match, but training/tuning budgets are not matched. Provisional for broader generalization because repeated seeds, broader families and sealed final tests have not run. No paired FNO/ROM/FOM latency panel exists.

| Model | Cases | Mean error % | Median error % | Worst error % | Cases above 5% |
| --- | --- | --- | --- | --- | --- |
| FNO small | 32 | 4.051765 | 2.644894 | 32.468105 | 4 |
| FNO medium | 32 | 3.515362 | 2.345553 | 28.277713 | 4 |
| FNO large | 32 | 3.450394 | 2.025227 | 29.374664 | 4 |
| Fresh ROM r128 | 32 | 7.886126 | 5.532976 | 40.113067 | 19 |
| Fresh ROM r256 | 32 | 8.369544 | 6.295649 | 55.496646 | 21 |

Sources: [worktrees/2026-09-14-no-poisson/experiments/neural-operator-poisson/runs/matched01/matched-summary.json](</home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-14-no-poisson/experiments/neural-operator-poisson/runs/matched01/matched-summary.json>) and [worktrees/2026-09-14-no-audit/experiments/neural-operator-audit/runs/fno_poisson01/field-audit.json](</home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-14-no-audit/experiments/neural-operator-audit/runs/fno_poisson01/field-audit.json>). The fresh ROM arms are rejected replacements: more bank capacity improved the bank projection floor but worsened deployed generalization. Historical weights have a different training history and must remain a labelled unmatched-data diagnostic. Investigate bank/head/generalization limits before another blind capacity increase. Direct DST remains the demanding efficient Poisson baseline; tolerance-tuned CG is an additional named control. Strong accuracy against an over-solved CG alone does not establish a competitive operating point.

## Retained Burgers same-job diagnosis

Audited development calibration evidence against the refined empirical numerical reference; provisional as a broader/continuum claim. The native ROM includes its compressed initial state in this diagnostic. These measurements show an efficient tested FOM faster and more accurate than the inherited ROM.

| Method | Median GPU ms | Median host ms | Worst fixed-initial error % | Failed stopping invocations | Latency outliers |
| --- | --- | --- | --- | --- | --- |
| rom | 54.630693 | 57.551909 | 1.867068 | 0 | 0 |
| same_nt1e-2_dt005 | 16.664419 | 18.758816 | 0.997803 | 0 | 1 |
| same_nt1e-4_dt005 | 19.060996 | 21.059535 | 1.381132 | 0 | 0 |
| same_nt1e-6_dt005 | 81.567216 | 83.902788 | 1.381082 | 0 | 0 |
| same_nt1e-2_dt01 | 9.296756 | 11.171961 | 1.864195 | 0 | 1 |
| coarse_half_dt005 | 18.534939 | 20.638627 | 1.899297 | 0 | 0 |
| coarse_quarter_dt01 | 13.841546 | 15.895978 | 3.639752 | 0 | 0 |

Source: [worktrees/2026-09-14-no-burgers/experiments/neural-operator-burgers/checks/refinement02-diagnosis-audit.json](</home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-14-no-burgers/experiments/neural-operator-burgers/checks/refinement02-diagnosis-audit.json>). All timing repetitions are retained. FOM arm identifiers encode the original same/coarse grid, Newton threshold (`nt`) and timestep (`dt`); inspect the immutable diagnosis protocol before changing any control. These are same-job comparisons, not ratios assembled across allocations.

Evolved-field representation diagnosis from already saved invocations; excludes the native fitted initial output. Best-found nonlinear fits are not certified global optima, and the error sources are not additive.

| Diagnostic | Median case maximum % | Worst % |
| --- | --- | --- |
| bank | 0.010239 | 0.108358 |
| nonlinear_best_found | 0.421101 | 0.842452 |
| online | 0.939970 | 1.658743 |

Saved iteration summary (existing invocations, no new solves):

```json
{
  "step_iteration_quantiles": {
    "min": 2.0,
    "q25": 3.0,
    "median": 3.0,
    "q75": 4.0,
    "q95": 8.0,
    "max": 42.0
  },
  "initial_fit_iterations_by_case": [
    120,
    9,
    11,
    178,
    28,
    44,
    16,
    48
  ]
}
```

## First implementation gaps and the next execution sequence

The source-generated plan is [worktrees/2026-09-14-no-audit/experiments/neural-operator-audit/checks/fixed-checkpoint-tuning-plan.json](</home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-14-no-audit/experiments/neural-operator-audit/checks/fixed-checkpoint-tuning-plan.json>), produced by [worktrees/2026-09-14-no-audit/experiments/neural-operator-audit/prepare_fixed_checkpoint_tuning.py](</home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-14-no-audit/experiments/neural-operator-audit/prepare_fixed_checkpoint_tuning.py>). Status: **proposed tests, not run**. The prepared [worktrees/2026-09-14-no-burgers/experiments/neural-operator-burgers/eq_ablation.py](</home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-14-no-burgers/experiments/neural-operator-burgers/eq_ablation.py>) is incomplete for the full requested study. Start with [accuracy_paths.py: make_rom stopping controls](</home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-14-no-burgers/experiments/mr-burgers2d/accuracy_paths.py>), [engines.py: weak/build_rom](</home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-14-no-burgers/experiments/mr-burgers2d/engines.py>), and [pilot.py: full-upwind component diagnostic](</home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-14-no-burgers/experiments/mr-burgers2d/pilot.py>). The fixed checkpoint is [worktrees/2026-09-14-no-burgers/experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl](</home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-14-no-burgers/experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl>); verify it against the hash below before execution. Its current scientific invariants are:

```json
{
  "checkpoint_sha256": "18f0266ae6f0454200ec0b7bf94a18cde531feac9d3170d5099adc5d68d6b589",
  "latent_dimension": 16,
  "bank_rank": 512,
  "weak_modes": 64,
  "intervals": 256,
  "dt": 0.005,
  "native_solver_config": {
    "ic_budget": 400,
    "step_budget": 180,
    "gtol": 1e-06
  },
  "initialization_policy": "same supplied-field Gaussian fit and candidate library"
}
```

After verified collection closure, the next single scientific owner may work in the approved Burgers worktree; audit, Poisson and consolidated trees remain read-only to that owner. Separate concurrent experiments need their own approved owners, worktrees and namespaces. Do not create new worktrees or assume new allocation authority from this handoff.

Concrete source gaps, taken directly from the saved plan:

- eq_ablation.py implements the three EQ sizes but has not run
- its budget pre-check does not bound NNLS fit runtime; enforce per-stage walltime and preserve partial evidence
- full-grid weak advection exists as a saved-state diagnostic, not a complete rollout arm
- make_rom hard-codes the evolution residual threshold; expose it without changing initial fitting
- make_rom shares gtol between initial fitting and evolution; separate those controls before a stationarity-tolerance sweep
- keep original numerical acceptance gates; label early-stopped approximations separately rather than treating them as stationary solves
- compare physical trajectory errors, convergence records and paired query cost; do not select using residual alone

Proposed stages and exact settings, generated from the plan. Each stage requires an immutable config, source hashes, bounded runtime, preserved partial evidence, and its stated control; no row is a completed result.

### converged full-upwind weak sentinel control — not run

```json
{
  "case_ids": [
    "burgers-calibration-00002",
    "burgers-calibration-00003"
  ],
  "purpose": "verify output stability under tighter solver settings at fixed timestep before a broader screen",
  "hold_fixed": "checkpoint, weak modes, timestep and Gauss cold start",
  "qualification": "numerical convergence of the reduced equations is not physical truth",
  "separate_initializer_check": "compare sampled and full supplied-field initial fits diagnostically; do not mix initializers in the EQ screen"
}
```

### quadrature control — not run

```json
{
  "eq_points": [
    256,
    512,
    1024
  ],
  "additional_control": "full-grid FOM-exact upwind advection projected onto the same weak modes",
  "hold_fixed": "checkpoint, modes, cold fit, timestep, solver tolerances and budgets",
  "fitting": "decoder-output training snapshots only; record each NNLS support, weights and fit error",
  "purpose": "determine whether EQ changes the complete trajectory or the high-effort accuracy endpoint"
}
```

### iteration screen — not run

```json
{
  "step_budgets": [
    2,
    4,
    8
  ],
  "reference": "converged control; retain its actual budget and stopping records",
  "hold_fixed": "one selected EQ rule, cold fit, residual threshold and stationarity threshold",
  "purpose": "measure deliberately early-stopped outputs and a converged reference"
}
```

### tolerance screen — not run

```json
{
  "evolution_normalized_gradient_tolerances": [
    0.001,
    1e-05,
    1e-06
  ],
  "step_budget": "same sufficiently large budget as the converged control",
  "hold_fixed": "same EQ rule, initial fitting and residual threshold",
  "normalization": "norm(J.T r)/(norm(J) norm(r)); not a physical-error tolerance",
  "optional_followup": "expose and screen normalized weak-residual thresholds separately if the initial screen warrants it"
}
```

### confirmation — not run

```json
{
  "rule": "freeze a small shortlist on calibration cases, then evaluate on common validation cases",
  "timing": "complete query with FNO and tolerance-matched efficient FOM controls in one GPU job; all repetitions retained",
  "outputs": "return supplied initial field exactly for every deployable method; retain native initial compression error separately"
}
```

Implement the full discrete-upwind weak rollout control before attributing an EQ-versus-ROM gap. Expose evolution residual and normalized-gradient thresholds independently of the initial-fit controls; preserve the original initial fitting defaults exactly. Verify reduced/full weak operator and gradient consistency on a saved case, stopping-record consistency, native/control output parity where settings agree, and requested-output contracts with bounded smoke work before any real batch.

Use the staged order instead of a large combined grid: establish a stable high-effort endpoint on diagnostic cases; compare EQ rules with unchanged solver and initializer; isolate cap; isolate tolerance; then freeze a shortlist on calibration data and evaluate common validation. An optional initial-fit study is separate. All deployable methods return the supplied initial field exactly, while native compression error remains separately reported. Early-stopped finite outputs may appear on a measured physical-error/cost curve with their actual stopping status; they do not inherit stationarity or relaxed acceptance gates. Residual reduction does not guarantee monotone physical accuracy.

Conditional scientific follow-up:

- If representation diagnostics miss the target, improve one checkpoint's reconstruction/trajectory training, then freeze and repeat this same tuning protocol.
- If best-found fits are accurate but the converged full-weak rollout is poor, isolate timestep and weak-projection/dynamics errors before attributing the gap to training.
- If EQ causes the trajectory gap, improve the decoder-output-trained rules while keeping the neural checkpoint fixed.

## FNO and broader campaign work still open

After complete Burgers data handoff, adapt and verify the operator training worker for the Burgers input/output contract. [worktrees/2026-09-14-no-audit/experiments/neural-operator-audit/README.md](</home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-14-no-audit/experiments/neural-operator-audit/README.md>) records that the existing fixed queue/staging is Poisson-specific even though model and smoke paths cover Burgers. The supplied initial field, coefficients and requested times are allowed inputs; generation descriptors, case IDs and truth sidecars are not. Keep case-disjoint train/validation splits, training-only normalization and validation-selected checkpoints. Retain full predictions, loss curves, parameters, complex intermediates and checkpoint hashes.

The current FNO screen is an initial capacity screen, not an exhaustively tuned baseline. Learning-rate refinement, repeated seeds, efficient alternatives such as TFNO and broader sampled-field families remain open. Saved optimizer/checkpoint state does not imply resumable training: continuation support in the current driver is not implemented. Add and verify explicit restart semantics before calling interrupted training resumed.

The later authorized hypotheses include supplied-field latent/coefficient prediction, direct same-bank prediction and prediction followed by weak correction, with trajectory-aware training if diagnosis warrants it. Freeze any improved checkpoint before repeating the same solver/EQ curve; do not mix endpoints from different weights. The eventual interleaved same-GPU panel must compare ROM, tuned operator and efficient FOM together, including initialization, interpolation, requested-time inference, full-field output and host transfer where reported. A faster method that misses the declared accuracy gate is a diagnostic, not the competitive target result.

## Environment, resource accounting and measurement rules

Continue the original bounded lanes; **do not assume a fresh GPU budget**. Use the captured accounting as a starting observation and refresh `sacct`/`squeue` for every owned historical and active allocation, including retries and running elapsed time, before planning any successor. Subtract actual usage from the original per-lane cap; the unspent time of another lane is not automatically a transferable allowance. If the existing authorized remainder cannot accommodate a bounded next stage, report the concrete needed extension. Preparation and read-only evidence review can proceed without extending compute.

Use only `/home/tahmid/Dev/.venv/bin/python` locally and `/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python` on Tufts. Local GPU work is smoke-only, under a minute, through `jaxrun` after sourcing `/etc/profile.d/jax-mem.sh`, with at most three concurrent instances. Real work belongs in unique paralab job directories on `gpu`, never `preempt`, with `JAX_DEFAULT_MATMUL_PRECISION=highest`, f64 and an explicit GPU-backend preflight that exits on CPU fallback. Check queue before and after submission. Check paralab disk before diagnosing missing output. Copy source directly to paralab and verify content hashes rather than ancestor `git` state. Generate data from seed on cluster; a checksummed copy of existing cluster-generated data is permitted, never replacement with local `data/`.

The retained environment audit records pinned additions with `--no-deps`, preserving the JAX stack. Torch/NeuralOperator passed allocated-GPU dtype/gradient/checkpoint checks despite recorded unsatisfied package pins; `pip check` is not claimed clean. Do not upgrade or reinstall the whole environment to resolve those declared pin mismatches. Preserve this provenance and rerun relevant allocated-GPU checks after any necessary environment change. The reviewed NeuralOperator spectral buffer fix and explicit complex parameter conversion are required for complex128, not just real `.double()` conversion.

Burn in before every timed block; compare complete queries in one job on one GPU; derive accuracy and cost from the same invocation; preserve repetition arrays and report medians, failures and outliers. Keep reference tolerance labels and like-for-like efficient baselines. Record seed, mesh, family, latent dimension, config, commit/content hashes, GPU and job ID. Reject CPU fallbacks, truncated/OOM/disk-full runs or unexplained captured-large-constant warnings. Checksum-pull and archive before exact completed-directory removal.

Minimize the weak residual projected onto smooth test modes, with $M > k$ comfortably and quadrature support approximately $m \approx 4M$; here $M$ is weak-mode count, $k$ is latent dimension and $m$ is quadrature-point count. Fit nonnegative EQ weights only on decoder-output training snapshots; preserve supports, weights and fit residuals for each rule and refit when mesh/modes change. Hyper-reduce cold initialization too. Keep the FOM-exact upwind operator within Burgers weak advection. Never replace these controls with pointwise/strong random collocation. Never issue bare `scancel`; only the repository's explicit numeric-ID cancellation helper for owned `ctol_*` jobs is permitted.

## Evidence and restoration map

- [worktrees/2026-09-13-nmrom-consolidated/consolidated/README.md](</home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-13-nmrom-consolidated/consolidated/README.md>) — Corrected consolidated entry point; implementations, selected checkpoints and self-contained evidence mirror.

- [worktrees/2026-09-14-no-poisson/experiments/neural-operator-poisson/runs/pilot01](</home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-14-no-poisson/experiments/neural-operator-poisson/runs/pilot01>) — Historical-checkpoint Poisson diagnosis, immutable raw archive, source/checkpoint/data evidence and collection audit.

- [worktrees/2026-09-14-no-poisson/experiments/neural-operator-poisson/runs/matched01](</home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-14-no-poisson/experiments/neural-operator-poisson/runs/matched01>) — Fresh common-data ROM training, per-rank predictions/checkpoints, full raw archive and matched summary.

- [worktrees/2026-09-14-no-audit/experiments/neural-operator-audit/runs/fno_poisson01](</home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-14-no-audit/experiments/neural-operator-audit/runs/fno_poisson01>) — FNO checkpoint/prediction/source archive, independent field audit, collection records and archive-parts manifest.

- [worktrees/2026-09-14-no-audit/experiments/neural-operator-audit/checks/environment01](</home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-14-no-audit/experiments/neural-operator-audit/checks/environment01>) — Original environment versions, additions, preservation audit and unsatisfied dependency record.

- [worktrees/2026-09-14-no-audit/experiments/neural-operator-audit/checks](</home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-14-no-audit/experiments/neural-operator-audit/checks>) — Local/allocated GPU f64 smoke, common-data audits and proposed fixed-checkpoint plan.

- [worktrees/2026-09-14-no-burgers/experiments/neural-operator-burgers/artifacts/calibration01](</home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-14-no-burgers/experiments/neural-operator-burgers/artifacts/calibration01>) — Retained original failed Burgers calibration; restore according to its README/archive manifest.

- [worktrees/2026-09-14-no-burgers/experiments/neural-operator-burgers/artifacts/refinement02/archive.json](</home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-14-no-burgers/experiments/neural-operator-burgers/artifacts/refinement02/archive.json>) — Ultimate committed raw archive manifest and parts after collector completion; do not infer readiness from local convenience views.

- [worktrees/2026-09-14-no-burgers/experiments/neural-operator-burgers/runs/refinement02](</home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-14-no-burgers/experiments/neural-operator-burgers/runs/refinement02>) — Live refined reference/diagnosis evidence and collector log; full final archive may still be pending.

- [worktrees/2026-09-14-no-burgers/experiments/neural-operator-burgers/checks](</home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-14-no-burgers/experiments/neural-operator-burgers/checks>) — Passing refined reference and saved-field diagnosis audits, dataset audits and durable collection phase.

- [worktrees/2026-09-14-no-burgers/experiments/neural-operator-burgers/cluster/collect_when_done.py](</home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-14-no-burgers/experiments/neural-operator-burgers/cluster/collect_when_done.py>) — Exact collector and recovery behavior; inspect without modifying until collection ends.

- [reports/2026-09-11-accuracy-improvements-and-wave-speed.md](</home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/reports/2026-09-11-accuracy-improvements-and-wave-speed.md>) — Prior selected accuracy round with rejected arms and limitations.

- [reports/2026-09-11-accuracy-integration.json](</home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/reports/2026-09-11-accuracy-integration.json>) — Selected-change provenance and original corrected worktree pointers.

- [reports/2026-09-11-iterative-fom-multiresolution.md](</home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/reports/2026-09-11-iterative-fom-multiresolution.md>) — Older named iterative-FOM comparisons; not a substitute for efficient paired operator panel.

- [Older Paper /neurips26__Copy_ (1).zip](</home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/Older Paper /neurips26__Copy_ (1).zip>) — Historical manuscript provenance; old single-model/speed claims require the archived corrections.

For split archives, read each archive's own manifest: concatenate parts in manifest order, verify the aggregate SHA256, inspect archive members, then extract into a fresh scratch directory. Verify file manifests and provenance before using restored content. Do not guess that similarly named archives share a schema. Poisson extracted `archive/` directories retain `MANIFEST.sha256`, `ARCHIVE.sha256`, source and output trees; collection JSONs record successful checks and exact remote deletion. FNO `archive-parts/manifest.json` pins ordered raw chunks; Burgers final raw chunks/manifest are not claimed present until its collector says complete. Preserve failed and superseded attempts alongside successful ones. Never overwrite `best-results/`.

The pre-reset wave evidence remains untrusted under the user's wave reset; only the fresh reflective-wave branch is trusted where independently audited. Absorbing waves remain excluded from the current selected comparison. Existing numerical findings are not newly retracted by this handoff; the nested-capacity recommendation was withdrawn as a response to the user's requested tuning mechanism.

## Pasteable continuation prompt

```text
Continue the accuracy/speed research in /home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude.
Read the canonical absolute LAB-LOG.md FIRST, then AGENTS.md, then reports/2026-09-14-claude-code-continuation-handoff.md and its manifest. The handoff is a snapshot; refresh live state before acting.

My objective is better physical accuracy and complete-query speed versus tuned FNO/other efficient neural operators AND an efficient FOM, together with CLASSIC ViT+CP fixed-checkpoint tunability: identical decoder weights, latent dimension and rank; vary Gauss-Newton iteration cap, stopping tolerance, and offline-fitted EQ rule. Do not substitute nested-capacity architecture or independently trained ranks for that mechanism. Do not promise a competitive result; measure it without relaxing gates.

First inspect the owned Burgers job/collector, lock, logs, durable status and resource accounting. Let the existing collector finish without editing its current-tree scientific dependencies, duplicating its process or racing archive/log writes. Verify full raw retention, audits, dataset handoff hashes and exact completed-job cleanup. Preserve original failed calibration. Branch heads may advance as it collects.

After safe collection closure, continue within the already approved Burgers worktree as its sole writer; other experiment trees stay read-only. Preserve main dirty files. Do not branch from frozen main, create a new worktree or merge branches without the relevant user decision. Final cohorts stay sealed. Continue only inside the original remaining per-lane GPU allowance; refresh usage before any bounded successor, and do not silently renew the budget.

Use the retained fixed-checkpoint-tuning-plan.json. Implement the missing complete full-upwind weak rollout and independently exposed evolution residual/gradient stopping controls while initial fitting stays unchanged. Bound NNLS and every stage, preserve partial evidence, then smoke-check operator/gradient parity, output and stopping records. Establish the converged sentinel; isolate EQ, cap and tolerance; freeze a shortlist before common validation and interleaved same-GPU FNO/ROM/efficient-FOM timing. Keep early-stop status explicit. Diagnose representation, EQ and dynamics separately; if training changes are warranted, improve one checkpoint, freeze it, and repeat this SAME fixed-checkpoint study.

Complete the Burgers common-data operator baseline and diagnose the fresh Poisson ROM generalization failures. Respect the existing FNO restart limitation, f64/complex128 checks, held-out input contract, training-only normalization and source hashes. Record all attempted outcomes, median complete-query times, physical errors, outliers and failures from matched invocations. Do not compare timing across jobs or call the inherited historical ROM matched-data evidence.

Use absolute venv paths, cluster GPU preflight, unique paralab directories, highest matmul precision, warmups, timing arrays, checksum archives and exact cleanup. Append the canonical LAB-LOG under its established lock before ending; keep reports source-generated. Ask whether to merge when experiments finish; do not merge unprompted. Start with the concrete safe next step rather than restarting the whole campaign.
```

## Glossary

- **ROM / NM-ROM:** reduced-order model / nonlinear-manifold reduced-order model; solve in a small latent space and decode the field.
- **FOM:** full-order numerical solver of the spatially discretized PDE. **PDE:** partial differential equation.
- **ViT:** vision-transformer encoder; it can infer coordinates from a supplied field, but is not itself the solver-effort tuning knob.
- **CP / bank / rank:** separable spatial representation / its learned field basis / number of basis or separable terms. **Latent dimension:** number of nonlinear coordinates solved online.
- **Checkpoint:** exact saved neural weights and supporting model state. **Capacity:** network size; separate capacities are distinct trained models.
- **FNO / TFNO:** Fourier neural operator / tensor-factorized Fourier neural operator.
- **GN / Gauss–Newton / cap:** iterative least-squares update / that algorithm / its maximum allowed iterations.
- **Weak modes / weak residual:** smooth test fields / PDE discrepancy projected onto those fields. **EQ:** empirical quadrature, a stored weighted subset of evaluation nodes. **NNLS:** nonnegative least squares, used offline to fit quadrature weights.
- **Stationarity / stopping tolerance:** sufficiently small optimization gradient / specified numerical threshold for stopping. Neither is a physical-error certificate. **gtol:** gradient stopping parameter; its exact normalization is stated in the proposed plan.
- **Initializer / cold fit:** procedure mapping the supplied initial field to latent coordinates before evolution. **Full upwind:** complete-grid evaluation of the FOM's discrete one-sided advection operator.
- **Bank projection / nonlinear best-found / online:** unrestricted fit in the learned spatial span / best discovered fit constrained by the decoder / deployable PDE-driven solve. Diagnostic fits use truth and are not inference baselines.
- **Model / method / diagnostic:** named trained network / solver configuration / truth-assisted or saved-field investigation.
- **Cases:** independent input problems; trajectories from one input belong to one case. **Calibration:** opened development cases for protocol/settings. **Validation:** separate development cases for model selection. **Final sealed cohort:** untouched cases reserved for final confirmation.
- **Mean / median / worst error %:** average / middle / largest case-relative field error, multiplied by one hundred. **Cases above 5%:** count exceeding the declared Poisson physical threshold. **Median case maximum % / worst %:** middle / largest of each Burgers case's largest evolved-time error.
- **Fixed-initial error:** trajectory field discrepancy normalized by the reference initial-field norm. **Physical candidate / empirical reference:** independently refined numerical reference, not a certified exact continuum solution.
- **Median GPU ms / median host ms:** middle retained query time on GPU / including host-side transfers and requested work, in milliseconds. **Failed stopping invocations:** attempts failing their numerical stopping contract. **Latency outliers:** times above the upper Tukey fence recorded by the audit. **Repetition:** separately timed invocation.
- **Quantiles / p95:** positions in a sorted sample / value at the ninety-fifth percentile. **Sentinel:** small diagnostic subset used before a broad screen.
- **DST / CG:** discrete sine transform direct solver / conjugate-gradient iterative linear solver. **Newton threshold / timestep:** nonlinear solver stopping setting / physical time increment.
- **f64 / complex128 / highest precision:** double-precision real / double-precision complex arithmetic / required JAX matrix-multiply precision. **Smoke:** bounded implementation check, not a scientific campaign.
- **Authorized / consumed / remaining seconds or minutes:** original per-lane cap / elapsed allocated GPU time / subtraction left for future work; not a reservation. **Expected / recorded cases / complete:** planned dataset size / current index entries / generation completion flag. **Node:** cluster machine. **Elapsed seconds:** allocation duration from Slurm. **Split:** training or validation subset.
- **Lane / allocation / job ID:** approved resource budget and owner / scheduled GPU reservation / Slurm job identifier. **Collector:** local process verifying and preserving remote job evidence.
- **Worktree / head / dirty state:** separate checkout / exact current Git commit / changes not yet committed. **Hash / SHA256 / manifest:** content fingerprint / hashing algorithm / list associating artifacts with hashes and provenance. **Archive parts:** ordered chunks reconstructing one raw archive.
