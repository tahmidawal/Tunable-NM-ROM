#!/home/tahmid/Dev/.venv/bin/python
"""Build a continuation snapshot from retained JSON evidence; never run a solver."""
from pathlib import Path
import argparse
import hashlib
import json
import statistics

ROOT = Path('/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude')
REPORT = ROOT / 'reports/2026-09-14-claude-code-continuation-handoff.md'
MANIFEST = REPORT.with_suffix('.json')
A = ROOT / 'worktrees/2026-09-14-no-audit/experiments/neural-operator-audit'
B = ROOT / 'worktrees/2026-09-14-no-burgers/experiments/neural-operator-burgers'
P = ROOT / 'worktrees/2026-09-14-no-poisson/experiments/neural-operator-poisson'
SOURCES = {}

def read(path):
    path = Path(path)
    raw = path.read_bytes()
    SOURCES[str(path)] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)

def link(path, label=None):
    return f'[{label or str(path.relative_to(ROOT))}](<{path}>)'

def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] + ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])

def block(value):
    return '```json\n' + json.dumps(value, indent=2) + '\n```'

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--snapshot', type=Path, required=True)
parser.add_argument('--check', action='store_true')
args = parser.parse_args()
snapshot = read(args.snapshot)
plan = read(A / 'checks/fixed-checkpoint-tuning-plan.json')
rom = read(P / 'runs/matched01/matched-summary.json')
fno = read(A / 'runs/fno_poisson01/field-audit.json')
burgers = read(B / 'checks/refinement02-diagnosis-audit.json')
launch = read(B / 'checks/collection-monitor-launch.json')
# Active collector status is snapshotted by the coordinator, not reread on rebuild.
parts = []
def add(text):
    parts.append(text.strip())

add('''# Claude Code continuation handoff: accuracy, speed and fixed-checkpoint tuning

This is a reproducible handoff snapshot of retained development evidence and unfinished work; it introduces no new GPU result. The canonical root `LAB-LOG.md` remains authoritative, especially when the active Burgers collector advances after this snapshot.''')
add(f'''## Read and preserve before editing

Read {link(ROOT / 'LAB-LOG.md')} first, then {link(ROOT / 'AGENTS.md')}. The live handoff observations are in {link(args.snapshot)}; the adjacent {link(MANIFEST, 'manifest')} pins every JSON used here. Regeneration reads retained evidence only:

```bash
/home/tahmid/Dev/.venv/bin/python reports/generate-claude-code-continuation-handoff.py --snapshot {args.snapshot} --check
```

Omit `--check` only when deliberately rebuilding this handoff and its manifest from an updated snapshot.

The user's objective has two linked parts: better physical accuracy and complete-query speed against tuned FNO/other efficient neural operators and efficient full-order solvers; and classic ViT + CP fixed-checkpoint tunability. Keep decoder weights, latent dimension and CP/bank rank identical within a tuning curve. Change Gauss–Newton iteration cap, stopping tolerance, and selection among offline-fitted EQ rules where implemented. The preceding nested-capacity recommendation was rejected as an answer to this request. Training separate ranks is a capacity experiment, not this inference-time tuning mechanism. No useful competitive joint operating point is established yet.

The scientific worktrees were approved from the corrected consolidated baseline. Continue in an existing approved tree after its current owner finishes; do not branch scientific work from frozen `main`, whose heat rollout is known broken. Preserve every pre-existing modified and untracked main file. Do not reset, stash, stage, commit or clean those files incidentally. Do not merge any experiment branch without the pending user merge decision. Final campaign cohorts remain sealed. The existing frozen archives remain untouched.''')
add('## Coordinator snapshot\n\nObserved at `' + snapshot['observed_at_utc'] + '`. These observations expire; recheck before acting. The date in the filename is the local session date. A collector can legitimately advance a branch after capture. Full command outputs and process identity are retained in the linked snapshot. Its archival commit is recorded in the canonical closing entry; the table records observed heads before that snapshot commit and the subsequent documentation commit, avoiding a self-referential hash.\n\n' + table(['Checkout','Absolute path','Branch','Observed HEAD'], [[name, '`'+v['path']+'`', '`'+v['branch']+'`', '`'+v['head']+'`'] for name,v in snapshot['trees'].items()]))
add('Pre-existing main status to preserve (captured before this handoff):\n\n```text\n'+snapshot['trees']['main']['status']+'\n```')
add('Completed allocation accounting at snapshot time; durations come from Slurm, not model query timings.\n\n'+table(['Job ID','Name','State','Elapsed seconds','Node'], [[v['job_id'],v['job_name'],v['state'],v['elapsed_seconds'],v['node']] for v in snapshot['accounting']]))
add('Original resource allowance, remaining rather than renewed:\n\n'+table(['Lane','Authorized seconds','Consumed seconds','Remaining seconds','Remaining minutes'], [[name,v['authorized_seconds'],v['consumed_seconds'],v['remaining_seconds'],f"{v['remaining_seconds']/60:.3f}"] for name,v in snapshot['budget']['lanes'].items()])+'\n\n'+snapshot['budget']['caution'])
add('Collector refreshed at `' + snapshot['collector_observed_at_utc'] + '`.\n\n'+table(['Job ID','Current PID','Phase','Repair source commit'], [[snapshot['collector']['status']['job_id'], snapshot['collector']['pid'], snapshot['collector']['status']['phase'], snapshot['collector']['restart']['source_commit']]]))
add('Burgers computation is complete in the scheduler record. Remote dataset indices at snapshot time:\n\n'+table(['Split','Complete','Recorded cases','Expected cases'], [[name,v.get('complete'),v.get('records'),v.get('count')] for name,v in snapshot['cluster']['dataset'].items()])+ '\n\nCollection, independent audit, raw archive retention, cache preparation and deletion are separate phases; follow the current collector record rather than inferring these from completed computation.')
add(f'''## Active Burgers collection: finish safely before implementation

The repaired collector is PID `{snapshot['collector']['pid']}`, observed in phase `{snapshot['collector']['status']['phase']}` for job `{launch['job_id']}`. Its command is `{snapshot['collector']['cmdline'].strip()}`. Its current log is {link(Path(snapshot['collector']['stdout']))}; durable current phase is {link(B / 'checks/refinement02-collection-status.json')}. Its bounded lifetime is recorded in the original launch evidence; an old PID or a `monitoring` file alone does not prove it is still alive. Inspect command identity, process state, scheduler state, logs and local lock before any takeover.

The old monitor retried an invalid completed-job queue lookup; the narrow repair queries the account queue and reaches accounting. Preserve `checks/collection-monitor-repair.json`, `checks/collection-monitor-restart.json`, the original monitor log and `runs/refinement02/collection-monitor-restarted.log`; use the restart log for current progress. This recovery submitted no GPU job.

**Do not edit the Burgers scientific source or collector dependencies while collection is pending.** `audit_diagnosis.py` compares archived diagnosis against current source hashes, including engines, accuracy paths and data provenance. Source edits could make valid evidence fail collection. The existing monitor owns its `runs/refinement02/collection-monitor.lock` single-instance lock, full raw archive/checksum verification, saved-field/reference/dataset audits, Git retention, exact remote cleanup, and locked canonical log append. Do not start a duplicate collector or a second writer. Let it complete; verify successful status, archive manifest/parts, retained audit results and exact job-directory removal before assuming completion.

The collector's planned cluster-generated dataset handoff cache is `{launch['handoff_cache']}`. Verify its copied index/case/sidecar hashes and calibration-provenance relocation before use by FNO. A partial training index or ready cache is not proof that validation and full raw collection finished. Do not delete the cache as if it were the completed job directory. If the monitor fails, preserve all evidence and diagnose its actual failed phase; do not rerun dataset generation, restart a job, or delete remote files blindly. Resume/recovery must fit remaining campaign resources and use exact owned paths.

The original failed calibration remains evidence, alongside the refined passing empirical reference. Do not silently replace its protocol or present refinement as a continuum-error theorem. No Burgers FNO training, learned step correction, trajectory retraining, or complete classic tuning sweep is claimed complete in this handoff.''')
add('## Retained Poisson common-data evidence')
rows=[]
for name, values in fno['models'].items():
    e=values['physical_candidate']
    rows.append([f'FNO {name}',len(values['physical_candidate_errors']),f"{100*e['mean']:.6f}",f"{100*e['median']:.6f}",f"{100*e['maximum']:.6f}",e['above_threshold_counts']['0.05']])
for rank, values in rom['results'].items():
    errors=[v['online'] for v in values['summary']['oracles']['physical_candidate']['cases']]
    rows.append([f'Fresh ROM r{rank}',len(errors),f'{100*statistics.mean(errors):.6f}',f'{100*statistics.median(errors):.6f}',f'{100*max(errors):.6f}',sum(v>.05 for v in errors)])
add('Single-seed development screen against the same empirical physical-reference candidates. Shared train/validation case and target hashes match, but training/tuning budgets are not matched. Provisional for broader generalization because repeated seeds, broader families and sealed final tests have not run. No paired FNO/ROM/FOM latency panel exists.\n\n'+table(['Model','Cases','Mean error %','Median error %','Worst error %','Cases above 5%'],rows))
add(f'''Sources: {link(P / 'runs/matched01/matched-summary.json')} and {link(A / 'runs/fno_poisson01/field-audit.json')}. The fresh ROM arms are rejected replacements: more bank capacity improved the bank projection floor but worsened deployed generalization. Historical weights have a different training history and must remain a labelled unmatched-data diagnostic. Investigate bank/head/generalization limits before another blind capacity increase. Direct DST remains the demanding efficient Poisson baseline; tolerance-tuned CG is an additional named control. Strong accuracy against an over-solved CG alone does not establish a competitive operating point.''')
add('## Retained Burgers same-job diagnosis')
rows=[]
for name,e in burgers['summary'].items():
    rows.append([name,f"{e['gpu_median_ms']:.6f}",f"{e['host_to_host_median_ms']:.6f}",f"{100*e['worst_fixed_initial_error']:.6f}",e['failed_stopping_invocations'],e['latency_upper_tukey_outliers']])
add('Audited development calibration evidence against the refined empirical numerical reference; provisional as a broader/continuum claim. The native ROM includes its compressed initial state in this diagnostic. These measurements show an efficient tested FOM faster and more accurate than the inherited ROM.\n\n'+table(['Method','Median GPU ms','Median host ms','Worst fixed-initial error %','Failed stopping invocations','Latency outliers'],rows))
add(f'Source: {link(B / "checks/refinement02-diagnosis-audit.json")}. All timing repetitions are retained. FOM arm identifiers encode the original same/coarse grid, Newton threshold (`nt`) and timestep (`dt`); inspect the immutable diagnosis protocol before changing any control. These are same-job comparisons, not ratios assembled across allocations.')
rows=[]
for name,e in plan['saved_evidence']['errors'].items():
    rows.append([name,f"{e['median_evolved_case_peak_percent']:.6f}",f"{e['worst_evolved_case_peak_percent']:.6f}"])
add('Evolved-field representation diagnosis from already saved invocations; excludes the native fitted initial output. Best-found nonlinear fits are not certified global optima, and the error sources are not additive.\n\n'+table(['Diagnostic','Median case maximum %','Worst %'],rows))
add('Saved iteration summary (existing invocations, no new solves):\n\n'+block({'step_iteration_quantiles':plan['saved_evidence']['step_iteration_quantiles'],'initial_fit_iterations_by_case':plan['saved_evidence']['initial_fit_iterations_by_case']}))
add('## First implementation gaps and the next execution sequence')
add(f'''The source-generated plan is {link(A / 'checks/fixed-checkpoint-tuning-plan.json')}, produced by {link(A / 'prepare_fixed_checkpoint_tuning.py')}. Status: **proposed tests, not run**. The prepared {link(B / 'eq_ablation.py')} is incomplete for the full requested study. Start with {link(B.parent / 'mr-burgers2d/accuracy_paths.py', 'accuracy_paths.py: make_rom stopping controls')}, {link(B.parent / 'mr-burgers2d/engines.py', 'engines.py: weak/build_rom')}, and {link(B.parent / 'mr-burgers2d/pilot.py', 'pilot.py: full-upwind component diagnostic')}. The fixed checkpoint is {link(B.parent / 'separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl')}; verify it against the hash below before execution. Its current scientific invariants are:

{block(plan['fixed'])}

After verified collection closure, the next single scientific owner may work in the approved Burgers worktree; audit, Poisson and consolidated trees remain read-only to that owner. Separate concurrent experiments need their own approved owners, worktrees and namespaces. Do not create new worktrees or assume new allocation authority from this handoff.''')
add('Concrete source gaps, taken directly from the saved plan:\n\n'+'\n'.join('- '+gap for gap in plan['readiness_gaps']))
add('Proposed stages and exact settings, generated from the plan. Each stage requires an immutable config, source hashes, bounded runtime, preserved partial evidence, and its stated control; no row is a completed result.\n\n'+ '\n\n'.join('### '+stage['name']+' — not run\n\n'+block({k:v for k,v in stage.items() if k!='name'}) for stage in plan['proposed_stages']))
add('''Implement the full discrete-upwind weak rollout control before attributing an EQ-versus-ROM gap. Expose evolution residual and normalized-gradient thresholds independently of the initial-fit controls; preserve the original initial fitting defaults exactly. Verify reduced/full weak operator and gradient consistency on a saved case, stopping-record consistency, native/control output parity where settings agree, and requested-output contracts with bounded smoke work before any real batch.

Use the staged order instead of a large combined grid: establish a stable high-effort endpoint on diagnostic cases; compare EQ rules with unchanged solver and initializer; isolate cap; isolate tolerance; then freeze a shortlist on calibration data and evaluate common validation. An optional initial-fit study is separate. All deployable methods return the supplied initial field exactly, while native compression error remains separately reported. Early-stopped finite outputs may appear on a measured physical-error/cost curve with their actual stopping status; they do not inherit stationarity or relaxed acceptance gates. Residual reduction does not guarantee monotone physical accuracy.''')
add('Conditional scientific follow-up:\n\n'+'\n'.join('- '+item for item in plan['conditional_followup']))
add(f'''## FNO and broader campaign work still open

After complete Burgers data handoff, adapt and verify the operator training worker for the Burgers input/output contract. {link(A / 'README.md')} records that the existing fixed queue/staging is Poisson-specific even though model and smoke paths cover Burgers. The supplied initial field, coefficients and requested times are allowed inputs; generation descriptors, case IDs and truth sidecars are not. Keep case-disjoint train/validation splits, training-only normalization and validation-selected checkpoints. Retain full predictions, loss curves, parameters, complex intermediates and checkpoint hashes.

The current FNO screen is an initial capacity screen, not an exhaustively tuned baseline. Learning-rate refinement, repeated seeds, efficient alternatives such as TFNO and broader sampled-field families remain open. Saved optimizer/checkpoint state does not imply resumable training: continuation support in the current driver is not implemented. Add and verify explicit restart semantics before calling interrupted training resumed.

The later authorized hypotheses include supplied-field latent/coefficient prediction, direct same-bank prediction and prediction followed by weak correction, with trajectory-aware training if diagnosis warrants it. Freeze any improved checkpoint before repeating the same solver/EQ curve; do not mix endpoints from different weights. The eventual interleaved same-GPU panel must compare ROM, tuned operator and efficient FOM together, including initialization, interpolation, requested-time inference, full-field output and host transfer where reported. A faster method that misses the declared accuracy gate is a diagnostic, not the competitive target result.''')
add('''## Environment, resource accounting and measurement rules

Continue the original bounded lanes; **do not assume a fresh GPU budget**. Use the captured accounting as a starting observation and refresh `sacct`/`squeue` for every owned historical and active allocation, including retries and running elapsed time, before planning any successor. Subtract actual usage from the original per-lane cap; the unspent time of another lane is not automatically a transferable allowance. If the existing authorized remainder cannot accommodate a bounded next stage, report the concrete needed extension. Preparation and read-only evidence review can proceed without extending compute.

Use only `/home/tahmid/Dev/.venv/bin/python` locally and `/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python` on Tufts. Local GPU work is smoke-only, under a minute, through `jaxrun` after sourcing `/etc/profile.d/jax-mem.sh`, with at most three concurrent instances. Real work belongs in unique paralab job directories on `gpu`, never `preempt`, with `JAX_DEFAULT_MATMUL_PRECISION=highest`, f64 and an explicit GPU-backend preflight that exits on CPU fallback. Check queue before and after submission. Check paralab disk before diagnosing missing output. Copy source directly to paralab and verify content hashes rather than ancestor `git` state. Generate data from seed on cluster; a checksummed copy of existing cluster-generated data is permitted, never replacement with local `data/`.

The retained environment audit records pinned additions with `--no-deps`, preserving the JAX stack. Torch/NeuralOperator passed allocated-GPU dtype/gradient/checkpoint checks despite recorded unsatisfied package pins; `pip check` is not claimed clean. Do not upgrade or reinstall the whole environment to resolve those declared pin mismatches. Preserve this provenance and rerun relevant allocated-GPU checks after any necessary environment change. The reviewed NeuralOperator spectral buffer fix and explicit complex parameter conversion are required for complex128, not just real `.double()` conversion.

Burn in before every timed block; compare complete queries in one job on one GPU; derive accuracy and cost from the same invocation; preserve repetition arrays and report medians, failures and outliers. Keep reference tolerance labels and like-for-like efficient baselines. Record seed, mesh, family, latent dimension, config, commit/content hashes, GPU and job ID. Reject CPU fallbacks, truncated/OOM/disk-full runs or unexplained captured-large-constant warnings. Checksum-pull and archive before exact completed-directory removal.

Minimize the weak residual projected onto smooth test modes, with $M > k$ comfortably and quadrature support approximately $m \\approx 4M$; here $M$ is weak-mode count, $k$ is latent dimension and $m$ is quadrature-point count. Fit nonnegative EQ weights only on decoder-output training snapshots; preserve supports, weights and fit residuals for each rule and refit when mesh/modes change. Hyper-reduce cold initialization too. Keep the FOM-exact upwind operator within Burgers weak advection. Never replace these controls with pointwise/strong random collocation. Never issue bare `scancel`; only the repository's explicit numeric-ID cancellation helper for owned `ctol_*` jobs is permitted.''')
add('## Evidence and restoration map')
evidence = [
(ROOT / 'worktrees/2026-09-13-nmrom-consolidated/consolidated/README.md','Corrected consolidated entry point; implementations, selected checkpoints and self-contained evidence mirror.'),
(P / 'runs/pilot01','Historical-checkpoint Poisson diagnosis, immutable raw archive, source/checkpoint/data evidence and collection audit.'),
(P / 'runs/matched01','Fresh common-data ROM training, per-rank predictions/checkpoints, full raw archive and matched summary.'),
(A / 'runs/fno_poisson01','FNO checkpoint/prediction/source archive, independent field audit, collection records and archive-parts manifest.'),
(A / 'checks/environment01','Original environment versions, additions, preservation audit and unsatisfied dependency record.'),
(A / 'checks','Local/allocated GPU f64 smoke, common-data audits and proposed fixed-checkpoint plan.'),
(B / 'artifacts/calibration01','Retained original failed Burgers calibration; restore according to its README/archive manifest.'),
(B / 'artifacts/refinement02/archive.json','Ultimate committed raw archive manifest and parts after collector completion; do not infer readiness from local convenience views.'),
(B / 'runs/refinement02','Live refined reference/diagnosis evidence and collector log; full final archive may still be pending.'),
(B / 'checks','Passing refined reference and saved-field diagnosis audits, dataset audits and durable collection phase.'),
(B / 'cluster/collect_when_done.py','Exact collector and recovery behavior; inspect without modifying until collection ends.'),
(ROOT / 'reports/2026-09-11-accuracy-improvements-and-wave-speed.md','Prior selected accuracy round with rejected arms and limitations.'),
(ROOT / 'reports/2026-09-11-accuracy-integration.json','Selected-change provenance and original corrected worktree pointers.'),
(ROOT / 'reports/2026-09-11-iterative-fom-multiresolution.md','Older named iterative-FOM comparisons; not a substitute for efficient paired operator panel.'),
(ROOT / 'Older Paper /neurips26__Copy_ (1).zip','Historical manuscript provenance; old single-model/speed claims require the archived corrections.'),
]
add('\n\n'.join(f'- {link(path)} — {desc}' for path,desc in evidence))
add('''For split archives, read each archive's own manifest: concatenate parts in manifest order, verify the aggregate SHA256, inspect archive members, then extract into a fresh scratch directory. Verify file manifests and provenance before using restored content. Do not guess that similarly named archives share a schema. Poisson extracted `archive/` directories retain `MANIFEST.sha256`, `ARCHIVE.sha256`, source and output trees; collection JSONs record successful checks and exact remote deletion. FNO `archive-parts/manifest.json` pins ordered raw chunks; Burgers final raw chunks/manifest are not claimed present until its collector says complete. Preserve failed and superseded attempts alongside successful ones. Never overwrite `best-results/`.

The pre-reset wave evidence remains untrusted under the user's wave reset; only the fresh reflective-wave branch is trusted where independently audited. Absorbing waves remain excluded from the current selected comparison. Existing numerical findings are not newly retracted by this handoff; the nested-capacity recommendation was withdrawn as a response to the user's requested tuning mechanism.''')
add(f'''## Pasteable continuation prompt

```text
Continue the accuracy/speed research in /home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude.
Read the canonical absolute LAB-LOG.md FIRST, then AGENTS.md, then reports/2026-09-14-claude-code-continuation-handoff.md and its manifest. The handoff is a snapshot; refresh live state before acting.

My objective is better physical accuracy and complete-query speed versus tuned FNO/other efficient neural operators AND an efficient FOM, together with CLASSIC ViT+CP fixed-checkpoint tunability: identical decoder weights, latent dimension and rank; vary Gauss-Newton iteration cap, stopping tolerance, and offline-fitted EQ rule. Do not substitute nested-capacity architecture or independently trained ranks for that mechanism. Do not promise a competitive result; measure it without relaxing gates.

First inspect the owned Burgers job/collector, lock, logs, durable status and resource accounting. Let the existing collector finish without editing its current-tree scientific dependencies, duplicating its process or racing archive/log writes. Verify full raw retention, audits, dataset handoff hashes and exact completed-job cleanup. Preserve original failed calibration. Branch heads may advance as it collects.

After safe collection closure, continue within the already approved Burgers worktree as its sole writer; other experiment trees stay read-only. Preserve main dirty files. Do not branch from frozen main, create a new worktree or merge branches without the relevant user decision. Final cohorts stay sealed. Continue only inside the original remaining per-lane GPU allowance; refresh usage before any bounded successor, and do not silently renew the budget.

Use the retained fixed-checkpoint-tuning-plan.json. Implement the missing complete full-upwind weak rollout and independently exposed evolution residual/gradient stopping controls while initial fitting stays unchanged. Bound NNLS and every stage, preserve partial evidence, then smoke-check operator/gradient parity, output and stopping records. Establish the converged sentinel; isolate EQ, cap and tolerance; freeze a shortlist before common validation and interleaved same-GPU FNO/ROM/efficient-FOM timing. Keep early-stop status explicit. Diagnose representation, EQ and dynamics separately; if training changes are warranted, improve one checkpoint, freeze it, and repeat this SAME fixed-checkpoint study.

Complete the Burgers common-data operator baseline and diagnose the fresh Poisson ROM generalization failures. Respect the existing FNO restart limitation, f64/complex128 checks, held-out input contract, training-only normalization and source hashes. Record all attempted outcomes, median complete-query times, physical errors, outliers and failures from matched invocations. Do not compare timing across jobs or call the inherited historical ROM matched-data evidence.

Use absolute venv paths, cluster GPU preflight, unique paralab directories, highest matmul precision, warmups, timing arrays, checksum archives and exact cleanup. Append the canonical LAB-LOG under its established lock before ending; keep reports source-generated. Ask whether to merge when experiments finish; do not merge unprompted. Start with the concrete safe next step rather than restarting the whole campaign.
```''')
add('''## Glossary

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
''')
content='\n\n'.join(parts)+'\n'
manifest={'kind':'generated continuation snapshot; canonical LAB-LOG authoritative','report':str(REPORT),'generator':str(Path(__file__).resolve()),'snapshot':str(args.snapshot.resolve()),'sources':SOURCES,'report_sha256':hashlib.sha256(content.encode()).hexdigest(),'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'result_status':'development evidence only; classic tuning stages not run; live state expires'}
manifest_content=json.dumps(manifest,indent=2)+'\n'
if args.check:
    assert REPORT.read_text()==content, 'Report differs from regeneration'
    assert MANIFEST.read_text()==manifest_content, 'Manifest differs from regeneration'
    print('Exact report and manifest regeneration passed')
else:
    REPORT.write_text(content)
    MANIFEST.write_text(manifest_content)
    print(REPORT)
