"""Generate this lane's report and `summary.json` from audited records only.

Nothing is typed. Every number comes from one of five hash-pinned sources:

* `runs/don01/audit.json` — this lane's independent NumPy audit of its own job;
* `../no-second/runs/{unet01,tsol01}/audit.json` — the U-Net and Transolver audits
  (jobs 3780138 / 3780139), inherited byte-for-byte at the fork commit;
* `../neural-operator-audit/runs/fno_burgers02/field-audit.json` — the FNO lane's audit;
* `../no-second/checks/refinement02-diagnosis-audit.json` — the Burgers ROM/FOM lane's
  matched eight-case audit (job 3702709), hash-pinned.

No speed ratio is formed anywhere. Timing rows are same-job only, one table per job, and
`cross_job=true` marks every row that came from another allocation.

    python reports/generate_report.py
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LANE = HERE.parent
WT = LANE.parents[1]
EXPERIMENTS = LANE.parent
FNO_AUDIT = EXPERIMENTS / 'neural-operator-audit/runs/fno_burgers02/field-audit.json'
DIAGNOSIS = EXPERIMENTS / 'no-second/checks/refinement02-diagnosis-audit.json'
DIAGNOSIS_SHA256 = 'ffa77d1b8bc44d2bd3d0ac2753444d2e1ff379898735258b60b9d68a41e3e187'
DIAGNOSIS_JOB = '3702709'
SECOND = {'unet01': EXPERIMENTS / 'no-second/runs/unet01/audit.json',
          'tsol01': EXPERIMENTS / 'no-second/runs/tsol01/audit.json'}
LABEL = {'unet': 'U-Net', 'transolver': 'Transolver', 'fno': 'FNO', 'deeponet': 'DeepONet'}
V1_BAR = 1.5  # DESIGN D1, inherited unchanged from no-second's V1


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pct(x):
    return '—' if x is None else f'{100 * x:.4f}'


def still_improving(best_epoch, epochs_completed, fraction=0.95):
    """True when the selected checkpoint sits in the last 5% of the epochs the arm ran: the
    stopping condition fired while validation was still improving, so the number is a lower
    bound on that configuration, not a converged value. `no-second`'s own definition."""
    return bool(best_epoch >= fraction * max(epochs_completed - 1, 1))


def capacity_of(config):
    family = config.get('family', 'fno')
    if family == 'unet':
        return f"base {config['base']}"
    if family == 'transolver':
        return f"dim {config['dim']}, {config['layers']} layers, patch {config['patch']}"
    if family == 'deeponet':
        return f"width {config['width']}, rank {config['rank']}, trunk {config['trunk_width']}"
    return f"width {config['width']}, {config['layers']} layers, {config['modes']} modes"


def arm_records(audit, source, source_sha, cross_job, job_id=None, gpu=None, attempt=None):
    """One dict per trained arm, family-agnostic, from an audit.json or the FNO field-audit."""
    out = []
    models = audit.get('arms', audit.get('models'))
    cohort = (audit.get('cohort') or {}).get('models') or (audit.get('diagnosis_cohort') or {}).get('models') or {}
    timing = (audit.get('timing') or {}).get('models') or {}
    for arm, r in models.items():
        if not (r.get('complete', True) and 'fixed_initial' in r):
            continue
        stop = r.get('stop_reason') or ('wall_budget' if r.get('stopped_by_wall_budget') else 'other')
        out.append(dict(
            arm=arm, family=r.get('family', 'fno'), config=r['config'], params=r['real_parameter_count'],
            dtype=r.get('parameter_dtype', 'torch.float64'), seed=r['config']['seed'],
            epochs=r['epochs_completed'], best_epoch=r['best_epoch'], stop_reason=stop,
            budget_s=r.get('wall_budget_seconds', 3000.0), training_s=r.get('training_seconds'),
            validation=r['fixed_initial'], cohort=(cohort.get(arm) or {}).get('fixed_initial'),
            timing=timing.get(arm), job_id=job_id or audit.get('job_id'),
            gpu=gpu or audit.get('gpu', 'NVIDIA A100 80GB PCIe'), attempt=attempt or audit.get('attempt'),
            source=source, source_sha256=source_sha, cross_job=cross_job,
            still_improving=still_improving(r['best_epoch'], r['epochs_completed'])))
    return out


def load_all():
    records, sources = [], {}

    def add(path, cross_job, **kw):
        audit = json.loads(Path(path).read_text())
        digest = sha(path)
        sources[str(Path(path).relative_to(WT))] = digest
        records.extend(arm_records(audit, str(Path(path).relative_to(WT)), digest, cross_job, **kw))
        return audit

    mine = add(LANE / 'runs/don01/audit.json', False)
    for attempt, path in SECOND.items():
        add(path, True)
    fno = add(FNO_AUDIT, True, job_id=json.loads(FNO_AUDIT.read_text()).get('job_id'), attempt='fno_burgers02')
    diagnosis = json.loads(DIAGNOSIS.read_text())
    assert sha(DIAGNOSIS) == DIAGNOSIS_SHA256, 'the pinned ROM/FOM diagnosis audit changed'
    sources[str(DIAGNOSIS.relative_to(WT))] = DIAGNOSIS_SHA256
    return mine, fno, diagnosis, records, sources


def selected(records, family):
    """DESIGN 3.1: argmin over ALL arms of that family, refine included, of the validation-32
    mean case-maximum fixed-initial error."""
    arms = [r for r in records if r['family'] == family]
    return min(arms, key=lambda r: r['validation']['mean']) if arms else None


def capacity_table(records):
    lines = ['| Run | Operator | Capacity | dtype | Real parameters | Epochs | Best epoch | Still improving? | '
             'Training s | Budget s | Ended by | Job |',
             '| --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- |']
    for r in records:
        t = '—' if r['training_s'] is None else f"{r['training_s']:.0f}"
        lines.append(f"| `{r['arm']}` | {LABEL[r['family']]} | {capacity_of(r['config'])} | "
                     f"{r['dtype'].replace('torch.', '')} | {r['params']} | {r['epochs']} | {r['best_epoch']} | "
                     f"{'yes' if r['still_improving'] else 'no'} | {t} | {r['budget_s']:.0f} | "
                     f"{r['stop_reason'].replace('_', ' ')} | {r['job_id']} |")
    return '\n'.join(lines)


def accuracy_table(records, key, title):
    lines = [f'**{title}**', '',
             '| Run | Operator | mean (%) | median (%) | worst (%) | cases > 5 % | Job |',
             '| --- | --- | ---: | ---: | ---: | ---: | --- |']
    for r in sorted((r for r in records if r[key]), key=lambda r: r[key]['mean']):
        s = r[key]
        lines.append(f"| `{r['arm']}` | {LABEL[r['family']]} | {pct(s['mean'])} | {pct(s['median'])} | "
                     f"{pct(s['maximum'])} | {s['above_threshold_counts']['0.05']} | {r['job_id']} |")
    return '\n'.join(lines)


def cohort_table(records, diagnosis):
    rows = [(r['arm'], LABEL[r['family']], r['cohort']['maximum'], r['job_id']) for r in records if r['cohort']]
    rows += [(name, 'ROM' if name == 'rom' else 'FOM', s['worst_fixed_initial_error'], DIAGNOSIS_JOB)
             for name, s in diagnosis['summary'].items()]
    lines = ['| Subject | Kind | worst fixed-initial error (%) | Job |', '| --- | --- | ---: | --- |']
    for arm, kind, value, job in sorted(rows, key=lambda t: t[2]):
        lines.append(f"| `{arm}` | {kind} | {pct(value)} | {job} |")
    return '\n'.join(lines)


def timing_table(records, attempt):
    rows = [r for r in records if r['attempt'] == attempt and r['timing']]
    if not rows:
        return '_No same-job timing block for this attempt._'
    lines = [f'| Arm | device query median (ms) | host transfer median (ms) | measurements |',
             '| --- | ---: | ---: | --- |']
    for r in sorted(rows, key=lambda r: r['timing']['device_pooled_median_ms']):
        t = r['timing']
        cases, reps = t['repetitions']  # the retained array's shape, not a count
        lines.append(f"| `{r['arm']}` | {t['device_pooled_median_ms']:.3f} | {t['host_pooled_median_ms']:.3f} | "
                     f"{reps} per case × {cases} cases = {reps * cases} |")
    return '\n'.join(lines)


def capacity_observation(mine_records):
    """Does more capacity buy accuracy here? Generated, because the answer decides whether an
    under-capacity reading is available."""
    caps = sorted((r for r in mine_records if r['arm'] != 'refine' and not r['arm'].endswith('-refine')),
                  key=lambda r: r['params'])
    if len(caps) < 2:
        return ''
    best, worst = min(caps, key=lambda r: r['validation']['mean']), max(caps, key=lambda r: r['validation']['mean'])
    monotone = all(caps[i]['validation']['mean'] <= caps[i + 1]['validation']['mean'] for i in range(len(caps) - 1))
    trend = ('accuracy gets **worse** monotonically as capacity grows' if monotone else
             f"the largest capacity is not the most accurate: `{best['arm']}` ({best['params']} parameters) "
             f"beats `{worst['arm']}` ({worst['params']})")
    tail = ('worst-case error does not follow that ordering' if not monotone_worst(caps) else
            'worst-case error follows the same ordering')
    return (f"Over the {len(caps)} capacities, {trend} "
            f"({' → '.join(f'{pct(r["validation"]["mean"])} %' for r in caps)} mean, smallest to largest); "
            f"{tail}. Three coupled configurations are not a capacity sweep, so this does not rule out "
            f"under-capacity — it says that making *these* knobs bigger, under this schedule, did not help.")


def budget_caveat(mine_records):
    stops = {r['stop_reason'] for r in mine_records}
    if stops == {'early_stopping'}:
        return ("One 3000 s budget per capacity, none of which was exhausted: every arm ran out of validation "
                "patience first, so what is untested here is a different stopping rule or schedule, not a "
                "longer run of this one.")
    if stops == {'wall_budget'}:
        return "One 3000 s budget per capacity, which every arm exhausted, so every error here is a lower bound."
    return ("One 3000 s budget per capacity; arms ended in more than one way (" +
            ', '.join(sorted(s.replace('_', ' ') for s in stops)) + "), see §1.")


def monotone_worst(caps):
    return all(caps[i]['validation']['maximum'] <= caps[i + 1]['validation']['maximum'] for i in range(len(caps) - 1))


def budget_paragraph(mine_records, records):
    """DESIGN 3: say what actually ended each arm, generated. `no-second` reported every Burgers
    arm ending on its wall budget; that must not be asserted here, it must be derived."""
    wall = [r['arm'] for r in mine_records if r['stop_reason'] == 'wall_budget']
    early = [r['arm'] for r in mine_records if r['stop_reason'] == 'early_stopping']
    other = [r['arm'] for r in mine_records if r['stop_reason'] not in ('wall_budget', 'early_stopping')]
    improving = [r['arm'] for r in mine_records if r['still_improving']]
    siblings = [r for r in records if r['cross_job']]
    sib_wall = sum(1 for r in siblings if r['stop_reason'] == 'wall_budget')
    parts = []
    if early:
        parts.append(
            f"**{len(early)} of {len(mine_records)} arms here ended by early stopping, not on the wall budget** "
            f"({', '.join('`%s`' % a for a in early)}): the patience rule fired, so training had stopped "
            f"improving for 250 consecutive epochs while budget remained. **These are the first Burgers arms "
            f"in this comparison to end that way** — {sib_wall} of {len(siblings)} arms in the FNO, U-Net and "
            f"Transolver jobs ended on their budget. For an early-stopped arm the error is **not** a lower "
            f"bound imposed by the budget: the budget was there and the schedule stopped anyway. What that "
            f"establishes is narrow and worth stating exactly — 250 consecutive epochs produced no new best "
            f"**validation selection score** under *this* schedule. It does not establish that no further "
            f"training could help; the training loss was still falling in all four histories, and a different "
            f"patience, learning-rate schedule or stopping rule is untested here.")
    if wall:
        parts.append(f"{len(wall)} arm(s) ended on the wall budget ({', '.join('`%s`' % a for a in wall)}); "
                     f"for those the budget binds and the error is a lower bound on that configuration.")
    if other:
        parts.append(f"{len(other)} arm(s) ended another way ({', '.join('`%s`' % a for a in other)}).")
    parts.append("No arm here was still improving when it stopped." if not improving else
                 f"Still improving when it stopped: {', '.join('`%s`' % a for a in improving)} — those errors "
                 f"are lower bounds.")
    return ' '.join(parts)


def criteria(records, don, fno_large):
    """DESIGN 4, evaluated in code. Every ratio is accuracy, never time."""
    out = []
    for metric, name in (('maximum', 'worst'), ('median', 'median')):
        ratio = don['validation'][metric] / fno_large['validation'][metric]
        out.append(dict(criterion='D1', comparison=f'{name} vs fno-large', ratio=ratio, bar=V1_BAR,
                        passed=bool(ratio <= V1_BAR)))
    rom = next(s['worst_fixed_initial_error'] for n, s in DIAG['summary'].items() if n == 'rom')
    if don['cohort']:
        out.append(dict(criterion='D2', comparison='matched-8 worst vs ROM', value=don['cohort']['maximum'],
                        rom=rom, passed=bool(don['cohort']['maximum'] < rom)))
    ranking = sorted({r['family'] for r in records},
                     key=lambda f: selected(records, f)['validation']['maximum'])
    out.append(dict(criterion='D3', comparison='family ranking by selected-arm validation worst',
                    ranking=[LABEL[f] for f in ranking],
                    deeponet_is_weakest=bool(ranking[-1] == 'deeponet')))
    return out


def tail_warning(records, chosen):
    """DESIGN 3.1: if an unselected sibling of the same family has a better worst case than the
    selected arm, that must be said beside the verdict, naming both arms."""
    siblings = [r for r in records if r['family'] == chosen['family'] and r['arm'] != chosen['arm']]
    better = [r for r in siblings if r['validation']['maximum'] < chosen['validation']['maximum']]
    if not better:
        return None
    best = min(better, key=lambda r: r['validation']['maximum'])
    return (f"The selection rule optimises the mean, not the tail, and it did so here: the selected "
            f"`{chosen['arm']}` has worst case {pct(chosen['validation']['maximum'])} % while the unselected "
            f"`{best['arm']}` has {pct(best['validation']['maximum'])} %. The rule is the FNO lane's own, "
            f"applied unchanged and pre-registered in DESIGN §3.1 before the job; it is not changed after "
            f"the fact, and the better arm is not quietly reported in its place.")


def rows_for_summary(records, criteria_rows):
    rows = []
    for r in records:
        base = {k: r[k] for k in ('arm', 'params', 'dtype', 'seed', 'budget_s', 'epochs', 'best_epoch',
                                  'stop_reason', 'job_id', 'gpu', 'attempt', 'source', 'source_sha256',
                                  'cross_job', 'still_improving')}
        base.update(operator=LABEL[r['family']], capacity=capacity_of(r['config']), pde='burgers')
        for cohort, stats in (('validation-32', r['validation']), ('diagnosis-8', r['cohort'])):
            if not stats:
                continue
            for metric, value in (('worst', stats['maximum']), ('median', stats['median']),
                                  ('mean', stats['mean']), ('p95', stats.get('p95'))):
                if value is not None:
                    rows.append(dict(base, cohort=cohort, metric=f'{metric}_fixed_initial_error', value=value,
                                     key=f"{r['arm']}|{r['job_id']}"))
        if r['timing'] and not r['cross_job']:  # DESIGN A4: this lane exports no other job's times
            for metric in ('device_pooled_median_ms', 'host_pooled_median_ms'):
                rows.append(dict(base, cohort='same-job timing', metric=metric, value=r['timing'][metric],
                                 key=f"{r['arm']}|{r['job_id']}",
                                 admissible_as_speed_claim=False,
                                 note='same-job, same-GPU device query; NOT divided by any other job'))
    rows += [dict(c, cohort='criterion', metric=c['criterion'], operator='DeepONet', pde='burgers') for c in criteria_rows]
    return rows


DIAG = None


def main():
    global DIAG
    mine, fno, DIAG, records, sources = load_all()
    don = selected(records, 'deeponet')
    fno_large = next(r for r in records if r['arm'] == 'fno-large')
    criteria_rows = criteria(records, don, fno_large)
    warning = tail_warning(records, don)
    mine_records = [r for r in records if not r['cross_job']]

    generated = dt.date.today().isoformat()
    text = f"""# A DeepONet on 2D viscous Burgers at 256², beside the FNO, U-Net and Transolver

Generated by `reports/generate_report.py` on {generated} from audited records only; every
**measured** number in this file — every error, every count, every time — is read from one of
the hash-pinned sources listed at the end and is never typed. Protocol constants and identifiers
that are not measurements (the 1.5× D1 bar, the patience, the ROM/FOM and sibling job ids, the
`ops-timing-panel` reference in §5) are literals in the generator, pinned by `DESIGN.md` and by
that lane's own record rather than by these five audits. Status: **final for the accuracy panel of job `{mine['job_id']}`**. The
timing column is same-job only and is not a speed claim — see §5.

DeepONet was the one operator named in the paper's abstract that had never been trained in
2D. This lane trains it on exactly the data, split, metric, budget, optimiser, selection
rule and seed that the U-Net and Transolver arms used (jobs 3780138 / 3780139) and that the
FNO arms used before them, and reports it beside them. The design was pre-registered in
`DESIGN.md` before the job; §3.1 fixed the selection metric and §4 the criteria.

## 1. What ran

Job `{mine['job_id']}` on {mine['gpu']}, source commit `{mine['source_commit'][:8]}`,
`jax_backend={mine['jax_backend']}`, {mine['data_files_verified']} data files verified in the
preamble, train/validation index hashes asserted equal to the FNO job's
(`{mine['train_index_sha256'][:8]}…` / `{mine['validation_index_sha256'][:8]}…`).

{capacity_table(mine_records)}

{budget_paragraph(mine_records, records)}

Nothing here is an architecture ceiling either way: no DeepONet-specific hyperparameter search
was run, the schedule is the U-Net's, and a family that early-stops under an inherited schedule
may simply need a different one.

## 2. Validation-32 accuracy, all four families

{accuracy_table(records, 'validation', 'Fixed-initial relative error, 32 held-out validation cases, '
                'recomputed from the saved prediction fields by an audit that imports neither torch nor jax')}

{capacity_observation(mine_records)}

Selected DeepONet arm, by the pre-registered rule (argmin validation mean case-maximum over
all arms including `refine`): **`{don['arm']}`**, {don['params']} real parameters,
{don['epochs']} epochs.

{warning or 'The selected arm also has the best worst case among its own family.'}

## 3. The matched eight-case ROM / FOM cohort

Rebuilt in-job from the same 4096-interval anchors, cohort index hash asserted equal to the
FNO job's. Accuracy is comparable across these jobs; **timing is not, and none is taken.**

{cohort_table(records, DIAG)}

## 4. Pre-registered criteria (DESIGN §4)

```json
{json.dumps(criteria_rows, indent=2)}
```

## 5. Timing — same job only

{timing_table(records, mine['attempt'])}

These are device-resident query times measured inside job `{mine['job_id']}` on its own GPU,
retained per repetition. **They are not a speed claim.** A speed number is admissible only
from a same-allocation panel in which the ROM, the operator and the FOM are timed in one job
on one GPU; no such panel has been run for these checkpoints, so **no speed statement about this
DeepONet is admissible from this lane**. No time here is divided by a time from any other job.

The admissible route exists and is prepared rather than run here: the `ops-timing-panel` lane
(job 4179247) already times the FNO, U-Net and Transolver checkpoints beside the NM-ROM, POD
and the named full-order solver in one allocation, and its `operators.json` extends by adding
rows. `reports/timing-handoff.json` in this lane carries exactly those rows — every DeepONet
checkpoint's path and SHA256, re-verified against the hash the training job recorded, plus the
`families.py` that harness needs to build the family. Running it there rather than copying the
harness here avoids a second copy of a 46-file harness for one extra family.

## 6. Caveats that must travel with these numbers

Single seed. One mesh (256 intervals). One Gaussian continuum family. {budget_caveat(mine_records)}
The eight-case cohort's
worst column is one case. Hyperparameters were inherited from the FNO lane and not re-tuned
per family; `refine` is the only family-level tuning. The float32 network gets more epochs
per second than the float64 FNO did — favourable to this lane, and the epoch counts are in
the table. The trunk is a coordinate MLP with sinusoidal features, the form this project's
3D lanes use; a different trunk is the first thing a reviewer would vary. And a DeepONet
compresses the whole 257² field through a small global bottleneck before its trunk, while the
FNO, U-Net and Transolver beside it are full-resolution field-to-field maps — that is what the
architecture is, not a defect of this implementation — but one implementation of one family,
on one schedule it did not choose, is evidence about this recipe and not a verdict on DeepONets.

## 7. Sources

| file | SHA256 |
| --- | --- |
""" + '\n'.join(f'| `{name}` | `{digest}` |' for name, digest in sorted(sources.items())) + f"""

Generator SHA256 `{sha(__file__)}`.

## 8. Glossary

- **arm** — one trained model: a capacity (`small`/`medium`/`large`) or the `refine` rerun of
  the selected capacity at a lower learning rate.
- **capacity** — the size knob of a family: U-Net `base` channels, Transolver `dim`, FNO
  `width`/`modes`, DeepONet branch `width`, output `rank` and `trunk` width.
- **refine** — a second training run of whichever capacity validation selected, at learning
  rate 3e-4 instead of 1e-3, on the same budget. It is an arm like any other and competes in
  the selection.
- **fixed-initial relative error** — the l2 discrepancy over the interior nodes between the
  predicted and reference field, divided by the l2 norm of the *supplied initial* field, so
  every output time is normalised by the same fixed quantity. The per-case number is the
  maximum over the six output times.
- **validation-32** — the 32 held-out validation cases of the shared Burgers dataset. Used for
  selection and for the headline table. Not the final cohort, which stays sealed.
- **matched eight-case cohort (diagnosis-8)** — eight cases the Burgers ROM/FOM lane also
  solved, rebuilt in this job from the same 4096-interval anchors, so the operator, the ROM
  and the full-order solver are graded on the same cases against the same reference.
- **ROM** — this project's reduced-order model, the subject the operators are being compared
  with. **FOM** — the full-order finite-difference solver; `same_nt1e-2_dt005` and its
  siblings are that solver run at looser tolerances, i.e. cheaper and less accurate settings.
- **worst / median / mean** — over the cases of a cohort, of the per-case maximum-over-time
  error. "cases > 5 %" counts how many cases exceed five percent.
- **still improving** — a heuristic flag: the best checkpoint fell in the last 5 % of the epochs
  the arm ran, i.e. validation was improving recently when the run ended. It is a hint that the
  run stopped mid-progress, not a proof that more training would have helped.
- **wall budget** — the fixed number of seconds each capacity is allowed to train. Equal
  budget, not equal epochs, is what is held constant across families.
- **cross-job** — a row measured in a different Slurm allocation. Accuracy may be read across
  jobs here because the data, split, metric and reference are identical; timing may not.
- **same-allocation panel** — a single job that times the ROM, the operator and the full-order
  solver on one GPU. The only construction from which a speed ratio may be quoted.
- **device query / host transfer** — the time to produce the complete trajectory in GPU
  memory, and separately the time to copy it back to the host.
- **fixed contract** — every family consumes the same feature tensor and emits the same five
  evolved fields, which are then masked to the zero boundary with the supplied initial state
  prepended; nothing between families differs except the network.
"""
    report = HERE / f'{generated}-ops-deeponet-b2d.md'
    report.write_text(text)
    (HERE / 'summary.json').write_text(json.dumps(dict(
        generated=generated, report=report.name, generator_sha256=sha(__file__), sources=sources,
        rule='every row carries its source file and that file SHA256; no row is typed',
        rows=rows_for_summary(records, criteria_rows)), indent=2) + '\n')
    print(report)
    print(json.dumps(criteria_rows, indent=2))
    if warning:
        print(warning)


if __name__ == '__main__':
    main()
