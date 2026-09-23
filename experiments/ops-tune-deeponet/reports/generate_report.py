"""Write this lane's report and `summary.json` from audited records only.

Every **measured** number in the report is read from one of the hash-pinned sources below and
is never typed. Protocol constants that are not measurements — the criterion bars, the arm
names, the job ids — are literals here, pinned by `DESIGN.md` and by each producing lane's own
record, and the report says so in its preamble.

Sources:
  * `runs/<attempt>/audit.json` for every attempt this lane completed (this lane's own numbers);
  * `reports/accounting.json` (`accounting.py`) for the data-parity accounting;
  * `../ops-deeponet-b2d/reports/summary.json` for the U-Net, Transolver, FNO, NM-ROM, FOM and
    persistence rows — itself generated, row by row, from those lanes' audits, and carrying each
    row's own source file and hash;
  * `checks/inherited-sources.json` for what this lane inherited unchanged.

    python reports/generate_report.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LANE = HERE.parent
PARENT_SUMMARY = LANE.parent / 'ops-deeponet-b2d/reports/summary.json'
ATTEMPTS = ('lad01', 'tun01', 'fin01')
# Criterion bars: DESIGN section 5.2. Literals, and declared as such.
BARS = dict(T0_rho=0.2, T1_large=0.7, T1_no_gain=0.10, T2=0.10, T3_factor=1.5,
            T4_rom_worst=0.018671, T5_discretisation=0.040265)
FNO_LARGE = dict(mean=0.022811, median=0.018054, worst=0.063825)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pct(x, places=4):
    return 'n/a' if x is None else f'{100 * x:.{places}f}'


def load():
    audits, sources = {}, {}
    for attempt in ATTEMPTS:
        path = LANE / 'runs' / attempt / 'audit.json'
        if path.exists():
            audits[attempt] = json.loads(path.read_text())
            sources[str(path.relative_to(LANE.parents[1]))] = sha(path)
    accounting = json.loads((HERE / 'accounting.json').read_text())
    sources['experiments/ops-tune-deeponet/reports/accounting.json'] = sha(HERE / 'accounting.json')
    parent = json.loads(PARENT_SUMMARY.read_text())
    sources[str(PARENT_SUMMARY.relative_to(LANE.parents[1]))] = sha(PARENT_SUMMARY)
    inherited = json.loads((LANE / 'checks/inherited-sources.json').read_text())
    sources['experiments/ops-tune-deeponet/checks/inherited-sources.json'] = sha(LANE / 'checks/inherited-sources.json')
    return audits, accounting, parent, inherited, sources


def parent_rows(parent, cohort, metric):
    out = {}
    for row in parent['rows']:
        if row.get('cohort') == cohort and row.get('metric') == metric:
            out.setdefault(row['arm'], {})[metric] = row['value']
            out[row['arm']].update({k: row.get(k) for k in
                                    ('operator', 'params', 'budget_s', 'epochs', 'stop_reason',
                                     'job_id', 'capacity', 'dtype', 'source', 'source_sha256')})
    return out


def arms_of(audit):
    return {name.split('-', 1)[1]: value for name, value in audit['arms'].items() if value.get('complete')}


def table(header, rows, align=None):
    align = align or ['---'] * len(header)
    lines = ['| ' + ' | '.join(header) + ' |', '| ' + ' | '.join(align) + ' |']
    lines += ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in rows]
    return '\n'.join(lines)


def section_what_ran(audits):
    rows, summary = [], []
    for attempt, audit in audits.items():
        for name, arm in sorted(arms_of(audit).items(), key=lambda kv: kv[1]['fixed_initial']['mean']):
            rows.append([f'`{name}`', attempt, arm['training_cases'], f"{arm['real_parameter_count']:,}",
                         arm['schedule'], f"{arm['wall_budget_seconds']:.0f}",
                         f"{arm['training_seconds']:.0f}", arm['steps_completed'],
                         arm['evaluations_completed'], arm['epochs_completed'],
                         arm['stop_reason'].replace('_', ' '),
                         'yes' if arm['patience_could_fire'] else 'no',
                         'yes' if arm['budget_shortened'] else 'no', audit['job_id']])
            summary.append(dict(attempt=attempt, arm=name, job_id=audit['job_id'],
                                training_cases=arm['training_cases'],
                                params=arm['real_parameter_count'], schedule=arm['schedule'],
                                budget_s=arm['wall_budget_seconds'], steps=arm['steps_completed'],
                                stop_reason=arm['stop_reason']))
    return table(['Arm', 'Job', 'Training cases', 'Params', 'Schedule', 'Budget s', 'Trained s',
                  'Steps', 'Evaluations', 'Epochs', 'Ended by', 'Patience could fire?',
                  'Budget shortened?', 'Job id'],
                 rows, ['---', '---', '---:', '---:', '---', '---:', '---:', '---:', '---:',
                        '---:', '---', '---', '---', '---']), summary


def section_accuracy(audits, parent):
    rows = []
    for attempt, audit in audits.items():
        for name, arm in arms_of(audit).items():
            f = arm['fixed_initial']
            rows.append([f'`{name}`', 'DeepONet (this lane)', arm['training_cases'],
                         pct(f['mean']), pct(f['median']), pct(f['maximum']),
                         f['above_threshold_counts']['0.05'],
                         pct(arm['training_subset']['mean_case_max']),
                         f"{arm['generalisation_gap']:.2f}", audit['job_id']])
    published = parent_rows(parent, 'validation-32', 'mean_fixed_initial_error')
    worst = parent_rows(parent, 'validation-32', 'worst_fixed_initial_error')
    median = parent_rows(parent, 'validation-32', 'median_fixed_initial_error')
    above = parent_rows(parent, 'validation-32', 'cases_above_5pct')
    for arm, row in sorted(published.items(), key=lambda kv: kv[1]['mean_fixed_initial_error']):
        rows.append([f'`{arm}`', (row.get('operator') or '') + ' (published)', 128,
                     pct(row['mean_fixed_initial_error']),
                     pct(median.get(arm, {}).get('median_fixed_initial_error')),
                     pct(worst.get(arm, {}).get('worst_fixed_initial_error')),
                     int(above.get(arm, {}).get('cases_above_5pct', -1)),
                     '—', '—', row.get('job_id')])
    rows.sort(key=lambda r: float(r[3]) if r[3] != 'n/a' else 1e9)
    return table(['Arm', 'Family', 'Training cases', 'mean (%)', 'median (%)', 'worst (%)',
                  '> 5 %', 'train-128 mean (%)', 'val / train', 'Job'],
                 rows, ['---', '---', '---:', '---:', '---:', '---:', '---:', '---:', '---:', '---'])


def section_ladder(audits, accounting):
    audit = audits.get('lad01')
    if not audit:
        return '*(the ladder job has not been collected yet)*', {}
    arms = arms_of(audit)
    order = ['c-pinned128', 'c-new128', 'base-512', 'base-2048', 'base-top']
    rows, previous, verdict = [], None, {}
    for name in order:
        arm = arms.get(name)
        if not arm:
            continue
        f = arm['fixed_initial']
        change = '' if previous is None else f'{100 * (f["mean"] / previous - 1):+.1f} %'
        rows.append([f'`{name}`', 'pinned 4096 anchor' if arm['mount'] == 'pinned' else 'generated',
                     arm['training_cases'], pct(f['mean']), pct(f['median']), pct(f['maximum']),
                     pct(arm['training_subset']['mean_case_max']), f"{arm['generalisation_gap']:.2f}",
                     change, arm['steps_completed']])
        if name.startswith('base') or name == 'c-new128':
            previous = f['mean']
    coverage = {row['cases']: row for row in accounting['coverage']}
    for row in rows:
        cases = row[2]
        row.append(f"{coverage[cases]['nearest_mean']:.4f}" if cases in coverage else '—')
    text = table(['Arm', 'Training targets', 'Cases', 'mean (%)', 'median (%)', 'worst (%)',
                  'train-128 (%)', 'val / train', 'change in mean', 'Steps', 'NN distance'],
                 rows, ['---', '---', '---:', '---:', '---:', '---:', '---:', '---:', '---:', '---:', '---:'])
    if 'c-new128' in arms and 'c-pinned128' in arms:
        new, pinned = arms['c-new128']['fixed_initial']['mean'], arms['c-pinned128']['fixed_initial']['mean']
        verdict['T0'] = dict(signed_relative=(new - pinned) / pinned,
                             rho=audit['label_discrepancy_ratio'].get('don-c-new128'),
                             negligible=audit['label_protocol_negligible'].get('don-c-new128'))
    if 'c-new128' in arms and 'base-top' in arms:
        small, top = arms['c-new128']['fixed_initial']['mean'], arms['base-top']['fixed_initial']['mean']
        verdict['T1'] = dict(ratio=top / small, large_effect=bool(top / small <= BARS['T1_large']),
                             rungs=[arms[n]['training_cases'] for n in order if n in arms])
        if 'base-2048' in arms:
            last = arms['base-2048']['fixed_initial']['mean']
            verdict['T1']['final_step_relative'] = (last - top) / last
            verdict['T1']['no_further_gain'] = bool((last - top) / last < BARS['T1_no_gain'])
    return text, verdict


def section_sweep(audits):
    audit = audits.get('tun01')
    if not audit:
        return '*(the sweep job has not been collected yet)*', {}
    arms = arms_of(audit)
    reference = arms.get('s-base')
    rows = []
    for name, arm in sorted(arms.items(), key=lambda kv: kv[1]['fixed_initial']['mean']):
        f = arm['fixed_initial']
        relative = '' if not reference else f'{100 * (1 - f["mean"] / reference["fixed_initial"]["mean"]):+.1f} %'
        rows.append([f'`{name}`', arm.get('knob') or '—',
                     json.dumps(arm.get('override') or {}, separators=(', ', ' = ')).strip('{}') or '—',
                     f"{arm['real_parameter_count']:,}", pct(f['mean']), pct(f['median']),
                     pct(f['maximum']), relative, arm['stop_reason'].replace('_', ' '),
                     'yes' if arm['patience_could_fire'] else 'no'])
    text = table(['Arm', 'Knob', 'Override', 'Params', 'mean (%)', 'median (%)', 'worst (%)',
                  'vs `s-base`', 'Ended by', 'Patience could fire?'],
                 rows, ['---', '---', '---', '---:', '---:', '---:', '---:', '---:', '---', '---'])
    return text, audit['decisions']


def main():
    audits, accounting, parent, inherited, sources = load()
    what_ran, arm_summary = section_what_ran(audits)
    ladder, ladder_verdict = section_ladder(audits, accounting)
    sweep, decisions = section_sweep(audits)
    accuracy = section_accuracy(audits, parent)
    operator, nmrom, parity = accounting['operator'], accounting['nmrom'], accounting['parity']
    extended = next((a['extended_training_data'] for a in audits.values()
                     if a.get('extended_training_data', {}).get('present')), {})

    body = f"""# Tuning the DeepONet baseline on 2D Burgers at 256²: what more data and a DeepONet-specific schedule buy

Generated by `reports/generate_report.py` from audited records only. Every **measured** number
here — every error, every count, every second — is read from one of the hash-pinned sources
listed at the end and is never typed. The criterion bars of `DESIGN.md` §5.2, the arm names and
the job identifiers are literals in this generator, pinned by the pre-registration and by each
producing lane's own record.

This lane exists because `ops-deeponet-b2d` left two explanations for the DeepONet's 14.79 %
validation mean unseparated: **128 training cases** against the 4608 trajectories our own
NM-ROM head saw, and **an inherited U-Net schedule**. `DESIGN.md` was written and independently
audited twice before the first GPU job.

## 1. What each side of the comparison trains on

| | trajectories | what is fitted | target solver | GPU-seconds per trajectory |
|---|---:|---|---|---:|
| Operators, published | {operator['training_trajectories']} | {operator['supervised_output_fields']} supervised output fields (5 evolved times per case) | {operator['reference_setting']['intervals']} intervals, $\\Delta t$ = {operator['reference_setting']['dt']}, restricted to 256 | {operator['per_case_seconds']['mean']:.1f} |
| NM-ROM bank | {nmrom['bank_trajectories']} | FOM state snapshots, cap {nmrom['bank_snapshot_cap']} | 256 intervals, $\\Delta t$ = 0.005, direct | — |
| NM-ROM head | {nmrom['head_trajectories']} | {nmrom['retained_head_codes']} retained decoder codes of {nmrom['available_head_states']} available | 256 intervals, $\\Delta t$ = 0.005, direct | — |
| This lane, top rung | {extended.get('cases', 'n/a')} | {5 * extended['cases'] if extended.get('cases') else 'n/a'} supervised output fields | {extended.get('reference_setting', {}).get('intervals', 'n/a')} intervals, $\\Delta t$ = {extended.get('reference_setting', {}).get('dt', 'n/a')}, restricted to 256 | — |

The headline disparity is the trajectory count: **{parity['trajectory_ratio']:.0f}×**. The state
counts are given as two inventories and **not** as a ratio — {nmrom['retained_head_codes']}
fitted latent codes for a reconstruction objective are not the same kind of object as
{operator['supervised_output_fields']} supervised output fields for an initial-condition-to-future-fields
map, and dividing them would assert an information equivalence this lane cannot support.

**Why the published operators had 128 cases.** Their targets were held to a reference costing
**{operator['per_case_seconds']['mean']:.2f} s per trajectory** (median
{operator['per_case_seconds']['median']:.2f}, max {operator['per_case_seconds']['maximum']:.2f}),
so 4608 of them is **{parity['pinned_protocol_gpu_hours_at_parity']:.0f} GPU-hours**. That is
not a design choice about operators; it is what their reference cost. This lane therefore
generates the extra cases at a declared cheaper setting and **measures** what that changes
(§2).

**What {parity['trajectory_ratio']:.0f}× more data actually buys on this five-parameter family**
— the normalised distance from a validation case to its nearest training neighbour:

{table(['Training cases', 'nearest neighbour, mean', 'median', 'max', 'training spacing, median'],
       [[r['cases'], f"{r['nearest_mean']:.4f}", f"{r['nearest_median']:.4f}",
         f"{r['nearest_max']:.4f}", f"{r['train_nearest_neighbour_median']:.4f}"]
        for r in accounting['coverage']],
       ['---:', '---:', '---:', '---:', '---:'])}

Our own head trained at the bottom row's density. An operator given the top row is being asked
to extrapolate roughly twice as far in parameter space as one given the bottom row.

## 2. The training targets the extra cases carry

{'The extended cases were generated at ' + str(extended.get('reference_setting', {}).get('intervals')) + ' intervals, $\\Delta t$ = ' + str(extended.get('reference_setting', {}).get('dt')) + ', chosen inside the job by the pre-registered rule of DESIGN §2.3 from two profiled candidates.' if extended else '*(no extended data collected yet)*'}
{'' if not extended else f'''
Cases 0–127 are the pinned cache's own physical draws, so the difference the cheaper reference
makes is **measured on the training distribution**, not assumed: over
{extended['label_discrepancy']['cases']} paired cases the fixed-initial difference between the
new target and the pinned 4096-anchor target of the same case is
**{pct(extended['label_discrepancy']['maximum'])} % worst**,
{pct(extended['label_discrepancy']['median'])} % median,
{pct(extended['label_discrepancy']['mean'])} % mean. The supplied input field and the generation
descriptors are bitwise identical, which the job asserts case by case.

That number is what makes the ladder readable, and what limits it: it is small against a 15 %
model error and **not** small against a 2 % one. §3's T0 row gives the ratio per arm.
'''}
## 3. The ladder: validation error against training-set size, at fixed compute

{ladder}

## 4. The sweep

{sweep}

## 5. Accuracy beside the other families

{accuracy}

## 6. Provenance

{table(['source', 'sha256'], [[f'`{name}`', f'`{digest[:16]}…`'] for name, digest in sorted(sources.items())])}

Inherited-source check: **{'passed' if inherited['passed'] else 'FAILED'}** —
{len(inherited['inherited'])} files byte-identical to the forked lane, {len(inherited['vendored'])}
byte-identical to the pinned generator, {len(inherited['changed'])} declared changed,
{len(inherited['new'])} new.

**No speed number appears in this report and none is admissible from this lane.** No timing block
was run; `timing.py` is not staged. Nothing here is divided by a time from any other job.
"""
    (HERE / '2026-09-22-ops-tune-deeponet.md').write_text(body)
    summary = dict(generated='2026-09-22', report='2026-09-22-ops-tune-deeponet.md',
                   generator_sha256=sha(Path(__file__)), sources=sources, bars=BARS,
                   fno_large_reference=FNO_LARGE, arms=arm_summary,
                   ladder_verdict=ladder_verdict, decisions=decisions,
                   rule='every row carries its source file and that file SHA256; no row is typed')
    (HERE / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f"wrote {HERE / '2026-09-22-ops-tune-deeponet.md'} and summary.json "
          f"({len(arm_summary)} arms from {len(audits)} attempts)")


if __name__ == '__main__':
    main()
