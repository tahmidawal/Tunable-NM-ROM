"""Generate the refinement report and scientific figures from pinned saved JSON evidence."""
from __future__ import annotations

import collections
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from inventory import ROOT, read, sha, write, identity

HERE = Path(__file__).resolve().parent
WTS = ROOT / 'worktrees'
FIG = HERE / 'figs'
PINS = {}


def load(p):
    p = Path(p)
    PINS[str(p)] = sha(p)
    return read(p)


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(str(c).replace('|', '\\|') for c in row) + ' |' for row in rows])


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f'{name}.png', dpi=170, bbox_inches='tight')
    fig.savefig(FIG / f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    inv = load(HERE / 'data/inventory.json.gz')
    timed = load(HERE / 'data/timing-rescore.json')
    dimsdir = WTS / '2026-10-18-jcp-dims/experiments/jcp-dims'
    speeddir = WTS / '2026-10-14-jcp-speed3d/experiments/jcp-speed3d'
    dims = load(dimsdir / 'checks/analysis-3d.json')
    speed = load(speeddir / 'checks/analysis-grid.json')
    qualify_q2 = load(WTS / '2026-10-18-jcp-qualify/experiments/jcp-qualify/runs/EXPORT.json')['threeD']['tested']
    mesh = load(speeddir / 'checks/analysis-pm.json')['m1']
    mesh_raw = load(speeddir / 'runs/m1/archive/output/result.json')
    budget = load(WTS / '2026-10-18-jcp-budget/experiments/jcp-budget/checks/b3-b3d.json')
    old_pod = load(WTS / '2026-10-06-jcp-hr-baselines/experiments/jcp-hr-baselines/checks/VERDICT-c3d64.json')
    pod_scope = dict(diagnostic=sum(v['diagnostic_only'] for v in old_pod['X'].values()),
                    qualified=sum(not v['diagnostic_only'] for v in old_pod['X'].values()),
                    dominated_all=sum('baseline dominates X:' in v['cell_line'] for v in old_pod['X'].values()),
                    total=len(old_pod['X']),
                    dominated_qualified=sum(not v['diagnostic_only'] and 'baseline dominates X:' in v['cell_line'] for v in old_pod['X'].values()),
                    withheld_qualified=sum(not v['diagnostic_only'] and 'withheld:' in v['cell_line'] for v in old_pod['X'].values()))
    qualified = {k: dims['configurations'].get(k, speed['configs'].get(k)) for k in dims['qualified_3d']}
    assert all(v is not None for v in qualified.values())
    bestmed = min(qualified, key=lambda k: qualified[k]['median'])
    bestworst = min(qualified, key=lambda k: qualified[k]['worst'])
    n = inv['summary_counts']
    values = dict(inventory_counts=n, best_median=dict(key=bestmed, **qualified[bestmed]),
                  best_worst=dict(key=bestworst, **qualified[bestworst]),
                  historical_ROM_comparisons=dims['h2q'], timing_population=timed['case_ids'],
                  historical_POD_scope=pod_scope, new_lanes={})

    # Exact historical timed population: cost and returned coefficients linked by original hashes.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ['median', 'worst']):
        for bank, colour in [('M2', '#a65c25'), ('W', '#27659a')]:
            points = [(k, v) for k, v in timed['records'].items() if k.startswith(bank)]
            ax.scatter([v['query_median_seconds'] * 1e3 for _, v in points],
                       [v[metric] * 100 for _, v in points], label=bank, c=colour, s=35)
        ax.set(xscale='log', yscale='log', xlabel='Total query, median over timed cases (ms)',
               ylabel=f'{metric.capitalize()} E0 ST error (%)', title=f'Historical timed workload: {metric} error')
        ax.grid(alpha=.2); ax.legend(title='Frozen bank')
    fig.suptitle(f"Exact timed case ids {timed['case_ids']} · H200 · fixed full-field output", fontsize=11)
    save(fig, 'timed_frontier')

    # Same-allocation mesh evidence. It is explicitly not fixed-M/full-field evidence.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    mesh_work = {}
    for label, vv in mesh['H41'].items():
        points = sorted((int(L), sec) for L, sec in vv['seconds'].items())
        key = mesh['H43'][label]['rom_key']
        q2 = dims['configurations'].get(key, {}).get('Q2', {}).get('label',
             dims['rounds'].get(key, {}).get('label', qualify_q2.get(key, 'unverified')))
        axes[0].plot([x for x, _ in points], [y * 1e3 for _, y in points], marker='o',
                     label=f'{label} (Q2 {q2})')
        ident = identity(key)
        sp = key.split('|', 2)[2]
        work = []
        for L, _ in points:
            rows = [r for r in mesh_raw['rows'] if r.get('setting') == ident['setting'] and
                    r.get('rule') == ident['rule'] and r.get('path') == 'mf' and r.get('n') == L + 1 and r.get('spec') == sp]
            assert len(rows) == 64
            counts = [sum(r['steps']['it'] if 'it' in r['steps'] else r['steps']['q']) for r in rows]
            work.append((L, float(np.median(counts))))
        mesh_work[label] = dict(rom_key=key, Q2=q2, median_iterations=work, source_job=mesh_raw['job_id'])
        axes[1].plot([x for x, _ in work], [y for _, y in work], marker='o', label=label)
    axes[0].set(xlabel='Mesh intervals per axis', ylabel='Median mesh-free query (ms)', yscale='log', title='Historical cost versus mesh')
    axes[1].set(xlabel='Mesh intervals per axis', ylabel='Median reported nonlinear corrections per trajectory', title='Historical nonlinear work versus mesh')
    for ax in axes: ax.grid(alpha=.2); ax.legend(fontsize=7)
    save(fig, 'mesh_cost_iterations')
    values['mesh_work'] = mesh_work

    fig, ax = plt.subplots(figsize=(7, 4))
    width_rows = []
    for r in [256, 512, 768, 1024]:
        name = f'W{r}M2052' if r != 512 else 'W512'
        key = f'{name}|lat65536|fs|CN|0.01|K1'
        v = dims['configurations'][key]
        width_rows.append(dict(key=key, r=r, M=v['M'], m=65536, median=v['median'], worst=v['worst'], qualified=v['qualified']))
    ax.plot([r['r'] for r in width_rows], [100 * r['median'] for r in width_rows], color='#27659a', alpha=.5)
    for label, admitted in [('qualified', True), ('unqualified / unverified', False)]:
        rows = [r for r in width_rows if r['qualified'] == admitted]
        ax.scatter([r['r'] for r in rows], [100 * r['median'] for r in rows],
                   facecolors='#27659a' if admitted else 'none', edgecolors='#27659a', label=label, s=65)
    ax.set(xlabel='Frozen bank width r', ylabel='Median E0 ST error (%)', yscale='log',
           title='Width at fixed M=2052, lattice m=65536, FS-CN step 0.01')
    ax.legend(); ax.grid(alpha=.2); save(fig, 'accuracy_width')
    values['fixed_test_width'] = width_rows

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    qrows = []
    for order in [16, 20, 24, 32, 40, 48, 56, 64]:
        key = f'W1024|gl{order}|fs|CN|0.01|K1'
        if key not in dims['configurations']: continue
        v = dims['configurations'][key]
        q = v['Q2']
        qrows.append(dict(key=key, m=order ** 3, Q1=v['Q1_pass'], Q2=q['label'],
                          distance=q.get('worst_dist_qp'), rho=q.get('rho_max')))
    for ax, field, threshold, label in [(axes[0], 'distance', 2.5e-4, 'Worst rollout successor distance'),
                                       (axes[1], 'rho', .116, 'Maximum own-state nonlinear discrepancy')]:
        rows = [r for r in qrows if r[field] is not None and np.isfinite(r[field])]
        ax.loglog([r['m'] for r in rows], [r[field] for r in rows], marker='o', color='#27659a')
        ax.axhline(threshold, color='#a65c25', linestyle='--', label='Registered threshold')
        ax.set(xlabel='Gauss quadrature points m', ylabel=label, title='W1024 FS-CN, step 0.01')
        ax.grid(alpha=.2); ax.legend()
    save(fig, 'quadrature_convergence')
    values['quadrature_historical'] = qrows

    fig, ax = plt.subplots(figsize=(9, 4))
    ingredients = ['Q', 'T', 'S', 'F', 'e_P']
    budget_rows = []
    members = list(budget['members'].items())
    x = np.arange(len(members))
    for ii, ingredient in enumerate(ingredients):
        vals = []
        for key, member in members:
            pc = [r for r in member['per_case'] if r['eligible']]
            a = [r['e_P'] if ingredient == 'e_P' else r['main'][ingredient] for r in pc]
            vals.append(float(np.median(a)))
            budget_rows.append(dict(key=key, ingredient=ingredient, cases=[r['case'] for r in pc], median=vals[-1]))
        ax.bar(x + (ii - 2) * .16, vals, .15, label=ingredient)
    ax.set(yscale='log', ylabel='Median field difference / initial-reference norm',
           title='Historical 3D error-budget interventions (own case population)', xticks=x,
           xticklabels=[f"{k.split('|')[0]}\n{k.split('|')[3]} {k.split('|')[4]}" for k, _ in members])
    ax.legend(title='Package'); ax.grid(axis='y', alpha=.2); save(fig, 'error_budget')
    values['historical_error_budget'] = budget_rows

    # In-range descriptive regime dependence; not a new OOD cohort or a one-factor experiment.
    metadata = {}
    for cid in range(64):
        p = WTS / f'2026-10-06-jcp-references/experiments/jcp-references/store/3d/val64/ST/c{cid:03d}.json'
        metadata[cid] = load(p)['desc']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    regime_sources = {}
    for key, label in [(bestmed, 'best historical median'), (bestworst, 'best historical worst')]:
        candidates = [o for o in inv['observations'] if o['config_key'] == key and o['cohort'] == 'val64' and
                      o['case_ids'] == list(range(64)) and o['n_scored'] == 64 and
                      abs(o['median'] - qualified[key]['median']) < 1e-10]
        assert candidates, key
        observation = sorted(candidates, key=lambda o: o['observation_id'])[0]
        source = load(observation['source'])['scores'][key]
        regime_sources[key] = observation['source']
        for ax, component, xlabel in [(axes[0], 'nu', 'Viscosity nu'), (axes[1], 'steepness', 'Nominal amplitude / viscosity')]:
            xx = [metadata[c]['nu'] if component == 'nu' else metadata[c]['A'] / metadata[c]['nu'] for c in range(64)]
            ax.scatter(xx, [100 * source[str(c)]['e'] for c in range(64)], s=22, alpha=.7, label=label)
            ax.set(xscale='log', yscale='log', xlabel=xlabel, ylabel='Per-case E0 ST error (%)', title='Validation cohort: descriptive association')
            ax.grid(alpha=.2); ax.legend(fontsize=8)
    save(fig, 'in_range_regime')
    values['regime_sources'] = regime_sources

    # New lane results remain separate, with exact scope and audit state supplied by their owners.
    lane_sections = []
    for name in ['solver', 'pod']:
        p = WTS / f'2026-10-26-b3d-{name}/experiments/b3d-{name}/EXPORT.json'
        if not p.is_file():
            p = p.parent / 'runs/EXPORT.json'
        if p.is_file():
            lane = load(p)
            values['new_lanes'][name] = lane
            status = lane.get('status', lane.get('results_audit', 'unrecorded'))
            text = f"### {name.capitalize()} lane\n\nAudit/status: `{status}`. Scope: {lane.get('scope', 'see export')}.\n\n"
            if name == 'solver':
                rows = [[o['key'], o['fresh_nonstationary_steps'],
                         f"{o['score']['e']*100:.7g}", f"{o['query_median_seconds']*1e3:.5g}"]
                        for o in lane.get('observations', [])]
                text += table(['Diagnostic setting/case', 'Fresh uncertified steps', 'E0 max error (%)',
                               'Raw median query (ms)'], rows) + '\n\n'
                text += ('These are selected-case diagnostics. A repaired certificate is not a full-trajectory '
                         'Q1 pass or a physical-accuracy improvement. Raw timings do not establish comparative '
                         'speed without the declared paired acceptance checks.\n\n')
                for control in lane.get('same_history', []):
                    text += (f"At `{control['key']}`, the returned Jacobian condition number is "
                             f"{control['diagnostics']['condition']:.6g}. The fresh tight solve uses "
                             f"{control['levels']['tight']['it']} iterations and reaches stationarity "
                             f"{control['levels']['tight']['fresh_gamma']:.6g}; its field distance to the "
                             f"returned baseline step is {control['levels']['tight']['baseline_distance']:.6g}. "
                             'This is a fresh restart from identical coefficient history, not restoration of '
                             'the original cached solver state.\n\n')
            else:
                rows = []
                for rule, q in lane.get('qualification', {}).get('whole_residual', {}).items():
                    rows.append([rule, f"{q['delta_g_max']:.7g}", f"{q['full_nonstat']}/{q['steps']}", q['pass_']])
                text += table(['Frozen support', 'Maximum full-gradient discrepancy', 'Full-objective uncertified steps',
                               'New HR admission'], rows) + '\n\n'
                rows = [[cid, f"{v['k8_to_k16']:.7g}", f"{v['k16_to_none']:.7g}", f"{v['k8_to_none']:.7g}"]
                        for cid, v in lane.get('case_field_distances', {}).items()]
                text += table(['Validation case', 'k8 to k16 distance', 'k16 to no-HR distance', 'k8 to no-HR distance'], rows) + '\n\n'
                text += ('These failures disqualify the frozen supports under the new conservative whole-objective '
                         'contract. They do not establish that the bank beats POD. The post-completion printing '
                         'failure and recovered checkpoint are retained in the execution record.\n\n')
            text += f"Machine export and hashed raw evidence: `{p}`. No cross-job speed-up is inferred.\n"
            lane_sections.append(text)
    if not lane_sections:
        lane_sections = ['Numerical diagnostic lanes are running or under independent results review. '
                         'No new GPU result is accepted in this version.\n']

    baseline_rows = []
    for label, key in [('Best observed median', bestmed), ('Best observed worst', bestworst)]:
        v = qualified[key]
        baseline_rows.append([label, f'`{key}`', f"{100*v['median']:.6g}", f"{100*v['worst']:.6g}"])
    timing_rows = [[f'`{k}`', f"{v['query_median_seconds']*1e3:.4g}", f"{100*v['median']:.5g}",
                    f"{100*v['worst']:.5g}", v['errors_above_1pct']] for k, v in sorted(timed['records'].items(), key=lambda kv: kv[1]['query_median_seconds'])]
    comparison_rows = [[k, f"{v['S']:.5g}", v['comparator_q2'], v['comparator_label']] for k,v in dims['h2q'].items()]
    ref_rows = []
    for cohort, v in inv['reference']['cohorts'].items():
        eps = [x['eps_ref'] for x in v['cases'].values()]
        ref_rows.append([cohort, f"{v['eligible']}/{len(v['cases'])}", f"{min(eps):.6g}–{max(eps):.6g}", v['selection_allowed']])
    report = f'''# Burgers 3D refinement: recovered evidence, numerical gaps and controlled replays

This report reconciles existing evidence and the bounded refinement campaign. The historical
numbers are conditional estimates against approved E0 ST references; new numerical results
are accepted only with their independent audit status. This generated version does not
promote missing or subset evidence into full-validation qualification.

## Recovered state

The current experimental base is `{inv['base_commit']}`. The inventory contains
{n['speed']} speed-screening summary records, {n['dims']} detailed dimension-study records
and {n['dims_rounds']} later quadrature-round records, plus {n['observations']} scored
observations and {n['scored_cases']} per-case records. These are different record types,
not a count of independent trained models or newly run experiments.

The nominal-width export reports {n['dims_nominal_width_qualified']} qualified entries;
the detailed configuration map contains {n['dims_qualified']}; its broader qualified union
contains {n['dims_qualified_union']}. Explicit test-space variants and later round-only
candidates explain why a single unlabelled count loses evidence. Raw score/coefficient case
ids are checked; intended registered populations are not reconstructed for every old schema.
The inventory exposes unknown fields and unsupported supporting schemas explicitly.

### Reference population and uncertainty

{table(['Cohort', 'ST eligible / stored', 'Conditional uncertainty range', 'Selection allowed'], ref_rows)}

Only development and validation populations can inform selection. Historical test observations
remain replication evidence. Sealed and reserved out-of-distribution cohorts remain closed.
Reference uncertainty is a conditional Richardson estimate, not a rigorous error bound.

### There is no single strongest bank configuration

{table(['Purpose', 'Qualified historical setting', 'Median error (%)', 'Worst error (%)'], baseline_rows)}

These full-validation figures select different objectives. A low median can coexist with a
large error on a difficult case. The test-space-expanded setting retains its explicit name;
it is not silently pooled with a nominal test-space configuration.

## Cost and accuracy on the exact saved timing workload

The timing-job coefficients were re-decoded directly with a NumPy evaluator and scored
against the approved reference fields. Its exact case ids are {timed['case_ids']}; it is
a historical workload, not the full validation population. All original timing repetitions,
payload identity links and source hashes are retained. Its timing source is job
`{timed['job_id']}` on `{timed['gpu']}`. Costs include mesh input projection, reduced
evolution and full-field output under that job's query contract; diagnostic certification
and offline setup are not query work.

{table(['Setting', 'Median full query (ms)', 'Median error (%)', 'Worst error (%)', 'Cases above 1% error'], timing_rows)}

![Accuracy versus total query cost on the exact historical timed population](figs/timed_frontier.png)

The full-validation accuracy and this timed subset are displayed separately. Combining the
former with the latter would hide the population difference and is not used as a new frontier.

## What the existing controls already distinguish

![Quadrature convergence with inherited thresholds](figs/quadrature_convergence.png)

Wide-bank fixed-sweep CN can pass nonlinear qualification while failing quadrature successor
agreement. Wide-bank large-step QN-BDF2 can instead pass quadrature but fail every-case
distance to a verified tight solve. Pooled stationarity alone can miss a small number of
important failed steps. Trust clipping is an observed counter; it is not by itself a cause.

Restarting from coefficient history resets the historical QN cache. Fresh-restart controls
are labelled separately; full-trajectory one-factor controls assess refresh-policy changes.
The completed quadrature trajectories are reused, and only missing successor evidence is run.

![Accuracy versus bank width with fixed tests and quadrature](figs/accuracy_width.png)

Open markers retain failed or unverified qualification. A decreasing raw error is not enough
to claim a qualified width dial. The square test-space control failed historically, so the
original registered causal test-space verdict remains blocked even where expansion helped.

![Error-budget interventions on their historical population](figs/error_budget.png)

Each bar is a median intervention distance on that member's own cases. The packages interact;
these bars are not additive contributions to total error. The representation floor and
time-discretisation package bind in different settings. The archived cohort limits remain.

## Mesh and input-regime dependence

![Historical same-allocation mesh cost and reported nonlinear work](figs/mesh_cost_iterations.png)

This mesh panel uses the mesh-free input/scoring-lattice contract and the historical deployment
mode policy, which may change the test list with the mesh. It establishes neither fixed-M
cost nor full-field output cost. Only the appropriately qualified curves support numerical
claims; unqualified curves remain visible. Full output and arbitrary mesh-sampled input
must be measured separately. Historical Jacobian counts are not reconstructed from iteration
counts where the original counter is absent.

![Descriptive in-range viscosity and steepness associations](figs/in_range_regime.png)

The regime plot uses only the existing validation population and the same E0 reference.
Amplitude divided by viscosity is a nominal steepness proxy. These associations are not
one-factor causal tests and do not qualify new steep-front or viscosity-extrapolation regimes.
The earlier two-dimensional steep-front failures cannot be silently promoted to 3D results.

## Three different comparisons

The historical confirmed ROM-to-ROM comparisons are:

{table(['Earlier comparator', 'Speed ratio', 'Comparator Q2', 'Qualification scope'], comparison_rows)}

They do not establish improvement over POD or a full solver. The historical POD headline
of {pod_scope['dominated_all']}/{pod_scope['total']} dominated pooled {pod_scope['diagnostic']}
diagnostic bank rows with {pod_scope['qualified']} qualified bank rows. In the qualified subset,
{pod_scope['dominated_qualified']}/{pod_scope['qualified']} were dominated and
{pod_scope['withheld_qualified']} neutral conclusions were withheld. Its hyper-reduction convergence was reported
rather than gated. Its measured accuracy/cost evidence is retained, with this qualification
gap stated explicitly. The new comparator lane tests the saved supports against the complete
unsampled reduced objective before any qualified comparison.

The earlier spectral result remains evidence against a general bank speed advantage on the
box. No new speed ratio versus a spectral or finite-difference solver is inferred from these
different jobs, outputs or populations. A repaired solver must face the same accuracy,
output and hardware contract, and POD receives the same numerical improvements.

## New bounded experiment results

{chr(10).join(lane_sections)}

## Completed, failed, deferred and open

- Completed recovery: canonical fork, separate historical populations/references, explicit
  qualification scopes, source hashes, saved coefficients and timing repetition evidence.
- Completed analysis: exact historical timed-workload re-scoring and generated scientific plots.
- Failed historical controls remain visible: under-resolved quadrature, incomplete nonlinear
  convergence and square test-space controls; post-hoc selections and retracted combined
  selection claims remain labelled in the source inventory.
- Numerical diagnostics and any frozen full-validation follow-up retain independent design,
  code, job and results gates. See lane exports and the job ledger for current completion.
- Deferred until supported: a qualified full-population repaired frontier, qualified POD
  hyper-reduction comparison, paired fastest-full-solver comparisons, fixed-mode scaling of
  a repaired candidate and separate full-field output/offline costs.
- No bank retraining, sealed-cohort opening or branch merge is implied.

## Reproduction and provenance

Run `inventory.py`, `rescore_timing.py`, then `make_report.py` with the repository's absolute
Python environment. These analysis scripts use CPU NumPy and saved trajectories. They do
not rerun the historical solver grid. `data/SOURCES.json` pins historical sources;
`PROVENANCE.json` pins this report's direct inputs, and `VALUES.json` retains extracted
numbers and figure inputs. Compressed inventory and per-case JSON lines are backed up on
the prescribed code-only mirror. Full numerical reruns use each lane's frozen manifest.

## Plain-language glossary

- Bank: frozen spatial functions that turn a short vector into a field; r is their count.
- M: number of smooth functions used to test the PDE; m: number of integration points.
- Mesh intervals: divisions along each spatial axis; nodes include the two boundary endpoints.
- E0 ST: approved space/time-refined reference; its uncertainty is conditional on convergence assumptions.
- Error: largest output-time field difference divided by the reference initial-field norm.
- Median / worst: middle / largest case error; they answer different accuracy questions.
- Outlier count: number of cases exceeding the table's stated error threshold.
- Full query: input projection, reduced time evolution and requested output decoding.
- Q1: nonlinear-solver validity, stationarity and agreement with tighter solves.
- Q2: quadrature successor and own-state nonlinear agreement, with the finer rule checked too.
- Qualified: satisfies both checks on the required population; unknown evidence is not a pass.
- CN / BDF2 / BE: Crank–Nicolson / second-order backward differentiation / backward Euler.
- FS / QN: fixed full-Jacobian correction sweeps / a solver that reuses its small normal matrix.
- Jacobian: derivative of the reduced residual; stationarity measures whether further correction remains.
- Trust radius / damping: controls on correction size / regularisation of a nonlinear correction.
- POD / ECSW: a standard linear reduced basis / weighted sampling of its residual equations.
- No-HR: evaluate the complete reduced objective without weighted spatial sampling.
- Frontier: settings that no other admissible setting beats in both cost and the chosen error metric.
- Diagnostic subset: selected cases for explaining a failure, not a whole-population performance estimate.
- Post hoc / retracted: selected after observing data / no longer an accepted earlier claim.
- Q/T/S/F/e_P: quadrature / time / nonlinear solve / reduced-form intervention / representation floor.
- Mesh-free: use method-specific input and output points; full-field output requests every mesh value.
- Conditional / provisional: valid only with stated numerical assumptions / awaiting required evidence.
'''
    (HERE / '2026-10-26-burgers3d-refinement.md').write_text(report)
    write(HERE / 'VALUES.json', values)
    write(HERE / 'PROVENANCE.json', PINS)
    print(HERE / '2026-10-26-burgers3d-refinement.md')


if __name__ == '__main__':
    main()
