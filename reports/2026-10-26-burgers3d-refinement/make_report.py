"""Generate the refinement report and scientific figures from pinned saved JSON evidence."""
from __future__ import annotations

import collections
import json
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from inventory import ROOT, read, sha, write, identity

HERE = Path(__file__).resolve().parent
WTS = ROOT / 'worktrees'
FIG = HERE / 'figs'
PINS = {}


def load(p):
    p = Path(p)
    PINS[str(p)] = sha(p)
    return read(p)


def load_frozen(name):
    path = HERE / 'data' / name
    gates = load(HERE / 'data/DATASET-GATES.json')
    gate = gates[name]
    assert sha(path) == gate['sha256'] and gate['verdict'].startswith('READY')
    audit = HERE / gate['audit_snapshot']
    assert sha(audit) == gate['audit_sha256']
    PINS[str(audit)] = sha(audit)
    return load(path)


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(str(c).replace('|', '\\|') for c in row) + ' |' for row in rows])


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f'{name}.png', dpi=170, bbox_inches='tight')
    fig.savefig(FIG / f'{name}.pdf', bbox_inches='tight', metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    inv = load(HERE / 'data/inventory.json.gz')
    closure_path = HERE / 'data/CAMPAIGN-CLOSURE.json'
    closure = load(closure_path) if closure_path.is_file() else None
    numbers_date = date.fromisoformat(closure['numbers_finalized_date']).isoformat() if closure else '2026-10-26'
    report_filename = f'{numbers_date}-burgers3d-refinement.md'
    report_status = ('Final numerical results for the bounded tested set; remaining scientific gaps are explicitly scoped.'
                     if closure else 'Provisional campaign checkpoint; final paired validation and publication review are pending.')
    timed = load(HERE / 'data/timing-rescore.json')
    dimsdir = WTS / '2026-10-18-jcp-dims/experiments/jcp-dims'
    speeddir = WTS / '2026-10-14-jcp-speed3d/experiments/jcp-speed3d'
    dims = load(dimsdir / 'checks/analysis-3d.json')
    historical_confirmation = load(dimsdir / 'runs/t3c/slim/output/d3run_t3c/result.json')
    first_confirmation = next(iter(historical_confirmation['timing'].values()))
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
                  historical_POD_scope=pod_scope, new_lanes={}, qualified_HR_POD_subjects=[])
    values['report_filename'] = report_filename
    values['numbers_finalized_date'] = numbers_date if closure else None

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
                     label=f'{label} (Q2 at 65 nodes: {q2})')
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

    test_rows = []
    for k, v in dims['configurations'].items():
        if k.startswith('W512') and v['spec'] == 'fs|CN|0.01|K1' and v['rule'] == 'lat32768':
            test_rows.append(dict(key=k, M=v['M'], r=v['r'], m=32768, median=v['median'], worst=v['worst'],
                                  Q1=v['Q1_pass'], Q2=v['Q2']['label'], qualified=v['qualified']))
    test_rows.sort(key=lambda v: v['M'])
    fig, ax = plt.subplots(figsize=(7, 4))
    for metric in ['median', 'worst']:
        ax.plot([r['M'] for r in test_rows], [r[metric] * 100 for r in test_rows], marker='o', label=metric)
    ax.set(xlabel='Actual smooth test functions M', ylabel='E0 ST error (%)', yscale='log',
           title='Frozen W512, lattice m=32768, FS-CN step 0.01')
    ax.legend(title='Full validation statistic'); ax.grid(alpha=.2)
    save(fig, 'test_space')
    values['test_space_fixed_quadrature'] = test_rows

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
    gap = dims['configurations']['W1024|gl48|fs|CN|0.01|K1']
    values['quadrature_vs_physical_example'] = dict(successor_distance=gap['Q2']['worst_dist_qp'], worst_E0=gap['worst'])

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
    regime_statistics = []
    for key, label in [(bestmed, 'best historical median'), (bestworst, 'best historical worst')]:
        candidates = [o for o in inv['observations'] if o['config_key'] == key and o['cohort'] == 'val64' and
                      o['case_ids'] == list(range(64)) and o['n_scored'] == 64 and
                      abs(o['median'] - qualified[key]['median']) < 1e-10]
        assert candidates, key
        observation = sorted(candidates, key=lambda o: o['observation_id'])[0]
        source = load(observation['source'])['scores'][key]
        regime_sources[key] = observation['source']
        errors = [source[str(c)]['e'] for c in range(64)]
        regime_statistics.append(dict(setting=key, cases=list(range(64)),
            viscosity_spearman=float(spearmanr([metadata[c]['nu'] for c in range(64)], errors).statistic),
            nominal_steepness_spearman=float(spearmanr([metadata[c]['A']/metadata[c]['nu'] for c in range(64)], errors).statistic),
            errors_above_1pct=sum(e > .01 for e in errors),
            largest_error_cases=sorted(range(64), key=lambda c: errors[c], reverse=True)[:3]))
        for ax, component, xlabel in [(axes[0], 'nu', 'Viscosity nu'), (axes[1], 'steepness', 'Nominal amplitude / viscosity')]:
            xx = [metadata[c]['nu'] if component == 'nu' else metadata[c]['A'] / metadata[c]['nu'] for c in range(64)]
            ax.scatter(xx, [100 * source[str(c)]['e'] for c in range(64)], s=22, alpha=.7, label=label)
            ax.set(xscale='log', yscale='log', xlabel=xlabel, ylabel='Per-case E0 ST error (%)', title='Validation cohort: descriptive association')
            ax.grid(alpha=.2); ax.legend(fontsize=8)
    save(fig, 'in_range_regime')
    values['regime_sources'] = regime_sources
    values['regime_statistics'] = regime_statistics
    new_frontier_text = ''
    frontier_path = HERE / 'data/frozen-frontier.json.gz'
    if not frontier_path.is_file():
        frontier_path = HERE / 'data/frozen-frontier.json'
    if frontier_path.is_file():
        import frontier_report
        frontier_data = load_frozen(frontier_path.name)
        rendered = frontier_report.render(frontier_data, FIG, table, save, inv['reference'])
        new_frontier_text = rendered.pop('text')
        values['full_validation_frontier'] = rendered
        values['qualified_HR_POD_subjects'] = [sid for sid, v in rendered['subjects'].items()
            if v['method'] == 'POD' and v['registered_qualified'] and
            v['qualification']['method_specific_extra_masks'].get('hyper_reduced') is True]
    new_mesh_text = ''
    mesh_path = HERE / 'data/frozen-mesh.json.gz'
    if not mesh_path.is_file():
        mesh_path = HERE / 'data/frozen-mesh.json'
    if mesh_path.is_file():
        import mesh_report
        rendered_mesh = mesh_report.render(load_frozen(mesh_path.name), inv['reference'], table, save)
        new_mesh_text = rendered_mesh.pop('text')
        values['fixed_mode_mesh'] = rendered_mesh

    # New lane results remain separate, with exact scope and audit state supplied by their owners.
    lane_sections = []
    for name in ['solver', 'pod']:
        p = WTS / f'2026-10-26-b3d-{name}/experiments/b3d-{name}/EXPORT.json'
        if not p.is_file():
            p = p.parent / 'runs/EXPORT.json'
        frozen_export = HERE / 'data' / f'lane-{name}-export.json'
        if frozen_export.is_file():
            p = frozen_export
        if p.is_file():
            lane = load(p)
            values['new_lanes'][name] = lane
            status = lane.get('status', lane.get('results_audit', 'unrecorded'))
            text = f"### {name.capitalize()} lane\n\nAudit/status: `{status}`. Scope: {lane.get('scope', 'see export')}.\n\n"
            if name == 'solver':
                groups = collections.defaultdict(list)
                for o in lane.get('observations', []):
                    width, policy, _ = o['key'].split('|')
                    groups[(o['source']['job_id'], width, policy)].append(o)
                rows = []
                for (job, width, policy), group in sorted(groups.items()):
                    errors = [o['score']['e'] for o in group]
                    distances = [o['distance_historical_tight'] for o in group if o.get('distance_historical_tight') is not None]
                    rows.append([job, f'{width}/{policy}', ', '.join(o['key'].split('|')[-1][4:] for o in group),
                        sum(o['fresh_nonstationary_steps'] for o in group), sum(o['nonlinear_iterations'] for o in group),
                        f'{np.median(errors)*100:.6g}/{max(errors)*100:.6g}',
                        'not measured' if not distances else f'{max(distances):.6g}',
                        f"{np.median([o['query_median_seconds'] for o in group])*1e3:.5g}"])
                text += table(['Job', 'Bank/policy', 'Case ids', 'Uncertified steps', 'Corrections',
                               'Median/worst error (%)', 'Max distance to tight', 'Raw median query (ms)'], rows) + '\n\n'
                text += ('These are selected-case diagnostics. A repaired certificate is not a full-trajectory '
                         'Q1 pass or a physical-accuracy improvement. Raw timings do not establish comparative '
                         'speed without the declared paired acceptance checks.\n\n')
                for control in lane.get('same_history', []):
                    usable = bool(control.get('anchor_usable') and all(v.get('verified') for v in control['levels'].values()))
                    text += (f"At `{control['key']}`, the returned Jacobian condition number is "
                             f"{control['diagnostics']['condition']:.6g}. The fresh tight solve uses "
                             f"{control['levels']['tight']['it']} iterations with reported stationarity "
                             f"{control['levels']['tight']['fresh_gamma']:.6g}; its field distance to the "
                             f"returned baseline step is {control['levels']['tight']['baseline_distance']:.6g}. "
                             f"Independent anchor usable: {usable}. "
                             'This is a fresh restart from identical coefficient history, not restoration of '
                             'the original cached solver state. The condition number measures this returned '
                             'step only; it does not bound conditioning across the trajectory or population.\n\n')
                historical_q = lane.get('historical_tight_quadrature', {}).get('data')
                if historical_q:
                    rows = [[v['solver'], f"{v['coarse']} to {v['fine']}", len(v['ordered_cases']),
                             f"{v['median']:.7g}", f"{v['maximum']:.7g}", v['maximum_case'],
                             len(v['cases_above_threshold'])] for v in historical_q['comparisons']]
                    text += ('The following re-analysis reuses saved full-validation trajectories. '
                             'Quadrature discrepancies persist with verified tight nonlinear solves, '
                             'so fixed-sweep work alone does not explain them.\n\n')
                    text += table(['Nonlinear solve', 'Quadrature transition', 'Cases', 'Median distance',
                                   'Maximum distance', 'Largest case', 'Cases over registered threshold'], rows) + '\n\n'
                finer = [v for v in lane.get('quadrature', [])
                         if v['source']['path'].startswith('results/qcheck801/')]
                if finer:
                    finer = sorted(finer, key=lambda v: int(v['key'].split('case')[-1]))
                    cases = [int(v['key'].split('case')[-1]) for v in finer]
                    bad = sum(sum(g > 1e-3 for g in v['fresh_gamma']) for v in finer)
                    steps = sum(len(v['fresh_gamma']) for v in finer)
                    text += (f"The missing inherited finer-rule check now covers the original case ids {cases}, "
                             f"with maximum successor distance {max(v['dist_gl64_gl80'] for v in finer):.7g}, "
                             f"maximum own-state rho {max(max(v['own_state_rho']) for v in finer):.7g}, and "
                             f"{bad}/{steps} fresh uncertified fine-rule steps. The independently approved join "
                             'to the saved complete-population target/successor evidence closes the legacy '
                             '`W1024|gl56|fs|CN|0.01|K1` qualification. It does not qualify the finest rule '
                             'as a new deployment or improve physical accuracy. Historical inventory status '
                             'counts above retain their original pre-campaign scope.\n\n')
                ts = lane.get('time_selection', {}).get('data')
                if ts:
                    tr = ts['rows']
                    values['time_refinement_diagnostic'] = ts
                    rows = [[v['dt'], ', '.join(map(str, v['cases'])), f"{100*v['median']:.7g}",
                             f"{100*v['median_lower']:.7g}–{100*v['median_upper']:.7g}", f"{100*v['worst']:.7g}",
                             v['corrections'], v['jacobian_contractions'], f"{v['fresh_bad_steps']}/{v['total_steps']}",
                             v['all_target_anchors_usable'] and v['all_own_chains_pass']] for v in tr]
                    text += ('#### Time refinement: median and worst trade off\n\n'
                             'These selected-case diagnostics hold bank, test space and target quadrature fixed. '
                             'All displayed settings pass their subset nonlinear and own-state quadrature controls. '
                             'Finer steps improve the median while worsening the worst case; neither a general '
                             'accuracy improvement nor a general failure describes this tradeoff. The intervals '
                             'are propagated conditional reference uncertainties, not statistical confidence intervals. '
                             'The inherited mode weights depend on the step size, so this is a time-step/method-package '
                             'intervention rather than isolated temporal truncation error.\n\n')
                    text += table(['Step size', 'Case ids', 'Median error (%)', 'Conditional median interval (%)',
                                   'Worst error (%)', 'Corrections (sum)', 'Jacobian contractions (sum)',
                                   'Uncertified steps', 'Subset controls pass'], rows) + '\n\n'
                    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
                    for metric, colour in [('median', '#27659a'), ('worst', '#a65c25')]:
                        y = np.array([v[metric] * 100 for v in tr])
                        lo = np.array([v[metric + '_lower'] * 100 for v in tr])
                        hi = np.array([v[metric + '_upper'] * 100 for v in tr])
                        axes[0].errorbar([v['dt'] for v in tr], y, yerr=[y-lo, hi-y],
                                         marker='o', capsize=4, color=colour, label=metric)
                    for metric, label in [('corrections', 'Nonlinear corrections'),
                                          ('jacobian_contractions', 'Jacobian contractions')]:
                        axes[1].plot([v['dt'] for v in tr], [v[metric] for v in tr], marker='o', label=label)
                    axes[0].set(ylabel='Conditional E0 ST error (%)', title='Selected-case accuracy tradeoff')
                    axes[1].set(ylabel='Sum over the diagnostic cases', title='Logical nonlinear work')
                    for ax in axes:
                        ax.set(xscale='log', xlabel='Time step (coarse to fine)'); ax.invert_xaxis()
                        ax.set_xticks([v['dt'] for v in tr], [str(v['dt']) for v in tr])
                        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
                        ax.grid(alpha=.2); ax.legend(fontsize=8)
                    save(fig, 'time_refinement_tradeoff')
                    text += ('![Time refinement accuracy and work](figs/time_refinement_tradeoff.png)\n\n'
                             f"Endpoint selection is explicitly post-diagnostic: {ts['selection']}. "
                             f"{ts['caution']} {ts['cost_scope']} "
                             'The earlier proposed worst-only rejection rule was not in the staged design '
                             'and was abandoned. Error cancellation is a hypothesis; this time-step control '
                             'alone does not identify a spatial, operator or representation cause.\n\n')
                pod_groups = collections.defaultdict(list)
                for v in lane.get('POD_tuning', []):
                    attempt = v['source']['path'].split('/')[1]
                    gate = lane.get('audits', {}).get('jobs', {}).get(attempt, {}).get('results', '')
                    if gate.startswith('READY'):
                        pod_groups[(v['source']['job_id'], v['subject'])].append(v)
                if pod_groups:
                    rows = []
                    for (job, subject), group in sorted(pod_groups.items()):
                        ids = sorted(v['case'] for v in group)
                        assert len(set(ids)) == len(ids) and set(ids) <= {0, 5, 30, 43}
                        if ids != [0, 5, 30, 43]:
                            rows.append([job, subject, str(ids), 'incomplete', 'incomplete', 'not established',
                                         'partial execution retained', 'partial'])
                            continue
                        errors = [v['score']['e'] for v in group]
                        common = all(v['common_rollout_check'] for v in group)
                        pooled = sum(v['nonstationary'] for v in group) <= .01 * sum(v['steps'] for v in group)
                        rows.append([job, subject, str(ids), f'{100*np.median(errors):.7g}', f'{100*max(errors):.7g}',
                            common and pooled, f"{max(v['certificate']['prod_to_tight_distance'] for v in group):.7g}",
                            sum(v['work']['nonlinear_iterations'] for v in group)])
                    text += ('#### Equal integration and nonlinear opportunities for POD\n\n'
                             'The unsampled POD diagnostics retain the original policies, their failures and '
                             'the prospectively frozen stronger-production control. Each row uses the same '
                             'selected cases; these are not complete-validation error estimates. The common '
                             'full-trajectory agreement gate remains unchanged. Cross-job times are not used '
                             'to choose between policies; the selected endpoints face the common allocation.\n\n')
                    text += table(['Job', 'POD schedule/policy', 'Case ids', 'Median error (%)', 'Worst error (%)',
                                   'Subset Q1/common pass', 'Max production–tight distance', 'Corrections (sum)'], rows) + '\n\n'
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
                fallback = lane.get('no_hr_full64')
                if fallback:
                    f = fallback['arms']['none']
                    text += (f"The unsampled POD fallback has audit status `{fallback['results_audit']}` on "
                             f"{len(fallback['ordered_case_ids'])} validation cases: Q1={f['Q1']}, "
                             f"{f['nonstationary_steps']}/{f['steps']} uncertified steps, "
                             f"{100*f['median_error']:.7g}% median and {100*f['worst_error']:.7g}% worst "
                             f"conditional E0 error, with {f['errors_above_one_percent']} cases above one percent. "
                             f"Maximum production-to-tight distance {f['maximum_tight_distance']:.7g}. "
                             'It is a qualified full-residual reduced model, not a qualified hyper-reduced model. '
                             'Its separate allocation is not used for cross-method timing ratios; the common '
                             'frontier gives POD the same integration improvements.\n\n')
                failed = lane.get('failed_support_extension')
                if failed:
                    text += (f"The sole larger-support construction has audit status `{failed['results_audit']}`. "
                             f"It stopped with {failed['terminal_support']} active points against its fixed "
                             f"target {failed['support_target']}; the off-support optimality gradient was "
                             f"{failed['kkt']['off_support_max_gradient']:.7g} against tolerance "
                             f"{failed['kkt']['tol']:.7g}, and {100*failed['banned_fraction']:.6g}% of "
                             f"candidate points were banned against a {100*failed['banned_fraction_limit']:.6g}% cap. "
                             f"The allocation completed normally after {failed['resource']['elapsed_seconds']} "
                             f"seconds ({failed['resource']['gpu_hours']:.6g} GPU-hours); this was not a time-budget "
                             'exhaustion. No larger support was admitted, and no rollout validation of it ran. '
                             'The permitted greedy path exhausted admissible choices under its fixed policy; '
                             'this does not prove that no valid larger support exists. The frozen stop rule '
                             'ends the new-support search. This rejected constructor supplies no qualified '
                             'hyper-reduced comparator and is not evidence that POD loses. Any separately '
                             'admitted improved-policy existing support is reported with the matched frontier.\n\n')
                a3 = lane.get('a3_replay')
                if a3:
                    assert a3['results_audit'].startswith('READY')
                    admission = a3['admission']
                    summary_path = HERE / 'data/pod-improved-support-summary.json'
                    assert sha(summary_path) == a3['summary_sha256']
                    support_summary = load(summary_path)
                    assert support_summary['result_sha256'] == a3['result_sha256']
                    assert support_summary['selection']['reference_store_version'] == inv['reference']['store_version']
                    scores = {}
                    for repetition in a3['timing_repetitions']:
                        for invocation in repetition['invocations']:
                            key = (str(repetition['case']), invocation['arm'])
                            value = invocation['e0']
                            if key in scores:
                                assert abs(scores[key]['e'] - value['e']) <= 1e-12
                            scores[key] = value
                    rows = []
                    for cid, distance in sorted(a3['common_field_distances'].items(), key=lambda kv: int(kv[0])):
                        own = support_summary['cases'][cid]
                        k_error, full_error = own['k16']['error'], own['none']['error']
                        eps = inv['reference']['cohorts']['val64']['cases'][cid]['eps_ref']
                        for arm, value in [('k16', k_error), ('none', full_error)]:
                            if (cid, arm) in scores:
                                timed_a3 = scores[(cid, arm)]
                                assert timed_a3['eligible'] is True and abs(timed_a3['e'] - value) <= 1e-12
                                assert np.isclose(timed_a3['eps_ref'], eps, rtol=1e-10, atol=0)
                        assert abs(k_error-full_error) <= eps
                        rows.append([cid, f'{distance:.7g}', admission['shared_field_case_mask'][cid],
                                     f"{a3['diagnostic_native_tight_lower_bounds'][cid]:.7g}",
                                     f"{abs(k_error-full_error):.7g}", f'{eps:.7g}'])
                    selected = a3['selected']
                    text += (f"#### Existing support under the tuned POD policy\n\n"
                             f"One frozen replay uses the existing k16 support with {selected['scheme']}, "
                             f"step {selected['dt']}, production tolerance {selected['gtol']:.7g}, on "
                             f"case ids {sorted(int(c) for c in a3['common_field_distances'])}. "
                             f"Own-objective Q1: k16={a3['own_objective_Q1']['k16']['q1']}, "
                             f"no-HR={a3['own_objective_Q1']['none']['q1']}. "
                             'The exact anchor populations and reproduction evidence remain in the export. '
                             'The common field check and the additional whole-objective checks are separate.\n\n')
                    text += table(['Case', 'k16 to no-HR field distance', 'Common field pass',
                                   'Tight-trajectory distance lower bound', 'Absolute scalar E0-error change',
                                   'Conditional reference uncertainty'], rows) + '\n\n'
                    text += (f"The common field limit is {admission['shared_field_bar']:.7g}; "
                             f"the additional gradient limit is {admission['extra_gradient_bar']:.7g}, "
                             f"with measured maximum {admission['delta_g_max']:.7g}. "
                             f"Full-objective stationarity misses {admission['extra_full_stationarity_bad_steps']}/"
                             f"{admission['extra_full_stationarity_steps']} steps against fraction limit "
                             f"{admission['extra_full_stationarity_fraction_limit']:.7g}. "
                             'The triangle lower bound subtracts both production-to-tight distances from '
                             'the observed model-to-model distance; the remaining failing case cannot be '
                             'explained solely by those production solve tolerances. Scalar E0-error changes '
                             'are within reference uncertainty on this subset, so this is a consistency '
                             'failure, not a demonstrated physical-accuracy loss or a bank advantage. '
                             'The frozen stop rule blocks full-validation HR release; no support was refitted.\n\n')
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
    regime_table = table(['Qualified historical setting', 'Spearman(error, viscosity)',
                          'Spearman(error, nominal steepness)', 'Cases above 1% error', 'Largest-error case ids'],
                         [[f"`{v['setting']}`", f"{v['viscosity_spearman']:.4g}",
                           f"{v['nominal_steepness_spearman']:.4g}", v['errors_above_1pct'],
                           ', '.join(str(c) for c in v['largest_error_cases'])] for v in regime_statistics])
    operator_sources = [WTS / '2026-10-26-b3d-solver/experiments/b3d-solver' / rel for rel in [
        'vendor/speed3d/vendor/quad3d/offmesh.py',
        'vendor/speed3d/vendor/quad3d/vendor/burgers3d-span/common.py',
        'vendor/speed3d/t2core.py']]
    for source in operator_sources:
        PINS[str(source)] = sha(source)
    operator_text = r'''### Which equations are actually being measured

The inherited bank kernel combines analytic continuum advection with mesh-dependent discrete
diffusion and discrete test moments. In the quadrature term,

$$
\mathcal{N}_q(c)=P_q^\top\big[(B_qc)\odot(D_qc)\big],\qquad
(\Lambda_N)_{\mathbf{k},\mathbf{k}}=
4(N-1)^2\sum_{d=1}^{3}\sin^2\!\left(\frac{\pi k_d}{2(N-1)}\right).
$$

Here $B_q$ evaluates the frozen bank, $D_q$ its analytic directional derivative, $P_q$ the
weighted tests, and $\Lambda_N$ supplies the discrete diffusion eigenvalues. The weak residual
uses $\mathcal{N}_q(c)+\nu\Lambda_NA_Nc$, with $A_N$ the mesh-derived test moments. $N$ denotes
nodes per axis here; historical plots using intervals have $L=N-1$.

The weak least-squares objective also has the inherited diagonal mode weights

$$
S_{\Delta t,\nu,\Lambda}=
\operatorname{diag}\!\left((a_0+\Delta t\,b_0\,\nu\,\lambda_{\mathbf{k}})^{-1}\right),
$$

where $a_0,b_0$ are coefficients of the time scheme. Changing the time step or diffusion
eigenvalues changes these weights as well as the time-discrete residual and test directions.
The measured time-step and eigenvalue interventions are method packages; even verified solves
do not isolate pure temporal truncation or physical diffusion bias from those weighting effects.

This archived formulation deviates from the general FOM-exact upwind rule in `AGENTS.md`.
The campaign preserves it to reconcile the actual historical method; it does not silently
substitute an upwind operator. POD's central finite-difference residual and the spectral full
solver use different discretizations. Their common E0 ST error permits an empirical accuracy–cost
comparison, but not a claim of identical discrete equations. Q1/Q2 certify the stated reduced
solve and its quadrature, not PDE or spatial consistency. Changing $N$ also changes test moments,
projection, Gram matrices and normalization; a composite mesh replay cannot isolate diffusion
or advection bias by itself. Any alternative operator is a separately named diagnostic control.

Unsampled POD restricts the trial state to its frozen span, $u=\Phi c$, and minimises
the full central finite-difference residual norm, $\|R_h(\Phi c)\|^2$. The residual
is not first projected by $\Phi^\top$. ECSW samples and weights that full residual.
Inherited metadata describing a residual projected onto the POD basis should be read with
this correction; the numerical kernels and archived results remain unchanged.

'''
    report = f'''# Burgers 3D refinement: recovered evidence, numerical gaps and controlled replays

Status: {report_status}
This report reconciles existing evidence and the bounded refinement campaign. The historical
numbers are conditional estimates against approved E0 ST references; new numerical results
are accepted only with their independent audit status. This generated version does not
promote missing or subset evidence into full-validation qualification.

## Problem and experimental controls

The problem is scalar viscous Burgers flow on the unit cube with zero boundary values.
The existing initial fields contain smooth Gaussian bumps with varying positions, widths,
amplitudes and viscosity. The frozen spatial bank represents the field using a short vector
of coefficients; these comparisons use linear bank prefixes, not a deployed nonlinear head.
The solver minimises the PDE residual tested against smooth spatial functions.

The independent controls separate bank width, test-space size, quadrature, nonlinear work
and time integration. They reuse existing trained banks and valid trajectories. Numerical
design, code/small controls, exact job manifests and interpretation pass separate independent
review gates. Failed controls remain in the ledger. The output contract and exact population
travel with each result; the common scoring schedule has
{len(inv['reference']['output_times'])} fields through time {inv['reference']['output_times'][-1]}
on the {inv['reference']['scoring_lattice']} lattice.

{operator_text}

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

The per-case score is the maximum evolved-output field discrepancy on the common scoring
lattice, divided by the fixed reference initial-field norm. Initial reconstruction is retained
separately and is not silently substituted for an evolved-output error.

### There is no single strongest bank configuration

{table(['Purpose', 'Qualified historical setting', 'Median error (%)', 'Worst error (%)'], baseline_rows)}

These full-validation figures select different objectives. A low median can coexist with a
large error on a difficult case. The test-space-expanded setting retains its explicit name;
it is not silently pooled with a nominal test-space configuration.

{new_frontier_text}

## Historical cost and accuracy on the exact saved timing workload

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

A failed numerical certificate does not establish the dominant physical error source.
For example, the historical wide-bank Gauss rule above has successor distance
{values['quadrature_vs_physical_example']['successor_distance']:.7g} while its worst conditional
E0 error is {100*values['quadrature_vs_physical_example']['worst_E0']:.7g} percent. Closing that
certificate alone need not remove the much larger physical error. The reported Jacobian
condition number below concerns one failed step, not the conditioning of the whole family.

Restarting from coefficient history resets the historical QN cache. Fresh-restart controls
are labelled separately; full-trajectory one-factor controls assess refresh-policy changes.
The completed quadrature trajectories are reused, and only missing successor evidence is run.

![Accuracy versus bank width with fixed tests and quadrature](figs/accuracy_width.png)

Open markers retain failed or unverified qualification. A decreasing raw error is not enough
to claim a qualified width dial. The square test-space control failed historically, so the
original registered causal test-space verdict remains blocked even where expansion helped.

![Test-space dependence at fixed bank, quadrature and integration](figs/test_space.png)

{table(['Setting', 'Actual M', 'Quadrature m', 'Q1', 'Q2', 'Median error (%)', 'Worst error (%)'],
       [[f"`{v['key']}`", v['M'], v['m'], v['Q1'], v['Q2'], f"{100*v['median']:.6g}", f"{100*v['worst']:.6g}"] for v in test_rows])}

This is a descriptive comparison of existing qualified settings at fixed bank width,
quadrature and integration. Increasing the test space changes worst-case error much more
than the median in the upper part of this tested range. It does not change the earlier
registered verdict whose square-system control failed, and it does not claim that still
larger test spaces must improve the result.

![Error-budget interventions on their historical population](figs/error_budget.png)

Each bar is a median intervention distance on that member's own cases. The packages interact;
these bars are not additive contributions to total error. The representation floor and
time-discretisation package bind in different settings. The archived cohort limits remain.

## Mesh and input-regime dependence

{new_mesh_text}

![Historical same-allocation mesh cost and reported nonlinear work](figs/mesh_cost_iterations.png)

This mesh panel uses the mesh-free input/scoring-lattice contract and the historical deployment
mode policy, which may change the test list with the mesh. It establishes neither fixed-M
cost nor full-field output cost. Only the appropriately qualified curves support numerical
claims at their tested reference mesh; there is no all-mesh Q2 join here. Failed curves remain visible. Full output and arbitrary mesh-sampled input
must be measured separately. Historical Jacobian counts are not reconstructed from iteration
counts where the original counter is absent.

The earlier general comparative multi-mesh advantage was withdrawn after continuous-POD
controls. Mesh-independent reduced evolution is not exclusive to the bank. The present
within-bank mesh measurements do not restore that comparative claim, and earlier
two-dimensional multi-mesh cost ratios are not relabelled as three-dimensional evidence.

![Descriptive in-range viscosity and steepness associations](figs/in_range_regime.png)

The regime plot uses only the existing validation population and the same E0 reference.
Amplitude divided by viscosity is a nominal steepness proxy. These associations are not
one-factor causal tests and do not qualify new steep-front or viscosity-extrapolation regimes.
The earlier two-dimensional steep-front failures cannot be silently promoted to 3D results.

{regime_table}

These rank correlations are descriptive on the same validation cases, without a family-wise
inference claim. Viscosity, amplitude and family shape vary together, so they do not identify
a separate causal effect of viscosity or front steepness.

## Three different comparisons

The historical confirmed ROM-to-ROM comparisons are:

The common selected winner is `{dims['h2q']['C_old']['winner']}`. Its confirmation timing
source is job `{historical_confirmation['job_id']}` on `{historical_confirmation['gpu']}`:
case ids {first_confirmation['cases']}, {first_confirmation['reps']} repetitions per case,
under the full-field mesh-path query at the historical reference mesh. These are historical
paired workload ratios, not the four-case re-scored timing curve or new full-validation times.

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

Run `build.py` with the repository's absolute Python environment to snapshot checkpoints,
generate the report, then build and verify the Research Update PDF in dependency order.
Use `inventory.py` and `rescore_timing.py` when rebuilding the historical evidence inputs.
These analysis scripts use CPU NumPy and saved trajectories. They do
not rerun the historical solver grid. `data/SOURCES.json` pins historical sources;
`PROVENANCE.json` pins this report's direct inputs, and `VALUES.json` retains extracted
numbers and figure inputs. Compressed inventory and per-case JSON lines are backed up on
the prescribed code-only mirror. Full numerical reruns use each lane's frozen manifest.

## Plain-language glossary

- Bank: frozen spatial functions that turn a short vector into a field; r is their count.
- M: number of smooth functions used to test the PDE; m: number of integration points.
- N / L: nodes per mesh axis / intervals per axis, with $L=N-1$.
- $B_q$, $D_q$, $P_q$: bank values, analytic directional derivatives, and weighted test values at quadrature points.
- $A_N$, $\\Lambda_N$, $c$, $\\nu$: mesh test moments, discrete diffusion eigenvalues, reduced coefficients, and viscosity.
- W512k8: a width-512 bank with approximately eight tests per coefficient, including mode-shell completion.
- M2_512: a separately trained frozen bank with its own source hash; it is not a prefix of the W bank.
- M2 / W1024: short labels for the specific frozen banks identified in the parameter tables; no new bank is trained in this campaign.
- POD k8/k16: support targets of eight/sixteen sampled residual rows per POD coefficient; these are not test-space labels.
- Bank keys: bank label, integration rule, nonlinear solver, time scheme, step size, and work policy, separated by vertical bars.
- FS K1 / QN k4: one fixed correction after the adaptive startup / matrix refresh every four steps, with the inherited slow-refresh logic retained.
- Gauss / gl / lattice / lat: tensor Gauss–Legendre rules / their short label / a rank-one integration point set / its short label.
- Mesh intervals: divisions along each spatial axis; nodes include the two boundary endpoints.
- E0 ST: approved space/time-refined reference; its uncertainty is conditional on convergence assumptions.
- Richardson estimate: numerical uncertainty inferred from reference refinements under an assumed convergence rate, rather than a rigorous bound.
- Error: largest output-time field difference divided by the reference initial-field norm.
- Median / worst: middle / largest case error; they answer different accuracy questions.
- Outlier count: number of cases exceeding the table's stated error threshold.
- Full query: input projection, reduced time evolution and requested output decoding.
- Descriptor / prepared native: supplied initial-condition parameters and canonical normalization / method-specific precomputed field or sine coefficients. These are distinct input contracts.
- mf / mesh / held: fixed Gauss input and fixed output / native mesh input and output at that mesh / supplied baseline coefficients held fixed for an untimed operator diagnostic.
- Device output: fully computed fields resident on the GPU; host transfer, scoring and saving are separate work.
- Phase: a separately measured part of a query; separate calls are not assumed additive to fused-query cost.
- Q1: nonlinear-solver validity, stationarity and agreement with tighter solves.
- Q2: quadrature successor and own-state nonlinear agreement, with the finer rule checked too.
- Mesh Q2 scope: successor field agreement on the common scoring lattice; growing-output native solve checks do not provide a finer-native-grid E0 reference certificate.
- Rho: relative change in the projected nonlinear term when the integration rule is refined at the same returned state.
- Qualified / admitted: satisfies the method's declared checks on the required population; nonapplicable checks are labelled, and unknown evidence is not a pass.
- CN / BDF2 / BE: Crank–Nicolson / second-order backward differentiation / backward Euler.
- FS / QN: fixed full-Jacobian correction sweeps / a solver that reuses its small normal matrix.
- Jacobian: derivative of the reduced residual; stationarity measures whether further correction remains.
- Contraction / residual-Jacobian pair: a declared tensor/matrix calculation / a requested joint residual-and-derivative evaluation. Counts describe logical source operations, not compiler kernels.
- Trust radius / damping: controls on correction size / regularisation of a nonlinear correction.
- Baseline / extra / refresh / trust / strict: original solver / larger correction budget with unchanged slow-refresh trigger / matrix refreshed each step / larger trust radius / tighter internal stopping.
- POD / ECSW: a standard linear reduced basis / weighted sampling of its residual equations.
- FOM / PS / FD: a full-state solver / the frozen sine pseudo-spectral solver / finite differences.
- Oversampled sine collocation: evaluate the nonlinear term on a larger grid before projecting it; the inherited PS implementation does not claim exact de-aliasing.
- Support / active points: integration points whose fitted weights are positive.
- NNLS / optimality check: fitting nonnegative weights / checking whether a permitted weight change could still improve that fit.
- KKT: the constrained optimality conditions used to check the nonnegative-weight fit.
- Banned points: candidates excluded by the greedy fit's fixed drop/ban policy; an excessive banned fraction fails its quality check.
- No-HR: evaluate the complete reduced objective without weighted spatial sampling.
- Frontier: settings that no other admissible setting beats in both cost and the chosen error metric.
- Joint accuracy match: both median and worst competitor upper conditional errors lie below the subject's lower errors on the same cases.
- Projection floor: best reference-field fit in the frozen span at a stored time; it does not identify the cause of a trajectory's larger error.
- Test moment / operator intervention: overlap of a bank function with a test / an explicitly changed equation component with other quantities held fixed.
- Mode weights: the inherited scaling of weak residual rows; they depend on the time scheme, step size, viscosity and diffusion eigenvalues.
- Shared / incremental setup: reused bank or basis preparation / additional preparation for one configuration; neither includes historical training unless stated.
- Cumulative peak memory: largest device allocation seen anywhere earlier in the job, not a standalone method's memory requirement.
- Diagnostic subset: selected cases for explaining a failure, not a whole-population performance estimate.
- Post hoc / retracted: selected after observing data / no longer an accepted earlier claim.
- Spearman correlation: association between the ranks of two quantities; not a causal effect.
- Q/T/S/F/e_P: quadrature / time / nonlinear solve / reduced-form intervention / representation floor.
- Mesh-free: use method-specific input and output points; full-field output requests every mesh value.
- Conditional / provisional: valid only with stated numerical assumptions / awaiting required evidence.
'''
    report_path = HERE / report_filename
    if closure:
        assert values.get('full_validation_frontier') and values.get('fixed_mode_mesh')
        assert closure['numerical_results_audit_ready'] is True
        final_queue = load(HERE / 'data/FINAL-QUEUE-AUDIT.json')
        assert not final_queue['live_campaign_jobs']
    report_path.write_text(report)
    draft = HERE / '2026-10-26-burgers3d-refinement.md'
    if closure and draft != report_path and draft.is_file():
        draft.unlink()  # Generated draft remains recoverable in the campaign's backed-up Git history.
    for name in ['inventory.py', 'rescore_timing.py', 'make_report.py', 'frontier_report.py',
                 'checkpoint.py', 'update_research.py', 'build.py', 'mesh_report.py']:
        code = HERE / name
        if code.is_file():
            PINS[str(code)] = sha(code)
    write(HERE / 'VALUES.json', values)
    write(HERE / 'PROVENANCE.json', PINS)
    print(report_path)


if __name__ == '__main__':
    main()
