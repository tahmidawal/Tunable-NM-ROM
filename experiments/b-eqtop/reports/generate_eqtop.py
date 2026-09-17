"""Generate the b-eqtop report and `summary.json` from the audit JSONs. Nothing is typed.

    python reports/generate_eqtop.py --j1 checks/bet101-audit.json [--j2 checks/bet201-audit.json]
                                     --out reports/2026-09-17-b-eqtop
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def f(x, d=4):
    return '—' if x is None else f'{x:.{d}f}'


def sci(x):
    return '—' if x is None else f'{x:.2e}'


def yn(x):
    return '—' if x is None else ('yes' if x else 'no')


def table(header, rows):
    out = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    out += ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]
    return '\n'.join(out) + '\n'


def rule_rows(rules, job):
    rows = []
    for x in rules:
        rows.append([x['q'], x['M'], x['source'], x['arm'], x['fit_states'], x['candidates'],
                     x['m'], x['m_target'], sci(x['relative_fit']), f(x['rho_max']), f(x['rho_p95']),
                     f(x['rho_median']), yn(x['certified_primary']), yn(x['certified_tight']),
                     yn(x['certified_secondary']), yn(x['truncated']),
                     f(x.get('fit_seconds'), 0), job])
    return rows


RULE_HDR = ['$q$', '$M$', 'source', 'arm', 'fit states', 'pool', '$m$', '$m$ target',
            'NNLS rel. fit', '$\\rho_{\\max}$', '$\\rho_{95}$', '$\\rho_{\\rm med}$',
            'primary', 'tight', 'secondary', 'truncated', 'fit (s)', 'job']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--j1', required=True)
    p.add_argument('--j2', default=None)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    A = json.loads(Path(a.j1).read_text())
    B = json.loads(Path(a.j2).read_text()) if a.j2 and Path(a.j2).exists() else None
    bars = A['bars']
    summary = []

    def row(**kw):
        summary.append(dict(kw))

    md = []
    v = A['verdict']
    lad = A['ladders']
    md.append(f"# b-eqtop — {'every rung of the EQ ladder is primary-certified' if v['every_rung_primary_certified'] else 'the top rungs are not all certifiable at m ≤ 6144'}; the primary ladder is {'monotone' if v['primary_ladder_monotone_evolved'] else 'NOT monotone'} on the evolved metric\n")
    md.append(f"Final numbers from job `{A['job_id']}` (attempt `{A['attempt']}`, `{A['gpu']}`, source `{A['commit']}`, elapsed {f(A['elapsed_seconds'], 0)} s; fit phase {f(A['fit_phase_seconds'], 0)} s)"
              + (f" and job `{B['job_id']}` (attempt `{B['attempt']}`, `{B['gpu']}`, source `{B['commit']}`, elapsed {f(B['elapsed_seconds'], 0)} s)" if B else '')
              + ". Frozen Burgers checkpoint `18f0266ae6f0…` at 256 intervals, six opened development cases, budget-600 block-damped variable projection, $M = 4(K+q)$. Pre-registered design: [`../DESIGN.md`](../DESIGN.md). Every number below is read from the audit JSONs.\n")
    md.append(f"Failed blocking gates, job 1: **{', '.join(A['failed']) or 'none'}**" + (f"; job 2: **{', '.join(B['failed']) or 'none'}**" if B else '') + ".\n")

    # ---- verdict block
    md.append('## Verdict against the pre-registered criteria\n')
    md.append(table(['criterion', 'holds', 'measured'], [
        ['P1: primary-certified rule at every rung', yn(v['every_rung_primary_certified']),
         f"rungs without one: {v['rungs_without_primary_rule'] or 'none'}"],
        ['P2: primary ladder monotone on evolved times, all converged, EQ cheaper than dense',
         yn(v['primary_ladder_passes']),
         f"monotone {yn(v['primary_ladder_monotone_evolved'])}; converged {yn(lad['primary']['all_converged']) if lad['primary'] else '—'}; cheaper than dense {yn(lad['primary']['cheaper_than_dense_where_measured']) if lad['primary'] else '—'}"],
        ['P3: tight ladder monotone on evolved times', yn(v['tight_ladder_monotone_evolved']),
         f"rungs present: {lad['tight']['q'] if lad['tight'] else '—'}"],
        ['hybrid ladder (certified EQ, dense where uncertified) monotone', yn(v['hybrid_ladder_monotone_evolved']),
         f"quadrature per rung: {lad['hybrid']['quadrature'] if lad['hybrid'] else '—'}"],
    ]))
    t = v['top_rung']
    md.append(f"Top rung $q = {t['q']}$: primary arm rule $m = {t['primary_rule']}$, $\\rho_{{\\max}} = {f(t['primary_rho_max'])}$, certified {yn(t['primary_certified'])}, evolved {f(t['evolved_percent'])} % against dense {f(t['dense_evolved_percent'])} %.\n")
    md.append('Predicted $m$ for each bar at the top rung from the per-arm log–log law $\\rho_{\\max} \\propto m^{-\\alpha}$:\n')
    md.append(table(['arm', '$m$ for primary (0.116)', '$m$ for tight (0.06)'],
                    [[arm, f(d.get('primary'), 0), f(d.get('tight'), 0)] for arm, d in t['m_for_bar_by_arm'].items()]))

    # ---- ladders
    md.append('## The rebuilt ladders (job 1, one allocation)\n')
    for key in ('primary', 'tight', 'hybrid', 'dense'):
        d = lad.get(key)
        if not d:
            continue
        md.append(f"**{d['name']}** — monotone evolved: **{yn(d['monotone_evolved'])}**; monotone all-times: {yn(d['monotone_all_times'])}; every rung converged: {yn(d['all_converged'])}; every EQ rung cheaper than its dense twin: {yn(d['cheaper_than_dense_where_measured'])}; violations: {d['violations'] or 'none'}.\n")
        rows = []
        for i, q in enumerate(d['q']):
            rows.append([q, d['M'][i], d['quadrature'][i], d['m'][i] if d['m'][i] is not None else '—',
                         f(d['rho_max'][i]), yn(d['certified_primary'][i]), yn(d['certified_tight'][i]),
                         f(d['worst_evolved_percent'][i]), f(d['worst_all_times_percent'][i]),
                         f(d['worst_t0_compression_percent'][i]), f(d['median_gpu_ms'][i], 1),
                         f(d['cost_vs_dense_twin'][i], 3) if d['cost_vs_dense_twin'][i] is not None else '—',
                         yn(d['converged'][i]), d['arms'][i]])
            for metric, val in (('worst_evolved_percent', d['worst_evolved_percent'][i]),
                                ('worst_all_times_percent', d['worst_all_times_percent'][i]),
                                ('t0_compression_percent', d['worst_t0_compression_percent'][i]),
                                ('median_gpu_ms', d['median_gpu_ms'][i])):
                row(table='ladder', ladder=key, arm=d['arms'][i], q=q, m=d['m'][i],
                    population=d['quadrature'][i], metric=metric, value=val,
                    certified_primary=d['certified_primary'][i], certified_secondary=None,
                    certified_tight=d['certified_tight'][i], rho_max=d['rho_max'][i],
                    job_id=A['job_id'], source_sha=A['commit'])
        md.append(table(['$q$', '$M$', 'quadrature', '$m$', '$\\rho_{\\max}$', 'primary', 'tight',
                         'worst evolved %', 'worst all-times %', '$t=0$ compression %',
                         'median GPU ms', 'cost / dense twin', 'converged', 'arm'], rows))

    # ---- every timed arm
    md.append('### Every timed arm and the same-job full-order controls\n')
    rows = []
    for x in A['arms']:
        rows.append([x['arm'], x['q'] if x['q'] is not None else '—', x['M'] or '—', x['m'] or '—',
                     x['quadrature'] or '—', f(x['rho_max']), yn(x['certified_primary']), yn(x['certified_tight']),
                     f(x['worst_evolved_percent']), f(x['worst_all_times_percent']),
                     f(x['worst_t0_compression_percent']), f(x['median_gpu_ms'], 1),
                     f(x['median_iterations'], 1), x['total_budget_exits'] if x['total_budget_exits'] is not None else '—',
                     sci(x['max_joint_stationarity']), yn(x['converged'])])
        if x['family'] == 'fom':
            for metric in ('worst_evolved_percent', 'worst_all_times_percent', 'median_gpu_ms'):
                row(table='controls', ladder=None, arm=x['arm'], q=None, m=None, population='fom',
                    metric=metric, value=x[metric], certified_primary=None, certified_secondary=None,
                    certified_tight=None, rho_max=None, job_id=A['job_id'], source_sha=A['commit'])
    md.append(table(['arm', '$q$', '$M$', '$m$', 'quadrature', '$\\rho_{\\max}$', 'primary', 'tight',
                     'worst evolved %', 'worst all-times %', '$t=0$ %', 'median GPU ms', 'median iters',
                     'budget exits', 'worst joint gradient', 'converged'], rows))
    md.append('### $\\rho$ against the evolved error at the top rung\n')
    md.append(table(['arm', '$m$', '$\\rho_{\\max}$', '$\\rho_{95}$', 'worst evolved %', 'primary', 'tight'],
                    [[x['arm'], x['m'], f(x['rho_max']), f(x['rho_p95']), f(x['evolved_percent']),
                      yn(x['certified_primary']), yn(x['certified_tight'])] for x in v['top_rung_rho_vs_evolved']]))

    # ---- rules
    md.append('## Every rule, its fit and its held-out $\\rho$\n')
    md.append(f"Bars: primary $\\rho_{{\\max}} \\le {bars['primary']}$, tight $\\rho_{{\\max}} \\le {bars['tight']}$, secondary $\\rho_{{95}} \\le {bars['primary']}$; 512 held-out reachable states per rung from 8 certification trajectories disjoint from the 24 fit trajectories. **The NNLS relative fit never certifies a rule.** Source `qrg304` rows are the archived rules re-certified in this job.\n")
    rows = rule_rows(A['rules'], A['job_id'])
    if B:
        rows += rule_rows(B['rules'], B['job_id'])
    md.append(table(RULE_HDR, rows))
    for job in ([A] + ([B] if B else [])):
        for x in job['rules']:
            for metric, val in (('rho_max', x['rho_max']), ('rho_p95', x['rho_p95']),
                                ('rho_median', x['rho_median']), ('relative_fit', x['relative_fit']),
                                ('fit_seconds', x.get('fit_seconds'))):
                row(table='rules', ladder=None, arm=x['arm'], q=x['q'], m=x['m'],
                    population=f"{x['source']}:{x['population']}", metric=metric, value=val,
                    certified_primary=x['certified_primary'], certified_secondary=x['certified_secondary'],
                    certified_tight=x['certified_tight'], rho_max=x['rho_max'],
                    job_id=job['job_id'], source_sha=job['commit'])

    # ---- laws
    md.append('## The empirical law $\\rho_{\\max} \\propto m^{-\\alpha}$ per rung and arm\n')
    rows = []
    for job in ([A] + ([B] if B else [])):
        for l in job['laws']:
            fit = l['fit_rho_max'] or {}
            rows.append([l['q'], l['arm'], ', '.join(str(m) for m in l['m']),
                         ', '.join(f(r) for r in l['rho_max']), f(fit.get('alpha')), fit.get('points', '—'),
                         f(l['m_for_bar'].get('primary'), 0), f(l['m_for_bar'].get('tight'), 0),
                         l['cheapest_certified_m'].get('primary') or '—', l['cheapest_certified_m'].get('tight') or '—',
                         job['job_id']])
            row(table='laws', ladder=None, arm=l['arm'], q=l['q'], m=None, population='reachable',
                metric='alpha', value=fit.get('alpha'), certified_primary=None, certified_secondary=None,
                certified_tight=None, rho_max=None, job_id=job['job_id'], source_sha=job['commit'])
            for b_, val in l['m_for_bar'].items():
                row(table='laws', ladder=None, arm=l['arm'], q=l['q'], m=None, population='reachable',
                    metric=f'm_for_{b_}_bar_by_law', value=val, certified_primary=None, certified_secondary=None,
                    certified_tight=None, rho_max=None, job_id=job['job_id'], source_sha=job['commit'])
    md.append(table(['$q$', 'arm', '$m$ grid fitted', '$\\rho_{\\max}$ per $m$', '$\\alpha$', 'points',
                     '$m$ for primary (law)', '$m$ for tight (law)', 'cheapest certified $m$ (primary)',
                     'cheapest certified $m$ (tight)', 'job'], rows))
    md.append('### Chains: how each (rung, arm) stopped\n')
    rows = []
    for job in ([A] + ([B] if B else [])):
        for k, c in (job['chains'] or {}).items():
            rows.append([k, c['stop_reason'], c['rules_fitted'], c['certified'], job['job_id']])
    md.append(table(['chain', 'stop reason', 'rules fitted', 'primary-certified', 'job'], rows))

    # ---- gates
    md.append('## Gates and cross-job fidelity\n')
    for job in ([A] + ([B] if B else [])):
        rows = [[k, yn(c['passed']), 'informational' if c.get('blocking') is False else 'blocking']
                for k, c in sorted(job['checks'].items())]
        md.append(f"### {job['attempt']} (job {job['job_id']})\n")
        md.append(table(['gate', 'passed', 'kind'], rows))
        fid = job.get('fidelity') or {}
        if fid:
            rows = []
            for arm, d_ in sorted(fid.items()):
                dd = d_['detail']
                rows.append([arm, dd.get('comparator', '—'), sci(dd.get('declared_tolerance')),
                             sci((dd.get('worst_all_times_percent') or {}).get('relative_difference')),
                             sci((dd.get('worst_evolved_percent') or {}).get('relative_difference')),
                             yn(d_['passed'])])
            md.append(table(['arm', 'qrg304 comparator', 'tolerance', 'rel. diff (all-times)',
                             'rel. diff (evolved)', 'passed'], rows))

    md.append('## Glossary\n')
    md.append('\n'.join([
        '- **$q$ / rung** — the number of fixed correction directions added to the frozen head; one value of $q$ is a rung; the solved state is $u = G(h_\\theta(z) + C_q y)$.',
        '- **$M$** — the number of weak test equations per step, $M = 4(K+q)$ sine modes.',
        '- **$m$** — the number of grid points the empirical-quadrature (EQ) rule keeps for the advection integral; dense arms use all 65 025 interior points.',
        '- **EQ / dense** — the advection integral replaced by an $m$-point nonnegative rule / integrated exactly on the grid. Same unknowns, same tests.',
        '- **$\\rho$** — the rule\'s relative error on the advection functional, $\\rho(u) = \\|\\sum_j w_j\\Phi(x_j)a(u)(x_j) - \\Phi^\\top a(u)\\| / \\|\\Phi^\\top a(u)\\|$, a property of a rule and a state.',
        '- **$\\rho_{\\max}$, $\\rho_{95}$, $\\rho_{\\rm med}$** — the maximum, 95th percentile and median of $\\rho$ over the 512 held-out reachable states of a rung.',
        '- **primary / tight / secondary bar** — $\\rho_{\\max} \\le 0.116$ / $\\rho_{\\max} \\le 0.06$ / $\\rho_{95} \\le 0.116$; a rule is certified on a bar if it meets it and is not truncated.',
        '- **reachable states** — every Levenberg–Marquardt iterate of the ROM\'s own dense-quadrature rollouts on training-family trajectories; **held-out** ones come from trajectories disjoint from the fit trajectories and from the six evaluation cases.',
        '- **arm (rule)** — how the fit design was formed: `std` (incumbent fit-state count, unit-norm rows), `fs64` (64 fit states, unit-norm rows, QR-compressed), `rhow64` (64 fit states, rows scaled per state so the objective is $\\sum_s \\rho_s^2$, QR-compressed). `qrg304` rows are the parent job\'s archived rules.',
        '- **fit states / pool / design rows** — the number of reachable states the rule is fitted on; the number of candidate grid points it may select from; states × $M$ equations.',
        '- **NNLS relative fit** — the residual of the nonnegative least-squares fit on its own design; reported, never used to certify.',
        '- **truncated** — the fitter hit its walltime cap before reaching the target $m$; disqualified.',
        '- **$\\alpha$ / law** — the slope of $\\log\\rho_{\\max}$ against $\\log m$ over a chain; the "$m$ for bar" columns are where that line crosses the bar (an extrapolation where no fitted point is below it).',
        '- **chain / stop reason** — the ascending-$m$ sequence of fits for one (rung, arm); `certified_and_confirmed` = a primary-certified rule plus one confirmation point, `grid_exhausted`, `truncated_at_walltime`, `submission_deadline`.',
        '- **primary / tight / hybrid ladder** — six rungs with, per rung, the cheapest rule certified on that bar; the hybrid uses dense quadrature at rungs with no primary-certified rule.',
        '- **worst evolved % / worst all-times % / $t=0$ compression %** — same-grid relative error (against the same job\'s converged `fft_tight` solve) worst over the six cases and over $t > 0$ / over all six output times / at $t = 0$ (the decoder\'s compression of the supplied field).',
        '- **median GPU ms** — median over three timed repetitions × six cases of the complete query, device-synchronised, after a 0.25 s burn-in; **cost / dense twin** is the ratio to the same-job dense arm at the same $q$ (never across jobs).',
        '- **converged** — every step exited on a convergence criterion (no budget exits) and the worst normalised joint gradient is $\\le 10^{-6}$.',
        '- **fft_tight / fft_loose / nt1e-2_dt01** — same-job full-order Newton solves; `fft_tight` is the same-grid reference so its own error is zero.',
        '- **cross-job fidelity** — a named arm of this job reproducing a named arm of `qrg304` to the declared tolerance: $10^{-9}$ where the direction matrix is bitwise (and always at $q = 0$), $10^{-3}$ otherwise.',
    ]) + '\n')
    text = '\n'.join(md)
    Path(a.out + '.md').write_text(text)
    (HERE / 'summary.json').write_text(json.dumps(dict(
        report=str(Path(a.out + '.md').name), report_sha256=hashlib.sha256(text.encode()).hexdigest(),
        rows=summary), indent=1) + '\n')
    print('wrote', a.out + '.md', 'rows', len(summary))


if __name__ == '__main__':
    main()
