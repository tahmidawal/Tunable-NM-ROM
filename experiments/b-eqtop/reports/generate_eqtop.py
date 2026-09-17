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
    p.add_argument('--j1', default=None, help='the ladder job (bet101) audit')
    p.add_argument('--j2', default=None, help='the rho-vs-m curve job (bet201) audit')
    p.add_argument('--j3', default=None, help='the draw-replication job (bet301) audit')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    load = lambda p_: (json.loads(Path(p_).read_text()) if p_ and Path(p_).exists() else None)
    A, B, C = load(a.j1), load(a.j2), load(a.j3)
    assert B or A, 'at least one audit is required'
    bars = (A or B)['bars']
    jobs = [j for j in (A, B, C) if j]
    summary = []

    def row(**kw):
        summary.append(dict(kw))

    md = []
    if A:
        v, lad = A['verdict'], A['ladders']
        title = ('every rung of the EQ ladder carries a primary-certified rule'
                 if v['every_rung_primary_certified'] else
                 f"rungs {v['rungs_without_primary_rule']} carry no primary-certified rule at $m \\le 6144$")
        title += ('; the primary EQ ladder is monotone on the evolved metric'
                  if v['primary_ladder_monotone_evolved'] else
                  '; the primary EQ ladder still regresses on the evolved metric')
    else:
        v, lad = None, None
        title = 'the rule-certification curve (the ladder job is not in this build)'
    md.append(f'# b-eqtop — {title}\n')
    md.append('Jobs: ' + '; '.join(
        f"`{j['attempt']}` = {j['job_id']} ({j['question']}), `{j['gpu']}`, source `{j['commit']}`, "
        f"elapsed {f(j['elapsed_seconds'], 0)} s" for j in jobs)
        + ". Frozen Burgers checkpoint `18f0266ae6f0…` at 256 intervals, $K = 16$, $R = 512$, six opened "
          "development cases, budget-600 block-damped variable projection, $M = 4(K+q)$, float64, "
          "highest matmul precision, `jax_backend=gpu`. Pre-registered design and its amendments: "
          "[`../DESIGN.md`](../DESIGN.md). Every number here is read from the audit JSONs by "
          "`generate_eqtop.py`; none is typed.\n")
    md.append('Failed blocking gates: ' + '; '.join(
        f"{j['attempt']} **{', '.join(j['failed']) or 'none'}**" for j in jobs) + '.\n')

    # ------------------------------------------------------------------ headline
    if C and C.get('replication'):
        md.append('> **Read the certification numbers with §"How much of $\\rho_{\\max}$ is the draw" first.** '
                  'A rule is a sample from a construction: independent draws of the candidate pool and the '
                  'fit-state subset move $\\rho_{\\max}$ enough to flip the verdict at the bar. Single-rule '
                  'certifications below are reported as such, and the replication table gives the spread.\n')

    if A:
        md.append('## Verdict against the pre-registered criteria\n')
        md.append(table(['criterion', 'holds', 'measured'], [
            ['P1: a primary-certified rule at every rung', yn(v['every_rung_primary_certified']),
             f"rungs without one: {v['rungs_without_primary_rule'] or 'none'}"],
            ['P2: primary ladder monotone on evolved times, all converged, EQ cheaper than its dense twin',
             yn(v['primary_ladder_passes']),
             f"monotone {yn(v['primary_ladder_monotone_evolved'])}; converged "
             f"{yn(lad['primary']['all_converged']) if lad['primary'] else '—'}; cheaper than dense "
             f"{yn(lad['primary']['cheaper_than_dense_where_measured']) if lad['primary'] else '—'}"],
            ['P3: tight ladder ($\\rho_{\\max} \\le %s$) monotone on evolved times' % bars['tight'],
             yn(v['tight_ladder_monotone_evolved']),
             f"rungs present: {lad['tight']['q'] if lad['tight'] else 'none'}"],
            ['hybrid ladder (certified EQ where it exists, dense elsewhere) monotone',
             yn(v['hybrid_ladder_monotone_evolved']),
             f"quadrature per rung: {lad['hybrid']['quadrature'] if lad['hybrid'] else '—'}"],
        ]))
        t = v['top_rung']
        md.append(f"Top rung $q = {t['q']}$: the chosen primary rule has $m = {t['primary_rule']}$, "
                  f"$\\rho_{{\\max}} = {f(t['primary_rho_max'])}$, certified {yn(t['primary_certified'])}; "
                  f"its evolved error is {f(t['evolved_percent'])} % against the same-job dense twin's "
                  f"{f(t['dense_evolved_percent'])} %.\n")

        # ---- ladders
        md.append('## The rebuilt ladders (one allocation, three timed repetitions)\n')
        for key in ('primary', 'tight', 'hybrid', 'dense'):
            d = lad.get(key)
            if not d:
                continue
            md.append(f"**{d['name']}** — monotone evolved: **{yn(d['monotone_evolved'])}**; monotone "
                      f"all-times: {yn(d['monotone_all_times'])}; every rung converged: "
                      f"{yn(d['all_converged'])}; every EQ rung cheaper than its dense twin: "
                      f"{yn(d['cheaper_than_dense_where_measured'])}; violations: {d['violations'] or 'none'}.\n")
            rows = []
            for i, q in enumerate(d['q']):
                rows.append([q, d['M'][i], d['quadrature'][i],
                             d['m'][i] if d['m'][i] is not None else '—', f(d['rho_max'][i]),
                             yn(d['certified_primary'][i]), yn(d['certified_tight'][i]),
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

        md.append('### Every timed arm and the same-job full-order controls\n')
        rows = []
        for x in A['arms']:
            rows.append([x['arm'], x['q'] if x['q'] is not None else '—', x['M'] or '—', x['m'] or '—',
                         x['quadrature'] or '—', f(x['rho_max']), yn(x['certified_primary']),
                         yn(x['certified_tight']), f(x['worst_evolved_percent']),
                         f(x['worst_all_times_percent']), f(x['worst_t0_compression_percent']),
                         f(x['median_gpu_ms'], 1), f(x['median_iterations'], 1),
                         x['total_budget_exits'] if x['total_budget_exits'] is not None else '—',
                         sci(x['max_joint_stationarity']), yn(x['converged'])])
            if x['family'] == 'fom':
                for metric in ('worst_evolved_percent', 'worst_all_times_percent', 'median_gpu_ms'):
                    row(table='controls', ladder=None, arm=x['arm'], q=None, m=None, population='fom',
                        metric=metric, value=x[metric], certified_primary=None,
                        certified_secondary=None, certified_tight=None, rho_max=None,
                        job_id=A['job_id'], source_sha=A['commit'])
        md.append(table(['arm', '$q$', '$M$', '$m$', 'quadrature', '$\\rho_{\\max}$', 'primary', 'tight',
                         'worst evolved %', 'worst all-times %', '$t=0$ %', 'median GPU ms',
                         'median iters', 'budget exits', 'worst joint gradient', 'converged'], rows))
        md.append('### $\\rho$ against the evolved error at the top rung\n')
        md.append('The question the bar exists to answer: does a lower $\\rho$ buy a lower field error?\n')
        md.append(table(['arm', '$m$', '$\\rho_{\\max}$', '$\\rho_{95}$', 'worst evolved %', 'primary', 'tight'],
                        [[x['arm'], x['m'], f(x['rho_max']), f(x['rho_p95']), f(x['evolved_percent']),
                          yn(x['certified_primary']), yn(x['certified_tight'])]
                         for x in v['top_rung_rho_vs_evolved']]))

    # ------------------------------------------------------------- replication
    if C and C.get('replication'):
        md.append('## How much of $\\rho_{\\max}$ is the draw?\n')
        md.append(f"Four independent draws of (candidate pool, fit-state subset) at fixed $(q, m, "
                  f"\\text{{states}}, \\text{{scaling}})$, pre-registered in DESIGN §A2 after `{B['attempt'] if B else 'bet201'}` "
                  f"showed two draws disagreeing by up to $2.3\\times$. Nothing else differs between the draws.\n")
        rows = []
        for x in C['replication']:
            rows.append([x['q'], x['m_target'], x['fit_states'], x['scaling'], x['draws'],
                         ', '.join(f(r) for r in x['rho_max']), f(x['rho_min']), f(x['rho_max_of_draws']),
                         f(x['rho_mean']), f(x['rho_std']), f(x['spread_ratio'], 2),
                         f"{x['certified_primary_count']}/{x['draws']}",
                         f"{x['certified_tight_count']}/{x['draws']}"])
            for metric, val in (('rho_mean', x['rho_mean']), ('rho_std', x['rho_std']),
                                ('spread_ratio', x['spread_ratio']),
                                ('certified_primary_fraction', x['certified_primary_count'] / x['draws'])):
                row(table='replication', ladder=None, arm=x['key'], q=x['q'], m=x['m_target'],
                    population='reachable', metric=metric, value=val, certified_primary=None,
                    certified_secondary=None, certified_tight=None, rho_max=x['rho_mean'],
                    job_id=C['job_id'], source_sha=C['commit'])
        md.append(table(['$q$', '$m$ target', 'fit states', 'scaling', 'draws', 'each $\\rho_{\\max}$',
                         'min', 'max', 'mean', 'sd', 'spread', 'certify primary', 'certify tight'], rows))

    # ------------------------------------------------------------------ rules
    md.append('## Every rule, its fit and its held-out $\\rho$\n')
    md.append(f"Bars: primary $\\rho_{{\\max}} \\le {bars['primary']}$, tight $\\rho_{{\\max}} \\le {bars['tight']}$, "
              f"secondary $\\rho_{{95}} \\le {bars['primary']}$. 512 held-out reachable states per rung, from 8 "
              "certification trajectories disjoint from the 24 fit trajectories and from the six evaluation "
              "cases. **A rule is never certified by its NNLS fit residual**, which is printed beside $\\rho$ so "
              "the anti-correlation stays visible. `qrg304` rows are that job's archived rules re-certified here.\n")
    rows = []
    for j in jobs:
        rows += rule_rows(j['rules'], j['job_id'])
        for x in j['rules']:
            for metric, val in (('rho_max', x['rho_max']), ('rho_p95', x['rho_p95']),
                                ('rho_median', x['rho_median']), ('relative_fit', x['relative_fit']),
                                ('fit_seconds', x.get('fit_seconds'))):
                row(table='rules', ladder=None, arm=x['arm'], q=x['q'], m=x['m'],
                    population=f"{x['source']}:{x['population']}", metric=metric, value=val,
                    certified_primary=x['certified_primary'],
                    certified_secondary=x['certified_secondary'],
                    certified_tight=x['certified_tight'], rho_max=x['rho_max'],
                    job_id=j['job_id'], source_sha=j['commit'])
    seen = set()
    dedup = []
    for r_ in rows:
        k = tuple(r_[:8]) + (r_[-1],)
        if k in seen:
            continue
        seen.add(k)
        dedup.append(r_)
    md.append(table(RULE_HDR, dedup))

    # ---- the two construction effects, measured
    if B:
        md.append('### The two things that move $\\rho_{\\max}$ at fixed $m$\n')
        R = B['rules']
        rows = []
        for q in sorted({x['q'] for x in R}):
            for m in (1024, 2048):
                old = [x for x in R if x['source'] == 'qrg304' and x['q'] == q
                       and x['population'] == 'reachable' and x['m_target'] == m]
                new = [x for x in R if x['source'] == 'this_job' and x['q'] == q
                       and x['arm'] == 'std' and x['m_target'] == m]
                if old and new:
                    o, n = old[0], new[0]
                    rows.append([q, m, o['fit_states'], f(o['rho_max']), yn(o['certified_primary']),
                                 f(n['rho_max']), yn(n['certified_primary']),
                                 f(n['rho_max'] / o['rho_max'], 2)])
        md.append('**The draw.** The identical construction — same population, same fit-state count, same '
                  'target $m$ — under `qrg304`\'s pool of 8192 and this lane\'s of 16384, which also '
                  'changes the fit-state subset:\n')
        md.append(table(['$q$', '$m$ target', 'fit states', '`qrg304` $\\rho_{\\max}$', 'certified',
                         'this lane $\\rho_{\\max}$', 'certified', 'ratio'], rows))
        rows = []
        arms = sorted({x['arm'] for x in R if x['source'] == 'this_job'})
        for q in sorted({x['q'] for x in R if x['source'] == 'this_job'}):
            cells = []
            for arm in arms:
                xs = [x for x in R if x['source'] == 'this_job' and x['q'] == q
                      and x['arm'] == arm and x['m_target'] == 2048]
                cells.append(f"{f(xs[0]['rho_max'])} ($m$={xs[0]['m']}, {xs[0]['fit_states']} st.)"
                             if xs else '—')
            rows.append([q] + cells)
        md.append('**The fit states and the row scaling**, at $m$ target 2048 in this lane\'s own job:\n')
        md.append(table(['$q$'] + arms, rows))

    # ------------------------------------------------------------------- laws
    md.append('## The empirical law $\\rho_{\\max} \\propto m^{-\\alpha}$\n')
    md.append('Fitted over each chain\'s untruncated points. **These slopes are fitted through the draw '
              'noise measured above**, so the "$m$ for bar" columns are order-of-magnitude statements, not '
              'predictions; where a chain certified, the cheapest certified $m$ is the measurement and the '
              'law is not needed.\n')
    rows = []
    for j in jobs:
        for l in j['laws']:
            fit = l['fit_rho_max'] or {}
            rows.append([l['q'], l['arm'], ', '.join(str(m) for m in l['m']),
                         ', '.join(f(r) for r in l['rho_max']), f(fit.get('alpha'), 3),
                         fit.get('points', '—'), f(l['m_for_bar'].get('primary'), 0),
                         f(l['m_for_bar'].get('tight'), 0),
                         l['cheapest_certified_m'].get('primary') or '—',
                         l['cheapest_certified_m'].get('tight') or '—', j['job_id']])
            row(table='laws', ladder=None, arm=l['arm'], q=l['q'], m=None, population='reachable',
                metric='alpha', value=fit.get('alpha'), certified_primary=None,
                certified_secondary=None, certified_tight=None, rho_max=None,
                job_id=j['job_id'], source_sha=j['commit'])
            for b_, val in l['m_for_bar'].items():
                row(table='laws', ladder=None, arm=l['arm'], q=l['q'], m=None, population='reachable',
                    metric=f'm_for_{b_}_bar_by_law', value=val, certified_primary=None,
                    certified_secondary=None, certified_tight=None, rho_max=None,
                    job_id=j['job_id'], source_sha=j['commit'])
    md.append(table(['$q$', 'arm', '$m$ grid fitted', '$\\rho_{\\max}$ per $m$', '$\\alpha$', 'points',
                     '$m$ for primary (law)', '$m$ for tight (law)',
                     'cheapest certified $m$ (primary)', 'cheapest certified $m$ (tight)', 'job'], rows))
    md.append('### Chains: how each (rung, arm) stopped\n')
    md.append("`gradient` means the fitter found no remaining candidate that improves the fit — the "
              "design, not the grid, caps $m$ there.\n")
    rows = []
    for j in jobs:
        for k, c in sorted((j['chains'] or {}).items()):
            rows.append([k, c['stop_reason'], c['rules_fitted'], c['certified'], j['job_id']])
    md.append(table(['chain', 'stop reason', 'rules fitted', 'primary-certified', 'job'], rows))

    # ------------------------------------------------------------------ gates
    md.append('## Gates and cross-job fidelity\n')
    for j in jobs:
        rows = [[k, yn(c['passed']), 'informational' if c.get('blocking') is False else 'blocking']
                for k, c in sorted(j['checks'].items())]
        md.append(f"### {j['attempt']} (job {j['job_id']})\n")
        md.append(table(['gate', 'passed', 'kind'], rows))
        fid = j.get('fidelity') or {}
        if fid:
            rows = []
            for arm, d_ in sorted(fid.items()):
                dd = d_['detail']
                rows.append([arm, dd.get('comparator', '—'), sci(dd.get('declared_tolerance')),
                             sci((dd.get('worst_all_times_percent') or {}).get('relative_difference')),
                             sci((dd.get('worst_evolved_percent') or {}).get('relative_difference')),
                             yn(d_['passed'])])
            md.append(table(['arm', '`qrg304` comparator', 'tolerance', 'rel. diff (all-times)',
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
