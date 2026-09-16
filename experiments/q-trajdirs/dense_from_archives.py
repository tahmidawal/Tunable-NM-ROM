"""Part 1 of the 2026-09-16 redirect: the DENSE ladder from data that already exists.

No GPU, no JAX, no new job. Every number is recomputed in NumPy from the saved output
fields of four already-collected, checksum-verified, independently audited jobs —
`cclad01` (cheap-corrections), `btq101`, `btq102`, `btq201` (b-ladder-top) — restored
from their Git-tracked archive chunks.

For each job, each DENSE arm and each case, the same-grid error against that job's own
converged `fft_tight` solve is recomputed at every output time, and the arm's
worst-over-cases all-times, evolved-times and t = 0 compression are formed exactly as
`audit_qtd.py` forms them. Dense ladders are then assembled by (job, test-count rule) and
read for monotonicity, converged non-dominated set, cost span and error span on both
metrics.

The known limits of this part, stated here rather than discovered later:

* the rungs of one ladder can come from ONE job only, so a cost span is only taken within
  a job; where a rung's converged twin lives in another job (the q = 256 rung, which
  converges only at per-step budget 600 in `btq102`) that is reported as a cross-job
  substitution and flagged, never folded into a span silently;
* `cclad01` and `qlad01` ran at per-step budget 180. Budget is an exit-reason fact, so
  arms with zero budget exits are unaffected by it and arms with budget exits are
  reported as not converged.

That is exactly why the redirect also runs one job: `qtd02` puts every dense rung at
budget 600 in one allocation on one GPU.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def median(x):
    return float(np.median(np.asarray(x, dtype=float)))


def mono(v):
    return bool(all(b <= a + 1e-12 for a, b in zip(v, v[1:])))


def nondominated(rows, cost, err):
    pts = [r for r in rows if r.get(cost) is not None and r.get(err) is not None]
    return sorted({r['arm'] for r in pts
                   if not any((o[cost] <= r[cost] and o[err] <= r[err] and
                               (o[cost] < r[cost] or o[err] < r[err])) for o in pts)})


def spans(rows, err):
    if len(rows) < 2:
        return dict(points=len(rows), cost_span=None, error_span=None)
    c = [x['gpu_ms'] for x in rows]
    v = [x[err] for x in rows]
    return dict(points=len(rows), cost_span=max(c) / max(min(c), 1e-300),
                error_span=max(v) / max(min(v), 1e-300))


def load_job(root, name):
    out = Path(root) / name / 'restored' / 'output'
    r = json.loads((out / 'result.json').read_text())
    cache = {}

    def fields(a):
        if a not in cache:
            cache[a] = np.load(out / a)['fields']
        return cache[a]

    refs = {x['case']: fields(x['artifact']) for x in r['reference']}
    base = {x['case']: x['artifact'] for x in r['invocations'] if x['name'] == 'fft_tight'}
    assert len(base) == len(refs), (name, sorted(base), sorted(refs))
    table, recheck = {}, 0.
    for x in r['invocations']:
        f = fields(x['artifact'])
        truth = refs[x['case']]
        n0 = np.linalg.norm(truth[0])
        ref = np.linalg.norm((f - truth).reshape(len(f), -1), axis=1) / n0
        recheck = max(recheck, abs(float(np.max(ref)) - x['error']['fixed_initial_max'])
                      / max(x['error']['fixed_initial_max'], 1e-300))
        sg = np.linalg.norm((f - fields(base[x['case']])).reshape(len(f), -1), axis=1) / n0
        t = table.setdefault(x['name'], dict(
            arm=x['name'], job=name, kind=x['kind'], q=x.get('q'), M=x.get('M'),
            m=x.get('m'), rule=x.get('rule'), quadrature=x.get('quadrature'),
            variant=x.get('variant') or x.get('fix'), dt=x.get('dt'),
            step_budget=x.get('step_budget'), trust_scale=x.get('trust_scale', 1.0),
            per_time={}, allt={}, evo={}, t0={}, gpu_ms=[], budget_exits=[], gj=[],
            converged=[], completed=[]))
        t['per_time'][x['case']] = sg.tolist()
        t['allt'][x['case']] = float(np.max(sg))
        t['evo'][x['case']] = float(np.max(sg[1:]))
        t['t0'][x['case']] = float(sg[0])
        t['gpu_ms'].append(x['gpu_seconds'] * 1e3)
        if x['kind'] == 'rom':
            t['budget_exits'].append(x.get('budget_exits', 0))
            g = x.get('worst_joint_stationarity')
            if g is None:
                g = max(float(np.max(x['step_stationarity'])), float(x['ic_stationarity']))
            t['gj'].append(float(g))
            t['completed'].append(bool(x.get('completed')))
            # One convergence rule for every job, applied here rather than trusted:
            # no budget exit anywhere, every exit reason in {1,2,4}, joint gradient <= 1e-6.
            t['converged'].append(bool(x.get('budget_exits', 0) == 0
                                       and bool(x.get('completed'))
                                       and float(g) <= 1e-6 * (1 + 1e-7)))
    rows = []
    for k, t in table.items():
        pt = np.array([t['per_time'][c] for c in sorted(t['per_time'])])
        rows.append(dict(
            arm=k, job=name, kind=t['kind'], q=t['q'], M=t['M'], m=t['m'], rule=t['rule'],
            quadrature=t['quadrature'], variant=t['variant'], dt=t['dt'],
            step_budget=t['step_budget'], trust_scale=t['trust_scale'],
            all_times=float(max(t['allt'].values()) * 100),
            evolved=float(max(t['evo'].values()) * 100),
            t0=float(max(t['t0'].values()) * 100),
            per_time=(np.max(pt, axis=0) * 100).tolist(),
            gpu_ms=median(t['gpu_ms']), reps=len(t['gpu_ms']),
            budget_exits=(int(np.sum(t['budget_exits'])) if t['budget_exits'] else None),
            max_joint_gradient=(float(np.max(t['gj'])) if t['gj'] else None),
            converged=(bool(all(t['converged'])) if t['converged'] else None)))
    return r, rows, recheck


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archives', required=True,
                   help='directory holding <job>/restored/output for each job')
    p.add_argument('--jobs', default='cclad01,btq101,btq102,btq201')
    p.add_argument('--out', required=True)
    a = p.parse_args()

    jobs, rows, checks = {}, [], {}
    for name in a.jobs.split(','):
        r, rr, recheck = load_job(a.archives, name)
        jobs[name] = dict(job_id=r.get('job_id'), gpu=r.get('gpu'), commit=r.get('commit'),
                          step_budget=(r['config'].get('strict') or {}).get('step_budget'),
                          elapsed_seconds=r.get('elapsed_seconds'),
                          reference_recomputation_relative=recheck,
                          physical_cases_sha256=hashlib.sha256(
                              np.ascontiguousarray(np.array(r['physical_cases'])).tobytes()
                          ).hexdigest())
        rows += rr
        checks[f'{name}_recomputes_archived_reference_error'] = dict(
            value=recheck, threshold=1e-9, passed=bool(recheck < 1e-9))

    cohorts = {v['physical_cases_sha256'] for v in jobs.values()}
    checks['all_jobs_share_one_cohort'] = dict(
        value=sorted(cohorts), passed=len(cohorts) == 1)

    dense = [x for x in rows if x['kind'] == 'rom' and x['quadrature'] == 'dense']
    fom = [x for x in rows if x['kind'] == 'fom']

    # One ladder per (job, test-count rule), retained solver variant only, budget as run.
    RETAINED = {'block', 'base', None}
    ladders = {}
    for job in jobs:
        for rule, label in (('m4', 'M = 4(K+q)'), ('m2', 'M = 2(K+q)'),
                            ('m256', 'fixed M = 256'), ('Mmax', 'M = 2112')):
            sel = [x for x in dense
                   if x['job'] == job and x['rule'] == rule
                   and x['variant'] in RETAINED and (x['dt'] in (None, 0.005))
                   and x.get('trust_scale', 1.0) == 1.0
                   and (x['step_budget'] in (None, 180, 600))]
            # Keep the LOWEST step budget per q so a ladder is one contract, and record
            # separately which rungs also have a higher-budget twin in the same job.
            best = {}
            for x in sel:
                if x['q'] not in best or (x['step_budget'] or 0) < (best[x['q']]['step_budget'] or 0):
                    best[x['q']] = x
            sel = sorted(best.values(), key=lambda x: x['q'])
            if len(sel) < 3:
                continue
            conv = [x for x in sel if x['converged']]
            lid = f'{job}_{rule}'
            ladders[lid] = dict(
                id=lid, job=job, rule=rule, label=f'{job}, dense, {label}',
                step_budgets=sorted({x['step_budget'] for x in sel}),
                rungs=[dict(q=x['q'], arm=x['arm'], M=x['M'], all_times=x['all_times'],
                            evolved=x['evolved'], t0=x['t0'], gpu_ms=x['gpu_ms'],
                            budget_exits=x['budget_exits'],
                            max_joint_gradient=x['max_joint_gradient'],
                            converged=x['converged'], per_time=x['per_time'])
                       for x in sel],
                monotone_evolved=mono([x['evolved'] for x in sel]),
                monotone_all_times=mono([x['all_times'] for x in sel]),
                all_converged=bool(sel and all(x['converged'] for x in sel)),
                converged_rungs=len(conv), rungs_total=len(sel),
                regressions_evolved=[dict(from_q=u['q'], to_q=w['q'], from_percent=u['evolved'],
                                          to_percent=w['evolved'])
                                     for u, w in zip(sel, sel[1:]) if w['evolved'] > u['evolved']],
                regressions_all_times=[dict(from_q=u['q'], to_q=w['q'],
                                            from_percent=u['all_times'], to_percent=w['all_times'])
                                       for u, w in zip(sel, sel[1:])
                                       if w['all_times'] > u['all_times']],
                nondominated_evolved=nondominated(
                    [dict(arm=x['arm'], gpu_ms=x['gpu_ms'], evolved=x['evolved']) for x in conv],
                    'gpu_ms', 'evolved'),
                nondominated_all_times=nondominated(
                    [dict(arm=x['arm'], gpu_ms=x['gpu_ms'], all_times=x['all_times'])
                     for x in conv], 'gpu_ms', 'all_times'))
            by = {x['arm']: x for x in conv}
            ladders[lid]['span_evolved'] = spans(
                [by[n] for n in ladders[lid]['nondominated_evolved']], 'evolved')
            ladders[lid]['span_all_times'] = spans(
                [by[n] for n in ladders[lid]['nondominated_all_times']], 'all_times')
            ladders[lid]['passes_redirected_criterion'] = bool(
                ladders[lid]['monotone_evolved'] and ladders[lid]['all_converged']
                and (ladders[lid]['span_evolved'].get('error_span') or 0) >= 2
                and (ladders[lid]['span_evolved'].get('cost_span') or 0) >= 2)

    # The cross-job substitution: the q = 256 dense rung that converges at budget 600.
    subs = [x for x in dense if x['job'] == 'btq102' and x['step_budget'] == 600
            and x['trust_scale'] == 1.0]

    out = dict(
        generated_from='already-collected, checksum-verified job archives; no GPU, no new job',
        jobs=jobs, checks=checks, ladders=ladders,
        dense_arms=sorted(dense, key=lambda x: (x['job'], x['rule'] or '', x['q'] or 0)),
        fom_controls=sorted(fom, key=lambda x: (x['job'], x['arm'])),
        budget600_dense_substitutes=subs,
        note=('a ladder is assembled inside ONE job so its cost span is a same-allocation '
              'ratio; cross-job rungs are listed separately and never folded into a span'))
    Path(a.out).write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(dict(
        jobs={k: v['job_id'] for k, v in jobs.items()},
        checks={k: v['passed'] for k, v in checks.items()},
        ladders={k: dict(monotone_evolved=v['monotone_evolved'],
                         monotone_all_times=v['monotone_all_times'],
                         all_converged=v['all_converged'],
                         span=v['span_evolved'],
                         passes=v['passes_redirected_criterion'])
                 for k, v in ladders.items()}), indent=2))
    print('WROTE', a.out)


if __name__ == '__main__':
    main()
