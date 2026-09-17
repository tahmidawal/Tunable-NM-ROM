"""A SECOND, independent re-derivation of the report's headline numbers from the raw fields.

    python checks/recheck_headline.py <archive/output> <audit.json> --out checks/<attempt>-recheck.json

The lane's pre-job Codex audit could not run (DESIGN.md A1: account quota exhausted until
2026-09-19). This stands in for part of it. It re-reads every saved field, recomputes the
same-grid errors, the medians, the non-dominated set and the two headline claims WITHOUT
importing `audit_panel.py`, the driver, or JAX, and compares against the audit JSON. It is a
different code path to the same quantities, not a rerun of the same one.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('fields')
    p.add_argument('audit')
    p.add_argument('--out', required=True)
    p.add_argument('--fno-name', default='fno-large')
    a = p.parse_args()
    A = Path(a.fields)
    r = json.loads((A / 'result.json').read_text())
    au = json.loads(Path(a.audit).read_text())
    cache = {}

    def f(n):
        if n not in cache:
            cache[n] = np.load(A / n)['fields']
        return cache[n]

    refs = {x['case']: f(x['artifact']) for x in r['reference']}
    base = {x['case']: x['artifact'] for x in r['invocations'] if x['name'] == 'fft_tight'}
    rows = {}

    def add(name, case, fl, ms):
        tr = refs[case]
        n0 = np.linalg.norm(tr[0])
        sg = np.linalg.norm((fl - f(base[case])).reshape(len(fl), -1), axis=1) / n0
        rf = np.linalg.norm((fl - tr).reshape(len(fl), -1), axis=1) / n0
        d = rows.setdefault(name, dict(ev=[], al=[], ref=[], ms=[]))
        d['ev'].append(float(sg[1:].max()))
        d['al'].append(float(sg.max()))
        d['ref'].append(float(rf.max()))
        d['ms'] += list(ms)

    for x in r['invocations']:
        add(x['name'], x['case'], f(x['artifact']), [x['gpu_seconds'] * 1e3])
    tj = A / f'{a.fno_name}-timing.json'
    if tj.exists():
        for c in json.loads(tj.read_text())['cases']:
            add(a.fno_name, int(c['case_index']), f(c['artifact']), [s * 1e3 for s in c['device_seconds']])

    tab = {k: dict(worst_evolved_percent=100 * max(v['ev']), worst_all_times_percent=100 * max(v['al']),
                   worst_reference_percent=100 * max(v['ref']), median_gpu_ms=float(np.median(v['ms'])))
           for k, v in rows.items()}
    fam = {x['arm']: x['family'] for x in au['arms']}
    worst, per = 0., {}
    for x in au['arms']:
        t = tab[x['arm']]
        d = {k: abs(x[k] - t[k]) / max(abs(t[k]), 1e-300) for k in t}
        per[x['arm']] = d
        worst = max(worst, max(d.values()))

    def nondom(names, ck='median_gpu_ms', ek='worst_evolved_percent'):
        return sorted(n for n in names if not any(
            tab[m][ck] <= tab[n][ck] and tab[m][ek] <= tab[n][ek]
            and (tab[m][ck] < tab[n][ck] or tab[m][ek] < tab[n][ek]) for m in names if m != n))

    reduced = [k for k in tab if fam.get(k) in ('rom', 'fast', 'pod', 'free')]
    best = min(reduced, key=lambda k: tab[k]['worst_evolved_percent'])
    beats = [k for k in tab if fam.get(k) == 'fom'
             and tab[k]['worst_evolved_percent'] <= tab[best]['worst_evolved_percent']
             and tab[k]['median_gpu_ms'] <= tab[best]['median_gpu_ms']]
    out = dict(
        source_result=str((A / 'result.json').resolve()), audit=str(Path(a.audit).resolve()),
        job_id=r.get('job_id'), intervals=r['intervals'],
        worst_relative_difference_against_audit=worst,
        agrees_with_audit=bool(worst < 1e-12), per_arm_relative_difference=per,
        recomputed=tab,
        nondominated_gpu_evolved_all=nondom(list(tab)),
        nondominated_gpu_evolved_reduced_only=nondom(reduced),
        best_reduced_by_evolved=dict(arm=best, **tab[best]),
        fom_cheaper_and_at_least_as_accurate=[dict(arm=k, **tab[k]) for k in beats],
        note=('a second independent code path over the same saved fields; imports neither the audit '
              'nor the driver nor JAX. Stands in for part of the Codex audit that could not run (A1).'))
    Path(a.out).write_text(json.dumps(out, indent=2) + '\n')
    print('worst relative difference against the audit:', worst)
    print('best reduced:', best, round(tab[best]['worst_evolved_percent'], 4), '%',
          round(tab[best]['median_gpu_ms'], 1), 'ms')
    print('FOM both cheaper and at least as accurate:', beats)
    print('WROTE', a.out)


if __name__ == '__main__':
    main()
