"""Freeze the selection (DESIGN 6, A1.0 step 5, A1.5, A2.4, A3.6). Writes checks/FROZEN-SELECTION.json and prints its
sha256 (committed before any budget job is staged).

2D (from analyze2d's output against the selection reference): the global scheme, per setting the accurate-equal-cost
and fast-equal-accuracy arms of every candidate scheme, and the BUDGET ARMS of each setting (A1.5): LSPG-BE dt0
(baseline), the global scheme's accurate-equal-cost and fast-equal-accuracy arms (flagged selected), GAL-CN dt0
(diagnostic). Fallbacks of a selected arm (A2.4): the same arm with the next two larger rules of its family, then
'unavailable'.
3D (from select3d's SELECTION-3D.json and, once available, analyze3d's fixed-sweep selection): deployed rules and their
fallbacks, and the fixed-sweep fast-equal-accuracy / accurate-equal-cost arms per setting with the 3D budget arms (A1.5).

    python freeze.py --analysis2d checks/analysis-c2grid-2d_first_order.json [--selection3d checks/SELECTION-3D.json]
                     [--analysis3d checks/analysis3d-c3grid.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DT0 = .005
FAM2 = dict(gauss=['gauss96', 'gauss128', 'gauss192', 'gauss256', 'gauss320'],
            fib=['fib1597', 'fib4181', 'fib6765', 'fib17711', 'fib46368'])


def fallbacks2(rule):
    for lad in FAM2.values():
        if rule in lad:
            i = lad.index(rule)
            return lad[i + 1:i + 3] + ['unavailable']
    return ['unavailable']


def arm_of(key):
    form, sc, dt, lev = key.split('|')
    return dict(form=form, scheme=sc, f=round(float(dt) / DT0, 6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--analysis2d')
    ap.add_argument('--selection3d')
    ap.add_argument('--analysis3d')
    ap.add_argument('--out', default=str(HERE / 'checks' / 'FROZEN-SELECTION.json'))
    a = ap.parse_args()
    dest = Path(a.out)
    fz = json.loads(dest.read_text()) if dest.exists() else {}
    if a.analysis2d:
        an = json.loads(Path(a.analysis2d).read_text())
        assert an['refs'] == '2d_first_order' and 'frozen' not in an['selection'], 'freeze only a fresh selection'
        g = an['selection']['global']['scheme']
        res = json.loads((Path(an['run']) / 'result.json').read_text())
        rules = {s['name']: s['rule'] for s in res['config']['settings']}
        budget = {}
        for s, rule in rules.items():
            arms = [dict(form='LSPG', scheme='BE', f=1, role='baseline'), dict(form='GAL', scheme='CN', f=1, role='diagnostic')]
            if g:
                for kind in ('accurate_equal_cost', 'fast_equal_accuracy'):
                    x = an['selection']['per_scheme_setting'][f'{s}|{g}'].get(kind)
                    if x:
                        form, sc, _, _ = x['key'].split('|')
                        arms.append(dict(form=form, scheme=sc, f=x['f'], role=kind, selected=True,   # exact factor (5/3 safe)
                                         fallback_rules=fallbacks2(rule)))
            uniq = []
            for x in arms:
                k = (x['form'], x['scheme'], x['f'])
                if k not in [(y['form'], y['scheme'], y['f']) for y in uniq]:
                    uniq.append(x)
                else:
                    y = next(y for y in uniq if (y['form'], y['scheme'], y['f']) == k)
                    y['role'] = f"{y['role']}+{x['role']}"
                    y['selected'] = y.get('selected') or x.get('selected')
                    y.setdefault('fallback_rules', x.get('fallback_rules'))
            budget[s] = dict(rule=rule, arms=uniq)
        fz['2d'] = dict(source_analysis=a.analysis2d,
                        source_sha256=hashlib.sha256(Path(a.analysis2d).read_bytes()).hexdigest(),
                        per_scheme_setting=an['selection']['per_scheme_setting'],
                        **{'global': an['selection']['global']}, tie=an['tie'], budget=budget)
    if a.selection3d:
        s3 = json.loads(Path(a.selection3d).read_text())
        ph = bool(s3.get('posthoc'))                  # A10: the effective converged rule of a post-hoc selection is its own
        fz['3d_rules'] = dict(source=a.selection3d, source_sha256=hashlib.sha256(Path(a.selection3d).read_bytes()).hexdigest(),
                              posthoc=ph, label=s3.get('label', 'pre-registered'), selection_arms=s3.get('selection_arms'),
                              settings={k: dict(status=v['status'],
                                                converged=v.get('posthoc_converged') if ph else v.get('converged'),
                                                pre_registered_converged=v.get('converged'), deployed=v.get('deployed'),
                                                fallbacks=v.get('fallbacks'), posthoc=ph,
                                                label=s3.get('label', 'pre-registered')) for k, v in s3['settings'].items()})
    if a.analysis3d:
        a3 = json.loads(Path(a.analysis3d).read_text())
        fz['3d_solver'] = dict(source=a.analysis3d, source_sha256=hashlib.sha256(Path(a.analysis3d).read_bytes()).hexdigest(),
                               selection=a3['selection'], budget=a3['budget_arms'])
    dest.write_text(json.dumps(fz, indent=1, default=float) + '\n')
    print(dest, hashlib.sha256(dest.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
