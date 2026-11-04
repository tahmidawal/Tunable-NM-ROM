"""Evaluate the R1b time and R1c space gates from compare.py outputs (DESIGN-G5 §4.4, G5-A2).

Usage: r1_gates.py --dir <collected compare dir> --out R1BC-GATES.json
Expects compare_<a>_<b>.json for (800,1600), (1600,3200), (3200,6400) and the startup
checks startup_<n>.json; space_<n>.json once R1c has run.
"""
import argparse
import json
import math
from pathlib import Path

TOL = {'e_u_max': 2e-3, 'e_p_max': 4e-3, 'e_D': 3e-3, 'e_L': 2e-4}
LADDER = [800, 1600, 3200, 6400]
FLOOR = 1e-12  # solver floor: differences below 100x this are not used for an order estimate


def passes(rec, scale=1.0):
    return {k: rec[k] <= scale * v for k, v in TOL.items()}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--dir', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args(); d = Path(a.dir)
    pairs = {}
    for c, f in zip(LADDER[:-1], LADDER[1:]):
        p = d / f'compare_{c}_{f}.json'
        if p.exists():
            pairs[c] = json.loads(p.read_text())
    res = {'tolerances': TOL, 'pairs': {}, 'orders': {}, 'startup': {}, 'space': {}}
    for c, rec in pairs.items():
        res['pairs'][str(c)] = {'metrics': {k: rec[k] for k in TOL}, 'passes': passes(rec)}
    for c in (800, 1600):
        if c in pairs and 2 * c in pairs:
            res['orders'][str(c)] = {k: (math.log2(pairs[c][k] / pairs[2 * c][k])
                                         if min(pairs[c][k], pairs[2 * c][k]) > 100 * FLOOR else None) for k in TOL}
    dt0 = None
    for c in (800, 1600, 3200):
        if str(c) in res['pairs'] and all(res['pairs'][str(c)]['passes'].values()):
            dt0 = c
            break
    res['dt0_denominator'] = dt0
    res['R1b_time_gate'] = 'pass' if dt0 else 'fail'
    if dt0 is not None:
        order_ok = True
        for c in (800, 1600):
            o = res['orders'].get(str(c))
            if o and c <= dt0:
                for k, v in o.items():
                    if v is not None and not 1.5 <= v <= 2.5:
                        order_ok = False
        res['empirical_order_check'] = 'pass' if order_ok else 'fail'
        if not order_ok:
            res['R1b_time_gate'] = 'fail_empirical_order'
    for n in (1600, 3200):
        p = d / f'startup_{n}.json'
        if p.exists():
            rec = json.loads(p.read_text())
            res['startup'][str(n)] = {'metrics': {k: rec[k] for k in TOL}, 'passes': passes(rec, 0.2)}
    for n in (800, 1600, 3200):
        p = d / f'space_{n}.json'
        if p.exists():
            rec = json.loads(p.read_text())
            res['space'][str(n)] = {'metrics': {k: rec[k] for k in TOL}, 'passes': passes(rec)}
    Path(a.out).write_text(json.dumps(res, indent=2) + '\n')
    print(json.dumps({k: res[k] for k in ('dt0_denominator', 'R1b_time_gate')} | {'pairs': {c: v['passes'] for c, v in res['pairs'].items()}}, indent=1))


if __name__ == '__main__':
    main()
