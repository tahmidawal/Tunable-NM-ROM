"""Tables and the PROTOCOL.md selection rule from the AMG-<case>.json files (no hand-typed numbers).

Usage: summarize.py --dev DIR [DIR ...] [--heldout DIR] --out SUMMARY.json  (also prints markdown tables)
"""
import argparse
import json
from pathlib import Path

import numpy as np

KEYS = ['e_u_rms', 'e_u_max', 'e_p_rms', 'St_rel', 'CD_max_rel', 'CL_max_abs']
RECORD_LU_SECONDS = 42.248898304998875  # SELECT-re105.json of rb-fb, c4_p3_dt300_frozen_oseen_K30


def load(dirs, case):
    out = {}
    for d in dirs:
        j = json.loads((Path(d) / f'AMG-{case}.json').read_text())
        for name, rec in j['entries'].items():
            out.setdefault(rec['solver']['name'], []).append(rec)
    return out


def row(rec):
    t = rec.get('timing', [])
    st = [r['solver_stats'] for r in t]
    tim = [r['timers'] for r in t]
    def med(f):
        return float(np.median([f(i) for i in range(len(t))])) if t else None
    return {
        'status': rec['status'], 'error': rec.get('error'),
        'targets_met': rec.get('targets_met'),
        'metrics': {k: (rec['targets'][k]['value'] if rec.get('targets') else None) for k in KEYS},
        'reps': len(t),
        'wall_median': rec.get('warm_seconds_median'), 'wall_min': rec.get('warm_seconds_min'),
        'wall_max': rec.get('warm_seconds_max'),
        'wall_all': [r['warm_seconds'] for r in t],
        'factor_or_setup_median': med(lambda i: tim[i]['factor']),
        'solve_median': med(lambda i: tim[i]['solve']),
        'conv_apply_median': med(lambda i: tim[i]['conv_apply']),
        'observe_median': med(lambda i: tim[i]['observe']),
        'outside_linear_solver_median': med(lambda i: t[i]['warm_seconds'] - tim[i]['factor'] - tim[i]['solve']),
        'amg_setup_median': med(lambda i: st[i]['amg_setup_seconds'] + st[i]['extract_seconds']),
        'krylov_median': med(lambda i: st[i]['krylov_seconds']),
        'outer_iters_mean': st[0].get('outer_iters_mean') if st else None,
        'outer_iters_max': st[0].get('outer_iters_max') if st else None,
        'inner_iters_per_solve': (st[0]['inner_iters'] / st[0]['solves']) if st and st[0].get('solves') else None,
        'refreshes': t[0]['lu_refreshes'] if t else None,
        'final_rel_max': st[0].get('final_rel_max') if st else None,
    }


def fmt(x, p=3):
    return '-' if x is None else (f'{x:.{p}g}' if isinstance(x, float) else str(x))


def table(rows, title):
    print(f'\n### {title}\n')
    print('| solver | status | targets | e_u RMS | e_u max | e_p RMS | St rel | CD_max rel | CL_max abs | wall s (median) | min-max | '
          'setup s | solve s | Krylov it/solve (max) | inner it/solve |')
    print('|' + '---|' * 15)
    for name, r in rows.items():
        m = r['metrics']
        print(f"| {name} | {r['status']} | {fmt(r['targets_met'])}/6 | " + ' | '.join(fmt(m[k]) for k in KEYS) +
              f" | {fmt(r['wall_median'], 4)} | {fmt(r['wall_min'], 4)}-{fmt(r['wall_max'], 4)} | {fmt(r['factor_or_setup_median'])} | "
              f"{fmt(r['solve_median'])} | {fmt(r['outer_iters_mean'])} ({fmt(r['outer_iters_max'])}) | {fmt(r['inner_iters_per_solve'])} |")


def select(rows, lu):
    """PROTOCOL rule: loosest rtol with 6/6 and every metric within 1 % (relative) of lu."""
    sel = {}
    for variant in ('amg1_sa_cc', 'amg2_sa_cc'):
        ok = []
        for name, r in rows.items():
            if not name.startswith(variant + '_r') or r['status'] != 'completed':
                continue
            rtol = float(name.split('_r', 1)[1].replace('e-', 'e-'))
            within = all(r['metrics'][k] is not None and lu['metrics'][k] is not None and
                         abs(r['metrics'][k] - lu['metrics'][k]) <= 0.01 * abs(lu['metrics'][k]) for k in KEYS)
            if r['targets_met'] == 6 and within:
                ok.append((rtol, name))
        sel[variant] = max(ok)[1] if ok else None
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dev', nargs='+', required=True)
    ap.add_argument('--heldout', nargs='*', default=[])
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    dev = load(a.dev, 're95')
    rows = {}
    for name, recs in dev.items():
        for i, rec in enumerate(recs):
            rows[name if i == 0 else f'{name} ({i + 1})'] = row(rec)
    table(rows, 'Development Re 95 (one run each)')
    lus = [r for n, r in rows.items() if n.startswith('lu')]
    lu_consistent = all(r['metrics'][k] is None or abs(r['metrics'][k] - lus[0]['metrics'][k]) <= 1e-9 * abs(lus[0]['metrics'][k])
                        for r in lus for k in KEYS)  # threaded assembly: agreement to rounding, not bitwise
    sel = select(rows, lus[0])
    print('\nLU runs identical across jobs:', lu_consistent, '\nselected:', sel)
    out = {'dev': rows, 'lu_identical_across_dev_jobs': lu_consistent, 'selected': sel,
           'record_lu_seconds_re105': RECORD_LU_SECONDS}
    if a.heldout:
        ho = load(a.heldout, 're105')
        hrows = {n: row(recs[0]) for n, recs in ho.items()}
        table(hrows, 'Held-out Re 105 (5 timed repetitions each, one allocation)')
        lu_med = [r['wall_median'] for n, r in hrows.items() if n.startswith('lu') and r['wall_median']]
        ratio = {n: r['wall_median'] / min(lu_med) for n, r in hrows.items() if r['wall_median'] and lu_med}
        print('\nwall / paired LU (faster LU block):', {n: round(v, 3) for n, v in ratio.items()})
        print('wall minus (factor/setup + solve) timers, i.e. the part no linear solver can remove:',
              {n: round(r['outside_linear_solver_median'], 2) for n, r in hrows.items() if r['wall_median']})
        print('wall / record 42.25 s:', {n: round(r['wall_median'] / RECORD_LU_SECONDS, 3) for n, r in hrows.items() if r['wall_median']})
        out.update(heldout=hrows, ratio_to_paired_lu=ratio)
    Path(a.out).write_text(json.dumps(out, indent=2, default=float) + '\n')


if __name__ == '__main__':
    main()
