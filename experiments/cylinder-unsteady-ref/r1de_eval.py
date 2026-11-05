"""R1d DFG comparison and R1e population checks from collected artifacts (DESIGN-G5 §4.4, G5-A5).

Usage: r1de_eval.py --records <json: {Re: attempt}> --settles <json: {Re: [attempt/run, ...]}>
       --gates artifacts/r1c-space02/results/compare/R1BC-GATES.json
       --space artifacts/r1c-space02/results/compare/space_3200.json
       --time  artifacts/r1b-compare01/results/compare/compare_3200_6400.json
       --dfg <coordinator sources/dfg2d2_reference.json> --out R1DE.json
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

import cycles as cyc

HERE = Path(__file__).resolve().parent
DIV_MAX, MASS_MAX, LIN_MAX = 1e-3, 1e-7, 1e-10
DFG_TOL = {'CD_max': ('rel', 2e-3), 'CL_max': ('abs', 2e-3), 'St': ('rel', 5e-3), 'dp_at_t0_plus_half_period': ('rel', 5e-3)}
RANGES_1996 = {'CD_max': 'CD_max', 'St': 'St', 'dp_1996_t0_at_CLmax_plus_half_period': 'dp_at_t0_plus_half_period'}


def run_dir(attempt, name):
    return HERE / 'artifacts' / attempt / 'results' / name


def record_checks(d):
    st = json.loads((d / 'stages.json').read_text())
    rec = [s for s in st['stages'] if s['name'] == 'record']
    out = {'status': st['status'], 'stages': [s['name'] for s in st['stages']]}
    ok = st['status'] == 'completed' and len(rec) == 1 and rec[0]['status'] == 'completed'
    h = np.load(d / 'history.npz')['rows']
    diag = json.loads((d / 'diagnostics.json').read_text())
    lin = np.load(d / 'linear_checks.npz')['rel']
    out['divergence_max'] = max(z['divergence'] for z in diag)
    out['mass_max'] = max(z['mass'] for z in diag)
    out['linear_residual_max'] = float(lin.max()) if lin.size else None
    out['diagnostic_times'] = len(diag)
    out['snapshots'] = sum(s['count'] for s in st['snapshots'])
    out['expected_record_snapshots'] = int(round(1.7 * 400)) + 1
    out['passes'] = {'completed': ok, 'divergence': out['divergence_max'] <= DIV_MAX, 'mass': out['mass_max'] <= MASS_MAX,
                     'linear': out['linear_residual_max'] is not None and out['linear_residual_max'] <= LIN_MAX,
                     'record_snapshots': out['snapshots'] >= out['expected_record_snapshots']}
    if rec:
        r = rec[0]
        rows = h[(h[:, 8] == st['stages'].index(r)) & np.isfinite(h[:, 1])]
        recs = cyc.cycles(rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 5] - rows[:, 6])
        out['cycles'] = cyc.summarize(recs) if recs else None
        out['per_cycle'] = recs
    return out


def main():
    ap = argparse.ArgumentParser()
    for k in ('--records', '--settles', '--gates', '--space', '--time', '--dfg', '--out'):
        ap.add_argument(k, required=True)
    a = ap.parse_args()
    records = json.loads(a.records); settles = json.loads(a.settles)
    res = {'R1e': {}, 'R1d': None, 'uncertainty': None}
    for Re, attempt in records.items():
        d = run_dir(attempt, f'record_re{float(Re):g}_p4')
        rc = record_checks(d)
        dev = []
        for path in settles[Re]:
            att, name = path.split('/')
            s = json.loads((run_dir(att, name) / 'stages.json').read_text())
            dev.append({'run': path, 'status': s['status'], 'periodic_reached': s['stages'][0].get('periodic_reached'),
                        't_end': s['stages'][0].get('t_end')})
        rc['settle_chain'] = dev
        rc['passes']['developed_before_record'] = bool(dev) and dev[-1]['status'] == 'completed' and dev[-1]['periodic_reached'] is True
        rc['R1e_pass'] = all(rc['passes'].values())
        res['R1e'][Re] = rc
    res['R1e_gate'] = 'pass' if all(v['R1e_pass'] for v in res['R1e'].values()) else 'fail'
    # R1d on Re 100
    dfg = json.loads(Path(a.dfg).read_text())
    c100 = res['R1e']['100']['cycles']
    rows = {}
    for k, (kind, tol) in DFG_TOL.items():
        ours, ref = c100[k]['mean'], dfg['finest'][k]
        err = abs(ours - ref) / abs(ref) if kind == 'rel' else abs(ours - ref)
        rows[k] = {'ours': ours, 'featflow_finest': ref, 'error': err, 'kind': kind, 'tolerance': tol, 'pass': err <= tol}
    ranges = dfg['schaefer_turek_1996_ranges']
    for ours_key, range_key in RANGES_1996.items():
        lo, hi = ranges[range_key]; v = c100[ours_key]['mean']
        rows[f'1996_range_{ours_key}'] = {'ours': v, 'range': [lo, hi], 'pass': v is not None and lo <= v <= hi}
    v = c100['CL_max']['mean']
    rows['1996_range_CL_max_non_gating'] = {'ours': v, 'range': ranges['CL_max'], 'inside': ranges['CL_max'][0] <= v <= ranges['CL_max'][1]}
    res['R1d'] = rows
    res['R1d_gate'] = 'pass' if all(r['pass'] for k, r in rows.items() if 'pass' in r) else 'fail'
    # Statistic-specific reference uncertainty (heuristic estimates, G5-A5 rev.2)
    sp = json.loads(Path(a.space).read_text()); tm = json.loads(Path(a.time).read_text())
    gap = lambda k: max(sp[k], tm[k])
    f = 1.0 / c100['period']['mean']; amp = 0.5 * (c100['CL_max']['mean'] - c100['CL_min']['mean'])
    prop = 1.7 / 0.35
    res['uncertainty'] = {
        'kind': 'heuristic estimate (linear propagation), not a bound',
        'field_and_force_over_horizon': {k: 2 * gap(k) * prop for k in ('e_u_max', 'e_p_max', 'e_D', 'e_L')},
        'cycle_statistics': {'CD_max_abs': 2 * gap('e_D') * sp['CD_maxabs_b'], 'CL_max_abs': 2 * gap('e_L'),
                             'dp_abs': 2 * max(sp['e_dp_abs'], tm['e_dp_abs']),
                             'St_rel': 2 * gap('e_L') / (2 * math.pi * f * amp * 0.35)}}
    gates = json.loads(Path(a.gates).read_text())
    res['R1b'] = gates['R1b_time_gate']; res['R1c'] = gates['R1c_space_gate']; res['dt0_denominator'] = gates['dt0_denominator']
    Path(a.out).write_text(json.dumps(res, indent=2) + '\n')
    print(json.dumps({k: res[k] for k in ('R1b', 'R1c', 'R1d_gate', 'R1e_gate', 'dt0_denominator')}))
    for k, r in rows.items():
        print(k, {kk: r[kk] for kk in r if kk in ('ours', 'featflow_finest', 'error', 'pass', 'inside', 'range')})


if __name__ == '__main__':
    main()
