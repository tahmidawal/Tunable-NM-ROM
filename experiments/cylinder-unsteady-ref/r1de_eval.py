"""R1d DFG comparison and R1e population checks (DESIGN-G5 §4.4, G5-A5 rev.3).

Validates the exact reduced population (G5-A3) and keeps the held-out case sealed: for
held-out Re only reference-QUALITY checks are computed/exported (completion, linear residuals,
divergence, mass, hashes, coverage, cadence, settle provenance) — no cycle statistics or histories.
Usage: r1de_eval.py --population <json> --out R1DE.json
population = {"records": {Re: attempt}, "settles": {Re: [attempt/run, ...] (in order)},
              "gates": path, "space": path, "time": path, "dfg": path}
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

import cycles as cyc

HERE = Path(__file__).resolve().parent
POP = {'90': 'train', '95': 'dev', '100': 'train', '105': 'heldout', '110': 'train'}
DT0, K0 = 1 / 3200, 4
DIV_MAX, MASS_MAX, LIN_MAX, SETTLE_CAP = 1e-3, 1e-7, 1e-10, 6.0
DFG_TOL = {'CD_max': ('rel', 2e-3), 'CL_max': ('abs', 2e-3), 'St': ('rel', 5e-3), 'dp_at_t0_plus_half_period': ('rel', 5e-3)}
RANGES_1996 = {'CD_max': 'CD_max', 'St': 'St', 'dp_1996_t0_at_CLmax_plus_half_period': 'dp_at_t0_plus_half_period'}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run_paths(attempt, name):
    return HERE / 'artifacts' / attempt / 'results' / name, HERE / 'retention' / attempt / 'results' / name


def check_record(Re, attempt, settle_chain):
    name = f'record_re{float(Re):g}_p4'
    d, ret = run_paths(attempt, name)
    cfg = json.loads((d / 'config.json').read_text()); st = json.loads((d / 'stages.json').read_text())
    p = {}
    p['config'] = (cfg['velocity_order'] == K0 and cfg['pressure_order'] == K0 - 1 and cfg['geometry_order'] == 4
                   and cfg['mesh_sha256'] == '528ffff38bb0f4d6d542f1abd4cba7c9ca2ba0427871caca806527c63ae6b092'
                   and cfg['scheme'] == 'frozen_oseen' and cfg['refactor_every'] == 40 and cfg['startup_substeps'] == 4
                   and cfg['assembly_bonus_order'] == 6 and cfg['threads'] == 16 and cfg['linear_relative_tolerance'] == 1e-10
                   and abs(cfg['nu'] - 0.1 / float(Re)) < 1e-15 and cfg['kick'] is None
                   and [s['name'] for s in cfg['stages']] == ['settle_dt0', 'record']
                   and all(abs(s['dt'] - DT0) < 1e-15 for s in cfg['stages'])
                   and abs(cfg['stages'][0]['duration'] - 0.7) < 1e-12 and abs(cfg['stages'][1]['duration'] - 1.7) < 1e-12)
    p['completed'] = st['status'] == 'completed' and all(s['status'] == 'completed' for s in st['stages'])
    # settle provenance: the record starts from the last settle's saved state, cumulative settle <= 6 s
    last = settle_chain[-1]; satt, sname = last.split('/')
    sd, sret = run_paths(satt, sname)
    sst = json.loads((sd / 'stages.json').read_text())
    state = sret / 'settle.npz'
    p['settle_state_linked'] = (cfg['initial'].get('sha256') == sha(state) == sst['states'][-1]['sha256'])
    total = 0.0; prev_state = None; links_ok = True
    for link in settle_chain:
        la, ln = link.split('/'); ld, lret = run_paths(la, ln)
        ls = json.loads((ld / 'stages.json').read_text()); lc = json.loads((ld / 'config.json').read_text())
        total += ls['stages'][0]['t_end'] - ls['stages'][0]['t_start']
        if prev_state is not None and lc['initial'].get('sha256') != prev_state:
            links_ok = False
        prev_state = sha(lret / 'settle.npz')
        if prev_state != ls['states'][-1]['sha256']:
            links_ok = False
    p['settle_chain_linked'] = links_ok
    p['settle_developed'] = sst['status'] == 'completed' and sst['stages'][0]['periodic_reached'] is True
    p['settle_within_cap'] = total <= SETTLE_CAP + 1e-9
    # snapshots: retained bytes match, exact 681 saves on the 1/400 grid in the record window
    rec = st['stages'][1]
    ts = []
    ok_hash = True
    for sn in st['snapshots']:
        if sha(ret / sn['file']) != sn['sha256']:
            ok_hash = False
        ts.append(np.load(ret / sn['file'])['t'])
    ts = np.concatenate(ts); tr = ts[(ts >= rec['t_start'] - 1e-9) & (ts <= rec['t_end'] + 1e-9)]
    grid = rec['t_start'] + np.arange(681) / 400
    p['snapshot_hashes'] = ok_hash
    p['record_coverage'] = len(tr) == 681 and np.max(np.abs(tr - grid)) < 1e-9
    diag = json.loads((d / 'diagnostics.json').read_text())
    drec = [z for z in diag if z.get('stage') == 1]
    dts = np.array([z['t'] for z in drec]); dgrid = rec['t_start'] + np.arange(171) * 4 / 400
    p['diagnostic_cadence'] = len(drec) == 171 and np.max(np.abs(dts - dgrid)) < 1e-9
    p['divergence'] = max(z['divergence'] for z in diag) <= DIV_MAX
    p['mass'] = max(z['mass'] for z in diag) <= MASS_MAX
    lin = np.load(d / 'linear_checks.npz')['rel']
    p['linear'] = lin.size > 0 and float(lin.max()) <= LIN_MAX
    p = {k: bool(v) for k, v in p.items()}
    out = {'split': POP[Re], 'passes': p, 'R1e_pass': all(p.values()), 'settle_seconds_total': total,
           'divergence_max': max(z['divergence'] for z in diag), 'mass_max': max(z['mass'] for z in diag),
           'linear_residual_max': float(lin.max()) if lin.size else None}
    if POP[Re] != 'heldout':
        h = np.load(d / 'history.npz')['rows']
        rows = h[(h[:, 8] == 1) & np.isfinite(h[:, 1])]
        recs = cyc.cycles(rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 5] - rows[:, 6])
        out['cycles'] = cyc.summarize(recs) if recs else None
        out['per_cycle'] = recs
    else:
        out['sealed'] = 'held-out: quality checks only; no cycle statistics or histories exported before R5'
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--population', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    pop = json.loads(Path(a.population).read_text())
    if set(pop['records']) != set(POP) or set(pop['settles']) != set(POP):
        raise ValueError(f'population must be exactly {sorted(POP)}')
    res = {'R1e': {Re: check_record(Re, pop['records'][Re], pop['settles'][Re]) for Re in sorted(POP, key=float)}}
    res['R1e_gate'] = 'pass' if all(v['R1e_pass'] for v in res['R1e'].values()) else 'fail'
    gates = json.loads(Path(pop['gates']).read_text())
    sp = json.loads(Path(pop['space']).read_text()); tm = json.loads(Path(pop['time']).read_text())
    res['R1b'], res['R1c'], res['dt0_denominator'] = gates['R1b_time_gate'], gates['R1c_space_gate'], gates['dt0_denominator']
    keys = ('e_u_max', 'e_p_max', 'e_D', 'e_L')
    same_as_gates = (gates.get('space', {}).get('metrics') == {k: sp[k] for k in keys}
                     and gates['candidates']['3200']['pair']['metrics'] == {k: tm[k] for k in keys})
    bound = (same_as_gates and sp.get('valid') and tm.get('valid') and sp['initial_state_sha256'] == tm['initial_state_sha256']
             and abs(sp['window'][0] - tm['window'][0]) < 1e-9 and res['R1b'] == 'pass' and res['R1c'] == 'pass'
             and res['dt0_denominator'] == 3200)
    res['uncertainty_inputs_bound'] = bool(bound)
    dfg = json.loads(Path(pop['dfg']).read_text())
    c100 = res['R1e']['100']['cycles']
    rows = {}
    for k, (kind, tol) in DFG_TOL.items():
        ours, ref = c100[k]['mean'], dfg['finest'][k]
        err = abs(ours - ref) / abs(ref) if kind == 'rel' else abs(ours - ref)
        rows[k] = {'ours': ours, 'featflow_finest': ref, 'error': float(err), 'kind': kind, 'tolerance': tol, 'pass': bool(err <= tol)}
    ranges = dfg['schaefer_turek_1996_ranges']
    for ours_key, range_key in RANGES_1996.items():
        lo, hi = ranges[range_key]; v = c100[ours_key]['mean']
        rows[f'1996_range_{ours_key}'] = {'ours': v, 'range': [lo, hi], 'pass': bool(v is not None and lo <= v <= hi)}
    v = c100['CL_max']['mean']
    rows['1996_range_CL_max_non_gating'] = {'ours': v, 'range': ranges['CL_max'], 'inside': bool(ranges['CL_max'][0] <= v <= ranges['CL_max'][1])}
    res['R1d'] = rows
    res['R1d_gate'] = 'pass' if all(r['pass'] for r in rows.values() if 'pass' in r) else 'fail'
    gap = lambda k: max(sp[k], tm[k])
    f = 1.0 / c100['period']['mean']; amp = 0.5 * (c100['CL_max']['mean'] - c100['CL_min']['mean'])
    prop = 1.7 / 0.35
    res['uncertainty'] = {
        'kind': 'heuristic estimate (linear propagation), not a bound',
        'field_and_force_over_horizon': {k: 2 * gap(k) * prop for k in ('e_u_max', 'e_p_max', 'e_D', 'e_L')},
        'cycle_statistics': {'CD_max_abs': 2 * gap('e_D') * sp['CD_maxabs_b'], 'CL_max_abs': 2 * gap('e_L'),
                             'dp_abs': 2 * max(sp['e_dp_abs'], tm['e_dp_abs']),
                             'St_rel': 2 * gap('e_L') / (2 * math.pi * f * amp * 0.35)}}
    Path(a.out).write_text(json.dumps(res, indent=2) + '\n')
    print(json.dumps({k: res[k] for k in ('R1b', 'R1c', 'R1d_gate', 'R1e_gate', 'dt0_denominator', 'uncertainty_inputs_bound')}))
    for k, r in rows.items():
        print(k, {kk: r[kk] for kk in r if kk in ('ours', 'featflow_finest', 'error', 'pass', 'inside', 'range')})


if __name__ == '__main__':
    main()
