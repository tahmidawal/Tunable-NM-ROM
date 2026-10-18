"""Offline picks and pair labels for one cell (DESIGN 6, 9, A1.6, A1.10-A1.12, A2.12, A2.17, A4.8, A5.3, A22.0,
A26.5), from the Stage-B job's data only. CPU, numpy. Writes checks/VERDICT-<cell>.json (+ .sha256).

Per admitted X (G-ours passed in the tuning and Stage-B jobs):
  - candidate set = Stage-B members Q1-qualified on the verdict cohort whose A-B-A timing against X is complete with
    every timed output equal to its scored output (<= 1e-12);
  - errors on the eligible case-kinds of the verdict cohort (ineligible ones shown, never ranked); W = worst, med =
    median; time = T (median over the 6 timing cases of the per-case median valid ratio t_B / mean(t_A1, t_A2));
  - roles B_acc / B_cost from the X-specific front in (T, W) (stats.picks); pair labels by stats.pair_label for the
    picks (registered) and for every other candidate whose worst envelope is not entirely above X's (exploratory);
  - cell flags carried into every line: adequacy / saturation, ECSW quality, budget relation (A2.5 / A25.2), 3D
    batch-1 diagnostic-only (A22.0).

Usage: python analyze.py <cell> <tune_output_dir> <stageb_output_dir>
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stats as ST  # noqa: E402

TIMED_TOL = 1e-12


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rho_of(cases_rec):
    """Valid ratios per timing case: valid triple (drift) with every output discrepancy finite and <= 1e-12."""
    out = {}
    for t, rc in cases_rec.items():
        vals = []
        for tr in rc['triples']:
            ok = tr['valid'] and all(np.isfinite(x) and x <= TIMED_TOL for x in (tr['dA1'], tr['dB'], tr['dA2']))
            if ok:
                vals.append(tr['tB'] / (0.5 * (tr['tA1'] + tr['tA2'])))
        out[int(t.split('|')[1])] = vals
    return out


def main():
    cell, tdir, bdir = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    T = json.loads((tdir / 'tune.json').read_text())
    B = json.loads((bdir / 'stageb.json').read_text())
    fz = HERE / B['config']['frozen']
    assert sha(fz) == B['config']['frozen_sha256']
    FZ = json.loads(fz.read_text())
    d = T['config']['d']
    if d == 2:
        exp = json.loads((HERE / 'inputs' / 'p1_analysis.json').read_text())['configurations']
        eX_all = {k: exp[k]['e0_per_case'] for k in B['ours']}
    else:
        exp = json.loads((HERE / 'inputs' / 'speed3d_batch1_e0.json').read_text())['configs']
        eX_all = {k: {f'val64|{j}': v for j, v in exp[k]['e0'].items()} for k in B['ours']}
    members = B['members']
    any_m = next(iter(members.values()), None)
    cases = any_m['cases'] if any_m else []
    elig = [c for c in cases if all(members[m]['e0'][c].get('eligible') for m in members)]
    eps = np.array([any_m['e0'][c]['eps_ref'] for c in elig]) if any_m else np.zeros(0)
    flags = dict(adequate=(T.get('adequacy') or {}).get('adequate'), saturation=(T.get('saturation') or {}).get('adequate'),
                 basis_used=T.get('basis_used'), diagnostic_only=bool(d == 3),
                 search_hours=T.get('search_hours'), allocation_hours=T['config'].get('search_allocation_hours'),
                 inventory=T.get('inventory_counts'), deferred=T.get('deferred'))
    if flags['search_hours'] is not None and flags['allocation_hours'] is not None:
        flags['budget'] = ('baseline search within ours\' allocation' if flags['search_hours'] <= flags['allocation_hours']
                           else 'baseline search exceeded ours\' allocation')
    out = dict(cell=cell, flags=flags, eligible_cases=elig, X={}, stageb_sha256=sha(bdir / 'stageb.json'),
               tune_sha256=sha(tdir / 'tune.json'), frozen_sha256=B['config']['frozen_sha256'])
    for key, xr in B['ours'].items():
        if not (xr.get('G_ours') and FZ['X'].get(key, {}).get('front') is not None):
            out['X'][key] = dict(label='not assessed', reason='G-ours failed or X not admitted')
            continue
        eX = np.array([eX_all[key][c] for c in elig])
        envX = ST.envelope(eX, eps)
        cand = []
        for m, mv in members.items():
            tm = B['timing'].get(f'{key}||{m}')
            if not (mv['q1']['q1'] and tm and tm.get('status') == 'complete'):
                continue
            rho = rho_of(tm['cases'])
            if sorted(rho) != list(ST.TIMING_CASES) or not all(rho.values()):
                continue
            eB = np.array([mv['e0'][c]['e'] for c in elig])
            ts = ST.time_stats(rho)
            kind, r, hp, scheme, dt, solver, gen = m.split('|')
            cand.append(dict(id=m, ratio=ts['median'], W=float(np.max(eB)), med=float(np.median(eB)), r=int(r),
                             kind=kind, hr_points=int(FZ['member_info'][m]['hr_points']), scheme=scheme, dt=float(dt),
                             solver=solver, eB=eB, rho=rho, quality=FZ['member_info'][m].get('quality')))
        if not cand:
            out['X'][key] = dict(label='baseline incomplete at this mesh', W_X=float(np.max(eX)) if len(eX) else None)
            continue
        b_acc, b_cost, F = ST.picks([{k: v for k, v in c.items() if k not in ('eB', 'rho')} for c in cand],
                                    float(np.max(eX)))
        byid = {c['id']: c for c in cand}
        rec = dict(W_X=float(np.max(eX)), med_X=float(np.median(eX)), front=[f['id'] for f in F], roles={},
                   exploratory={})
        for role, p in (('B_acc', b_acc), ('B_cost', b_cost)):
            if p is None:
                rec['roles'][role] = dict(label='no baseline in role')
                continue
            c = byid[p['id']]
            lab = ST.pair_label(eX, c['eB'], eps, c['rho'])
            rec['roles'][role] = dict(id=c['id'], quality=c['quality'], T=c['ratio'], W=c['W'], **{
                k: v for k, v in lab.items() if k in ('label', 'time', 'envX', 'envB', 'qualifier', 'reason')})
        for c in cand:
            envB = ST.envelope(c['eB'], eps)
            if envB['worst'][0] > envX['worst'][1]:
                continue                                  # worst envelope entirely above X's: not labelled
            lab = ST.pair_label(eX, c['eB'], eps, c['rho'])
            rec['exploratory'][c['id']] = dict(label=lab['label'] + ' (exploratory, unadjusted)', T=c['ratio'], W=c['W'],
                                               quality=c['quality'])
        dom_reg = [r_ for r_, v in rec['roles'].items() if v.get('label', '').startswith('baseline dominates')]
        dom_exp = [i for i, v in rec['exploratory'].items() if v['label'].startswith('baseline dominates')]
        rec['cell_line'] = ('observed domination of X by a registered pick' if dom_reg else
                            'observed domination of X by a non-registered Stage-B configuration' if dom_exp else
                            f'no observed domination among the {len(cand)} configurations compared')
        if not flags.get('adequate') and not dom_reg and not dom_exp:
            rec['cell_line'] += ' (withheld: basis possibly under-sampled, A2.2)'
        out['X'][key] = rec
    p = HERE / 'checks' / f'VERDICT-{cell}.json'
    p.write_text(json.dumps(out, indent=1, default=float) + '\n')
    (HERE / 'checks' / f'VERDICT-{cell}.sha256').write_text(f'{sha(p)}  {p.name}\n')
    for k, v in out['X'].items():
        print(k, v.get('cell_line', v.get('label')), {r_: x.get('label') for r_, x in v.get('roles', {}).items()})


if __name__ == '__main__':
    main()
