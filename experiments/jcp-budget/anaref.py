"""Offline analysis of this lane's references (DESIGN 3, A1.5, A2.3, A3.3; CPU, NumPy only). Recomputes from the pulled
lattice arrays (never trusts a job's verdict fields alone):

  * G-R1  e0repro level fields vs the E0 store's level fields (same L, dt, prod purpose): per time ||d|| / n0; bar 1e-12
          on E0's GPU type, else the G-R2 bar
  * G-R2  the horizon run's FINAL triple (each final level matched on (L, dt, purpose) with an E0 attempt), first six
          outputs: t0 bitwise, evolved times <= min(1e-8, 0.01 eps_ref(t_k)); 'not evaluable' unless the finest final
          level is matched; a T = 1.0 reference must also reproduce the T = 0.5 one on its first eleven outputs
  * judge recomputation on the final triple (whole run and per window) vs the stored records
  * G-R3  recomputed from the local fine file when present (fine vs lattice at the shared nodes) and the stored flag
  * C-R1  x1 result: prerequisites (solver, floor, t0_shared) true and order false
  * PS cross-check per window (verdict.crosscheck, and the pre-A10 rule) on the selected PS ladder (escalation attempt
          if present), against the matching FD final reference; per dimension and window the tier validated /
          unvalidated / suspended (A2.3) and, per case and window, eligibility = accepted & G-R2 & (not a designated
          cross-check case or its PASS) -- the tier is reported beside it
Writes checks/refs-<name>.json.

    python anaref.py --run runs/<a>/archive/output [--run runs/<b>/archive/output ...] [--out ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'ref'))
import verdict as V  # noqa: E402  (this lane's copy, output count generalised)

E0 = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-references/experiments/jcp-references')
WINDOWS = {'W1': (1, 5), 'W2': (6, 10), 'W3': (11, 20)}
FINEST = {2: 8192, 3: 512}
DESIGNATED = {2: ('val32', (0, 1, 5, 19)), 3: ('val64', (0, 43))}          # A1.5 / A2.3 cross-check cases
FINE = {2: 1024, 3: 128}
LAT = {2: 256, 3: 64}


def e0_raw(dim, coh, case):
    sys.path.insert(0, str(E0))
    import refstore as RS
    sys.path.pop(0)
    return RS._load_raw(dim, coh, case, 'ST'), RS


def tnorm(a):
    a = np.asarray(a, float)
    return np.sqrt(np.sum(a.reshape(a.shape[0], -1) ** 2, axis=1))


def final_levels(arr, meta):
    att = {a['id']: a for a in meta['attempts']}
    return [dict(L=att[i]['L'], lat=np.asarray(arr[i]), tau=att[i]['tau'], steps_ok=att[i]['steps_ok'],
                 umax=att[i]['umax_running'], id=i, dt=att[i]['dt'], purpose=att[i]['purpose']) for i in meta['final']]


def g_r2(arr, meta, d):
    (earr, emeta), RS = e0_raw(d, meta['cohort'], meta['case'])
    n0 = float(np.linalg.norm(np.asarray(earr['t0']).ravel()))
    eps_t = emeta['judge']['eps_t']
    t0_bitwise = bool(np.array_equal(np.asarray(arr['t0']), np.asarray(earr['t0'])))
    rows = []
    for lv in final_levels(arr, meta):
        b = next((x for x in emeta['attempts'] if x['L'] == lv['L'] and abs(x['dt'] - lv['dt']) < 1e-15 and
                  x['purpose'] == lv['purpose']), None)
        if b is None:
            rows.append(dict(new=lv['id'], e0=None, matched=False))
            continue
        x, y = lv['lat'][:6], np.asarray(earr[b['id']])
        dd = (tnorm(x - y) / n0).tolist()
        bars = [0.] + [min(1e-8, .01 * e) if e is not None else 0. for e in eps_t]
        rows.append(dict(new=lv['id'], e0=b['id'], matched=True, d_t=dd, bars=bars, t0_bitwise=bool(np.array_equal(x[0], y[0])),
                         passed=bool(np.array_equal(x[0], y[0]) and all(dd[k] <= bars[k] for k in range(1, 6)))))
    fin = final_levels(arr, meta)[2]
    evaluable = any(r_['matched'] and r_['new'] == fin['id'] for r_ in rows)
    passed = bool(t0_bitwise and evaluable and all(r_['passed'] for r_ in rows if r_['matched']))
    return dict(t0_bitwise=t0_bitwise, pairs=rows, evaluable=evaluable, passed=passed, e0_store_version=RS.store_version())


def judges(arr, meta):
    t0 = np.asarray(arr['t0'])
    lv = final_levels(arr, meta)
    nu = meta['desc']['nu']
    jd = V.judge(lv, t0, nu)
    agree = bool(jd['outcome'] == meta['judge']['outcome'] and
                 all((a is None and b is None) or (a is not None and b is not None and abs(a - b) <= 1e-12 * max(abs(b), 1e-300))
                     for a, b in zip(jd['eps_t'], meta['judge']['eps_t'])))
    n = lv[0]['lat'].shape[0]
    win = {}
    for w, (a, b) in WINDOWS.items():
        if b >= n:
            continue
        idx = [0] + list(range(a, b + 1))
        jw = V.judge([dict(l, lat=l['lat'][idx]) for l in lv], t0, nu)
        stored = (meta.get('judge_windows') or {}).get(w, {})
        win[w] = dict(outcome=jw['outcome'], eps=jw['eps_ref'], record=jw, stored_outcome=stored.get('outcome'),
                      agree=bool(jw['outcome'] == stored.get('outcome')))
    return jd, agree, win


def g_r3_local(js, arr, meta, d):
    fl = Path(str(js)[:-5] + '_fine.npz')
    if not fl.exists():
        return None
    z = np.load(fl)
    F = np.asarray(z['fine_' + meta['final'][2]])
    Lt = np.asarray(arr[meta['final'][2]])
    sf = FINE[d] // LAT[d]
    if d == 2:
        return bool(np.array_equal(F[:, ::sf, ::sf], Lt))
    mf = FINE[d] - 1
    Fc = F.reshape((F.shape[0],) + (mf,) * 3)[:, sf - 1::sf, sf - 1::sf, sf - 1::sf]
    return bool(np.array_equal(Fc.reshape(F.shape[0], -1), Lt))


def crosscheck_windows(ps_meta, ps_arr, fd_case):
    """E0 crosscheck (and the pre-A10 rule) per window on the selected PS ladder vs the FD final reference."""
    runs = {r['id']: r for r in ps_meta['runs']}
    sel_sp = ps_meta['selected_spatial']
    full = next(i for i in sel_sp if runs[i]['N'] == ps_meta['Ns'][-1])
    u3_all = fd_case['u3']
    out = {}
    for w, (a, b) in WINDOWS.items():
        if b >= u3_all.shape[0]:
            continue
        idx = [0] + list(range(a, b + 1))
        jw = fd_case['windows'][w]['record']
        ps = dict(spatial=[(runs[i]['N'], np.asarray(ps_arr[i])[idx]) for i in sel_sp], half=np.asarray(ps_arr[ps_meta['selected_half']])[idx],
                  full=np.asarray(ps_arr[full])[idx], finite=ps_meta['finite'], gates_ok=ps_meta['gates_ok'],
                  stability_ok=ps_meta['stability_ok'])
        r = V.crosscheck(jw, u3_all[idx], ps, jw['n0'])
        out[w] = dict(verdict=r['verdict'], reason=r.get('reason'), old_rule=r.get('old_rule_verdict'), x_ps=r.get('x_ps'),
                      eps=r.get('eps_ref'), sigma=r.get('sigma'))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, action='append')
    ap.add_argument('--e0-gpu', default='NVIDIA A100 80GB PCIe', help="GPU type of E0's job for the G-R1 cases")
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    rep = dict(runs=a.run, G_R1=[], cases=[], x1=[], ps=[], tiers={}, prefix_T1=[])
    fd = {}
    psl = []
    for run in map(Path, a.run):
        for step in sorted(p for p in run.iterdir() if p.is_dir() and p.name.startswith('ref_refrun')):
            res = json.loads((step / 'result.json').read_text())
            mode, d = res['config']['mode'], int(res['config']['dim'])
            gpu = res['job']['gpu']
            if mode == 'e0repro':
                for js in sorted(step.rglob('E0REPRO/c*.json')):
                    m = json.loads(js.read_text())
                    z = np.load(str(js)[:-5] + '.npz')
                    (earr, emeta), RS = e0_raw(d, m['cohort'], m['case'])
                    n0 = float(np.linalg.norm(np.asarray(earr['t0']).ravel()))
                    same = gpu == a.e0_gpu
                    rows = []
                    for at in m['attempts']:
                        eid = next((b['id'] for b in emeta['attempts'] if b['L'] == at['L'] and b['purpose'] == 'prod'), None)
                        if eid is None:
                            rows.append(dict(L=at['L'], evaluable=False, passed=False))
                            continue
                        dd = (tnorm(np.asarray(z[f"L{at['L']}"]) - np.asarray(earr[eid])) / n0).tolist()
                        bars = [0.] + [(1e-12 if same else min(1e-8, .01 * (e or 0.))) for e in emeta['judge']['eps_t']]
                        rows.append(dict(L=at['L'], e0=eid, d_t=dd, bars=bars,
                                         passed=bool(dd[0] == 0. and all(dd[k] <= bars[k] for k in range(1, 6)))))
                    rep['G_R1'].append(dict(cohort=m['cohort'], case=m['case'], gpu=gpu, e0_gpu=a.e0_gpu, same_gpu=same,
                                            levels=rows, passed=bool(rows and all(r_['passed'] for r_ in rows))))
            elif mode == 'fd':
                for js in sorted(step.rglob('T*/c*.json')):
                    if js.parent.name == 'PS':
                        continue
                    m = json.loads(js.read_text())
                    z = np.load(str(js)[:-5] + '.npz')
                    jd, agree, win = judges(z, m)
                    g3 = g_r3_local(js, z, m, d)
                    rec = dict(dim=d, cohort=m['cohort'], case=m['case'], horizon=m['horizon'], outcome=m['outcome'],
                               refined=m['refined'], tight=m['tight_used'], G_R3_flag=m.get('G_R3'), G_R3_local=g3,
                               judge=dict(outcome=jd['outcome'], eps_ref=jd['eps_ref'], agree=agree,
                                          windows={w: {k: v for k, v in x.items() if k != 'record'} for w, x in win.items()}),
                               fine_windows={w: (v or {}).get('outcome') for w, v in (m.get('judge_fine_windows') or {}).items()},
                               fine_eps={w: (v or {}).get('eps_ref') for w, v in (m.get('judge_fine_windows') or {}).items()},
                               seconds=m.get('seconds'), gpu=gpu)
                    rec['G_R2'] = g_r2(z, m, d)
                    rep['cases'].append(rec)
                    fd[(d, m['cohort'], m['case'], m['horizon'])] = dict(u3=np.asarray(z[m['final'][2]]), windows=win,
                                                                       levels=final_levels(z, m), eps_t=jd['eps_t'])
            elif mode == 'x1':
                for x in res.get('x1', []):
                    ch = x['checks']
                    rep['x1'].append(dict(dim=d, case=x['case'], outcome=x['outcome'],
                                          fired=bool(ch['solver'] and ch['floor'] and ch['t0_shared'] and not ch['order']),
                                          evaluable=bool(ch['solver'] and ch['floor'] and ch['t0_shared'])))
            elif mode == 'ps':
                for js in sorted(step.rglob('PS/c*_a*.json')):
                    psl.append((d, js))
    # T = 1.0 vs T = 0.5 prefix (A1.5)
    for (d, coh, c, T), v in fd.items():
        if T > .5 and (d, coh, c, .5) in fd:
            h = fd[(d, coh, c, .5)]
            rows = []
            for lv in v['levels']:
                m = next((x for x in h['levels'] if x['L'] == lv['L'] and abs(x['dt'] - lv['dt']) < 1e-15 and
                          x['purpose'] == lv['purpose']), None)
                if m is None:
                    continue
                n0 = float(np.linalg.norm(m['lat'][0].ravel()))
                dd = (tnorm(lv['lat'][:11] - m['lat']) / n0).tolist()
                bars = [0.] + [min(1e-8, .01 * e) if e is not None else 0. for e in h['eps_t']]
                rows.append(dict(L=lv['L'], passed=bool(all(dd[k] <= bars[k] for k in range(1, 11)) and dd[0] == 0.)))
            rep['prefix_T1'].append(dict(dim=d, cohort=coh, case=c, rows=rows, passed=bool(rows and all(r_['passed'] for r_ in rows))))
    # PS cross-checks: the highest attempt per case decides
    best = {}
    for d, js in psl:
        m = json.loads(js.read_text())
        k = (d, m['cohort'], m['case'], m['horizon'])
        if k not in best or m['ladder_attempt'] > best[k][0]['ladder_attempt']:
            best[k] = (m, js)
    for (d, coh, c, T), (m, js) in sorted(best.items()):
        fdc = fd.get((d, coh, c, T))
        if fdc is None:
            rep['ps'].append(dict(dim=d, cohort=coh, case=c, horizon=T, windows=None, note='no FD reference for this case'))
            continue
        z = np.load(str(js)[:-5] + '.npz')
        rep['ps'].append(dict(dim=d, cohort=coh, case=c, horizon=T, attempt=m['ladder_attempt'],
                              windows=crosscheck_windows(m, z, fdc)))
    for d in (2, 3):
        dc, cs = DESIGNATED[d]
        for T in sorted({x['horizon'] for x in rep['ps'] if x['dim'] == d}):
            for w in WINDOWS:
                vs = [next((x['windows'][w]['verdict'] for x in rep['ps'] if x['dim'] == d and x['horizon'] == T and
                            x['cohort'] == dc and x['case'] == c and x['windows'] and w in x['windows']), 'MISSING') for c in cs]
                if all(v == 'MISSING' for v in vs):
                    continue
                tier = 'suspended' if 'FAIL' in vs else ('validated' if all(v == 'PASS' for v in vs) else 'unvalidated')
                rep['tiers'].setdefault(f'{d}d', {}).setdefault(f'T{T:g}', {})[w] = dict(tier=tier, verdicts=dict(zip(map(str, cs), vs)))
    # per case and window eligibility (the tier is reported beside it; it decides verdict vs provisional vs descriptive)
    for rec in rep['cases']:
        d = rec['dim']
        dc, cs = DESIGNATED[d]
        desig = rec['cohort'] == dc and rec['case'] in cs
        elig = {}
        for w, x in rec['judge']['windows'].items():
            xc = None
            if desig:
                p = next((y for y in rep['ps'] if y['dim'] == d and y['cohort'] == rec['cohort'] and y['case'] == rec['case']
                          and y['horizon'] == rec['horizon'] and y['windows']), None)
                xc = p['windows'].get(w, {}).get('verdict') if p else 'MISSING'
            tier = rep['tiers'].get(f'{d}d', {}).get(f"T{rec['horizon']:g}", {}).get(w, {}).get('tier', 'unvalidated (no cross-check)')
            elig[w] = dict(accepted=x['outcome'] == 'accepted', G_R2=rec['G_R2']['passed'], designated=desig, crosscheck=xc,
                           tier=tier, eligible=bool(x['outcome'] == 'accepted' and rec['G_R2']['passed'] and (not desig or xc == 'PASS')))
        rec['eligibility'] = elig
    rep['summary'] = dict(G_R1=all(x['passed'] for x in rep['G_R1']) if rep['G_R1'] else None,
                          G_R2=all(c['G_R2']['passed'] for c in rep['cases']) if rep['cases'] else None,
                          G_R3=all(c['G_R3_flag'] is True and c['G_R3_local'] is not False for c in rep['cases']) if rep['cases'] else None,
                          judge_agree=all(c['judge']['agree'] and all(w['agree'] for w in c['judge']['windows'].values())
                                          for c in rep['cases']) if rep['cases'] else None,
                          C_R1=all(x['fired'] for x in rep['x1']) if rep['x1'] else None,
                          prefix_T1=all(x['passed'] for x in rep['prefix_T1']) if rep['prefix_T1'] else None,
                          tiers=rep['tiers'])
    dest = Path(a.out)
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(rep, indent=1, default=float) + '\n')
    print(dest, json.dumps(rep['summary']))


if __name__ == '__main__':
    main()
