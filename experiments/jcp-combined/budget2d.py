"""Offline 2D error budget from a `c2bud` result (DESIGN 5 with A1.4, A2.3, A2.4, A3.5, A3.6; A1.10 gates B0-B3).

Every quantity is a measured distance per output time t_k (k = 1..5) on the 257^2 nodes, divided by n0 = ||u0|| of the
physical initial field at the nodes. Coefficients are decoded through the bank at the node coordinates (score.bank2d);
references are read through the authenticated registry (score.load_ref2d), so --refs switches the reference.

Per setting, budget arm a (form f, scheme s, step h) and case:
  e_tot  = ||u_a - u_ref||                     delta_quad  = ||u_a - u_a[r->r_inf]||
  e_star = ||u_star - u_ref||                  delta_solve = ||u_a - u_a[tight]||
  e_P    = ||(I - Pi) u_ref||  (floor)         delta_form  = ||u_a[tight] - u_a[tight, GAL]||   (0 for GAL arms)
  e_dyn  = ||u_star - Pi u_ref||               delta_time  = ||u_a - u_a[(BDF2, h0/8)]||       (package, A1.1)
  D_a    = ||u_a - u_star|| (direct)           refine      = ||u_a[tight] - u_a[tight, h/2]||   (within-scheme)
  I_time_quad, I_time_solve: four-corner mixed differences (A1.4, A2.3)
  context sensitivity: deployed-end delta_quad vs ideal-end ||u_star - u_star[r_inf->r]||; deployed-end form sensitivity vs
  ideal-end ||u_star - u_star[GAL->LSPG]|| (only when both exceed 1e-8)
  U_star(t) = ||star(h0/8) - star(h0/16)|| + tight-vs-tighter of both; a component < 3 U_star is 'unresolved'.
Gates: B0 per time e_star^2 = e_P^2 + e_dyn^2 (relative 1e-10) with the analytic negative fixtures of A3.5;
B1 sham bitwise (job); B2 control rule exceeds tau on most check cases (job); B3 delta_time(LSPG-BE dt0) > 1e-3 on most
cases; converged-rule status (job); finalists (selected arms): delta_quad <= tau and ||r_inf - check|| <= 2.5e-5 on every
case; in-job error reproduction (per time, every run) |x - y| <= 1e-13 + 1e-10 |y| against the job's ST/S/TX.
Summaries per arm: max over t of each component per case; cohort median / worst; the orthogonal pair at
t* = argmax_k e_star(t_k); 'largest measured sensitivity or discrepancy' counts over cases (A1.1(v)).

    python budget2d.py --run runs/c2bud/archive/output --refs 2d_first_order
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DT0 = .005
TAU = 2.5e-4
H8, H16 = DT0 / 8, DT0 / 16


def k_(role, form, sc, dt, lev):
    return f'{role}|{form}|{sc}|{dt:.8g}|{lev}'


def b0_fixtures():
    """A3.5 analytic negative controls: each must give |defect| > 1e-6 with the stated exact value."""
    defect = lambda v, y, P: float(np.linalg.norm(v - y) ** 2 - np.linalg.norm(y - P @ y) ** 2 - np.linalg.norm(v - P @ y) ** 2)
    e1, e2 = np.array([1., 0.]), np.array([0., 1.])
    f1 = defect(e1 + 1e-3 * e2, e2, np.diag([1., 0.]))
    f2 = defect(e1, e2, np.array([[1., .5], [0., 0.]]))
    return dict(orthogonal_out_of_span=f1, expected1=-2e-3, oblique=f2, expected2=.5,
                passed=bool(abs(f1 + 2e-3) < 1e-12 and abs(f2 - .5) < 1e-12 and abs(f1) > 1e-6 and abs(f2) > 1e-6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--refs', default='2d_first_order')
    ap.add_argument('--allow-smoke', action='store_true')
    ap.add_argument('--out')
    a = ap.parse_args()
    import score
    run = Path(a.run)
    res = json.loads((run / 'result.json').read_text())
    if res['config'].get('smoke') and not a.allow_smoke:
        raise SystemExit('smoke result refused')
    ent = score.validate_entry(a.refs)
    assert res['checkpoint_sha256'] == ent['bank_sha256']
    cfg = res['config']
    cohorts = res['cohorts']
    Q, G = score.bank2d(max(int(s['Rp']) for s in cfg['settings']))
    out = dict(run=str(run), refs=a.refs, result_sha256=hashlib.sha256((run / 'result.json').read_bytes()).hexdigest(),
               provisional=True, B0_fixtures=b0_fixtures(), settings={}, gates={})
    rows = {(r['setting'], r['run'], r['cohort'], r['case']): r for r in res['rows']}
    repro = dict(compared=0, failures=0, worst=0.)
    tags = list(ent['tags']) + list(ent.get('derived', {}))
    for s in cfg['settings']:
        sname, Rp = s['name'], int(s['Rp'])
        z = np.load(run / f'WB_{sname}.npz')
        cks = [str(x) for x in z['cases']]
        S_ = res['setup'][sname]
        Qb, _ = np.linalg.qr(G[:, :Rp])                                         # orthonormal basis of the span at the nodes
        dec = lambda W: score.decode2d(G, W[None] if W.ndim == 1 else W)
        armres = {}
        for x in s['arms']:
            f, sc, h = x['form'], x['scheme'], DT0 * x['f']
            K = dict(a=k_('main', f, sc, h, 'prod'), conv=k_('conv', f, sc, h, 'prod'), tight=k_('main', f, sc, h, 'tight'),
                     tgal=k_('main', 'GAL', sc, h, 'tight') if f == 'LSPG' else None, time=k_('main', f, 'BDF2', H8, 'prod'),
                     half=k_('main', f, sc, h / 2, 'tight'), c1=k_('conv', f, 'BDF2', H8, 'prod'),
                     c2=k_('main', f, 'BDF2', H8, 'tight'), check=k_('check', f, sc, h, 'prod') if x.get('selected') else None)
            per = []
            for ci, ck in enumerate(cks):
                coh, c = ck.split('|')
                c = int(c)
                n0 = float(np.linalg.norm(Q.e.initial(256, np.asarray(cohorts[coh]['physical'][c]))))
                U = {nm: dec(z[key][ci]) / n0 for nm, key in K.items() if key and key in z.files}
                st = {nm: dec(z[k_(*kk)][ci]) / n0 for nm, kk in
                      dict(star=('conv', 'GAL', 'BDF2', H8, 'tight'), star16=('conv', 'GAL', 'BDF2', H16, 'tight'),
                           starx=('conv', 'GAL', 'BDF2', H8, 'tighter'), star16x=('conv', 'GAL', 'BDF2', H16, 'tighter'),
                           starR=('main', 'GAL', 'BDF2', H8, 'tight'), starL=('conv', 'LSPG', 'BDF2', H8, 'tight')).items()}
                refs = score.load_ref2d(ent, coh, c)
                nt = lambda X: [float(np.linalg.norm(X[j])) for j in range(6)]
                d = {}
                d['delta_quad'] = nt(U['a'] - U['conv'])
                d['delta_solve'] = nt(U['a'] - U['tight'])
                d['delta_form'] = nt(U['tight'] - U['tgal']) if 'tgal' in U else [0.] * 6
                d['delta_time'] = nt(U['a'] - U['time'])
                d['refine'] = nt(U['tight'] - U['half'])
                d['D_a'] = nt(U['a'] - st['star'])
                d['I_time_quad'] = nt(U['c1'] - U['conv'] - U['time'] + U['a'])
                d['I_time_solve'] = nt(U['c2'] - U['tight'] - U['time'] + U['a'])
                d['ctx_quad_ideal_end'] = nt(st['star'] - st['starR'])
                d['ctx_form_ideal_end'] = nt(st['star'] - st['starL'])
                d['U_star'] = [float(np.linalg.norm(st['star'][j] - st['star16'][j]) + np.linalg.norm(st['star'][j] - st['starx'][j])
                                     + np.linalg.norm(st['star16'][j] - st['star16x'][j])) for j in range(6)]
                if 'check' in U:
                    d['finalist_check'] = nt(U['conv'] - U['check'])
                for t in tags:
                    Rn = refs[t] / n0
                    inner = Rn[:, 1:-1, 1:-1].reshape(6, -1)
                    PR = (inner @ Qb) @ Qb.T                                     # Pi u_ref on the interior nodes
                    ustar = st['star'][:, 1:-1, 1:-1].reshape(6, -1)
                    eP = np.linalg.norm(inner - PR, axis=1)
                    edyn = np.linalg.norm(ustar - PR, axis=1)
                    es = np.linalg.norm(ustar - inner, axis=1)
                    d[f'e_tot_{t}'] = nt(U['a'] - Rn)
                    d[f'e_star_{t}'] = es.tolist()
                    d[f'e_P_{t}'] = eP.tolist()
                    d[f'e_dyn_{t}'] = edyn.tolist()
                    d[f'B0_{t}'] = [float(abs(es[j] ** 2 - eP[j] ** 2 - edyn[j] ** 2) / max(es[j] ** 2, 1e-300)) for j in range(1, 6)]
                    d[f'init_fit_{t}'] = float(np.linalg.norm(U['a'][0] - Rn[0]))
                # in-job reproduction (per time, all three tags)
                for nm, key in K.items():
                    r = rows.get((sname, key, coh, c)) if key else None
                    if r is None or nm not in U:
                        continue
                    for t in ('ST', 'S', 'TX'):
                        if t not in refs:
                            continue
                        mine = nt(U[nm] - refs[t] / n0)
                        for j in range(6):
                            y = r[f'e_{t}_per_time'][j]
                            repro['compared'] += 1
                            repro['worst'] = max(repro['worst'], abs(mine[j] - y))
                            repro['failures'] += 0 if abs(mine[j] - y) <= 1e-13 + 1e-10 * abs(y) else 1
                per.append(dict(case=ck, **d))
            # summaries
            mx = lambda v: max(v[1:])
            summ = {}
            for comp in ('delta_quad', 'delta_solve', 'delta_form', 'delta_time', 'refine', 'D_a', 'I_time_quad',
                         'I_time_solve', 'U_star') + tuple(f'{p}_{t}' for p in ('e_tot', 'e_star', 'e_P', 'e_dyn') for t in tags):
                v = [mx(p_[comp]) for p_ in per]
                summ[comp] = dict(median=float(np.median(v)), worst=float(max(v)))
            largest = {}
            for t in tags:
                cnt = {}
                for p_ in per:
                    comps = dict(e_P=mx(p_[f'e_P_{t}']), e_dyn=mx(p_[f'e_dyn_{t}']), delta_quad=mx(p_['delta_quad']),
                                 delta_time=mx(p_['delta_time']), delta_form=mx(p_['delta_form']), delta_solve=mx(p_['delta_solve']))
                    w = max(comps, key=comps.get)
                    cnt[w] = cnt.get(w, 0) + 1
                largest[t] = cnt
            unres = {comp: sum(mx(p_[comp]) < 3 * mx(p_['U_star']) for p_ in per)
                     for comp in ('delta_quad', 'delta_solve', 'delta_form', 'delta_time')}
            b0 = max(max(p_[f'B0_{t}']) for p_ in per for t in tags)
            fin = None
            if x.get('selected'):
                fin = dict(delta_quad_worst=summ['delta_quad']['worst'],
                           check_worst=max(mx(p_['finalist_check']) for p_ in per) if 'finalist_check' in per[0] else None)
                fin['qualified'] = bool(fin['delta_quad_worst'] <= TAU and fin['check_worst'] is not None and
                                        fin['check_worst'] <= cfg['conv_bar'])
            ctx = dict(quad=dict(deployed_end=summ['delta_quad']['median'],
                                 ideal_end=float(np.median([mx(p_['ctx_quad_ideal_end']) for p_ in per]))),
                       form=dict(deployed_end=summ['delta_form']['median'],
                                 ideal_end=float(np.median([mx(p_['ctx_form_ideal_end']) for p_ in per]))))
            for v in ctx.values():
                v['flag'] = ('below floor' if min(v['deployed_end'], v['ideal_end']) <= 1e-8 else
                             ('context-sensitive' if max(v['deployed_end'], v['ideal_end']) > 2 * min(v['deployed_end'], v['ideal_end'])
                              else 'consistent'))
            armres[f"{f}|{sc}|{x['f']:g}"] = dict(arm=x, keys=K, summary=summ, largest=largest, unresolved_vs_Ustar=unres,
                                                  B0_worst_relative=b0, finalist=fin, context=ctx, per_case=per)
        b3 = None
        be = armres.get('LSPG|BE|1')
        if be:
            dt_ = [max(p_['delta_time'][1:]) for p_ in be['per_case']]
            b3 = dict(exceeds=sum(v > 1e-3 for v in dt_), cases=len(dt_), discriminating=bool(sum(v > 1e-3 for v in dt_) > len(dt_) / 2))
        out['settings'][sname] = dict(rule=s['rule'], conv_status=S_.get('converged_rule_status'), conv_check=S_.get('conv_check'),
                                      conv_check_successor=S_.get('conv_check_successor'), B1=S_.get('B1_sham_bitwise'),
                                      B2=S_.get('B2_control_rule'), B3=b3, arms=armres)
    b0w = max(v['B0_worst_relative'] for s_ in out['settings'].values() for v in s_['arms'].values())
    out['gates'] = dict(B0=dict(worst_relative=b0w, passed=bool(b0w <= 1e-10)), B0_fixtures=out['B0_fixtures']['passed'],
                        reproduction=dict(repro, passed=bool(repro['compared'] > 0 and repro['failures'] == 0)))
    dest = Path(a.out or HERE / 'checks' / f"budget2d-{run.parents[1].name}-{a.refs}.json")
    dest.write_text(json.dumps(out, default=float) + '\n')
    print(dest, json.dumps(out['gates'], default=float))


if __name__ == '__main__':
    main()
