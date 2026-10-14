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

import offline  # noqa: F401  (A8.1: CPU, highest, x64 - before any JAX import)

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
    ap.add_argument('--grid', help='c2grid output dir: budget arm runs must reproduce the grid runs (fields <= 1e-8)')
    ap.add_argument('--out')
    a = ap.parse_args()
    import score
    run = Path(a.run)
    res = json.loads((run / 'result.json').read_text())
    if res['config'].get('smoke') and not a.allow_smoke:
        raise SystemExit('smoke result refused')
    if not res['config'].get('smoke') and not a.grid:
        raise SystemExit('production budgets require --grid (grid reproduction, A8.5)')
    ent = score.validate_entry(a.refs)
    assert res['checkpoint_sha256'] == ent['bank_sha256']
    cfg = res['config']
    cohorts = res['cohorts']
    Q, G = score.bank2d(max(int(s['Rp']) for s in cfg['settings']))
    out = dict(run=str(run), refs=a.refs, result_sha256=hashlib.sha256((run / 'result.json').read_bytes()).hexdigest(),
               analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               provisional=True, B0_fixtures=b0_fixtures(), settings={}, gates={})
    rows = {(r['setting'], r['run'], r['cohort'], r['case']): r for r in res['rows']}
    repro = dict(compared=0, failures=0, worst=0., applicable=(a.refs == '2d_first_order'))
    gridz = {}
    grid_rows = {}
    if a.grid:
        gres = json.loads((Path(a.grid) / 'result.json').read_text())
        grid_rows = {(r['setting'], r['run'], r['cohort'], r['case']): r for r in gres['rows']}
    grepro = dict(compared=0, failures=0, worst=0.)
    tags = list(ent['tags']) + list(ent.get('derived', {}))
    import c2bud as C2B
    cfg_cases = [(coh, c) for coh in cfg['cohorts'] for c in
                 (cfg.get('case_subset', {}).get(coh) if cfg.get('case_subset', {}).get(coh) is not None
                  else range(cohorts[coh]['n']))]
    check_cases = [x for x in cfg_cases if x[0] == 'dev6'] + [x for x in cfg_cases if x[0] == 'val32'][:cfg['check_val32']]
    inv_mismatch = 0
    for s in cfg['settings']:
        sname, Rp = s['name'], int(s['Rp'])
        z = np.load(run / f'WB_{sname}.npz')
        cks = [str(x) for x in z['cases']]
        S_ = res['setup'][sname]
        Qb, _ = np.linalg.qr(G[:, :Rp])                                         # orthonormal basis of the span at the nodes
        dec = lambda W: score.decode2d(G, W[None] if W.ndim == 1 else W)
        # configured inventory (A9.3; codex-code-7 N14): plan runs on every case, the check-rule copies of every r_inf run
        # type and the control run on the check cases; must equal the JSON rows and the archived (non-NaN) entries
        plan = C2B.budget_runs(s['arms'])
        expk = {(C2B.key(r), coh, c) for r in plan for coh, c in cfg_cases}
        expk |= {(C2B.key(('check',) + tuple(r[1:])), coh, c) for r in plan if r[0] == 'conv' for coh, c in check_cases}
        expk |= {(C2B.key(('ctrl', 'LSPG', 'BE', DT0, 'prod')), coh, c) for coh, c in check_cases}
        rowk = {(r_['run'], r_['cohort'], r_['case']) for r_ in res['rows'] if r_['setting'] == sname}
        arck = {(k2, *ck.split('|')[:1], int(ck.split('|')[1])) for k2 in z.files if k2 not in ('w0', 'cases')
                for ci, ck in enumerate(cks) if not np.isnan(z[k2][ci]).all()}
        nrow_dup = len([1 for r_ in res['rows'] if r_['setting'] == sname]) - len(rowk)
        inv_mismatch += len(expk ^ rowk) + len(expk ^ arck) + nrow_dup
        # in-job reproduction: every row of this setting ONCE, original reference only (A1.8; codex-code-5 N4)
        if repro['applicable']:
            for ci, ck in enumerate(cks):
                coh, c = ck.split('|')
                c = int(c)
                n0 = float(np.linalg.norm(Q.e.initial(256, np.asarray(cohorts[coh]['physical'][c]))))
                refs = score.load_ref2d(ent, coh, c)
                for key in [k2 for k2 in z.files if k2 not in ('w0', 'cases')]:
                    r = rows.get((sname, key, coh, c))
                    if r is None:
                        continue
                    Ux = dec(z[key][ci]) / n0
                    for t in ('ST', 'S', 'TX'):
                        mine = [float(np.linalg.norm((Ux - refs[t] / n0)[j])) for j in range(6)]
                        for j in range(6):
                            y = np.nan if r[f'e_{t}_per_time'][j] is None else r[f'e_{t}_per_time'][j]
                            repro['compared'] += 1
                            if np.isfinite(mine[j]) and np.isfinite(y):
                                repro['worst'] = max(repro['worst'], abs(mine[j] - y))
                                repro['failures'] += 0 if abs(mine[j] - y) <= 1e-13 + 1e-10 * abs(y) else 1
                            else:                                   # invalid rollout: accounted separately (N7)
                                repro['invalid'] = repro.get('invalid', 0) + 1
                                repro['failures'] += 0 if (not np.isfinite(mine[j]) and not np.isfinite(y) and not r['verified']) else 1
        kb = k_('main', 'LSPG', 'BE', DT0, 'prod')
        valid_ci = [ci for ci, ck in enumerate(cks) if rows.get((sname, kb, ck.split('|')[0], int(ck.split('|')[1])), {}).get('verified')
                    and rows[(sname, kb, ck.split('|')[0], int(ck.split('|')[1]))].get('e_ST') is not None]
        if repro['applicable'] and len(valid_ci) < 2:
            repro.setdefault('controls', {})[sname] = dict(all_break=False, unavailable='fewer than two valid baseline rows')
        if repro['applicable'] and len(valid_ci) > 1:       # G-score negative controls for the budget archive (A9.6)
            coh, c = cks[valid_ci[0]].split('|')
            c = int(c)
            r0 = rows[(sname, kb, coh, c)]
            n0 = float(np.linalg.norm(Q.e.initial(256, np.asarray(cohorts[coh]['physical'][c]))))
            refs0 = score.load_ref2d(ent, coh, c)
            U0_ = dec(z[kb][valid_ci[0]]) / n0
            e_ = lambda U_, R_: max(float(np.linalg.norm(U_[j] - R_[j] / n0)) for j in range(1, 6))
            ctl = dict(case_swap=abs(e_(dec(z[kb][valid_ci[1]]) / n0, refs0['ST']) - r0['e_ST']),
                       time_shift=abs(e_(np.roll(U0_, 1, 0), refs0['ST']) - r0['e_ST']),
                       ST_S_swap=abs(e_(U0_, refs0['S']) - r0['e_ST']))
            repro.setdefault('controls', {})[sname] = dict(ctl, all_break=bool(all(v > 1e-6 for v in ctl.values())))
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
                ok_ = {nm: bool(rows.get((sname, key, coh, c), {}).get('verified')) for nm, key in K.items() if key}
                stk = dict(star=('conv', 'GAL', 'BDF2', H8, 'tight'), star16=('conv', 'GAL', 'BDF2', H16, 'tight'),
                           starx=('conv', 'GAL', 'BDF2', H8, 'tighter'), star16x=('conv', 'GAL', 'BDF2', H16, 'tighter'),
                           starR=('main', 'GAL', 'BDF2', H8, 'tight'), starL=('conv', 'LSPG', 'BDF2', H8, 'tight'))
                ok_.update({nm: bool(rows.get((sname, k_(*kk), coh, c), {}).get('verified')) for nm, kk in stk.items()})
                need = lambda *nms: all(ok_.get(nm, False) for nm in nms)
                if a.grid:                                   # grid reproduction of the arm run (same arm, other job)
                    gk = '|'.join(K['a'].split('|')[1:])
                    if sname not in gridz and (Path(a.grid) / f'W_{sname}.npz').exists():
                        gz = np.load(Path(a.grid) / f'W_{sname}.npz')
                        gridz[sname] = (gz['W'], [str(x) for x in gz['runs']], [str(x) for x in gz['cases']])
                    if sname in gridz and gk in gridz[sname][1] and ck in gridz[sname][2]:
                        Wg = gridz[sname][0][gridz[sname][2].index(ck), gridz[sname][1].index(gk)]
                        dg = max(float(np.linalg.norm((dec(Wg) / n0 - U['a'])[j])) for j in range(1, 6))
                        grepro['compared'] += 1
                        grepro['worst'] = max(grepro['worst'], dg)
                        grepro['failures'] += 0 if dg <= 1e-8 else 1
                st = {nm: dec(z[k_(*kk)][ci]) / n0 for nm, kk in
                      dict(star=('conv', 'GAL', 'BDF2', H8, 'tight'), star16=('conv', 'GAL', 'BDF2', H16, 'tight'),
                           starx=('conv', 'GAL', 'BDF2', H8, 'tighter'), star16x=('conv', 'GAL', 'BDF2', H16, 'tighter'),
                           starR=('main', 'GAL', 'BDF2', H8, 'tight'), starL=('conv', 'LSPG', 'BDF2', H8, 'tight')).items()}
                refs = score.load_ref2d(ent, coh, c)
                nt = lambda X: [float(np.linalg.norm(X[j])) for j in range(6)]
                NA = [None] * 6
                d = {}
                # each component is available only if every trajectory it uses is verified (code audit 4 item 5)
                d['delta_quad'] = nt(U['a'] - U['conv']) if need('a', 'conv') else NA
                d['delta_solve'] = nt(U['a'] - U['tight']) if need('a', 'tight') else NA
                d['delta_form'] = (nt(U['tight'] - U['tgal']) if need('tight', 'tgal') else NA) if 'tgal' in U else [0.] * 6
                d['delta_time'] = nt(U['a'] - U['time']) if need('a', 'time') else NA
                d['refine'] = nt(U['tight'] - U['half']) if need('tight', 'half') else NA
                d['D_a'] = nt(U['a'] - st['star']) if need('a', 'star') else NA
                d['I_time_quad'] = nt(U['c1'] - U['conv'] - U['time'] + U['a']) if need('c1', 'conv', 'time', 'a') else NA
                d['I_time_solve'] = nt(U['c2'] - U['tight'] - U['time'] + U['a']) if need('c2', 'tight', 'time', 'a') else NA
                d['ctx_quad_ideal_end'] = nt(st['star'] - st['starR']) if need('star', 'starR') else NA
                d['ctx_form_ideal_end'] = nt(st['star'] - st['starL']) if need('star', 'starL') else NA
                d['U_star'] = ([float(np.linalg.norm(st['star'][j] - st['star16'][j]) + np.linalg.norm(st['star'][j] - st['starx'][j])
                                      + np.linalg.norm(st['star16'][j] - st['star16x'][j])) for j in range(6)]
                               if need('star', 'star16', 'starx', 'star16x') else NA)
                if 'check' in U:
                    d['finalist_check'] = nt(U['conv'] - U['check']) if need('conv', 'check') else NA
                for t in tags:
                    Rn = refs[t] / n0
                    inner = Rn[:, 1:-1, 1:-1].reshape(6, -1)
                    PR = (inner @ Qb) @ Qb.T                                     # Pi u_ref on the interior nodes
                    ustar = st['star'][:, 1:-1, 1:-1].reshape(6, -1)
                    eP = np.linalg.norm(inner - PR, axis=1)
                    edyn = np.linalg.norm(ustar - PR, axis=1)
                    es = np.linalg.norm(ustar - inner, axis=1)
                    d[f'e_tot_{t}'] = nt(U['a'] - Rn) if need('a') else NA
                    d[f'e_star_{t}'] = es.tolist() if need('star') else NA
                    d[f'e_P_{t}'] = eP.tolist()                                   # reference-only quantity
                    d[f'e_dyn_{t}'] = edyn.tolist() if need('star') else NA
                    d[f'B0_{t}'] = [float(abs(es[j] ** 2 - eP[j] ** 2 - edyn[j] ** 2) / max(es[j] ** 2, 1e-300)) for j in range(1, 6)]
                    # B0 fixture (iii) on real data: one term with a mismatched normaliser must break the identity
                    # A12.1: RELATIVE defect (the B0 gate's own normalisation); the absolute defect is ~ 99 e_P^2, which is
                    # below 1e-6 whenever the projection floor is small, so the absolute test was non-discriminating
                    d[f'B0_mismatch_{t}'] = float(abs(es[5] ** 2 - (eP[5] / .1) ** 2 - edyn[5] ** 2) / max(es[5] ** 2, 1e-300))
                    d[f'init_fit_{t}'] = float(np.linalg.norm(U['a'][0] - Rn[0]))
                    ts = 1 + int(np.argmax(es[1:]))
                    d[f'orthogonal_pair_at_tstar_{t}'] = (dict(t_star=ts, e_star=float(es[ts]), e_P=float(eP[ts]), e_dyn=float(edyn[ts]))
                                                         if need('star') else None)
                    if d['D_a'][1] is not None:                     # triangle bound |e_tot - e_star| <= D_a per time
                        et = nt(U['a'] - Rn)
                        d[f'triangle_ok_{t}'] = all(abs(et[j] - es[j]) <= d['D_a'][j] + 1e-14 for j in range(1, 6))
                per.append(dict(case=ck, **d))
            # summaries
            mx = lambda v: None if v[1] is None else max(v[1:])
            summ = {}
            for comp in ('delta_quad', 'delta_solve', 'delta_form', 'delta_time', 'refine', 'D_a', 'I_time_quad',
                         'I_time_solve', 'U_star') + tuple(f'{p}_{t}' for p in ('e_tot', 'e_star', 'e_P', 'e_dyn') for t in tags):
                v = [mx(p_[comp]) for p_ in per if mx(p_[comp]) is not None]
                summ[comp] = (dict(median=float(np.median(v)), worst=float(max(v)), available=len(v), cases=len(per))
                              if v else dict(median=None, worst=None, available=0, cases=len(per)))
            clos = []
            for p_ in per:
                dd = [mx(p_[k]) for k in ('delta_quad', 'delta_solve', 'delta_form', 'delta_time')]
                if all(x is not None for x in dd) and mx(p_['D_a']) is not None:
                    sq, sm = float(np.sqrt(sum(x * x for x in dd))), float(sum(dd))
                    clos.append(dict(D_over_rss=(mx(p_['D_a']) / sq) if sq > 0 else ('both zero' if mx(p_['D_a']) == 0 else 'undefined'),
                                     D_over_sum=(mx(p_['D_a']) / sm) if sm > 0 else ('both zero' if mx(p_['D_a']) == 0 else 'undefined')))
            ctx_case = []
            for p_ in per:
                for comp, ide in (('delta_quad', 'ctx_quad_ideal_end'), ('delta_form', 'ctx_form_ideal_end')):
                    dep_, ide_ = mx(p_[comp]), mx(p_[ide])
                    if dep_ is None or ide_ is None:
                        continue
                    ctx_case.append((comp, 'below floor' if min(dep_, ide_) <= 1e-8 else
                                     ('context-sensitive' if max(dep_, ide_) > 2 * min(dep_, ide_) else 'consistent')))
            largest = {}
            for t in tags:
                cnt = {}
                for p_ in per:
                    comps = dict(e_P=mx(p_[f'e_P_{t}']), e_dyn=mx(p_[f'e_dyn_{t}']), delta_quad=mx(p_['delta_quad']),
                                 delta_time=mx(p_['delta_time']), delta_form=mx(p_['delta_form']), delta_solve=mx(p_['delta_solve']))
                    w = max(comps, key=comps.get) if None not in comps.values() else 'unavailable (incomplete comparison)'
                    cnt[w] = cnt.get(w, 0) + 1
                largest[t] = cnt
            unres = {}
            for comp in ('delta_quad', 'delta_solve', 'delta_form', 'delta_time', 'refine', 'D_a', 'I_time_quad', 'I_time_solve'):
                pairs_ = [(mx(p_[comp]), mx(p_['U_star'])) for p_ in per]
                unres[comp] = dict(unresolved=sum(1 for v, u in pairs_ if v is not None and u is not None and v < 3 * u),
                                   unavailable=sum(1 for v, u in pairs_ if v is None or u is None))
            b0 = max(max(p_[f'B0_{t}']) for p_ in per for t in tags)
            b0m = per[0][f'B0_mismatch_{tags[0]}'] if per else None      # pinned fixture: first case, t = 0.25 (A3.5(iii))
            fin = None
            if x.get('selected'):
                fq = [mx(p_['delta_quad']) for p_ in per]
                fc = [mx(p_['finalist_check']) for p_ in per] if 'finalist_check' in per[0] else [None]
                fin = dict(delta_quad_worst=None if (None in fq or not fq) else max(fq),
                           check_worst=None if (None in fc or not fc) else max(fc),
                           unavailable_cases=sum(v is None for v in fq + fc))
                fin['qualified'] = bool(fin['delta_quad_worst'] is not None and fin['delta_quad_worst'] <= TAU and
                                        fin['check_worst'] is not None and fin['check_worst'] <= cfg['conv_bar'])
            med_ = lambda k: (lambda v: float(np.median(v)) if v else None)([mx(p_[k]) for p_ in per if mx(p_[k]) is not None])
            ctx = dict(quad=dict(deployed_end=summ['delta_quad']['median'], ideal_end=med_('ctx_quad_ideal_end')),
                       form=dict(deployed_end=summ['delta_form']['median'], ideal_end=med_('ctx_form_ideal_end')))
            for v in ctx.values():
                if v['deployed_end'] is None or v['ideal_end'] is None:
                    v['flag'] = 'unavailable'
                    continue
                v['flag'] = ('below floor' if min(v['deployed_end'], v['ideal_end']) <= 1e-8 else
                             ('context-sensitive' if max(v['deployed_end'], v['ideal_end']) > 2 * min(v['deployed_end'], v['ideal_end'])
                              else 'consistent'))
            armres[f"{f}|{sc}|{x['f']:g}"] = dict(arm=x, keys=K, summary=summ, largest=largest, unresolved_vs_Ustar=unres,
                                                  B0_worst_relative=b0, B0_mismatch_min_defect=b0m, closure=clos,
                                                  context=ctx, context_per_case={k: sum(1 for c_, v in ctx_case if c_ == k and v == 'context-sensitive')
                                                                                 for k in ('delta_quad', 'delta_form')},
                                                  finalist=fin, per_case=per)
        b3 = None
        be = armres.get('LSPG|BE|1')
        if be:
            dt_ = [None if p_['delta_time'][1] is None else max(p_['delta_time'][1:]) for p_ in be['per_case']]
            comp_ = None not in dt_
            b3 = dict(exceeds=sum(v > 1e-3 for v in dt_ if v is not None), cases=len(dt_), complete=comp_,
                      discriminating=bool(comp_ and sum(v > 1e-3 for v in dt_) > len(dt_) / 2))
        # B2 offline (codex-code-6 N8): delta_quad with the CONTROL rule as r, i.e. ||u(ctrl) - u(r_inf)|| for the baseline arm
        kc_, kv_ = k_('ctrl', 'LSPG', 'BE', DT0, 'prod'), k_('conv', 'LSPG', 'BE', DT0, 'prod')
        b2d = []
        for coh, c in check_cases:                         # the configured check population (codex-code-7 N13)
            ck = f'{coh}|{c}'
            ci = cks.index(ck) if ck in cks else None
            if ci is None or kc_ not in z.files or kv_ not in z.files:
                b2d.append(float('nan'))
                continue
            n0 = float(np.linalg.norm(Q.e.initial(256, np.asarray(cohorts[coh]['physical'][c]))))
            okk = rows.get((sname, kc_, coh, c), {}).get('verified') and rows.get((sname, kv_, coh, c), {}).get('verified')
            b2d.append(max(float(np.linalg.norm((dec(z[kc_][ci]) / n0 - dec(z[kv_][ci]) / n0)[j])) for j in range(1, 6))
                       if okk else float('nan'))
        b2 = dict(rule=cfg['control_rule'], per_case=b2d, complete=bool(b2d and np.isfinite(b2d).all()),
                  discriminating=bool(b2d and np.isfinite(b2d).all() and sum(v > TAU for v in b2d) > len(b2d) / 2))
        withheld = []
        if not b2.get('discriminating'):
            withheld.append('quadrature attribution (B2 control non-discriminating)')
        if not (b3 or {}).get('discriminating'):
            withheld.append('time attribution (B3 control non-discriminating)')
        if S_.get('converged_rule_status') == 'converged rule unverified':
            withheld.append('quadrature attribution (converged rule unverified)')
        out['settings'][sname] = dict(rule=s['rule'], conv_status=S_.get('converged_rule_status'), conv_check=S_.get('conv_check'),
                                      conv_check_successor=S_.get('conv_check_successor'), B1=S_.get('B1_sham_bitwise'),
                                      B2=b2, B3=b3, withheld=withheld, arms=armres)
    b0w = max(v['B0_worst_relative'] for s_ in out['settings'].values() for v in s_['arms'].values())
    b0m = next(iter(next(iter(out['settings'].values()))['arms'].values()))['B0_mismatch_min_defect']
    nrows = sum(1 for r in res['rows'])
    out['gates'] = dict(B0=dict(worst_relative=b0w, passed=bool(b0w <= 1e-10)), B0_fixtures=out['B0_fixtures']['passed'],
                        B0_real_mismatch=dict(min_defect=b0m, passed=bool(b0m > 1e-6)),
                        B1=all((s_['B1'] or {}).get('all_equal') for s_ in out['settings'].values()),
                        reproduction=dict(repro, rows=nrows,
                                          passed=(bool(repro['compared'] == 18 * nrows and repro['failures'] == 0 and inv_mismatch == 0 and
                                                       all(v['all_break'] for v in repro.get('controls', {}).values()) and
                                                       len(repro.get('controls', {})) == len(cfg['settings']))
                                                  if repro['applicable'] else 'not applicable (new reference)')),
                        inventory=dict(mismatch=inv_mismatch, passed=bool(inv_mismatch == 0)),
                        grid_reproduction=dict(grepro, expected=sum(len(s_['arms']) for s_ in cfg['settings']) * len(cfg_cases),
                                               passed=(bool(grepro['failures'] == 0 and grepro['compared'] ==
                                                            sum(len(s_['arms']) for s_ in cfg['settings']) * len(cfg_cases))
                                                       if a.grid else 'not run')))
    dest = Path(a.out or HERE / 'checks' / f"budget2d-{run.parents[1].name}-{a.refs}.json")
    dest.write_text(json.dumps(out, default=float) + '\n')
    print(dest, json.dumps(out['gates'], default=float))


if __name__ == '__main__':
    main()
