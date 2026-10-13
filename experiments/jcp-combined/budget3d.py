"""Offline 3D error budget from a `c3run` bud result (DESIGN 5 with A1.4, A1.5, A2.3, A2.4, A3.5, A3.6; gates B0-B2).

Per setting (deployed rule r, converged rule r_inf, next finer rule) and budget arm a (fixed-sweep spec or vendor):
components per output time on the 63^3 lattice (coefficients decoded through the bank at the lattice coordinates),
relative to ||u0|| of the physical initial field at the lattice:
  delta_quad  = ||u_a - u_a[r -> r_inf]||            delta_solve = ||u_a - u_a[adaptive tight LSPG, same scheme/step]||
  delta_form  = ||u_a[tight] - u_a[tight, GAL]||     delta_time  = ||u_a - u_a[(BDF2, h0/8), same K]||   (package)
  refine      = ||u_a[tight] - u_a[tight, h/2]||     D_a = ||u_a - u_star||
  I_time_quad, I_time_solve (four corners), context sensitivities, U_star (h0/8 vs h0/16 + tight-vs-tighter)
  e_tot, e_star, e_P, e_dyn against the 3D reference (reference-limited); B0 per time; fixture (iii) mismatched normaliser.
Every component is available only where all its trajectories are valid. Converged-rule check (A2.4): every r_inf run
type vs the same run with the next finer rule on the budget cases (bar 2.5e-5). Finalists (A3.6): selected arms'
delta_quad <= tau and r_inf-vs-finer <= 2.5e-5 on EVERY case. Reproduction: per time, every row (original reference only);
grid reproduction of each arm run against the c3grid run of the same arm (fields <= 1e-8, A1.11).

    python budget3d.py --run runs/c3bud/archive/output --grid runs/c3grid/archive/output
"""
from __future__ import annotations

import offline  # noqa: F401  (A8.1: CPU, highest, x64 - before any JAX import)

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DT0 = .01
TAU, BAR = 2.5e-4, 2.5e-5
H8, H16 = DT0 / 8, DT0 / 16


def sk(spec):
    if spec[0] == 'fs':
        return f'fs|{spec[1]}|{spec[2]:.8g}|K{spec[3]}'
    if spec[0] == 'vendor':
        return 'vendor|BE|0.01|K1'
    return f'ad|{spec[1]}|{spec[2]}|{spec[3]:.8g}|{spec[4]}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--grid')
    ap.add_argument('--refs', default='3d_first_order')
    ap.add_argument('--allow-smoke', action='store_true')
    ap.add_argument('--out')
    a = ap.parse_args()
    import score
    import c3common as C3
    run = Path(a.run)
    res = json.loads((run / 'result.json').read_text())
    if res['config'].get('smoke') and not a.allow_smoke:
        raise SystemExit('smoke result refused')
    if not res['config'].get('smoke') and not a.grid:
        raise SystemExit('production budgets require --grid (grid reproduction, A8.5)')
    import c3run as C3R
    ent = score.validate_entry(a.refs)
    z = np.load(ent['file'])
    cfg = res['config']
    tab = C3.C.table(cfg['cohort_seed'], cfg['cohort_count'])
    i65 = C3.lattice65_index(65)
    X65 = np.stack(np.meshgrid(*(np.arange(1, 64) / 64.,) * 3, indexing='ij'), -1).reshape(-1, 3)
    Wz = np.load(run / 'W_bud.npz')
    rows = {(r['setting'], r['rule'], r['spec'], r['case']): r for r in res['rows']}
    gz = np.load(Path(a.grid) / 'W_grid.npz') if a.grid else None
    out = dict(run=str(run), refs=a.refs, result_sha256=hashlib.sha256((run / 'result.json').read_bytes()).hexdigest(),
               provisional='reference-limited (BE reference at dt0/4)', settings={}, gates={})
    repro = dict(compared=0, failures=0, worst=0., applicable=(a.refs == '3d_first_order'))
    grepro = dict(compared=0, failures=0, worst=0.)
    for s in cfg['settings']:
        sname, Rp = s['name'], int(s['Rp'])
        rr = dict(r=s['rule'], conv=s['conv'], finer=s['finer'])
        bk = C3.load_bank(s['bank'])
        G = np.asarray(C3.TB.feature_rows(bk['params'], bk['T'][:, :Rp], X65))
        Qb, _ = np.linalg.qr(G.T)

        def traj(rule, spec):
            key = f'{sname}|{rule}|{sk(spec)}'
            if key not in Wz.files:
                return None
            ids = [int(x) for x in Wz[f'cases|{key}']]
            return dict(zip(ids, Wz[key])), key

        n0 = {}
        allids = sorted({int(x) for k in Wz.files if k.startswith(f'cases|{sname}|') for x in Wz[k]})
        for j in allids:
            n0[j] = float(np.linalg.norm(np.asarray(C3.C.initial_interior(65, tab, j))[i65]))
        dec = lambda W, j: (W @ G) / n0[j]
        valid = lambda key, j: bool(rows.get((sname, key.split('|')[1], key.split('|', 2)[2], j), {}).get('valid'))
        # reproduction of every row (original reference only)
        if repro['applicable']:
            for key in [k for k in Wz.files if k.startswith(f'{sname}|')]:
                ids = [int(x) for x in Wz[f'cases|{key}']]
                for ci, j in enumerate(ids):
                    r = rows.get((sname, key.split('|')[1], key.split('|', 2)[2], j))
                    if r is None:
                        continue
                    F = Wz[key][ci] @ G
                    R = np.asarray(z[f'c{j}'])
                    for k in range(6):
                        x, y = float(np.linalg.norm(F[k] - R[k])) / n0[j], (np.nan if r['e_ref_per_time'][k] is None else r['e_ref_per_time'][k])
                        repro['compared'] += 1
                        if np.isfinite(x) and np.isfinite(y):
                            repro['worst'] = max(repro['worst'], abs(x - y))
                        if not (np.isfinite(x) and np.isfinite(y)):
                            repro['invalid'] = repro.get('invalid', 0) + 1
                        repro['failures'] += 0 if (np.isfinite(x) and np.isfinite(y) and abs(x - y) <= 1e-13 + 1e-10 * abs(y)) \
                            or (not np.isfinite(x) and not np.isfinite(y) and not r['valid']) else 1
        # converged-rule check: every r_inf run type vs the finer rule on the cases where both exist
        conv_d, conv_ok, missing = 0., True, []
        budget_ids = list(range(cfg['budget_cases'])) if not cfg.get('case_subset') else list(cfg['case_subset'])[:cfg['budget_cases']]
        plan_conv = sorted({sk(sp) for role, sp in C3R.bud_specs(s['arms']) if role == 'conv'})
        for spec_key in plan_conv:                         # the required inventory comes from the plan, not the archive
            key = f"{sname}|{rr['conv']}|{spec_key}"
            fk = f"{sname}|{rr['finer']}|{spec_key}"
            if key not in Wz.files:
                missing.append(spec_key + ' (r_inf run)')
                conv_ok = False
                continue
            if fk not in Wz.files:                          # every r_inf run type needs its finer counterpart (A2.4)
                missing.append(spec_key)
                conv_ok = False
                continue
            A_, B_ = dict(zip([int(x) for x in Wz[f'cases|{key}']], Wz[key])), dict(zip([int(x) for x in Wz[f'cases|{fk}']], Wz[fk]))
            if not set(budget_ids) <= (set(A_) & set(B_)):
                missing.append(spec_key + ' (cases)')
                conv_ok = False
            for j in set(A_) & set(B_):
                d = float((np.linalg.norm(dec(A_[j] - B_[j], j), axis=1))[1:].max())
                conv_ok = conv_ok and valid(key, j) and valid(fk, j) and np.isfinite(d)
                conv_d = max(conv_d, d) if np.isfinite(d) else float('inf')
        arms = {}
        for x in s['arms']:
            spec = tuple(x['spec'])
            sc_, dt, K = ('BE', DT0, 1) if spec[0] == 'vendor' else spec[1:]
            names = dict(a=(rr['r'], spec), conv=(rr['conv'], spec), tight=(rr['r'], ('ad', 'LSPG', sc_, dt, 'tight')),
                         tgal=(rr['r'], ('ad', 'GAL', sc_, dt, 'tight')), time=(rr['r'], ('fs', 'BDF2', H8, K)),
                         half=(rr['r'], ('ad', 'LSPG', sc_, dt / 2, 'tight')), c1=(rr['conv'], ('fs', 'BDF2', H8, K)),
                         c2=(rr['r'], ('ad', 'LSPG', 'BDF2', H8, 'tight')), finer=(rr['finer'], spec),
                         star=(rr['conv'], ('ad', 'GAL', 'BDF2', H8, 'tight')), star16=(rr['conv'], ('ad', 'GAL', 'BDF2', H16, 'tight')),
                         starx=(rr['conv'], ('ad', 'GAL', 'BDF2', H8, 'tighter')),
                         star16x=(rr['conv'], ('ad', 'GAL', 'BDF2', H16, 'tighter')),
                         starR=(rr['r'], ('ad', 'GAL', 'BDF2', H8, 'tight')), starL=(rr['conv'], ('ad', 'LSPG', 'BDF2', H8, 'tight')))
            T = {nm: traj(*v) for nm, v in names.items()}
            per = []
            for j in sorted(T['a'][0]) if T['a'] else []:
                U = {nm: dec(t[0][j], j) for nm, t in T.items() if t and j in t[0]}
                ok = {nm: valid(t[1], j) for nm, t in T.items() if t and j in t[0]}
                need = lambda *ns: all(ok.get(n_, False) for n_ in ns)
                nt = lambda X: [float(np.linalg.norm(X[k])) for k in range(6)]
                NA = [None] * 6
                d = dict(case=j)
                d['delta_quad'] = nt(U['a'] - U['conv']) if need('a', 'conv') else NA
                d['finalist_finer'] = nt(U['conv'] - U['finer']) if need('conv', 'finer') else NA
                d['delta_solve'] = nt(U['a'] - U['tight']) if need('a', 'tight') else NA
                d['delta_form'] = nt(U['tight'] - U['tgal']) if need('tight', 'tgal') else NA
                d['delta_time'] = nt(U['a'] - U['time']) if need('a', 'time') else NA
                d['refine'] = nt(U['tight'] - U['half']) if need('tight', 'half') else NA
                d['D_a'] = nt(U['a'] - U['star']) if need('a', 'star') else NA
                d['I_time_quad'] = nt(U['c1'] - U['conv'] - U['time'] + U['a']) if need('c1', 'conv', 'time', 'a') else NA
                d['I_time_solve'] = nt(U['c2'] - U['tight'] - U['time'] + U['a']) if need('c2', 'tight', 'time', 'a') else NA
                d['ctx_quad_ideal_end'] = nt(U['star'] - U['starR']) if need('star', 'starR') else NA
                d['ctx_form_ideal_end'] = nt(U['star'] - U['starL']) if need('star', 'starL') else NA
                d['U_star'] = ([float(np.linalg.norm(U['star'][k] - U['star16'][k]) + np.linalg.norm(U['star'][k] - U['starx'][k]) +
                                      np.linalg.norm(U['star16'][k] - U['star16x'][k])) for k in range(6)]
                               if need('star', 'star16', 'starx', 'star16x') else NA)
                Rn = np.asarray(z[f'c{j}']) / n0[j]
                d['e_tot'] = nt(U['a'] - Rn) if need('a') else NA
                if need('star'):
                    PR = (Rn @ Qb) @ Qb.T
                    eP, edyn, es = (np.linalg.norm(X, axis=1) for X in (Rn - PR, U['star'] - PR, U['star'] - Rn))
                    d.update(e_P=eP.tolist(), e_dyn=edyn.tolist(), e_star=es.tolist(),
                             B0=[float(abs(es[k] ** 2 - eP[k] ** 2 - edyn[k] ** 2) / max(es[k] ** 2, 1e-300)) for k in range(1, 6)],
                             B0_mismatch=float(abs(es[5] ** 2 - (eP[5] / .1) ** 2 - edyn[5] ** 2)))       # pinned A3.5(iii)
                    ts = 1 + int(np.argmax(es[1:]))
                    d['orthogonal_pair_at_tstar'] = dict(t_star=ts, e_star=float(es[ts]), e_P=float(eP[ts]), e_dyn=float(edyn[ts]))
                    if d['D_a'][1] is not None and d['e_tot'][1] is not None:
                        d['triangle_ok'] = all(abs(d['e_tot'][k] - es[k]) <= d['D_a'][k] + 1e-14 for k in range(1, 6))
                if gz is not None and f"{sname}|{rr['r']}|{sk(spec)}" in gz.files:
                    gids = [int(v) for v in gz[f"cases|{sname}|{rr['r']}|{sk(spec)}"]] if f"cases|{sname}|{rr['r']}|{sk(spec)}" in gz.files \
                        else list(range(gz[f"{sname}|{rr['r']}|{sk(spec)}"].shape[0]))
                    if j in gids:
                        dg = float(np.linalg.norm(dec(gz[f"{sname}|{rr['r']}|{sk(spec)}"][gids.index(j)], j) - U['a'], axis=1)[1:].max())
                        grepro['compared'] += 1
                        grepro['worst'] = max(grepro['worst'], dg)
                        grepro['failures'] += 0 if dg <= 1e-8 else 1
                per.append(d)
            mx = lambda v: None if v is None or v[1] is None else max(v[1:])
            fin_per = per                                         # finalist: every case (A3.6)
            per = [p_ for p_ in per if p_['case'] in budget_ids]  # budget statistics: the budget population only (N9)
            summ = {}
            for comp in ('delta_quad', 'delta_solve', 'delta_form', 'delta_time', 'refine', 'D_a', 'I_time_quad', 'I_time_solve',
                         'U_star', 'e_tot', 'e_P', 'e_dyn', 'e_star'):
                v = [mx(p_.get(comp)) for p_ in per if mx(p_.get(comp)) is not None]
                summ[comp] = dict(median=float(np.median(v)) if v else None, worst=float(max(v)) if v else None,
                                  available=len(v), cases=len(per))
            fin = None
            if x.get('selected'):
                fq = [mx(p_['delta_quad']) for p_ in fin_per]
                ff = [mx(p_['finalist_finer']) for p_ in fin_per]
                fin = dict(cases=len(fin_per), delta_quad_worst=None if None in fq else max(fq),
                           finer_worst=None if None in ff else max(ff))
                fin['qualified'] = bool(fin['delta_quad_worst'] is not None and fin['delta_quad_worst'] <= TAU and
                                        fin['finer_worst'] is not None and fin['finer_worst'] <= BAR and
                                        len(fin_per) == len(res['config'].get('case_subset') or range(cfg['cohort_count'])))
            largest = {}
            for p_ in per:
                comps = {k: mx(p_.get(k)) for k in ('e_P', 'e_dyn', 'delta_quad', 'delta_time', 'delta_form', 'delta_solve')}
                comps = {k: v for k, v in comps.items() if v is not None}
                w = max(comps, key=comps.get) if comps else 'unavailable'
                largest[w] = largest.get(w, 0) + 1
            b0 = [max(p_['B0']) for p_ in per if 'B0' in p_]
            clos, ctxc, unres = [], {'delta_quad': 0, 'delta_form': 0}, {}
            for p_ in per:
                dd = [mx(p_[k]) for k in ('delta_quad', 'delta_solve', 'delta_form', 'delta_time')]
                Dv = mx(p_['D_a'])
                if None not in dd and Dv is not None:
                    sq, sm = float(np.sqrt(sum(v * v for v in dd))), float(sum(dd))
                    clos.append(dict(D_over_rss=(Dv / sq) if sq > 0 else ('both zero' if Dv == 0 else 'undefined'),
                                     D_over_sum=(Dv / sm) if sm > 0 else ('both zero' if Dv == 0 else 'undefined')))
                for comp, ide in (('delta_quad', 'ctx_quad_ideal_end'), ('delta_form', 'ctx_form_ideal_end')):
                    dep_, ide_ = mx(p_[comp]), mx(p_[ide])
                    if dep_ is not None and ide_ is not None and min(dep_, ide_) > 1e-8 and max(dep_, ide_) > 2 * min(dep_, ide_):
                        ctxc[comp] += 1
            for comp in ('delta_quad', 'delta_solve', 'delta_form', 'delta_time', 'refine', 'D_a', 'I_time_quad', 'I_time_solve'):
                pr = [(mx(p_[comp]), mx(p_['U_star'])) for p_ in per]
                unres[comp] = dict(unresolved=sum(1 for v, u in pr if v is not None and u is not None and v < 3 * u),
                                   unavailable=sum(1 for v, u in pr if v is None or u is None))
            arms[sk(spec)] = dict(arm=x, summary=summ, finalist=fin, largest=largest, closure=clos,
                                  context_sensitive_cases=ctxc, unresolved_vs_Ustar=unres,
                                  B0_worst_relative=max(b0) if b0 else None,
                                  B0_mismatch_min_defect=(per[0].get('B0_mismatch') if per else None),     # pinned fixture
                                  per_case=fin_per)
        S_ = res['settings'][sname]
        base_key = 'fs|BE|0.01|K1'
        tc_, tv_ = traj(cfg['control_rule'], ('fs', 'BE', DT0, 1)), traj(rr['conv'], ('fs', 'BE', DT0, 1))
        b2d = []
        for j in budget_ids:
            if tc_ and tv_ and j in tc_[0] and j in tv_[0] and valid(tc_[1], j) and valid(tv_[1], j):
                b2d.append(float(np.linalg.norm(dec(tc_[0][j] - tv_[0][j], j), axis=1)[1:].max()))
            else:
                b2d.append(float('nan'))
        b2 = dict(rule=cfg['control_rule'], per_case=b2d, complete=bool(b2d and np.isfinite(b2d).all()),
                  discriminating=bool(b2d and np.isfinite(b2d).all() and sum(v > TAU for v in b2d) > len(b2d) / 2))
        bt_ = arms.get(base_key)
        dtv = [None if p_['delta_time'][1] is None else max(p_['delta_time'][1:]) for p_ in (bt_['per_case'] if bt_ else [])
               if p_['case'] in budget_ids]
        b3 = dict(exceeds=sum(v > 1e-3 for v in dtv if v is not None), cases=len(dtv), complete=bool(dtv and None not in dtv),
                  discriminating=bool(dtv and None not in dtv and sum(v > 1e-3 for v in dtv) > len(dtv) / 2))
        withheld = []
        if not b2['discriminating']:
            withheld.append('quadrature attribution (B2 control non-discriminating)')
        if not b3['discriminating']:
            withheld.append('time attribution (B3 control non-discriminating)')
        ctl = {}
        tb_ = traj(rr['r'], ('fs', 'BE', DT0, 1))
        if repro['applicable'] and tb_ and len(tb_[0]) > 1:      # G-score controls on the budget archive (A9.6)
            j0, j1 = sorted(tb_[0])[:2]
            r0 = rows[(sname, rr['r'], base_key, j0)]
            R0 = np.asarray(z[f'c{j0}'])
            ee = lambda F_: max(float(np.linalg.norm(F_[k] - R0[k])) / n0[j0] for k in range(1, 6))
            F0 = tb_[0][j0] @ G
            ctl = dict(case_swap=abs(ee(tb_[0][j1] @ G) - r0['e_ref']), time_shift=abs(ee(np.roll(F0, 1, 0)) - r0['e_ref']))
            ctl['all_break'] = bool(all(v > 1e-6 for v in ctl.values()))
        if not (conv_ok and conv_d <= BAR):
            withheld.append('quadrature attribution (converged rule not verified against the finer rule)')
        out['settings'][sname] = dict(rules=rr, conv_check=dict(worst=conv_d, all_valid=conv_ok, missing=missing,
                                                                passed=bool(conv_ok and conv_d <= BAR)),
                                      B1=S_.get('B1_sham_bitwise'), B2=b2, B3=b3, gscore_controls=ctl, withheld=withheld, arms=arms)
    b0w = [v['B0_worst_relative'] for s_ in out['settings'].values() for v in s_['arms'].values() if v['B0_worst_relative'] is not None]
    b0m = [v['B0_mismatch_min_defect'] for s_ in out['settings'].values() for v in s_['arms'].values() if v['B0_mismatch_min_defect'] is not None]
    nrows = len(res['rows'])
    ncases_all = len(cfg.get('case_subset') or range(cfg['cohort_count']))
    n_exp = sum((ncases_all if x.get('selected') else int(cfg['budget_cases'])) for s_ in cfg['settings'] for x in s_['arms'])
    import budget2d as B2D
    fx = B2D.b0_fixtures()
    pinned = b0m[0] if b0m else None
    out['gates'] = dict(B0=dict(worst_relative=max(b0w) if b0w else None, passed=bool(b0w and max(b0w) <= 1e-10)),
                        B0_fixtures=fx['passed'],
                        B0_real_mismatch=dict(pinned_defect=pinned, passed=bool(pinned is not None and pinned > 1e-6)),
                        B1=all((s_['B1'] or {}).get('all_equal') for s_ in out['settings'].values()),
                        reproduction=dict(repro, rows=nrows,
                                          passed=(bool(repro['compared'] == 6 * nrows and repro['failures'] == 0 and
                                                       all(s_['gscore_controls'].get('all_break') for s_ in out['settings'].values()))
                                                  if repro['applicable'] else 'not applicable')),
                        grid_reproduction=dict(grepro, expected=n_exp,
                                               passed=(bool(grepro['failures'] == 0 and grepro['compared'] == n_exp) if a.grid
                                                       else 'not run')))
    dest = Path(a.out or HERE / 'checks' / f"budget3d-{run.parents[1].name}-{a.refs}.json")
    dest.write_text(json.dumps(out, default=float) + '\n')
    print(dest, json.dumps(out['gates'], default=float))


if __name__ == '__main__':
    main()
