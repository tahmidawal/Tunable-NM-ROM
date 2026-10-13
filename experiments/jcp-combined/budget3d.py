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
                        x, y = float(np.linalg.norm(F[k] - R[k])) / n0[j], r['e_ref_per_time'][k]
                        repro['compared'] += 1
                        if np.isfinite(x) and np.isfinite(y):
                            repro['worst'] = max(repro['worst'], abs(x - y))
                        repro['failures'] += 0 if (np.isfinite(x) and np.isfinite(y) and abs(x - y) <= 1e-13 + 1e-10 * abs(y)) \
                            or (not np.isfinite(x) and not np.isfinite(y) and not r['valid']) else 1
        # converged-rule check: every r_inf run type vs the finer rule on the cases where both exist
        conv_d, conv_ok = 0., True
        for key in [k for k in Wz.files if k.startswith(f"{sname}|{rr['conv']}|")]:
            spec_key = key.split('|', 2)[2]
            fk = f"{sname}|{rr['finer']}|{spec_key}"
            if fk not in Wz.files:
                continue
            A_, B_ = dict(zip([int(x) for x in Wz[f'cases|{key}']], Wz[key])), dict(zip([int(x) for x in Wz[f'cases|{fk}']], Wz[fk]))
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
                d['e_tot'] = nt(U['a'] - Rn)
                if need('star'):
                    PR = (Rn @ Qb) @ Qb.T
                    eP, edyn, es = (np.linalg.norm(X, axis=1) for X in (Rn - PR, U['star'] - PR, U['star'] - Rn))
                    d.update(e_P=eP.tolist(), e_dyn=edyn.tolist(), e_star=es.tolist(),
                             B0=[float(abs(es[k] ** 2 - eP[k] ** 2 - edyn[k] ** 2) / max(es[k] ** 2, 1e-300)) for k in range(1, 6)],
                             B0_mismatch=float(abs(es[1] ** 2 - (eP[1] / 1.01) ** 2 - edyn[1] ** 2)))
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
            summ = {}
            for comp in ('delta_quad', 'delta_solve', 'delta_form', 'delta_time', 'refine', 'D_a', 'I_time_quad', 'I_time_solve',
                         'U_star', 'e_tot', 'e_P', 'e_dyn', 'e_star'):
                v = [mx(p_.get(comp)) for p_ in per if mx(p_.get(comp)) is not None]
                summ[comp] = dict(median=float(np.median(v)) if v else None, worst=float(max(v)) if v else None,
                                  available=len(v), cases=len(per))
            fin = None
            if x.get('selected'):
                fq = [mx(p_['delta_quad']) for p_ in per]
                ff = [mx(p_['finalist_finer']) for p_ in per]
                fin = dict(cases=len(per), delta_quad_worst=None if None in fq else max(fq), finer_worst=None if None in ff else max(ff))
                fin['qualified'] = bool(fin['delta_quad_worst'] is not None and fin['delta_quad_worst'] <= TAU and
                                        fin['finer_worst'] is not None and fin['finer_worst'] <= BAR and
                                        len(per) == len(res['config'].get('case_subset') or range(cfg['cohort_count'])))
            largest = {}
            for p_ in per:
                comps = {k: mx(p_.get(k)) for k in ('e_P', 'e_dyn', 'delta_quad', 'delta_time', 'delta_form', 'delta_solve')}
                comps = {k: v for k, v in comps.items() if v is not None}
                w = max(comps, key=comps.get) if comps else 'unavailable'
                largest[w] = largest.get(w, 0) + 1
            b0 = [max(p_['B0']) for p_ in per if 'B0' in p_]
            arms[sk(spec)] = dict(arm=x, summary=summ, finalist=fin, largest=largest,
                                  B0_worst_relative=max(b0) if b0 else None,
                                  B0_mismatch_min_defect=min((p_['B0_mismatch'] for p_ in per if 'B0_mismatch' in p_), default=None),
                                  per_case=per)
        S_ = res['settings'][sname]
        withheld = []
        if not (S_.get('B2_control_rule') or {}).get('discriminating'):
            withheld.append('quadrature attribution (B2 control non-discriminating)')
        out['settings'][sname] = dict(rules=rr, conv_check=dict(worst=conv_d, all_valid=conv_ok,
                                                                passed=bool(conv_ok and conv_d <= BAR)),
                                      B1=S_.get('B1_sham_bitwise'), B2=S_.get('B2_control_rule'), withheld=withheld, arms=arms)
    b0w = [v['B0_worst_relative'] for s_ in out['settings'].values() for v in s_['arms'].values() if v['B0_worst_relative'] is not None]
    b0m = [v['B0_mismatch_min_defect'] for s_ in out['settings'].values() for v in s_['arms'].values() if v['B0_mismatch_min_defect'] is not None]
    nrows = len(res['rows'])
    out['gates'] = dict(B0=dict(worst_relative=max(b0w) if b0w else None, passed=bool(b0w and max(b0w) <= 1e-10)),
                        B0_real_mismatch=dict(min_defect=min(b0m) if b0m else None, passed=bool(b0m and min(b0m) > 1e-6)),
                        B1=all((s_['B1'] or {}).get('all_equal') for s_ in out['settings'].values()),
                        reproduction=dict(repro, rows=nrows, passed=(bool(repro['compared'] == 6 * nrows and repro['failures'] == 0)
                                                                     if repro['applicable'] else 'not applicable')),
                        grid_reproduction=dict(grepro, passed=(bool(grepro['compared'] > 0 and grepro['failures'] == 0) if a.grid
                                                               else 'not run')))
    dest = Path(a.out or HERE / 'checks' / f"budget3d-{run.parents[1].name}-{a.refs}.json")
    dest.write_text(json.dumps(out, default=float) + '\n')
    print(dest, json.dumps(out['gates'], default=float))


if __name__ == '__main__':
    main()
