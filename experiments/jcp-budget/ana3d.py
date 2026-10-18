"""Offline 3D B2 / B7 analysis (DESIGN 4, 5, A1.2, A2.1, A2.2, A3.1, A3.2, A3.5; CPU, NumPy only).

From a pulled b3run output (W_b3.npz, GLAT_<setting>.npy, result.json), the E0 store (3D val64, ST primary) and this
lane's 3D references (T = 0.5 lattice files): per member and case, relative to n0 = ||R(t0)|| on the 63^3 lattice:
  corners 0, Q, T, S, QT, QS, TS, QTS (factorial over Q, T, S); form panel F|S, F|QS, F|TS, F|QTS at the tight end;
  Moebius effects, F effect and its pairs with Q and T at the tight end (F x S undefined, A3.1); floor e_P, ideal
  dynamics (u_star = the star run), U_star, chi / chi*, G-B0, G-B4 (check rule, per corner type, solver-specific
  validity from the job rows), G-B3 (qualify's q3d archives), controls C-Q / C-T / C-S / G-Q / C-F; B7 Q1 / Q2 at T
  and the windows / growth / stability as in ana7.

    python ana3d.py --run runs/<attempt>/archive/output/b3run_<cfg> [--refs <dir with 3d/...>] [--anaref <json>]
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ana2d as A2  # noqa: E402
import ana7 as A7  # noqa: E402

RS = A2.RS
WT = A2.WT
ING3 = ('Q', 'T', 'S')
TAU, CONV = 2.5e-4, 2.5e-5
COHORT = {923801: 'val64'}


def nt(X, n0):
    return np.linalg.norm(np.asarray(X).reshape(len(X), -1), axis=1) / n0


class Basis:
    def __init__(self, G):
        G = np.asarray(G)
        self.G = G.T if G.shape[0] < G.shape[1] else G                   # (63^3, R')
        self.Q, _ = np.linalg.qr(self.G)

    def dec(self, W):
        return np.asarray(W) @ self.G.T

    def proj(self, F):
        return (F @ self.Q) @ self.Q.T


def rows_valid(res, tag):
    """Solver-specific validity of every run of a tag (A3.5): adaptive -> the job's 'valid' (nfail == 0); fs / qn ->
    valid and non-stationary steps <= 1 % pooled over the run's steps."""
    out = {}
    for r in res['rows']:
        if r['tag'] != tag:
            continue
        ok = bool(r['valid'])
        if 'nonstationary' in r:
            ok = ok and r['nonstationary'] <= .01 * max(1, r['steps_n'])
        out[r['case']] = ok
    return out


def b2_member(res, Wz, B, m, cases, refs, star_ok_cases, cfg_chk=()):
    key = m['key']
    b = res['b2'][key]
    names = list(b['corners']) + list(b['form'])
    W = {}
    V = {}
    for nm in names:
        spec = (b['corners'].get(nm) or b['form'].get(nm))['spec']
        tag = f'{key}#{nm}'
        W[nm] = Wz[f'{tag}|{spec}']
        V[nm] = rows_valid(res, tag)
    sname = key.split('|')[0]
    st = {k: (Wz[f'{sname}#{k}|' + res['b2'][f'{sname}#star'][k]['spec']], rows_valid(res, f'{sname}#{k}'))
          for k in ('star', 'star16', 'starx', 'star16x')}
    arch = {}
    for f in glob.glob(f'{WT}/2026-10-18-jcp-qualify/experiments/jcp-qualify/runs/q3d*/archive/output/*/W_q2.npz'):
        z = np.load(f)
        kq = [k for k in z.files if k.startswith(f'{key}#q|')]
        if kq:
            cs = z[f'cases|{kq[0]}']
            for i, c in enumerate(cs):
                arch[int(c)] = z[kq[0]][i]
    per = []
    gb0 = 0.
    for ci, j in enumerate(cases):
        R, n0, eps, elig = refs[j]
        F = {nm: B.dec(W[nm][ci]) for nm in names}
        val = {nm: V[nm].get(j, False) for nm in names}
        PR = B.proj(R)
        eP = nt(R - PR, n0)
        d = dict(case=j, eligible=elig, eps_ref=eps, valid=val, e_P=float(eP[1:].max()), e_P_t0=float(eP[0]))
        for nm in names:
            if val[nm]:
                ev, ed = nt(F[nm] - R, n0), nt(F[nm] - PR, n0)
                gb0 = max(gb0, float((np.abs(ev ** 2 - eP ** 2 - ed ** 2) / np.maximum(ev ** 2, 1e-300))[1:].max()))
        mx = lambda X: float(nt(X, n0)[1:].max())
        fac = {nm: F[nm] for nm in ('0', 'Q', 'T', 'S', 'QT', 'QS', 'TS', 'QTS')}
        eff = A2.mobius(fac, {k: val[k] for k in fac})
        d['main'] = {x: (mx(eff[x]) if x in eff else None) for x in ING3}
        d['pairs'] = {p: (mx(eff[p]) if p in eff else None) for p in ('QT', 'QS', 'TS')}
        d['triple'] = mx(eff['QTS']) if 'QTS' in eff else None
        # form at the tight end and its pairs with Q, T; F x S undefined (A3.1)
        if val['S'] and val['F|S']:
            Ff = F['F|S'] - F['S']
            d['main']['F'] = mx(Ff)
            if val['QS'] and val['F|QS']:
                d['pairs']['FQ'] = mx(F['F|QS'] - F['QS'] - Ff)
            if val['TS'] and val['F|TS']:
                d['pairs']['FT'] = mx(F['F|TS'] - F['TS'] - Ff)
        d['main'].setdefault('F', None)
        for p in ('FQ', 'FT'):
            d['pairs'].setdefault(p, None)
        d['pairs']['FS'] = 'undefined'
        d['e_tot'] = mx(F['0'] - R) if val['0'] else None
        if all(st[k][1].get(j) for k in st) and j in star_ok_cases:
            Fs = {k: B.dec(st[k][0][ci]) for k in st}
            Us = nt(Fs['star'] - Fs['star16'], n0) + nt(Fs['star'] - Fs['starx'], n0) + nt(Fs['star16'] - Fs['star16x'], n0)
            d['U_star'] = float(Us[1:].max())
            d['e_dyn_star'] = mx(Fs['star'] - PR)
            if val['0']:
                d['D_a'] = mx(F['0'] - Fs['star'])
                # closure: u_star - u_0 = (u_QTS - u_0) + (u_star - u_QTS), the first by the Moebius sum
                if 'QTS' in eff and val['QTS']:
                    tot = sum(eff[k] for k in eff)
                    d['closure_rel'] = float(np.linalg.norm(F['QTS'] - F['0'] - tot) / max(np.linalg.norm(F['QTS'] - F['0']), 1e-300))
            if val.get('F|QTS'):
                d['star_vs_FQTS'] = mx(Fs['star'] - F['F|QTS'])            # the same spec run twice (determinism)
            d['chi_star'] = {x: float(2 * np.max(np.abs(np.sum(eff[x] * (Fs['star'] - PR), axis=1)[1:])) / n0 ** 2)
                             for x in ING3 if x in eff}
            if val['0'] and all(x in eff for x in ING3) and d['main'].get('F') is not None and j in cfg_chk:
                d['C_F'] = control_floor3(R, PR, eP, n0, eps or 0., d['U_star'], B, Fs['star'], eff, d)
        else:
            d['U_star'] = d['e_dyn_star'] = None
        if j in arch and val['0']:
            d['G_B3'] = mx(F['0'] - B.dec(arch[j]))
        # G-B4: check-rule replays on the Q-containing corners
        g4 = {}
        for nm in names:
            if 'Q' not in nm:
                continue
            tag = f'{key}#{nm}#check'
            spec = (b['corners'].get(nm) or b['form'].get(nm))['spec']
            kk = f'{tag}|{spec}'
            if kk in Wz.files:
                ids = list(Wz[f'cases|{kk}'])
                if j in ids:
                    vc = rows_valid(res, tag).get(j, False)
                    g4[nm] = mx(B.dec(Wz[kk][ids.index(j)]) - F[nm]) if (vc and val[nm]) else None
        d['G_B4'] = g4
        per.append(d)
    return per, gb0


def control_floor3(R, PR, eP, n0, eps, U, B, Fstar, eff, d):
    """C-F in 3D (A2.2): seed sin(q pi x) sin(q pi y) sin(q pi z) on the 63^3 interior lattice, q = 63, 62, ..."""
    x = np.arange(1, 64) / 64.
    edyn = nt(Fstar - PR, n0)
    mains = [nt(v, n0) for k, v in eff.items() if len(k) == 1]
    R2, ident = R.copy(), []
    for k in range(1, R.shape[0]):
        maxc = max([eP[k], edyn[k]] + [m[k] for m in mains] + ([d['main']['F']] if d['main'].get('F') else []))
        eta = n0 * (10 * maxc + 2 * eps + 3 * U)
        rk = R[k] - PR[k]
        for q in (63, 62, 61, 60):
            s1 = np.sin(q * np.pi * x)
            h = np.einsum('i,j,k->ijk', s1, s1, s1).reshape(-1)
            hp = h - B.Q @ (B.Q.T @ h)
            if np.linalg.norm(hp) < 1e-8 * np.linalg.norm(h):
                continue
            nr = np.linalg.norm(rk)
            v = hp - (hp @ rk) / nr ** 2 * rk if nr > 0 else hp
            if np.linalg.norm(v) < 1e-8 * np.linalg.norm(hp):
                continue
            v /= np.linalg.norm(v)
            break
        else:
            return dict(available=False)
        R2[k] = R[k] + eta * v
        eP2 = np.linalg.norm(R2[k] - B.Q @ (B.Q.T @ R2[k])) / n0
        want = np.sqrt(eP[k] ** 2 + (eta / n0) ** 2)
        ident.append(abs(eP2 - want) / want)
    PR2 = B.proj(R2)
    d2 = dict(d, e_P=float(nt(R2 - PR2, n0)[1:].max()), e_dyn_star=float(nt(Fstar - PR2, n0)[1:].max()))
    lab = largest3(d2)
    return dict(available=True, identity_worst=float(max(ident)), label=lab,
                fired=bool(max(ident) <= 1e-10 and lab == 'e_P'))


def largest3(d):
    U, eps = d.get('U_star'), d['eps_ref'] or 0.
    cand = {'e_P': ('ref', d['e_P']), 'e_dyn_star': ('ref', d['e_dyn_star']),
            **{x: ('traj', d['main'].get(x)) for x in ('Q', 'T', 'S', 'F')}}
    if U is None or any(v is None for _, v in cand.values()):
        return 'incomplete'
    res = {k: (v >= 3 * U if kd == 'traj' else (v > 2 * eps if k == 'e_P' else (v > 2 * eps and v >= 3 * U)))
           for k, (kd, v) in cand.items()}
    order = sorted(((v, k) for k, (_, v) in cand.items() if res[k]), reverse=True)
    if not order:
        return 'none resolved'
    if len(order) > 1:
        (va, ka), (vb, kb) = order[:2]
        ta, tb = cand[ka][0], cand[kb][0]
        tau = 3 * U if ta == tb == 'traj' else (2 * eps if ta == tb == 'ref' else 2 * eps + 3 * U)
        if va - vb <= tau:
            return 'tied'
    return order[0][1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--refs')
    ap.add_argument('--anaref')
    ap.add_argument('--minpop', type=int, default=8)
    ap.add_argument('--out')
    a = ap.parse_args()
    run = Path(a.run)
    res = json.loads((run / 'result.json').read_text())
    cfg = res['config']
    Wz = np.load(run / 'W_b3.npz')
    coh = COHORT[int(cfg['cohort_seed'])]
    cases = list(cfg['case_subset'])
    refs = {}
    for j in cases:
        R, meta = RS.load(3, coh, j)
        st = RS.reference_status(3, coh, j)
        refs[j] = (np.asarray(R), float(meta['judge']['n0']), st['eps_ref'], st['eligible'])
    out = dict(run=str(run), members={}, gates={}, controls={}, b7={})
    gb0 = 0.
    bases = {}
    for m in cfg['members']:
        sname = m['key'].split('|')[0]
        if sname not in bases:
            bases[sname] = Basis(np.load(run / f'GLAT_{sname}.npy'))
        B = bases[sname]
        if 'b2' in cfg:
            # star family check (G-B4 for u_star)
            sres = res['b2'][f'{sname}#star']
            sok = {}
            for k in ('star', 'star16', 'starx', 'star16x'):
                spec = sres[k]['spec']
                kk, kc = f'{sname}#{k}|{spec}', f'{sname}#{k}#check|{spec}'
                ids_c = list(Wz[f'cases|{kc}'])
                vq, vc = rows_valid(res, f'{sname}#{k}'), rows_valid(res, f'{sname}#{k}#check')
                for i, j in enumerate(ids_c):
                    ci = cases.index(j)
                    ok = vq.get(j) and vc.get(j) and float(np.max(nt(B.dec(Wz[kc][i]) - B.dec(Wz[kk][ci]), refs[j][1])[1:])) <= CONV
                    sok[k] = sok.get(k, True) and bool(ok)
            star_ok = bool(sok) and all(sok.values())
            per, g0 = b2_member(res, Wz, B, m, cases, refs, set(cases) if star_ok else set(),
                                set(cfg['b2'].get('check_cases', range(4))))
            gb0 = max(gb0, g0)
            labs = [largest3(d) for d in per if d['eligible']]
            comp = [l for l in labs if l != 'incomplete']
            el = [d for d in per if d['eligible']]
            if len(el) < a.minpop:
                dom = 'inconclusive (population)'
            elif len(comp) < .75 * len(el):
                dom = 'inconclusive (incomplete)'
            else:
                top = max(set(comp), key=comp.count) if comp else None
                dom = top if (top and top not in ('tied', 'none resolved') and comp.count(top) > .5 * len(comp)) else 'no single dominant'
            g4 = {}
            for d in per:
                for c, v in d['G_B4'].items():
                    g4.setdefault(c, []).append(v)
            out['members'][m['key']] = dict(per_case=per, counts={l: labs.count(l) for l in set(labs)}, dominant=dom,
                                            G_B4={c: bool(all(v is not None and v <= CONV for v in vs)) for c, vs in g4.items()},
                                            star_family_check=sok, ideal_end='verified' if star_ok else 'quadrature-unverified')
    out['gates']['G_B0'] = dict(worst=gb0, passed=bool(gb0 <= 1e-10))
    gb3 = [d['G_B3'] for v in out['members'].values() for d in v['per_case'] if 'G_B3' in d]
    out['gates']['G_B3'] = dict(worst=max(gb3) if gb3 else None, n=len(gb3), passed=bool(gb3 and max(gb3) <= 1e-8))
    # controls
    if 'controls' in cfg:
        B = bases.get('W256') or Basis(np.load(run / 'GLAT_W256.npy'))
        ids = list(cfg['controls'].get('cases', range(4)))
        def W_(tag, spec):
            k = f'{tag}|{spec}'
            return Wz[k], list(Wz[f'cases|{k}'])
        cq, cqi = W_('CTRL#Q#ctrl', 'fs|CN|0.01|K1')
        qi, _ = W_('CTRL#Q#qinf', 'fs|CN|0.01|K1')
        dq = [float(np.max(nt(B.dec(cq[i]) - B.dec(qi[i]), refs[j][1])[1:])) for i, j in enumerate(cqi)]
        out['controls']['C_Q'] = dict(dist=dq, fired=bool(sum(x > TAU for x in dq) >= 1))
        tb, tbi = W_('CTRL#T', 'fs|BE|0.01|K1')
        tt, _ = W_('CTRL#T', 'fs|BDF2|0.00125|K1')
        dt_ = [float(np.max(nt(B.dec(tb[i]) - B.dec(tt[i]), refs[j][1])[1:])) for i, j in enumerate(tbi)]
        out['controls']['C_T'] = dict(dist=dt_, fired=bool(sum(x > 1e-3 for x in dt_) > len(dt_) / 2))
        s0, s0i = W_('CTRL#S#q0', 'qn|BDF2|0.025|k4|q0')
        st_, _ = W_('CTRL#S#tight', 'ad|LSPG|BDF2|0.025|tight')
        ds = [float(np.max(nt(B.dec(s0[i]) - B.dec(st_[i]), refs[j][1])[1:])) for i, j in enumerate(s0i)]
        out['controls']['C_S'] = dict(dist=ds, fired=bool(sum(x > TAU for x in ds) > len(ds) / 2))
        out['controls']['G_Q'] = res['controls'].get('G_Q')
    cf = [d['C_F'] for v in out['members'].values() for d in v['per_case'] if d.get('C_F', {}).get('available')]
    out['controls']['C_F'] = dict(n=len(cf), fired=bool(cf and all(x['fired'] for x in cf)),
                                  labels=sorted({x['label'] for x in cf}))
    # B7 (3D): Q1 / Q2 at T and windows against this lane's references
    for mk, v in res.get('b7', {}).items():
        sname = mk.split('|')[0]
        B = bases[sname]
        T = v['horizon']
        g = lambda nm: (Wz[f"{mk}#B7T{T:g}#{nm}|" + spec_of(mk, nm)], list(Wz[f"cases|{mk}#B7T{T:g}#{nm}|" + spec_of(mk, nm)]))
        P, Pi = g('prod')
        Tt, _ = g('tight')
        rows = []
        for i, j in enumerate(Pi):
            n0 = refs[j][1]
            row = dict(case=j, valid=v['valid'][i], tight_valid=v['tight_valid'][i],
                       dist_tight=float(np.max(nt(B.dec(P[i]) - B.dec(Tt[i]), n0)[1:])))
            if a.refs:
                nr = A7.load_newref3(a.refs, coh, j, T) if hasattr(A7, 'load_newref3') else None
                if nr is not None:
                    meta, Rn = nr
                    n0n = float(np.linalg.norm(Rn[0]))
                    F = B.dec(P[i])
                    et = [float(np.linalg.norm(F[k] - Rn[k])) / n0n for k in range(len(F))]
                    row.update(e_t_new=et, e_W1=max(et[1:6]), e_W2=max(et[6:11]),
                               E_rom=[.5 * float(np.sum(F[k] ** 2)) for k in range(len(F))])
            rows.append(row)
        TT, TTi = g('tighter')
        dtt = [float(np.max(nt(B.dec(Tt[Pi.index(j)]) - B.dec(TT[i]), refs[j][1])[1:])) for i, j in enumerate(TTi)]
        QP, QPi = g('qp')
        dqp = [float(np.max(nt(B.dec(P[Pi.index(j)]) - B.dec(QP[i]), refs[j][1])[1:])) for i, j in enumerate(QPi)]
        QQ, QQi = g('qpp')
        dqq = [float(np.max(nt(B.dec(QP[QPi.index(j)]) - B.dec(QQ[i]), refs[j][1])[1:])) for i, j in enumerate(QQi)]
        rho = [x for rr in v['rho'] for x in rr[1:]]
        q1 = dict(all_valid=all(v['valid']), nonstationary_fraction=v['nonstationary'] / max(1, v['steps']),
                  worst_dist_tight=max(r_['dist_tight'] for r_ in rows), worst_tt=max(dtt) if dtt else None,
                  tight_valid=all(v['tight_valid']), tighter_valid=all(v['tighter_valid']))
        q1['pass'] = bool(q1['all_valid'] and q1['tight_valid'] and q1['tighter_valid'] and q1['nonstationary_fraction'] <= .01
                          and q1['worst_dist_tight'] <= TAU and q1['worst_tt'] is not None and q1['worst_tt'] <= CONV)
        q2 = dict(worst_dist_qp=max(dqp) if dqp else None, rho_max=max(rho) if rho else None,
                  worst_qp_qpp=max(dqq) if dqq else None, valid=all(v['qp_valid']) and all(v['qpp_valid']))
        ok = q2['valid'] and q2['worst_dist_qp'] is not None and q2['worst_dist_qp'] <= TAU and q2['rho_max'] is not None \
            and np.isfinite(q2['rho_max']) and q2['rho_max'] <= .116
        q2['label'] = 'pass' if (ok and q2['worst_qp_qpp'] is not None and q2['worst_qp_qpp'] <= TAU) else \
            ('unverified (finer rule not converged)' if ok else 'fail')
        out['b7'][mk] = dict(Q1=q1, Q2=q2, qualified_at_T=bool(q1['pass'] and q2['label'] == 'pass'), rows=rows)
    dest = Path(a.out or HERE / 'checks' / f"b3-{run.parents[1].name}.json")
    dest.write_text(json.dumps(out, indent=1, default=float) + '\n')
    print(dest, json.dumps(out['gates']), json.dumps({k: v.get('fired') for k, v in out['controls'].items() if isinstance(v, dict)}))


def spec_of(mk, nm):
    import s3common as S
    sname, rule, *spec = mk.split('|')
    sp = S.parse_key('|'.join(spec))
    if nm in ('tight', 'tighter'):
        return S.spec_key(('ad', sp[1], sp[2], nm))
    return S.spec_key(sp)


if __name__ == '__main__':
    main()
