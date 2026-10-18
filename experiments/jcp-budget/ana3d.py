"""Offline 3D B2 / B7 analysis (DESIGN 4, 5, A1.2, A2.1, A2.2, A2.4, A3.1, A3.2, A3.5; CPU, NumPy only).

From a pulled b3run output (W_b3.npz, GLAT_<setting>.npy = the lattice basis used in-job, result.json), the E0 store
(3D val64, ST primary, identity-checked) and this lane's 3D references (manifest- and identity-checked), per member and
case, relative to n0 = ||R(t0)|| on the 63^3 lattice:

  pass 1  run acceptance (A3.5): adaptive runs -> the job's validity (nfail == 0); fs / qn runs -> validity and
          non-stationary steps <= 1 % POOLED over the run type's steps (all cases of that tag); G-B4: check-rule replays
          of every Q-containing corner type and of the u_star family; failed corner types are invalid on every case
  pass 2  corners 0, Q, T, S, QT, QS, TS, QTS; form panel F|S, F|QS, F|TS, F|QTS (tight end); Moebius effects and the
          G-B2 closure; F effect and its pairs with Q, T (F x S undefined, A3.1); e_P, e_dyn*, U_star, D_a, signed
          chi / chi* / background; G-B0 with counts; G-B3 (qualify q3d archives, coverage); C-F
  decide  largest / dominance / pair labels with the 2D rules (A1.4, A2.4, A3.2)
  B7      Q1 / Q2 at T (pooled certification, q+ vs q++ convergence, explicit non-finite handling), G-L1 (first six
          outputs vs the archives), stability, growth (tiers from anaref), front order
  controls C-Q / C-T / C-S (comparator validity), G-Q, C-score (identity errors on the scoring path)

    python ana3d.py --run <b3run output> --refs <dir with 3d/...> --manifest <sha256 file> --anaref <json> [--minpop 8]
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
ING3 = ('Q', 'T', 'S', 'F')
TAU, CONV = 2.5e-4, 2.5e-5
COHORT = {923801: 'val64'}
METRIC3 = RS.V.METRIC_ID[3]
STARS = ('star', 'star16', 'starx', 'star16x')


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


def load_e0_3d(j, coh='val64', time_tag=None):
    R, meta = RS.load(3, coh, int(j))
    if (meta['dim'], meta['cohort'], meta['case'], meta['kind']) != (3, coh, int(j), 'ST'):
        raise A2.IdentityError('3D reference identity')
    if time_tag is not None and time_tag != A2.TIME_TAG:
        raise A2.IdentityError('time shift')
    st = RS.reference_status(3, coh, int(j))
    return np.asarray(R), float(meta['judge']['n0']), st['eps_ref'], st['eligible']


def tag_acceptance(res):
    """{tag: {case: accepted}} with the A3.5 solver-specific rule (pooled non-stationary fraction per tag)."""
    by = {}
    for r in res['rows']:
        by.setdefault(r['tag'], []).append(r)
    out = {}
    for tag, rows in by.items():
        cert = [r for r in rows if 'nonstationary' in r]
        pooled_ok = True
        if cert:
            pooled_ok = sum(r['nonstationary'] for r in cert) <= .01 * max(1, sum(r['steps_n'] for r in cert))
            if any(not np.isfinite(g) for r in cert for g in (r.get('gam_cert') or [])):
                pooled_ok = False
        out[tag] = {r['case']: bool(r['valid'] and pooled_ok) for r in rows}
    return out


def gb4(res, Wz, B, key, names, b, refs, ACC, corners_spec):
    out = {}
    for nm in names:
        if 'Q' not in nm:
            continue
        spec = corners_spec[nm]
        kk, k0 = f'{key}#{nm}#check|{spec}', f'{key}#{nm}|{spec}'
        if kk not in Wz.files:
            out[nm] = dict(passed=False, worst=None, note='no check run')
            continue
        ids_c, ids_0 = list(Wz[f'cases|{kk}']), list(Wz[f'cases|{k0}'])
        vals = []
        for i, j in enumerate(ids_c):
            ok = ACC.get(f'{key}#{nm}#check', {}).get(j) and ACC.get(f'{key}#{nm}', {}).get(j)
            vals.append(float(np.max(nt(B.dec(Wz[kk][i]) - B.dec(Wz[k0][ids_0.index(j)]), refs[j][1])[1:])) if ok else None)
        out[nm] = dict(worst=max([v for v in vals if v is not None], default=None),
                       passed=bool(vals and all(v is not None and v <= CONV for v in vals)))
    return out


def star_check(res, Wz, B, sname, refs, ACC):
    sres = res['b2'][f'{sname}#star']
    vals = []
    for k in STARS:
        spec = sres[k]['spec']
        kk, kc = f'{sname}#{k}|{spec}', f'{sname}#{k}#check|{spec}'
        ids0, idc = list(Wz[f'cases|{kk}']), list(Wz[f'cases|{kc}'])
        for i, j in enumerate(idc):
            ok = ACC.get(f'{sname}#{k}', {}).get(j) and ACC.get(f'{sname}#{k}#check', {}).get(j)
            vals.append(float(np.max(nt(B.dec(Wz[kc][i]) - B.dec(Wz[kk][ids0.index(j)]), refs[j][1])[1:])) if ok else None)
    return dict(worst=max([v for v in vals if v is not None], default=None),
                passed=bool(vals and all(v is not None and v <= CONV for v in vals)))


def archive_q3d(key):
    arch, srcs = {}, []
    for f in glob.glob(f'{WT}/2026-10-18-jcp-qualify/experiments/jcp-qualify/runs/q3d*/archive/output/*/W_q2.npz'):
        z = np.load(f)
        kq = [k for k in z.files if k.startswith(f'{key}#q|')]
        if kq:
            srcs.append(f)
            for i, c in enumerate(z[f'cases|{kq[0]}']):
                arch[int(c)] = z[kq[0]][i]
    return arch, srcs


def b2_member(res, Wz, B, m, cases, refs, ACC, failed, star_ok, chk):
    key = m['key']
    b = res['b2'][key]
    names = list(b['corners']) + list(b['form'])
    spec_of = {nm: (b['corners'].get(nm) or b['form'].get(nm))['spec'] for nm in names}
    W = {nm: Wz[f'{key}#{nm}|{spec_of[nm]}'] for nm in names}
    ids_of = {nm: list(Wz[f'cases|{key}#{nm}|{spec_of[nm]}']) for nm in names}
    sname = key.split('|')[0]
    sres = res['b2'][f'{sname}#star']
    st = {k: (Wz[f'{sname}#{k}|' + sres[k]['spec']], list(Wz[f'cases|{sname}#{k}|' + sres[k]['spec']])) for k in STARS}
    arch, srcs = archive_q3d(key)
    per = []
    S = dict(gb0=0., gb0_n=0, gb2=0., gb2_n=0, gb3=[], gb3_missing=0)
    for j in cases:
        R, n0, eps, elig = refs[j]
        raw = {nm: bool(ACC.get(f'{key}#{nm}', {}).get(j)) for nm in names}
        val = {nm: raw[nm] and nm not in failed for nm in names}
        F = {nm: B.dec(W[nm][ids_of[nm].index(j)]) for nm in names}
        PR = B.proj(R)
        eP = nt(R - PR, n0)
        d = dict(case=j, eligible=elig, eps_ref=eps, valid=val, e_P=float(eP[1:].max()), e_P_t0=float(eP[0]))
        for nm in names:
            if raw[nm]:
                ev, ed = nt(F[nm] - R, n0), nt(F[nm] - PR, n0)
                S['gb0'] = max(S['gb0'], float((np.abs(ev ** 2 - eP ** 2 - ed ** 2) / np.maximum(ev ** 2, 1e-300))[1:].max()))
                S['gb0_n'] += 5
        mx = lambda X: float(nt(X, n0)[1:].max())
        fac = {nm: F[nm] for nm in ('0', 'Q', 'T', 'S', 'QT', 'QS', 'TS', 'QTS')}
        effr = A2.mobius(fac, {k: raw[k] for k in fac})
        if 'QTS' in effr:
            rel = float(np.linalg.norm(F['QTS'] - F['0'] - sum(effr.values())) / max(np.linalg.norm(F['QTS'] - F['0']), 1e-300))
            S['gb2'], S['gb2_n'] = max(S['gb2'], rel), S['gb2_n'] + 1
        eff = A2.mobius(fac, {k: val[k] for k in fac})
        d['main'] = {x: (mx(eff[x]) if x in eff else None) for x in ('Q', 'T', 'S')}
        d['pairs'] = {p: (mx(eff[p]) if p in eff else None) for p in ('QT', 'QS', 'TS')}
        d['triple'] = mx(eff['QTS']) if 'QTS' in eff else None
        Ff = None
        if val['S'] and val['F|S']:
            Ff = F['F|S'] - F['S']
            d['main']['F'] = mx(Ff)
            d['pairs']['QF'] = mx(F['F|QS'] - F['QS'] - Ff) if (val['QS'] and val['F|QS']) else None
            d['pairs']['TF'] = mx(F['F|TS'] - F['TS'] - Ff) if (val['TS'] and val['F|TS']) else None
        else:
            d['main']['F'] = None
            d['pairs']['QF'] = d['pairs']['TF'] = None
        d['pairs']['SF'] = 'undefined (A3.1)'
        d['e_tot'] = mx(F['0'] - R) if val['0'] else None
        sv = {k: ACC.get(f'{sname}#{k}', {}).get(j) for k in STARS}
        if star_ok and all(sv.values()):
            Fs = {k: B.dec(st[k][0][st[k][1].index(j)]) for k in STARS}
            Us = nt(Fs['star'] - Fs['star16'], n0) + nt(Fs['star'] - Fs['starx'], n0) + nt(Fs['star16'] - Fs['star16x'], n0)
            d['U_star'] = float(Us[1:].max())
            d['e_dyn_star'] = mx(Fs['star'] - PR)
            es, ed = nt(Fs['star'] - R, n0), nt(Fs['star'] - PR, n0)
            S['gb0'] = max(S['gb0'], float((np.abs(es ** 2 - eP ** 2 - ed ** 2) / np.maximum(es ** 2, 1e-300))[1:].max()))
            S['gb0_n'] += 5
            if val['0']:
                d['D_a'] = mx(F['0'] - Fs['star'])
                if val.get('QTS'):
                    d['closure_to_star'] = dict(QTS_minus_0=mx(F['QTS'] - F['0']), star_minus_QTS=mx(Fs['star'] - F['QTS']))
            if val.get('F|QTS'):
                d['star_vs_FQTS'] = mx(Fs['star'] - F['F|QTS'])                   # the same specification run twice
            d['chi'], d['chi_star'], d['chi_background'] = {}, {}, {}
            effs = dict(eff)
            if Ff is not None:
                effs['F'] = Ff
            for x in ING3:
                if x in effs and val['0']:
                    ca = 2 * np.sum(effs[x] * (F['0'] - PR), axis=1) / n0 ** 2
                    cs = 2 * np.sum(effs[x] * (Fs['star'] - PR), axis=1) / n0 ** 2
                    k = 1 + int(np.argmax(np.abs(ca[1:])))
                    d['chi'][x] = float(ca[k])
                    d['chi_star'][x] = float(cs[1 + int(np.argmax(np.abs(cs[1:])))])
                    d['chi_background'][x] = float((ca - cs)[k])
            if j in chk and val['0'] and all(x in eff for x in ('Q', 'T', 'S')) and d['main']['F'] is not None:
                d['C_F'] = control_floor3(R, PR, eP, n0, eps or 0., d['U_star'], B, Fs['star'], eff, d)
        else:
            d['U_star'] = d['e_dyn_star'] = None
        if val['0']:
            if j in arch:
                g = mx(F['0'] - B.dec(arch[j]))
                d['G_B3'] = g
                S['gb3'].append(g)
            else:
                S['gb3_missing'] += 1
        per.append(d)
    return per, S, srcs


def control_floor3(R, PR, eP, n0, eps, U, B, Fstar, eff, d):
    """C-F in 3D (A2.2): seed sin(q pi x) sin(q pi y) sin(q pi z) on the 63^3 interior lattice, q = 63, 62, ..."""
    x = np.arange(1, 64) / 64.
    edyn = nt(Fstar - PR, n0)
    mains = [nt(v, n0) for k, v in eff.items() if len(k) == 1]
    R2, ident = R.copy(), []
    for k in range(1, R.shape[0]):
        maxc = max([eP[k], edyn[k]] + [m[k] for m in mains] + [d['main']['F']])
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
    dyn_same = bool(abs(d2['e_dyn_star'] - float(edyn[1:].max())) <= 1e-12 * max(float(edyn[1:].max()), 1e-300))
    return dict(available=True, identity_worst=float(max(ident)), label=lab, dyn_unchanged=dyn_same,
                fired=bool(max(ident) <= 1e-10 and dyn_same and lab == 'e_P'))


def largest3(d):
    U, eps = d.get('U_star'), d['eps_ref'] or 0.
    cand = {'e_P': ('ref', d['e_P']), 'e_dyn_star': ('ref', d['e_dyn_star']), **{x: ('traj', d['main'].get(x)) for x in ING3}}
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


def pair_label3(d, p):
    U = d.get('U_star')
    a, b, v = d['main'].get(p[0]), d['main'].get(p[1]), d['pairs'].get(p)
    if U is None or a is None or b is None or v is None or isinstance(v, str):
        return 'unavailable'
    big = max(a, b)
    if v > .25 * big and v >= 3 * U:
        return 'interacting'
    if v <= .25 * big and big >= 3 * U:
        return 'additive'
    return 'unresolved'


def summarise3(per, minpop):
    el = [d for d in per if d['eligible']]
    lab = [largest3(d) for d in el]
    comp = [l for l in lab if l != 'incomplete']
    out = dict(n_eligible=len(el), n_complete=len(comp), counts={l: lab.count(l) for l in set(lab)})
    if len(comp) < minpop:
        out['dominant'] = 'inconclusive (population)'
    elif len(comp) < .75 * len(el):
        out['dominant'] = 'inconclusive (incomplete)'
    else:
        top = max(set(comp), key=comp.count)
        out['dominant'] = top if (top not in ('tied', 'none resolved') and comp.count(top) > .5 * len(comp)) else 'no single dominant'
    pl = {'SF': 'undefined (A3.1)'}
    for p in ('QT', 'QS', 'TS', 'QF', 'TF'):
        labs = [pair_label3(d, p) for d in el]
        av = [l for l in labs if l != 'unavailable']
        if len(av) < minpop:
            pl[p] = 'inconclusive (population)'
        elif sum(l == 'interacting' for l in labs) > .25 * len(el):
            pl[p] = 'interacting'
        elif sum(l == 'additive' for l in labs) >= .75 * len(el):
            pl[p] = 'additive'
        else:
            pl[p] = 'mixed'
    out['pairs'] = pl
    med = lambda v: float(np.median(v)) if v else None
    out['median_main'] = {x: med([d['main'][x] for d in el if d['main'].get(x) is not None]) for x in ING3}
    for nm in ('chi', 'chi_star', 'chi_background'):
        out[f'median_{nm}'] = {x: med([d[nm][x] for d in el if d.get(nm, {}).get(x) is not None]) for x in ING3}
    for k in ('e_P', 'e_dyn_star', 'U_star', 'D_a', 'e_tot'):
        v = [d[k] for d in el if d.get(k) is not None]
        out[f'median_{k}'], out[f'worst_{k}'] = med(v), (float(max(v)) if v else None)
    return out


def b7_3d(res, Wz, B, mk, v, refs, ACC, refdir, EL, minpop):
    T = v['horizon']
    import s3common as S
    sname, rule, *spec = mk.split('|')
    sp = S.parse_key('|'.join(spec))
    key = S.spec_key(sp)
    tg = lambda nm: f"{mk}#B7T{T:g}#{nm}"
    sk = {'prod': key, 'tight': S.spec_key(('ad', sp[1], sp[2], 'tight')), 'tighter': S.spec_key(('ad', sp[1], sp[2], 'tighter')),
          'qp': key, 'qpp': key}
    g = lambda nm: (Wz[f'{tg(nm)}|{sk[nm]}'], list(Wz[f'cases|{tg(nm)}|{sk[nm]}']))
    P, Pi = g('prod')
    Tt, Ti = g('tight')
    TT, TTi = g('tighter')
    QP, QPi = g('qp')
    QQ, QQi = g('qpp')
    acc = lambda nm, j: bool(ACC.get(tg(nm), {}).get(j))
    arch, srcs = archive_q3d(mk)
    rows = []
    for i, j in enumerate(Pi):
        n0 = refs[j][1]
        F = B.dec(P[i])
        row = dict(case=j, valid=acc('prod', j))
        row['dist_tight'] = float(np.max(nt(F - B.dec(Tt[Ti.index(j)]), n0)[1:])) if acc('tight', j) else float('inf')
        if j in arch:
            row['G_L1'] = float(np.max(nt(F[:6] - B.dec(arch[j]), n0)[1:]))
        row['E_rom'] = [.5 * float(np.sum(F[k] ** 2)) for k in range(len(F))]
        nr = A7.load_newref(refdir, (COHORT[923801], j), T, dim=3) if refdir else None
        if nr is not None:
            meta, Rn = nr
            A7.window_rows(row, F, Rn, meta, len(F), EL, f'{COHORT[923801]}|{j}')
        rows.append(row)
    dtt = [float(np.max(nt(B.dec(Tt[Ti.index(j)]) - B.dec(TT[i]), refs[j][1])[1:])) if (acc('tight', j) and acc('tighter', j))
           else float('inf') for i, j in enumerate(TTi)]
    dqp = [float(np.max(nt(B.dec(P[Pi.index(j)]) - B.dec(QP[i]), refs[j][1])[1:])) if acc('qp', j) else float('inf')
           for i, j in enumerate(QPi)]
    dqq = [float(np.max(nt(B.dec(QP[QPi.index(j)]) - B.dec(QQ[i]), refs[j][1])[1:])) if (acc('qp', j) and acc('qpp', j))
           else float('inf') for i, j in enumerate(QQi)]
    rho = np.asarray([x for rr in v['rho'] for x in rr], float)
    q1 = dict(all_valid=all(r_['valid'] for r_ in rows),
              nonstationary_fraction=v['nonstationary'] / max(1, v['steps']),
              worst_dist_tight=max(r_['dist_tight'] for r_ in rows), worst_tt=max(dtt) if dtt else None)
    q1['pass'] = bool(q1['all_valid'] and q1['nonstationary_fraction'] <= .01 and q1['worst_dist_tight'] <= TAU and
                      q1['worst_tt'] is not None and q1['worst_tt'] <= CONV)
    fin = rho[np.isfinite(rho)]
    q2 = dict(population=len(QPi), worst_dist_qp=max(dqp) if dqp else None, rho_max=float(fin.max()) if fin.size else None,
              rho_nonfinite=int(np.isinf(rho).sum()), rho_trivial=int(np.isnan(rho).sum()), worst_qp_qpp=max(dqq) if dqq else None)
    ok = (q2['worst_dist_qp'] is not None and q2['worst_dist_qp'] <= TAU and q2['rho_nonfinite'] == 0 and
          q2['rho_max'] is not None and q2['rho_max'] <= .116)
    q2['label'] = 'pass' if (ok and q2['worst_qp_qpp'] is not None and q2['worst_qp_qpp'] <= TAU) else \
        ('unverified (finer rule not converged)' if ok else 'fail')
    gl1 = [r_['G_L1'] for r_ in rows if 'G_L1' in r_]
    return dict(Q1=q1, Q2=q2, qualified_at_T=bool(q1['pass'] and q2['label'] == 'pass'), rows=rows,
                G_L1=dict(worst=max(gl1) if gl1 else None, n=len(gl1), expected=len(rows), sources=srcs,
                          passed=bool(gl1 and len(gl1) == len(rows) and max(gl1) <= 1e-8)),
                stability=A7.stability_of(rows, q1['all_valid']), growth=A7.growth_of(rows, len(P[0]), minpop))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--refs')
    ap.add_argument('--manifest')
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
    refs = {j: load_e0_3d(j, coh) for j in cases}
    if a.manifest:
        A7.MAN.update(A7.load_manifest(a.manifest))
    EL = A7.elig_of(json.loads(Path(a.anaref).read_text()), float(cfg['b7']['horizon']), dim=3) if (a.anaref and 'b7' in cfg) else {}
    ACC = tag_acceptance(res)
    out = dict(run=str(run), members={}, gates={}, controls={}, b7={})
    bases = {}
    S = dict(gb0=0., gb0_n=0, gb2=0., gb2_n=0, gb3=[], gb3_missing=0)
    chk = set(cfg.get('b2', {}).get('check_cases', range(4)))
    for m in cfg['members']:
        sname = m['key'].split('|')[0]
        if sname not in bases:
            bases[sname] = Basis(np.load(run / f'GLAT_{sname}.npy'))
        B = bases[sname]
        if 'b2' not in cfg:
            continue
        b = res['b2'][m['key']]
        names = list(b['corners']) + list(b['form'])
        spec_of = {nm: (b['corners'].get(nm) or b['form'].get(nm))['spec'] for nm in names}
        g4 = gb4(res, Wz, B, m['key'], names, cfg['b2'], refs, ACC, spec_of)
        sc = star_check(res, Wz, B, sname, refs, ACC)
        failed = {nm for nm, v in g4.items() if not v['passed']}
        per, s_, srcs = b2_member(res, Wz, B, m, cases, refs, ACC, failed, sc['passed'], chk)
        for k in ('gb0', 'gb2'):
            S[k] = max(S[k], s_[k])
            S[k + '_n'] += s_[k + '_n']
        S['gb3'] += s_['gb3']
        S['gb3_missing'] += s_['gb3_missing']
        out['members'][m['key']] = dict(per_case=per, summary=summarise3(per, a.minpop), G_B4=g4, star_family=sc,
                                        failed=sorted(failed), archive_sources=srcs,
                                        ideal_end='verified' if sc['passed'] else 'quadrature-unverified')
    out['gates'] = dict(G_B0=dict(worst=S['gb0'], n=S['gb0_n'], passed=bool(S['gb0_n'] > 0 and S['gb0'] <= 1e-10)),
                        G_B2=dict(worst=S['gb2'], n=S['gb2_n'], passed=bool(S['gb2_n'] > 0 and S['gb2'] <= 1e-12)),
                        G_B3=dict(worst=max(S['gb3']) if S['gb3'] else None, n=len(S['gb3']), missing=S['gb3_missing'],
                                  passed=bool(S['gb3'] and S['gb3_missing'] == 0 and max(S['gb3']) <= 1e-8)))
    if 'controls' in cfg:
        B = bases.get('W256') or Basis(np.load(run / 'GLAT_W256.npy'))

        def W_(tag, spec):
            k = f'{tag}|{spec}'
            return Wz[k], list(Wz[f'cases|{k}'])
        acc = lambda tag, j: bool(ACC.get(tag, {}).get(j))
        cq, cqi = W_('CTRL#Q#ctrl', 'fs|CN|0.01|K1')
        qi, qii = W_('CTRL#Q#qinf', 'fs|CN|0.01|K1')
        dq = [float(np.max(nt(B.dec(cq[i]) - B.dec(qi[qii.index(j)]), refs[j][1])[1:])) if acc('CTRL#Q#qinf', j) else None
              for i, j in enumerate(cqi)]
        out['controls']['C_Q'] = dict(dist=dq, fired=bool(sum(x is not None and x > TAU for x in dq) >= 1))
        tb, tbi = W_('CTRL#T', 'fs|BE|0.01|K1')
        tt, tti = W_('CTRL#T', 'fs|BDF2|0.00125|K1')
        ok_t = acc('CTRL#T', tbi[0]) if tbi else False
        dt_ = [float(np.max(nt(B.dec(tb[i]) - B.dec(tt[tti.index(j)]), refs[j][1])[1:])) for i, j in enumerate(tbi)]
        out['controls']['C_T'] = dict(dist=dt_, comparator_accepted=ok_t, fired=bool(ok_t and sum(x > 1e-3 for x in dt_) > len(dt_) / 2),
                                      fired_any=bool(ok_t and any(x > 1e-3 for x in dt_)))
        s0, s0i = W_('CTRL#S#q0', 'qn|BDF2|0.025|k4|q0')
        stt, sti = W_('CTRL#S#tight', 'ad|LSPG|BDF2|0.025|tight')
        ds = [float(np.max(nt(B.dec(s0[i]) - B.dec(stt[sti.index(j)]), refs[j][1])[1:])) if acc('CTRL#S#tight', j) else None
              for i, j in enumerate(s0i)]
        out['controls']['C_S'] = dict(dist=ds, fired=bool(sum(x is not None and x > TAU for x in ds) > len(ds) / 2),
                                      fired_any=bool(sum(x is not None and x > TAU for x in ds) >= 1))
        out['controls']['G_Q'] = res['controls'].get('G_Q')
        raised = {}
        for nm, kw in (('case_swap', dict(j=cases[0], coh='val64', time_tag=A2.TIME_TAG, want=cases[-1] if len(cases) > 1 else 1)),
                       ('time_shift', dict(j=cases[0], coh='val64', time_tag='t1,t2,t3,t4,t5,t6', want=cases[0]))):
            try:
                R_, *_ = load_e0_3d(kw['j'], kw['coh'], kw['time_tag'])
                if kw['want'] != kw['j']:
                    raise A2.IdentityError('case swap')
                raised[nm] = False
            except A2.IdentityError:
                raised[nm] = True
        out['controls']['C_score'] = dict(raised=raised, fired=bool(all(raised.values())))
    cf = [d['C_F'] for v in out['members'].values() for d in v['per_case'] if d.get('C_F', {}).get('available')]
    out['controls']['C_F'] = dict(n=len(cf), fired=bool(cf and all(x['fired'] for x in cf)), labels=sorted({x['label'] for x in cf}))
    for mk, v in res.get('b7', {}).items():
        out['b7'][mk] = b7_3d(res, Wz, bases[mk.split('|')[0]], mk, v, refs, ACC, a.refs, EL, a.minpop)
    if out['b7']:
        mks = list(out['b7'])
        out['front_order'] = A7.front_order(out['b7'], mks, a.minpop)
    dest = Path(a.out or HERE / 'checks' / f"b3-{run.parents[1].name}.json")
    dest.write_text(json.dumps(out, indent=1, default=float) + '\n')
    print(dest, json.dumps({k: v.get('passed') for k, v in out['gates'].items()}),
          json.dumps({k: (v or {}).get('fired') for k, v in out['controls'].items()}))


if __name__ == '__main__':
    main()
