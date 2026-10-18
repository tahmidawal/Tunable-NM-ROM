"""Offline 2D B2 analysis (DESIGN 4, A1.1-A1.6, A2.1-A2.5, A3.1-A3.2; CPU, NumPy only, independent decode npcheck.decode257).

From a pulled b2run output directory (B2_*.npz, B2CHK_*.npz, STAR_*.npz, STARCHK_*.npz, CTRL_b2.npz, result.json) and the
E0 store (refstore, primary reference), per member and case, per output time k = 1..5, relative to n0 = ||R(t0)||:
  corners F_S (decoded), Moebius effects I_S, main effects ||I_x||, pairs ||I_xy||, closure (G-B2), floor e_P,
  ideal dynamics e_dyn* = ||F_star - Pi R||, chi_x = 2<I_x, F_0 - Pi R>, chi*_x = 2<I_x, F_star - Pi R>, U_star,
  D_a, delta_form_tight = ||F_SF - F_S||, the B0 identity for every corner (G-B0), G-B4 (check rule), G-B3 (qualify /
  P1 archive), the controls C-Q, C-S, C-I, C-T, C-F, sham G-B1, C-score; then the A1.4 / A2.4 / A3.2 decision rules.
Every quantity is a max over k = 1..5 unless named otherwise. Writes checks/budget2d-<attempt>-<mesh>.json.

    python ana2d.py --run runs/<attempt>/archive/output/b2run_<cfg> [--cases dev6]
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
import npcheck as NC  # noqa: E402

E0 = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-references/experiments/jcp-references')
sys.path.insert(0, str(E0))
import refstore as RS  # noqa: E402
sys.path.pop(0)
WT = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees')
ING = ('Q', 'T', 'F', 'S', 'I', 'M')
TAU, CONV_BAR = 2.5e-4, 2.5e-5
METRIC_ID = RS.V.METRIC_ID[2]
TIME_TAG = 't0,t1,t2,t3,t4,t5'


class IdentityError(ValueError):
    pass


def score(F, rec):
    """C-score: identity-checked E0 score (ST; S only for LSPG-BE at dt0). rec: case_id, requested (cohort, case),
    time_tag, kind, scheme, dt."""
    coh, c = rec['requested']
    if rec['case_id'] != f'{coh}|{c}':
        raise IdentityError('case swap')
    if rec['time_tag'] != TIME_TAG:
        raise IdentityError('time shift')
    if rec.get('kind', 'ST') == 'S' and not (rec['scheme'] == 'BE' and abs(rec['dt'] - .005) < 1e-15):
        raise IdentityError('reference-kind swap')
    s = RS.score(F, 2, coh, int(c), rec.get('kind', 'ST'))
    if s['metric_id'] != METRIC_ID:
        raise IdentityError('metric')
    return s


def b0_fixtures():
    """combined A3.5 analytic negative fixtures (must break the identity)."""
    defect = lambda v, y, P: float(np.linalg.norm(v - y) ** 2 - np.linalg.norm(y - P @ y) ** 2 - np.linalg.norm(v - P @ y) ** 2)
    e1, e2 = np.array([1., 0.]), np.array([0., 1.])
    f1 = defect(e1 + 1e-3 * e2, e2, np.diag([1., 0.]))
    f2 = defect(e1, e2, np.array([[1., .5], [0., 0.]]))
    return dict(f1=f1, f2=f2, passed=bool(abs(f1 + 2e-3) < 1e-12 and abs(f2 - .5) < 1e-12))


class Span:
    """Orthonormal basis of the bank prefix at the 255^2 interior lattice nodes (Pi of DESIGN 4.2)."""

    def __init__(self, r):
        NC.decode257(np.zeros((1, r)), r)
        G = NC._G257[r].reshape(257, 257, r)[1:-1, 1:-1].reshape(-1, r)
        self.Q, _ = np.linalg.qr(G)

    def proj(self, F):
        """F (n, 257, 257) -> Pi F on the interior (n, 255^2)."""
        X = np.asarray(F)[:, 1:-1, 1:-1].reshape(len(F), -1)
        return (X @ self.Q) @ self.Q.T


def inner(F):
    return np.asarray(F)[:, 1:-1, 1:-1].reshape(len(F), -1)


def nt(X, n0):
    return np.linalg.norm(np.asarray(X).reshape(len(X), -1), axis=1) / n0


def mobius(corners, avail):
    """{subset-key: vector-valued effect} for every subset whose sub-corners are all available."""
    keys = list(corners)
    sets = {k: frozenset('' if k == '0' else k) for k in keys}
    inv = {v: k for k, v in sets.items()}
    out = {}
    for k, S in sets.items():
        if not S:
            continue
        subs = [frozenset(c) for n in range(len(S) + 1) for c in itertools.combinations(sorted(S), n)]
        if all(T in inv and avail[inv[T]] for T in subs):
            out[k] = sum(((-1) ** (len(S) - len(T))) * corners[inv[T]] for T in subs)
    return out


def archive_f257(mesh, member):
    """jcp-qualify / P1 archived fields of the same configuration (G-B3): {case_id: f257 (6, 257, 257)}."""
    r, form, scheme, fac = member['Rp'], member['form'], member['scheme'], float(member['factor'])
    pats = [f'{WT}/2026-10-18-jcp-qualify/experiments/jcp-qualify/runs/*{mesh}/archive/output/*/R_R{r}_{form}_{scheme}_{fac:g}.npz']
    if form == 'LSPG' and fac == 1.:
        pats.append(f'{WT}/2026-10-17-jcp-p1-scaling/experiments/jcp-p1-scaling/runs/acc{mesh}/archive/output/*/R_R{r}_{scheme}.npz')
    out = {}
    for pat in pats:
        for f in sorted(glob.glob(pat)):
            if f'/fb{mesh}/' in f:
                continue                      # fallback configurations use another rule
            z = np.load(f)
            for i, c in enumerate(z['case_ids']):
                if np.isfinite(z['f257'][i]).all():
                    out.setdefault(str(c), np.asarray(z['f257'][i]))
    return out


def analyse_member(run, mk, member, mesh, span, star, refs, ids_keep):
    z = np.load(run / f"B2_{mk.replace('|', '_')}.npz")
    ck = [str(x) for x in z['case_ids']]
    corners = [str(x) for x in z['corners']]
    r = int(member['Rp'])
    arch = archive_f257(mesh, member)
    chk = run / f"B2CHK_{mk.replace('|', '_')}.npz"
    zc = np.load(chk) if chk.exists() else None
    per = []
    gb0, gb2 = 0., 0.
    for ci, cid in enumerate(ck):
        if ids_keep and cid not in ids_keep:
            continue
        R, n0, eps, elig = refs[cid]
        valid = {c: bool(z['valid'][ci, j]) for j, c in enumerate(corners)}
        F = {c: NC.decode257(z['W'][ci, j], r) for j, c in enumerate(corners)}
        PR = span.proj(R)
        RI = inner(R)
        eP = nt(RI - PR, n0)
        d = dict(case=cid, eligible=elig, eps_ref=eps, valid=valid)
        # G-B0 for every valid corner: e_v^2 = e_P^2 + ||v - Pi R||^2 per time
        for c in corners:
            if not valid[c]:
                continue
            ev = nt(inner(F[c]) - RI, n0)
            ed = nt(inner(F[c]) - PR, n0)
            dfc = np.abs(ev ** 2 - eP ** 2 - ed ** 2) / np.maximum(ev ** 2, 1e-300)
            gb0 = max(gb0, float(dfc[1:].max()))
        d['e_tot'] = float(nt(inner(F['0']) - RI, n0)[1:].max()) if valid['0'] else None
        d['e_P'] = float(eP[1:].max())
        d['e_P_t0'] = float(eP[0])
        # Moebius effects
        Fi = {c: inner(F[c]) for c in corners}
        eff = mobius(Fi, valid)
        mx = lambda X: float(nt(X, n0)[1:].max())
        d['main'] = {x: (mx(eff[x]) if x in eff else None) for x in ING}
        d['pairs'] = {''.join(p): (mx(eff[''.join(p)]) if ''.join(p) in eff else None)
                      for p in itertools.combinations(ING, 2)}
        d['higher'] = {k: mx(v) for k, v in eff.items() if len(k) >= 3}
        d['e_t0'] = float(nt(Fi['0'] - RI, n0)[0]) if valid['0'] else None
        if 'I' in Fi and valid.get('I'):
            d['e_t0_ls'] = float(nt(Fi['I'] - RI, n0)[0])
        # closure (G-B2) on the full factorial over QTFSI
        full = 'QTFSI' if 'QTFSI' in corners else None
        if full and all(valid[c] for c in corners if set(c) <= set('QTFSI') or c == '0'):
            tot = sum(v for k, v in eff.items() if set(k) <= set('QTFSI'))
            rel = np.linalg.norm(Fi[full] - Fi['0'] - tot) / max(np.linalg.norm(Fi[full] - Fi['0']), 1e-300)
            gb2 = max(gb2, float(rel))
            d['closure_rel'] = float(rel)
        # ideal end, U_star, D_a, chi, chi*
        si = star['ids'].get(cid)
        if si is not None and all(star['valid'][k][si] for k in ('star', 'star16', 'starx', 'star16x')):
            Fs = {k: inner(NC.decode257(star['W'][k][si], r)) for k in ('star', 'star16', 'starx', 'star16x')}
            Us = nt(Fs['star'] - Fs['star16'], n0) + nt(Fs['star'] - Fs['starx'], n0) + nt(Fs['star16'] - Fs['star16x'], n0)
            d['U_star'] = float(Us[1:].max())
            d['e_dyn_star'] = mx(Fs['star'] - PR)
            es = nt(Fs['star'] - RI, n0)
            ed = nt(Fs['star'] - PR, n0)
            gb0 = max(gb0, float((np.abs(es ** 2 - eP ** 2 - ed ** 2) / np.maximum(es ** 2, 1e-300))[1:].max()))
            if valid['0']:
                d['D_a'] = mx(Fi['0'] - Fs['star'])
            d['chi_star'] = {}
            for x in ING:
                if x in eff:
                    ch = 2 * np.sum(eff[x] * (Fs['star'] - PR), axis=1) / n0 ** 2
                    k = 1 + int(np.argmax(np.abs(ch[1:])))
                    d['chi_star'][x] = float(ch[k])
        else:
            d['U_star'] = d['e_dyn_star'] = None
        d['chi'] = {}
        if valid['0']:
            for x in ING:
                if x in eff:
                    ch = 2 * np.sum(eff[x] * (Fi['0'] - PR), axis=1) / n0 ** 2
                    k = 1 + int(np.argmax(np.abs(ch[1:])))
                    d['chi'][x] = float(ch[k])
        if valid.get('S') and valid.get('FS'):
            d['delta_form_tight'] = mx(Fi['FS'] - Fi['S'])
        # G-B3: deployed corner vs the archived run
        if cid in arch and valid['0']:
            d['G_B3'] = float(max(np.linalg.norm(F['0'][k] - arch[cid][k]) for k in range(1, 6)) / n0)
        # G-B4: check-rule replays
        if zc is not None and cid in [str(x) for x in zc['case_ids']]:
            cj = [str(x) for x in zc['case_ids']].index(cid)
            g4 = {}
            for kk, c in enumerate([str(x) for x in zc['corners']]):
                if zc['valid'][cj, kk] and valid[c]:
                    g4[c] = float(max(np.linalg.norm((NC.decode257(zc['W'][cj, kk], r) - F[c])[k]) for k in range(1, 6)) / n0)
                else:
                    g4[c] = None
            d['G_B4'] = g4
        per.append(d)
    return per, gb0, gb2


def resolved_thresholds(d):
    U = d.get('U_star')
    eps = d['eps_ref'] or 0.
    return U, eps


def largest(d, q_ok=True, star_ok=True):
    """A1.4 / A2.4 / A3.2: the largest candidate of a case with tie thresholds; 'incomplete' if any is unavailable."""
    U, eps = resolved_thresholds(d)
    cand = {'e_P': ('ref', d['e_P']), 'e_dyn_star': ('ref', d['e_dyn_star'] if star_ok else None)}
    for x in ING:
        v = d['main'].get(x)
        if x == 'Q' and not q_ok:
            v = None
        cand[x] = ('traj', v)
    if U is None or any(v is None for _, v in cand.values()):
        return dict(label='incomplete', available={k: v for k, (_, v) in cand.items() if v is not None})
    res = {}
    for k, (kind, v) in cand.items():
        if kind == 'traj':
            res[k] = v >= 3 * U
        elif k == 'e_P':
            res[k] = v > 2 * eps
        else:
            res[k] = v > 2 * eps and v >= 3 * U
    order = sorted(((v, k) for k, (_, v) in cand.items() if res[k]), reverse=True)
    if not order:
        return dict(label='none resolved')
    if len(order) > 1:
        (va, ka), (vb, kb) = order[0], order[1]
        ta, tb = cand[ka][0], cand[kb][0]
        tau = 3 * U if ta == tb == 'traj' else (2 * eps if ta == tb == 'ref' else 2 * eps + 3 * U)
        if va - vb <= tau:
            return dict(label='tied', top=[ka, kb])
    return dict(label=order[0][1])


def pair_label(d, x, y):
    U = d.get('U_star')
    a, b, p = d['main'].get(x), d['main'].get(y), d['pairs'].get(''.join(sorted((x, y), key=ING.index)))
    if U is None or a is None or b is None or p is None:
        return 'unavailable'
    big = max(a, b)
    if p > .25 * big and p >= 3 * U:
        return 'interacting'
    if p <= .25 * big and big >= 3 * U:
        return 'additive'
    return 'unresolved'


def summarise(per, q_ok, star_ok, minpop):
    el = [d for d in per if d['eligible']]
    lab = [largest(d, q_ok, star_ok)['label'] for d in el]
    comp = [l for l in lab if l != 'incomplete']
    out = dict(n_eligible=len(el), n_complete=len(comp), counts={l: lab.count(l) for l in set(lab)})
    if len(el) < minpop:
        out['dominant'] = 'inconclusive (population)'
    elif len(comp) < .75 * len(el):
        out['dominant'] = 'inconclusive (incomplete)'
    else:
        top = max(set(comp), key=comp.count) if comp else None
        out['dominant'] = top if (top and top not in ('tied', 'none resolved') and comp.count(top) > .5 * len(comp)) \
            else 'no single dominant'
    pl = {}
    for x, y in itertools.combinations(ING, 2):
        labs = [pair_label(d, x, y) for d in el]
        av = [l for l in labs if l != 'unavailable']
        if len(av) < minpop:
            pl[x + y] = 'inconclusive (population)'
        elif sum(l == 'interacting' for l in av) > .25 * len(av):
            pl[x + y] = 'interacting'
        elif sum(l == 'additive' for l in av) >= .75 * len(av):
            pl[x + y] = 'additive'
        else:
            pl[x + y] = 'mixed'
    out['pairs'] = pl
    med = lambda k: (lambda v: float(np.median(v)) if v else None)([d['main'][k] for d in el if d['main'].get(k) is not None])
    out['median_main'] = {x: med(x) for x in ING}
    mains = {x: v for x, v in out['median_main'].items() if v is not None}
    out['two_largest'] = sorted(mains, key=mains.get, reverse=True)[:2]
    for k in ('e_P', 'e_dyn_star', 'U_star', 'D_a', 'e_tot', 'delta_form_tight'):
        v = [d[k] for d in el if d.get(k) is not None]
        out[f'median_{k}'] = float(np.median(v)) if v else None
        out[f'worst_{k}'] = float(np.max(v)) if v else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--cases', default=None, help='restrict to these case ids (comma list)')
    ap.add_argument('--minpop', type=int, default=8)
    ap.add_argument('--out')
    a = ap.parse_args()
    run = Path(a.run)
    res = json.loads((run / 'result.json').read_text())
    cfg = res['config']
    mesh = int(cfg['mesh'])
    keep = set(a.cases.split(',')) if a.cases else None
    out = dict(run=str(run), mesh=mesh, store_version=RS.store_version(), label='E0 R_ST, conditional Richardson eps_ref',
               B0_fixtures=b0_fixtures(), members={}, gates={}, controls={})
    refs = {}

    def ref(cid):
        if cid not in refs:
            coh, c = cid.split('|')
            R, meta = RS.load(2, coh, int(c))
            st = RS.reference_status(2, coh, int(c))
            refs[cid] = (R, float(meta['judge']['n0']), st['eps_ref'], st['eligible'])
        return refs[cid]

    gb0 = gb2 = 0.
    gb3, gb4 = [], {}
    for rk, rr in res['b2'].items():
        r = int(rk[1:])
        span = Span(r)
        zs = np.load(run / f'STAR_R{r}.npz')
        star = dict(ids={str(c): i for i, c in enumerate(zs['case_ids'])},
                    W={k: zs[f'W_{k}'] for k in ('star', 'star16', 'starx', 'star16x')},
                    valid={k: zs[f'valid_{k}'] for k in ('star', 'star16', 'starx', 'star16x')})
        # u_star check (G-B4 for the star family)
        zsc = np.load(run / f'STARCHK_R{r}.npz')
        sc = []
        for i, c in enumerate(zsc['case_ids']):
            j = star['ids'][str(c)]
            for k in ('star', 'star16', 'starx', 'star16x'):
                if zsc[f'valid_{k}'][i] and star['valid'][k][j]:
                    n0 = ref(str(c))[1]
                    sc.append(float(max(np.linalg.norm((NC.decode257(zsc[f'W_{k}'][i], r) - NC.decode257(star['W'][k][j], r))[t])
                                        for t in range(1, 6)) / n0))
                else:
                    sc.append(None)
        star_ok = bool(sc and all(v is not None and v <= CONV_BAR for v in sc))
        for c in zs['case_ids']:
            ref(str(c))
        for mk, mm in rr['members'].items():
            per, g0, g2 = analyse_member(run, mk, mm['member'], mesh, span, star, refs, keep)
            gb0, gb2 = max(gb0, g0), max(gb2, g2)
            gb3 += [d['G_B3'] for d in per if 'G_B3' in d]
            g4 = {}
            for d in per:
                for c, v in (d.get('G_B4') or {}).items():
                    g4.setdefault(c, []).append(v)
            g4ok = {c: bool(all(v is not None and v <= CONV_BAR for v in vs)) for c, vs in g4.items()}
            q_ok = bool(g4ok.get('Q', False))
            gb4[mk] = dict(per_corner=g4ok, worst={c: max([v for v in vs if v is not None], default=None) for c, vs in g4.items()},
                           star_family=star_ok, star_worst=max([v for v in sc if v is not None], default=None))
            mp = a.minpop if mesh != 4096 else 6
            out['members'][mk] = dict(member=mm['member'], diagnostic=mm['member']['form'] == 'GAL',
                                      summary=summarise(per, q_ok, star_ok, mp), per_case=per,
                                      quadrature=('verified on the check cases, extrapolated to the rest' if q_ok else 'Q withheld'),
                                      ideal_end=('verified' if star_ok else 'quadrature-unverified'))
    out['gates'] = dict(G_B0=dict(worst=gb0, passed=bool(gb0 <= 1e-10)), G_B2=dict(worst=gb2, passed=bool(gb2 <= 1e-12)),
                        G_B3=dict(worst=max(gb3) if gb3 else None, n=len(gb3), passed=bool(gb3 and max(gb3) <= 1e-8)),
                        G_B4=gb4, G_B5=res['setup'], B0_fixtures=out['B0_fixtures']['passed'])
    out['controls'] = controls(run, res, ref)
    dest = Path(a.out or HERE / 'checks' / f"budget2d-{run.parents[1].name}-{mesh}.json")
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(out, indent=1, default=float) + '\n')
    print(dest, json.dumps({k: (v.get('passed') if isinstance(v, dict) and 'passed' in v else '...') for k, v in out['gates'].items()}),
          json.dumps({k: v.get('fired') for k, v in out['controls'].items() if isinstance(v, dict)}))


def controls(run, res, ref):
    out = {}
    p = run / 'CTRL_b2.npz'
    if not p.exists():
        return out
    z = np.load(p)
    ids = [str(x) for x in z['case_ids']]
    r = int(res['config']['controls'].get('Rp', 384))
    n0 = {c: ref(c)[1] for c in ids}
    dmax = lambda A, B, c: float(max(np.linalg.norm((NC.decode257(A, r) - NC.decode257(B, r))[k]) for k in range(1, 6)) / n0[c])
    q = [dmax(z['Q_ctrl'][i], z['Q_qinf'][i], c) for i, c in enumerate(ids)]
    out['C_Q'] = dict(dist=q, fired=bool(sum(v > TAU for v in q) >= 1))
    s = [dmax(z['S_b0'][i], z['S_tight'][i], c) for i, c in enumerate(ids)]
    out['C_S'] = dict(dist=s, fired=bool(sum(v > TAU for v in s) > len(s) / 2))
    ci = []
    for i, c in enumerate(ids):
        Fp, F0 = NC.decode257(z['I_pert'][i][:1], r), NC.decode257(z['S_prod'][i][:1], r)
        got = float(np.linalg.norm(Fp[0] - F0[0])) / n0[c]
        want = .01 * float(np.linalg.norm(F0[0])) / n0[c]
        ci.append(dict(got=got, want=want, rel=abs(got - want) / want if want > 0 else None))
    out['C_I'] = dict(rows=ci, fired=bool(all(x['rel'] is not None and x['rel'] <= 1e-12 and x['got'] > 0 for x in ci)))
    ctl = res.get('controls', {})
    out['G_B1'] = dict(sham_rule_bitwise=ctl.get('b2_controls', {}).get('sham_rule_bitwise'),
                       prod_rerun_bitwise=ctl.get('b2_controls', {}).get('prod_rerun_bitwise'),
                       offline_bitwise=bool(np.array_equal(z['sham'], z['S_prod']) and np.array_equal(z['S_prod2'], z['S_prod'])))
    out['G_Q'] = {k: v for k, v in ctl.items() if k.startswith('GQ')}
    # C-score: identity errors must be raised on the actual scoring path
    c0 = ids[0]
    coh, cc = c0.split('|')
    F = NC.decode257(z['S_prod'][0], r)
    raised = {}
    for nm, rec in (('case_swap', dict(case_id=ids[1] if len(ids) > 1 else 'x|0', requested=(coh, cc), time_tag=TIME_TAG)),
                    ('time_shift', dict(case_id=c0, requested=(coh, cc), time_tag='t1,t2,t3,t4,t5,t6')),
                    ('ST_S_swap', dict(case_id=c0, requested=(coh, cc), time_tag=TIME_TAG, kind='S', scheme='CNR', dt=.005))):
        try:
            score(F, rec)
            raised[nm] = False
        except IdentityError:
            raised[nm] = True
    out['C_score'] = dict(raised=raised, fired=bool(all(raised.values())))
    return out


if __name__ == '__main__':
    main()
