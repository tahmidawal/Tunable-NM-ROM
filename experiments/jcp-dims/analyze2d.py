"""jcp-dims 2D analysis (DESIGN 3, 6, A1.2, A1.5, A2.2, A2.5, A3.2; CPU only). From the collected dacc archives of
f2a, f2b, f2k, f2l, b2f, b2m it computes, per configuration: offline Q1 (dscore.q1), E0 scores (ST; S for LSPG-BE at dt0,
labelled), ladder Q2 (dscore.ladder_q2) and the numerical-qualification label; an independent NumPy rho on dev6|0 for
the registered subset (A6.1); then every A4 m*, B3 family verdict and cross-family gate, A3 / B6 transition, M = r
control, GAL-vs-LSPG and LMF verdict, and the accuracy part of the deployed-path Shapley decomposition. Writes
checks/analysis-2d.json (and checks/mstar-2d.json for the timing-config generator). Per-configuration results are
cached in checks/cache2d/ (keyed by the archive file sha256).

    JAX_PLATFORMS=cpu python analyze2d.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

import dscore as DSC
import dstats as DS
import npcheck as NC
import qscore as SC

HERE = Path(__file__).resolve().parent
CFG = HERE / 'configs'
CACHE = HERE / 'checks' / 'cache2d'
JOBS = ('f2a', 'f2b', 'f2k', 'f2l', 'b2f', 'b2m')
R_LIST = (128, 256, 384, 512)
KAPPAS = (1, 2, 3, 4, 8)
PRIMARY = 'gauss256'
RHO_SUBSET_MAX = 4e7                  # A6.1: independent rho recomputed when m * M <= 4e7 (dev6|0, six outputs)


def sha_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def tag_of(kind):
    return {'F2': '', 'K2': 'k', 'LMF': 'lmf', 'B3': 'f', 'B3ii': 'f'}[kind]


# ------------------------------------------------------------------------------------------- per configuration
def config_entry(att, d, res, setting, arm):
    form, scheme, fac = arm['form'], arm['scheme'], float(arm['factor'])
    p = d / f'R_{setting}_{form}_{scheme}_{fac:g}.npz'
    if not p.exists():
        return None
    sh = sha_file(p)
    cp = CACHE / f'{att}__{setting}__{form}_{scheme}_{fac:g}.json'
    if cp.exists():
        c = json.loads(cp.read_text())
        if c.get('sha256') == sh:
            return c
    R = dict(np.load(p))
    rows = [r_ for r_ in res['rows'] if r_['setting'] == setting and r_['arm'] == f'{form}_{scheme}_{fac:g}']
    q = DSC.q1(R, rows)
    sc = DSC.scores(R)
    e = dict(sha256=sh, job=att, setting=setting, form=form, scheme=scheme, factor=fac, r=int(R['Rp']), M=int(R['M']),
             policy=str(R['policy']), rule=str(R['rule']), square=bool(R['square']), case_ids=[str(x) for x in R['case_ids']],
             Q1=q['Q1'], disagreements=[str(x) for x in q['disagreements'][:5]], pooled=q['pooled_noncertified'],
             steps=q['pooled_steps'], worst_dist_tight=max(v['dist_tight'] for v in q['per_case'].values()),
             valid={c: v['valid'] for c, v in q['per_case'].items()},
             e0={c: {k: dict(e=v[k]['e'], eps_ref=v[k]['eps_ref'], eligible=bool(v[k]['eligible'])) for k in v}
                 for c, v in sc.items()},
             rho_job_max=float(np.nanmax(R['rho'])) if np.isfinite(R['rho']).any() else None)
    m = int(len(np.asarray(R['A'])))       # M (rows of A)
    if bool(R['rho_only']) and str(R['qp_rule']) != 'None':
        import rules2d
        X, w = rules2d.points(str(R['rule']))
        if len(X) * m <= RHO_SUBSET_MAX:
            Xp, wp = rules2d.points(str(R['qp_rule']))
            r = int(R['Rp'])
            t, tp = NC.tables(X, w, r, R['kx'], R['ky'], 1024), NC.tables(Xp, wp, r, R['kx'], R['ky'], 1024)
            off = [NC.rho(np.asarray(R['W'][0, k]), t, tp) for k in range(6)]
            job = [float(x) for x in R['rho'][0]]
            e['rho_offline'] = dict(case=e['case_ids'][0], offline=off, job=job,
                                    agree=all((np.isnan(a) and np.isnan(b)) or SC.agree(a, b) for a, b in zip(off, job)))
    CACHE.mkdir(parents=True, exist_ok=True)
    cp.write_text(json.dumps(e, default=float) + '\n')
    return e


def load_all():
    C, RAW = {}, {}
    for att in JOBS:
        d = DSC.out_dir(att, 'dacc', att)
        if d is None:
            continue
        res = json.loads((d / 'result.json').read_text())
        for s in res['config']['settings']:
            for a in s['arms']:
                k = f"{s['name']}|{a['form']}|{a['scheme']}"
                e = config_entry(att, d, res, s['name'], a)
                if e is not None and res['complete'] and s['name'] not in res.get('memory_rejected', []):
                    C[k] = e
                    RAW[k] = (d, s['name'], a)
        if res.get('fixgate'):
            C.setdefault('_fixgate', {})[att] = res['fixgate']
        C.setdefault('_settings', {}).update({k: dict(A_rank=v['A_rank'], A_cond=v['A_cond'], M=v['M']) for k, v in res['settings'].items()})
    return C, RAW


def R_of(RAW, k):
    if k not in RAW:
        return None
    d, s, a = RAW[k]
    return DSC.load(d, s, a['form'], a['scheme'], a['factor'])


# ------------------------------------------------------------------------------------------- ladders
def ladder_labels(C, RAW, lad):
    tag = tag_of(lad['kind'])
    form, scheme, _ = lad['scheme'].split('|')
    keys = [f"R{lad['r']}M{lad['M']}{tag}{rl}|{form}|{scheme}" for rl in lad['rungs']]
    Rs = [None] * len(keys)
    out = []
    for i, k in enumerate(keys):
        e = C.get(k)
        if e is None:
            out.append(dict(key=k, label='missing', Q1=None, Q2=None))
            continue
        for j in (i, i + 1, i + 2):
            if j < len(keys) and Rs[j] is None:
                Rs[j] = R_of(RAW, keys[j])
        top = i == len(keys) - 1
        if top:
            q2 = dict(label='unverified (ladder top)')
        else:
            q2 = DSC.ladder_q2(Rs[i], Rs[i + 1], Rs[i + 2] if i + 2 < len(keys) else None, second_to_top=(i == len(keys) - 2))
        qual = bool(e['Q1']['pass'] and q2['label'] == 'pass')
        out.append(dict(key=k, rule=lad['rungs'][i], Q1=e['Q1']['pass'], Q2=q2, qualified=qual,
                        label='qualified' if qual else 'not qualified'))
        if i >= 1:
            Rs[i - 1] = None
    return out


def m_of(rule):
    import rules2d
    return len(rules2d.points(rule)[0])


def mstar_of(lab, lad):
    cand = lad['candidates']
    labels = [x['label'] if x['label'] == 'missing' else ('qualified' if x.get('qualified') else 'not qualified')
              for x in lab if x.get('rule', x['key'].split('|')[0]) is not None][:len(cand)]
    sizes = [m_of(r) for r in cand]
    return DS.mstar(labels, sizes), labels, sizes


# ------------------------------------------------------------------------------------------- comparisons
def arrays(C, kX, kY, kind='ST'):
    eX, eY = C[kX]['e0'], C[kY]['e0']
    ids = [c for c in C[kX]['case_ids'] if c in eY]
    a = np.array([eX[c][kind]['e'] for c in ids])
    b = np.array([eY[c][kind]['e'] for c in ids])
    eps = np.array([eX[c][kind]['eps_ref'] for c in ids])
    el = np.array([eX[c][kind]['eligible'] and C[kX]['valid'][c] and C[kY]['valid'][c] for c in ids])
    return a, b, eps, el


def trans(C, Q, kX, kY):
    if kX not in C or kY not in C:
        return dict(label='inconclusive (missing)')
    a, b, eps, el = arrays(C, kX, kY)
    lab, p = DS.transition(a, b, eps, el, qualified=(Q.get(kX, False), Q.get(kY, False)))
    return dict(label=lab, stats=p, X=kX, Y=kY)


def t2_label(e):
    return f"1024|R{e['r']}|M{e['M']}|{e['rule']}|{e['form']}|{e['scheme']}|1"


def timing_costs():
    """DESIGN 4 / A1.1: per subject of t2: cost c = median over valid fused triplets of t_fused / mean(t_A); per-case
    medians (for the case bootstrap); usable iff <= 1/3 of its triplets invalid; timing-job qualification on dev6
    (validity, pooled certificate, G-cert, tight / tighter, q+ rollout and rho, q++ convergence)."""
    d = DSC.out_dir('t2', 'dtime', 't2')
    if d is None:
        return {}
    res = json.loads((d / 'result.json').read_text())
    out = {}
    for lb, s in res['subjects'].items():
        fz = [x for x in res['fused'] if x['B'] == lb]
        ok = [x for x in fz if x['valid']]
        per = {}
        for x in ok:
            per.setdefault(x['case'], []).append(x['ratio'])
        rows = s['certification']
        steps = int(s['steps']) * len(rows)
        q1 = bool(rows and all(r_['valid'] and r_['tight_valid'] and r_['tighter_valid'] for r_ in rows)
                  and sum(r_['noncertified'] for r_ in rows) <= .01 * steps and all(r_['dist_tight'] <= 2.5e-4 for r_ in rows)
                  and all(r_['dist_tight_tighter'] <= 2.5e-5 for r_ in rows))
        q2 = True
        if s['spec'].get('qp'):
            q2 = all(r_.get('qp_valid') and r_.get('dist_qp', 1) <= 2.5e-4 for r_ in rows)
            z = np.load(d / f"S_{lb.replace('|', '_')}.npz")
            rh = np.asarray(z['rho'], float)
            q2 = q2 and bool(np.nanmax(rh) <= .116)
            if s['spec'].get('qpp'):
                q2 = q2 and all(r_.get('qpp_valid') and r_.get('dist_qp_qpp', 1) <= 2.5e-4 for r_ in rows)
            else:
                q2 = False                                 # no check rung: unverified, not qualified
        out[lb] = dict(c=float(np.median([x['ratio'] for x in ok])) if ok else None,
                       c_ev=float(np.median([x['ratio'] for x in res['invocations'] if x['B'] == lb and x['valid']] or [np.nan])),
                       per_case={k: float(np.median(v)) for k, v in per.items()}, usable=bool(fz and len(ok) >= len(fz) * 2 / 3),
                       qualified=bool(q1 and q2), q1=q1, q2=q2)
    return out


def shapley(C, Q, out, cost, B=10000):
    """6.1c + A1.5 + A2.5: deployed-path decomposition between consecutive r over the corners (r, M, m) of F2."""
    res = {}
    rng = np.random.default_rng(0)
    for ra, rb in zip(R_LIST[:-1], R_LIST[1:]):
        ms = {}
        for r in (ra, rb):
            lid = f"F2|R{r}|M{4 * r}|LSPG|CNR|1|gauss|s0|frozen"
            ms[r] = out['mstar'].get(lid, {}).get('mstar', {}).get('m')
        if not all(ms.values()):
            res[f'{ra}->{rb}'] = dict(outcome='decomposition not reached (no m*)')
            continue
        key = lambda i, j, k: f"R{(ra, rb)[i]}M{4 * (ra, rb)[j]}gauss{int(round(ms[(ra, rb)[k]] ** .5))}|LSPG|CNR"
        corners = {(i, j, k): key(i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)}
        if not all(c in C and Q.get(c) for c in corners.values()):
            res[f'{ra}->{rb}'] = dict(outcome='decomposition not reached (unqualified corner)',
                                      corners={str(k): v for k, v in corners.items()})
            continue
        ids = [c for c in C[corners[(0, 0, 0)]]['case_ids']
               if all(C[v]['e0'][c]['ST']['eligible'] and C[v]['valid'][c] for v in corners.values())]
        E = {kk: np.array([C[v]['e0'][c]['ST']['e'] for c in ids]) for kk, v in corners.items()}
        eps = float(np.median([C[corners[(0, 0, 0)]]['e0'][c]['ST']['eps_ref'] for c in ids]))
        f_acc = {kk: float(np.median(v)) for kk, v in E.items()}
        boot = []
        for _ in range(B):
            ix = rng.integers(0, len(ids), len(ids))
            boot.append(DS.shapley3({kk: float(np.median(v[ix])) for kk, v in E.items()}))
        boot = np.array(boot)
        lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
        r_ = dict(population=len(ids), acc=DS.shapley3(f_acc).tolist(), acc_interval=[lo.tolist(), hi.tolist()],
                  acc_labels=[DS.sign_label(l, h) for l, h in zip(lo, hi)], d_acc=f_acc[(1, 1, 1)] - f_acc[(0, 0, 0)],
                  corners={str(k): v for k, v in corners.items()})
        r_['acc_shares'] = (np.array(r_['acc']) / r_['d_acc']).tolist() if abs(r_['d_acc']) > 2 * eps else None
        lbs = {kk: t2_label(C[v]) for kk, v in corners.items()}
        if cost and all(lbs[kk] in cost and cost[lbs[kk]]['per_case'] for kk in corners):
            cases = sorted(set.intersection(*[set(cost[lbs[kk]]['per_case']) for kk in corners]))
            Cc = {kk: np.array([cost[lbs[kk]]['per_case'][c] for c in cases]) for kk in corners}
            f_c = {kk: float(np.median(v)) for kk, v in Cc.items()}
            bc = np.array([DS.shapley3({kk: float(np.median(v[ix])) for kk, v in Cc.items()})
                           for ix in (rng.integers(0, len(cases), len(cases)) for _ in range(B))])
            clo, chi = np.percentile(bc, [2.5, 97.5], axis=0)
            dtot = np.array([f_c[(1, 1, 1)] - f_c[(0, 0, 0)]])
            dbo = np.array([b.sum() for b in bc])
            r_.update(cost=DS.shapley3(f_c).tolist(), cost_interval=[clo.tolist(), chi.tolist()],
                      cost_labels=[DS.sign_label(l, h, 'cost') for l, h in zip(clo, chi)], d_cost=float(dtot[0]))
            dlo, dhi = np.percentile(dbo, [2.5, 97.5])
            r_['cost_shares'] = (np.array(r_['cost']) / dtot[0]).tolist() if (abs(dtot[0]) / f_c[(0, 0, 0)] > .05 and (dlo > 0 or dhi < 0)) else None
        r_['wording'] = 'a selection-conditioned description of the difference between two deployed configurations, not a decomposition of the error norm'
        res[f'{ra}->{rb}'] = r_
    return res


def main():
    C, RAW = load_all()
    man = json.loads((CFG / 'MANIFEST-cells.json').read_text())
    out = dict(label='E0 R_ST: conditional Richardson estimate (E0 A10; combined A10, A11, retracted A15 carried)',
               configurations={k: {kk: v[kk] for kk in ('Q1', 'pooled', 'steps', 'worst_dist_tight', 'rule', 'r', 'M', 'policy',
                                                         'form', 'scheme', 'square', 'disagreements', 'rho_offline')
                                   if kk in v} for k, v in C.items() if not k.startswith('_')},
               ladders=[], mstar={}, families={}, gates={}, A3={}, B6={}, LMF={}, shapley_acc={})
    Q = {}
    for lad in man['ladders']:
        lab = ladder_labels(C, RAW, lad)
        for x in lab:
            if x.get('qualified') is not None:
                Q[x['key']] = x['qualified']
                if x['key'] in out['configurations']:
                    out['configurations'][x['key']].update(Q2=x['Q2'], numerically_qualified=x['qualified'])
        ms = mstar_of(lab, lad) if lad['candidates'] else (None, None, None)
        lid = f"{lad['kind']}|R{lad['r']}|M{lad['M']}|{lad['scheme']}|{lad['family']}|s{lad['draw']}|{lad.get('policy', 'frozen')}"
        out['ladders'].append(dict(id=lid, labels=lab, mstar=ms[0]))
        if ms[0] is not None:
            out['mstar'][lid] = dict(mstar=ms[0], labels=ms[1], sizes=ms[2])
    # ---- B3 families
    for r, M in ((128, 512), (384, 1536)):
        fam = {}
        for lad, L in zip(man['ladders'], out['ladders']):
            if lad['kind'] in ('B3', 'B3ii', 'F2') and lad['r'] == r and lad['M'] == M and lad['scheme'] == 'LSPG|CNR|1':
                if L['mstar'] is None:
                    continue
                f = lad['family']
                fam.setdefault(f, {})[lad['draw']] = out['mstar'][L['id']]
        res = {}
        for f, dr in fam.items():
            if len(dr) == 1:
                res[f] = dict(family=dr[0]['mstar'], per_draw=[dr[0]['mstar']])
            else:
                labs = [dr[s]['labels'] for s in sorted(dr)]
                res[f] = DS.family_mstar(labs, dr[0]['sizes'])
        # cross-family gate: (b) from the fixgate block, (a) finest rung vs the gauss384 target (rollout)
        fg = {}
        for att in ('b2f', 'b2m'):
            for blk in C.get('_fixgate', {}).get(att, []):
                if blk['r'] == r and blk['M'] == M:
                    fg.update(blk['rules'])
        tgt = f'R{r}M{M}gauss384|LSPG|CNR'
        gates = {}
        Rt = R_of(RAW, tgt)
        for f, dr in fam.items():
            for s in sorted(dr):
                finest = {'fib': 'fib121393', 'sobol': f'sobol{2 ** 17}', 'halton': f'halton{2 ** 17}', 'mc': f'mc{2 ** 17}',
                          'smolyak': 'smolyak14', 'gauss': 'gauss384'}[f]
                fr = finest if (s == 0 and f not in ('mc',)) else f'{finest}s{s}'
                kf = f'R{r}M{M}{"" if f == "gauss" else "f"}{fr}|LSPG|CNR'
                rb = fg.get(fr, {}).get('rho_tgt', [])
                avail = kf in C and tgt in C and C[kf]['Q1']['pass'] and C[tgt]['Q1']['pass'] and all(C[kf]['valid'].values())
                dist = np.inf
                if avail and Rt is not None:
                    Rf = R_of(RAW, kf)
                    dist = max(NC.fdist(NC.decode257(Rf['W'][j], r), NC.decode257(Rt['W'][j], r), SC.n0_of(Rt, j))
                               for j in range(len(Rt['W'])))
                gates[f'{f}|s{s}'] = dict(rule=fr, gate=DS.cross_family_gate(rb, avail, dist), worst_dist_to_target=dist,
                                          rho_tgt_max=max(rb) if rb else None)
        bad = fg.get('badlat121393', {})
        gates['badlat121393 (control)'] = dict(gate=DS.cross_family_gate(bad.get('rho_tgt', []), False, np.inf),
                                               rho_tgt_max=max(bad['rho_tgt']) if bad.get('rho_tgt') else None)
        pe = {}
        names = sorted(res)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                ga = all(v['gate'] == 'pass' for kk, v in gates.items() if kk.startswith(a + '|'))
                gb = all(v['gate'] == 'pass' for kk, v in gates.items() if kk.startswith(b + '|'))
                v = DS.point_efficiency(res[a]['family']['m'], res[b]['family']['m'], names=(a, b))
                if not (ga and gb):
                    v = 'inconclusive (gate flagged or unavailable)'
                pe[f'{a} vs {b}'] = v
        out['families'][f'R{r}M{M}'] = dict(mstar=res, point_efficiency=pe)
        out['gates'][f'R{r}M{M}'] = gates
    # ---- A3 panels (LSPG-CN-R)
    A3 = {}
    for name, cells in (('fixed (M=2048, gauss256)', [(r, 2048) for r in R_LIST]), ('fixed m gauss256, M=4r', [(r, 4 * r) for r in R_LIST])):
        ks = [f'R{r}M{M}{PRIMARY}|LSPG|CNR' for r, M in cells]
        tr = [trans(C, Q, ks[i], ks[i + 1]) for i in range(len(ks) - 1)]
        A3[name] = dict(transitions=tr, range=DS.range_verdict([t['label'] for t in tr]))
    out['A3'] = A3
    # ---- B6
    B6 = {}
    for r in R_LIST:
        for form, sch, tag in (('LSPG', 'CNR', ''), ('LSPG', 'CN', 'k'), ('GAL', 'CN', 'k'), ('LSPG', 'BE', 'k')):
            ks = {k: f'R{r}M{k * r}{tag}{PRIMARY}|{form}|{sch}' for k in KAPPAS}
            tr = {f'{a}->{b}': trans(C, Q, ks[a], ks[b]) for a, b in ((1, 2), (2, 3), (3, 4), (4, 8))}
            ctl = trans(C, Q, ks[1], ks[4])
            control = DS.control_mr(ctl['label'], (Q.get(ks[1], False), Q.get(ks[4], False)))
            labs = [tr[x]['label'] for x in ('2->3', '3->4')]
            B6[f'R{r}|{form}-{sch}'] = dict(transitions=tr, control_1_to_4=ctl, control=control,
                                             range_2_4=DS.range_verdict(labs) if control == 'fired' else
                                             ('test-space sensitivity not demonstrated at this r' if control == 'did not fire'
                                              else 'blocked (control unavailable)'),
                                             range_1_8=DS.range_verdict([tr[x]['label'] for x in tr]) if control == 'fired' else None)
        for k in KAPPAS:
            g = f'R{r}M{k * r}k{PRIMARY}|GAL|CN'
            for ref, nm in ((f'R{r}M{k * r}k{PRIMARY}|LSPG|CN', 'GAL vs LSPG-CN (primary)'), (f'R{r}M{k * r}{PRIMARY}|LSPG|CNR', 'GAL vs LSPG-CN-R')):
                if g in C and ref in C:
                    a, b, eps, el = arrays(C, g, ref)
                    lab, p = DS.two_way(a, b, eps, el, qualified=(Q.get(g, False), Q.get(ref, False)), names=('GAL', 'LSPG'))
                    B6.setdefault(f'R{r}|kappa{k}|GAL-vs-LSPG', {})[nm] = dict(label=lab, stats=p)
            if k == 1 and g in C and f'R{r}M{r}k{PRIMARY}|LSPG|CN' in C:
                Rg, Rl = R_of(RAW, g), R_of(RAW, f'R{r}M{r}k{PRIMARY}|LSPG|CN')
                dd = [NC.fdist(NC.decode257(Rg['W'][j], r), NC.decode257(Rl['W'][j], r), SC.n0_of(Rl, j)) for j in range(len(Rl['W']))]
                B6[f'R{r}|kappa1|GAL-vs-LSPG']['measured distance (no gate, A1.2)'] = dict(worst=max(dd), median=float(np.median(dd)))
    out['B6'] = B6
    # ---- LMF
    lm = {}
    for f, rungs_def in (('gauss', 'F2'), ('smolyak', 'B3')):
        lid_l = next((L['id'] for lad, L in zip(man['ladders'], out['ladders']) if lad['kind'] == 'LMF' and lad['family'] == f), None)
        lid_d = next((L['id'] for lad, L in zip(man['ladders'], out['ladders']) if lad['kind'] == rungs_def and lad['r'] == 384
                      and lad['M'] == 1536 and lad['family'] == f and lad['scheme'] == 'LSPG|CNR|1' and lad['draw'] == 0), None)
        if lid_l in out['mstar'] and lid_d in out['mstar']:
            lm[f] = dict(lmf=out['mstar'][lid_l]['mstar'], default=out['mstar'][lid_d]['mstar'],
                         efficiency=DS.point_efficiency(out['mstar'][lid_l]['mstar']['m'], out['mstar'][lid_d]['mstar']['m'],
                                                        names=('LMF', 'default')))
    sl, sd = C.get('_settings', {}).get('R384M1536lmfgauss256'), C.get('_settings', {}).get('R384M1536gauss256')
    kl, kd = 'R384M1536lmfgauss256|LSPG|CNR', 'R384M1536gauss256|LSPG|CNR'
    stab = dict(rank_ok=bool(sl and sl['A_rank'] == 384), cond_ok=bool(sl and sd and sl['A_cond'] <= 10 * sd['A_cond']),
                valid=bool(kl in C and all(C[kl]['valid'].values())), q1_gauss256=bool(kl in C and C[kl]['Q1']['pass']))
    lm['stability'] = dict(stab, passed=all(stab.values()))
    if lm['stability']['passed'] and kl in C and kd in C:
        a, b, eps, el = arrays(C, kl, kd)
        lab, p = DS.two_way(a, b, eps, el, qualified=(Q.get(kl, False), Q.get(kd, False)), names=('LMF', 'default'))
        lm['accuracy_gauss256'] = dict(label=lab, stats=p)
    elif not lm['stability']['passed']:
        lm['outcome'] = 'unstable test space (no verdict)'
    out['LMF'] = lm
    # ---- secondary B6 at each cell's own m* (LSPG-CN-R, GAL-CN, LSPG-CN)
    for r in R_LIST:
        for form, sch, tag, kind in (('LSPG', 'CNR', '', 'F2'), ('LSPG', 'CN', 'k', 'K2'), ('GAL', 'CN', 'k', 'K2')):
            ks = {}
            for k in KAPPAS:
                lid = f"{kind}|R{r}|M{k * r}|{form}|{sch}|1|gauss|s0|frozen"
                m = out['mstar'].get(lid, {}).get('mstar', {}).get('m')
                ks[k] = f'R{r}M{k * r}{tag}gauss{int(round(m ** .5))}|{form}|{sch}' if m else None
            tr2 = {f'{a}->{b}': (trans(C, Q, ks[a], ks[b]) if ks[a] and ks[b] else dict(label='inconclusive (no m*)'))
                   for a, b in ((1, 2), (2, 3), (3, 4), (4, 8))}
            B6.setdefault(f'R{r}|{form}-{sch}', {})['secondary_at_mstar'] = tr2
    # ---- timing (t2): per-cell cost, timing-job qualification, elasticity, deployed-path Shapley
    cost = timing_costs()
    out['costs'] = cost
    if cost:
        cells = [(C[k]['r'], C[k]['M'], m_of(C[k]['rule']), cost[lb]['c']) for k in C if not k.startswith('_')
                 for lb in [t2_label(C[k])] if lb in cost and cost[lb]['usable'] and cost[lb]['qualified'] and Q.get(k)]
        if cells:
            r_, M_, m_, c_ = (np.array(x, float) for x in zip(*cells))
            out['elasticity'] = DS.elasticity(r_, M_, m_, c_)
    out['shapley'] = shapley(C, Q, out, cost)
    # ---- m* map for the timing generator
    mst = {lid: v['mstar']['m'] for lid, v in out['mstar'].items()}
    (HERE / 'checks').mkdir(exist_ok=True)
    (HERE / 'checks' / 'mstar-2d.json').write_text(json.dumps(mst, indent=1) + '\n')
    (HERE / 'checks' / 'analysis-2d.json').write_text(json.dumps(out, indent=1, default=str) + '\n')
    print('configs', len(out['configurations']), 'ladders', len(out['ladders']))


if __name__ == '__main__':
    main()
