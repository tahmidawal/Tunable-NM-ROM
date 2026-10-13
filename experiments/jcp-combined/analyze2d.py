"""Offline analysis of a `c2grid` result (DESIGN 6, A1.6, A1.8, A1.9, A2.6, A2.11, A3.1, A3.2, A3.3).

Inputs: <run>/result.json, W_<setting>.npz, fom rows; a scores file from score.py for the chosen reference registry
entry (errors are read from it, so a new reference is one switch: --scores). Field distances between rollouts (anchor
discrepancy, solve sensitivity, H3 vectors) are computed by decoding W through the bank at the 257^2 nodes.
Outputs checks/analysis-<attempt>-<refs>.json:
  arms        per (setting, run): cohort worst / median e_ST, e_S, e_TX, verified count, eligible, anchor discrepancy
              (median over resolved cases, resolved count), per-setting paired ratio (median, IQR) and common-panel ratio
  eps_ref     per case heuristic temporal tie threshold max_t ||ST - TX|| / n0 (A1.8)
  selection   per scheme x setting accurate-equal-cost and fast-equal-accuracy arms (A3.2); global scheme (A2.6)
  H1, H2, H3, H5 verdicts with their numbers
  timing      panel validity
All accuracy numbers PROVISIONAL (first-order references).

    python analyze2d.py --run runs/c2grid/archive/output --scores checks/scores-c2grid-2d_first_order.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DT0 = .005
CANDIDATES = [('LSPG', 'BDF2'), ('LSPG', 'CNR'), ('GAL', 'CN')]          # fixed tie order (A2.6)
DIAL = ['R128', 'R256', 'R384', 'R512']


def rkey(form, sc, f, lev='prod'):
    return f'{form}|{sc}|{DT0 * f:.8g}|{lev}'


class Fields:
    """Decode W (6, R') to the 257^2 nodes through the bank (same path as score.py)."""

    def __init__(self, Rmax):
        import score
        self.Q, self.G = score.bank2d(Rmax)
        self.decode = score.decode2d

    def dist(self, Wa, Wb, n0):
        U = self.decode(self.G, Wa - Wb)
        return [float(np.linalg.norm(U[j])) / n0 for j in range(6)]


def tie_case(x, y, eps):
    return abs(x - y) <= 2 * eps


def beaten(a, b, thr):
    """a is beaten by b (b strictly smaller by more than thr)."""
    return b < a - thr


def summarize_timing(panel):
    by = {}
    for inv in panel['invocations']:
        by.setdefault(inv['B'], []).append(inv['ratio'])
    return {k: dict(median=float(np.median(v)), q25=float(np.quantile(v, .25)), q75=float(np.quantile(v, .75)), n=len(v))
            for k, v in by.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--scores', required=True)
    ap.add_argument('--out')
    a = ap.parse_args()
    run = Path(a.run)
    res = json.loads((run / 'result.json').read_text())
    sc = json.loads(Path(a.scores).read_text())
    assert sc['gate'].get('passed', True), 'scores did not reproduce the in-job errors (G-score)'
    S = sc['scores']
    cfg = res['config']
    settings = [s['name'] for s in cfg['settings']]
    cohorts = res['cohorts']
    cases = [(coh, c) for coh in cfg['cohorts'] for c in
             (cfg.get('case_subset', {}).get(coh) if cfg.get('case_subset', {}).get(coh) is not None
              else range(cohorts[coh]['n']))]
    Rmax = max(res['setup'][s]['Rp'] for s in settings)
    F = Fields(Rmax)
    n0 = {}
    for coh, c in cases:
        ph = np.asarray(cohorts[coh]['physical'][c])
        n0[(coh, c)] = float(np.linalg.norm(F.Q.e.initial(256, ph)))
    # heuristic temporal tie threshold per case (A1.8): max_t ||ST - TX|| / n0 = max_t ||ST - S|| / (15 n0)
    import score
    ent = score.REG[sc['refs']]
    eps = {}
    for coh, c in cases:
        R = score.load_ref2d(ent, coh, c)
        eps[(coh, c)] = (max(float(np.linalg.norm(R['ST'][j] - R['S'][j])) for j in range(1, 6)) / 15. / n0[(coh, c)]
                         if 'S' in R and 'ST' in R else 0.)
    rows = {(r['setting'], r['run'], r['cohort'], r['case']): r for r in res['rows']}
    out = dict(run=str(run), refs=sc['refs'], provisional=True, arms={}, eps_ref={f'{k[0]}|{k[1]}': v for k, v in eps.items()},
               timing={}, selection={}, hypotheses={})
    panels = {k: dict(valid=v.get('valid'), median_drift=v.get('median_drift'), hashes_ok=v.get('hashes_ok'),
                      subjects=summarize_timing(v)) for k, v in res['timing'].items()}
    out['timing'] = {k: dict(valid=v['valid'], median_drift=v['median_drift'], hashes_ok=v['hashes_ok']) for k, v in panels.items()}
    per_case = {}
    for s in settings:
        z = np.load(run / f'W_{s}.npz')
        W, keys, cks = z['W'], [str(k) for k in z['runs']], [str(x) for x in z['cases']]
        kix = {k: i for i, k in enumerate(keys)}
        anc8, anc16 = 'GAL|BDF2|0.000625|tight', 'GAL|BDF2|0.0003125|tight'
        anc8x, anc16x = 'GAL|BDF2|0.000625|tighter', 'GAL|BDF2|0.0003125|tighter'
        U_anc, anc_d = {}, {}
        for ci, ck in enumerate(cks):
            coh, c = ck.split('|')
            c = int(c)
            nn = n0[(coh, c)]
            d = lambda k1, k2: max(F.dist(W[ci, kix[k1]], W[ci, kix[k2]], nn)[1:])
            U_anc[(coh, c)] = d(anc8, anc16) + d(anc8, anc8x) + d(anc16, anc16x)
            for k in keys:
                anc_d[(k, coh, c)] = d(k, anc8)
        for k in keys:
            vals = []
            for coh, c in cases:
                sk = f'{s}|{k}|{coh}|{c}'
                r = rows[(s, k, coh, c)]
                e = {t: S[sk][t]['worst'] for t in S[sk]}
                vals.append(dict(case=f'{coh}|{c}', verified=r['verified'], anchor=anc_d[(k, coh, c)], U_anc=U_anc[(coh, c)],
                                 resolved=bool(anc_d[(k, coh, c)] > 3 * U_anc[(coh, c)]), **{f'e_{t}': v for t, v in e.items()}))
            per_case[(s, k)] = vals
            ent_ = dict(cases=len(vals), verified=sum(v['verified'] for v in vals))
            ent_['eligible'] = ent_['verified'] == len(vals)
            for t in ('ST', 'S', 'TX'):
                xs = [v[f'e_{t}'] for v in vals if f'e_{t}' in v]
                if xs:
                    ent_[f'worst_{t}'] = max(xs)
                    ent_[f'median_{t}'] = float(np.median(xs))
            rs = [v['anchor'] for v in vals if v['resolved']]
            ent_['anchor_median_resolved'] = float(np.median(rs)) if rs else None
            ent_['anchor_resolved'] = len(rs)
            ent_['anchor_median_all'] = float(np.median([v['anchor'] for v in vals]))
            pt = panels.get(f'setting|{s}', {}).get('subjects', {}).get(f'{s}|{k}')
            ent_['ratio_setting'] = pt
            if k == 'LSPG|BE|0.005|prod':
                ent_['ratio_setting'] = dict(median=1.0, q25=1.0, q75=1.0, n=0)
            ent_['ratio_common'] = panels.get('common', {}).get('subjects', {}).get(f'{s}|{k}')
            if s == cfg['common_baseline'] and k == 'LSPG|BE|0.005|prod':
                ent_['ratio_common'] = dict(median=1.0, q25=1.0, q75=1.0, n=0)
            out['arms'][f'{s}|{k}'] = ent_
    epsv = list(eps.values())
    thr_med, thr_worst = 2 * float(np.median(epsv)), 2 * max(epsv)
    out['tie_thresholds'] = dict(median=thr_med, worst=thr_worst, per_case='2 eps_ref(case)')
    arms = out['arms']

    # ---- selection (A3.2) ----
    sel = {}
    for s in settings:
        base = arms[f'{s}|LSPG|BE|0.005|prod']
        pvalid = bool(panels.get(f'setting|{s}', {}).get('valid'))       # A4.4: invalid panels unusable for selection
        for form, scn in CANDIDATES:
            cand = []
            if not pvalid:
                sel[f'{s}|{form}-{scn}'] = dict(accurate_equal_cost=None, fast_equal_accuracy=None, reason='timing-invalid')
                continue
            for f in cfg['grid']['dt_factors']:
                k = rkey(form, scn, f)
                x = arms.get(f'{s}|{k}')
                if x is None or not x['eligible'] or x.get('ratio_setting') is None:
                    continue
                cand.append(dict(key=k, f=f, cost=x['ratio_setting']['median'], med=x['median_TX'], worst=x['worst_TX'],
                                 medST=x['median_ST'], worstST=x['worst_ST']))
            feas = [c for c in cand if c['cost'] <= 1.0]
            unbeaten = [c for c in feas if not any(beaten(c['med'], d['med'], thr_med) for d in feas)]
            aec = min(unbeaten, key=lambda c: (c['cost'], -c['f'])) if unbeaten else None
            # feasible iff none of worst/median ST/TX is beaten by the setting's LSPG-BE at dt0 (A3.2)
            fe = [c for c in cand if not (c['worstST'] > base['worst_ST'] + thr_worst or c['medST'] > base['median_ST'] + thr_med
                                          or c['worst'] > base['worst_TX'] + thr_worst or c['med'] > base['median_TX'] + thr_med)]
            fea = min(fe, key=lambda c: (c['cost'], c['med'], -c['f'])) if fe else None
            sel[f'{s}|{form}-{scn}'] = dict(accurate_equal_cost=aec, fast_equal_accuracy=fea)
    out['selection']['per_scheme_setting'] = sel
    points = {f'{f}-{s_}': 0 for f, s_ in CANDIDATES}
    sums = {f'{f}-{s_}': 0. for f, s_ in CANDIDATES}
    complete = {f'{f}-{s_}': True for f, s_ in CANDIDATES}
    winners = {}
    for s in DIAL:
        if s not in settings:
            continue
        meds = {k_: sel[f'{s}|{k_}']['accurate_equal_cost']['med'] for k_ in points if sel[f'{s}|{k_}']['accurate_equal_cost']}
        for k_ in points:
            if k_ not in meds:
                complete[k_] = False
            else:
                sums[k_] += meds[k_]
        w = [k_ for k_ in meds if not any(beaten(meds[k_], meds[o], thr_med) for o in meds)]
        winners[s] = w
        for k_ in w:
            points[k_] += 1
    order = [f'{f}-{s_}' for f, s_ in CANDIDATES]
    glob = sorted(order, key=lambda k_: (-points[k_], 0 if complete[k_] else 1, sums[k_], order.index(k_)))[0] if winners else None
    out['selection']['global'] = dict(points=points, sums=sums, complete=complete, winners=winners, scheme=glob)

    # ---- H1 (A1.9, A2.6): global scheme's accurate-equal-cost arms along the dial ----
    def compare(s1, s2, k1, k2, tag):
        a1, a2 = per_case[(s1, k1)], per_case[(s2, k2)]
        res_ = []
        for x, y in zip(a1, a2):
            e_ = eps[tuple(x['case'].split('|')[:1]) + (int(x['case'].split('|')[1]),)]
            if tie_case(x[f'e_{tag}'], y[f'e_{tag}'], e_):
                res_.append(0)
            else:
                res_.append(1 if y[f'e_{tag}'] < x[f'e_{tag}'] else -1)
        resolved = [r for r in res_ if r != 0]
        A1, A2 = arms[f'{s1}|{k1}'], arms[f'{s2}|{k2}']
        return dict(improved=sum(r > 0 for r in res_), worsened=sum(r < 0 for r in res_), ties=res_.count(0),
                    resolved=len(resolved), cases=len(res_), median_from=A1[f'median_{tag}'], median_to=A2[f'median_{tag}'],
                    worst_from=A1[f'worst_{tag}'], worst_to=A2[f'worst_{tag}'],
                    median_change=(A2[f'median_{tag}'] / A1[f'median_{tag}'] - 1.))

    if glob:
        f_, s_ = glob.split('-')
        h1 = {}
        for s1, s2 in zip(DIAL[:-1], DIAL[1:]):
            if s1 not in settings or s2 not in settings:
                continue
            x1, x2 = sel[f'{s1}|{glob}']['accurate_equal_cost'], sel[f'{s2}|{glob}']['accurate_equal_cost']
            if not x1 or not x2:
                h1[f'{s1}->{s2}'] = dict(verdict='unavailable')
                continue
            ent_ = {tag: compare(s1, s2, x1['key'], x2['key'], tag) for tag in ('TX', 'ST')}
            be_ = {tag: compare(s1, s2, 'LSPG|BE|0.005|prod', 'LSPG|BE|0.005|prod', tag) for tag in ('TX', 'ST')}
            if s1 != 'R384':
                v = all(ent_[t]['median_change'] <= -.25 and not (abs(ent_[t]['median_to'] - ent_[t]['median_from']) <= thr_med)
                        for t in ('TX', 'ST'))
                verdict = 'pass' if v else 'fail'
            else:
                lim = any(ent_[t]['resolved'] < ent_[t]['cases'] / 2 for t in ('TX', 'ST'))
                v = all(ent_[t]['resolved'] > 0 and ent_[t]['improved'] >= .6 * ent_[t]['resolved'] and
                        ent_[t]['median_change'] <= -.10 and ent_[t]['worst_to'] - ent_[t]['worst_from'] <= 0.001 for t in ('TX', 'ST'))
                verdict = 'reference-limited' if lim else ('pays off' if v else 'does not pay off')
            h1[f'{s1}->{s2}'] = dict(verdict=verdict, scheme=ent_, be_dt0=be_, arms=[x1['key'], x2['key']])
        out['hypotheses']['H1'] = h1
        # ---- H2 (A3.1, A2.6) ----
        def pts(pred, err):
            P = []
            for s in settings:
                for k_, x in arms.items():
                    if not k_.startswith(s + '|') or not x['eligible'] or x.get('ratio_common') is None:
                        continue
                    if pred(k_.split('|', 1)[1]):
                        P.append((x['ratio_common']['median'], x[err], k_))
            return P
        isBE = lambda k: k.startswith('LSPG|BE|') and k.endswith('prod')
        isC = lambda k: any(k.startswith(f'{f}|{s2}|') for f, s2 in CANDIDATES) and k.endswith('prod')
        h2a = {}
        for err, thr in (('worst_TX', thr_worst), ('median_TX', thr_med)):
            cov = []
            for cq, eq, kq in pts(isBE, err):
                ok = any(cp <= cq and not beaten(ep, eq, thr) for cp, ep, kp in pts(isC, err))
                cov.append(dict(arm=kq, covered=ok))
            h2a[err] = dict(covered=sum(c['covered'] for c in cov), total=len(cov), uncovered=[c['arm'] for c in cov if not c['covered']])
        dial = []
        for s in DIAL:
            x = sel.get(f'{s}|{glob}', {}).get('accurate_equal_cost')
            if x and s in settings:
                ac = arms[f'{s}|{x["key"]}']
                dial.append(dict(setting=s, arm=x['key'], cost_common=(ac['ratio_common'] or {}).get('median'),
                                 median_TX=ac['median_TX']))
        b = dial[:3]
        h2b = len(b) == 3 and all(b[i]['cost_common'] is not None and b[i + 1]['cost_common'] is not None and
                                  b[i + 1]['cost_common'] > b[i]['cost_common'] and
                                  b[i + 1]['median_TX'] <= b[i]['median_TX'] + thr_med for i in range(2))
        out['hypotheses']['H2'] = dict(a=h2a, a_pass=all(v['covered'] == v['total'] for v in h2a.values()), dial=dial,
                                       b_pass=bool(h2b))
    # ---- H3 (A1.9, A2.7, A3.3): fast worst case ----
    if 'fast' in settings:
        h3 = {}
        z = np.load(run / 'W_fast.npz')
        W, keys, cks = z['W'], [str(k) for k in z['runs']], [str(x) for x in z['cases']]
        kix = {k: i for i, k in enumerate(keys)}
        kbe, kcn, kan = 'LSPG|BE|0.005|prod', 'LSPG|CN|0.005|prod', 'GAL|BDF2|0.000625|tight'
        for tag in ('ST', 'TX'):
            worst_case = max(per_case[('fast', kcn)], key=lambda v: v[f'e_{tag}'])['case']
            allc = {}
            for ci, ck in enumerate(cks):
                coh, c = ck.split('|')
                c = int(c)
                nn = n0[(coh, c)]
                R = score.load_ref2d(ent, coh, c)[tag]
                Ub = F.decode(F.G, W[ci, kix[kbe]]) / nn
                Uc = F.decode(F.G, W[ci, kix[kcn]]) / nn
                Ua = F.decode(F.G, W[ci, kix[kan]]) / nn
                Rn = R / nn
                eI = [float(np.linalg.norm(Ua[j] - Rn[j])) for j in range(6)]
                ts = int(np.argmax(eI[1:])) + 1
                dl, ep = (Ub[ts] - Ua[ts]).ravel(), (Ua[ts] - Rn[ts]).ravel()
                E, D = float(np.linalg.norm(ep)), float(np.linalg.norm(dl))
                eBE, eCN = float(np.linalg.norm(Ub[ts] - Rn[ts])), float(np.linalg.norm(Uc[ts] - Rn[ts]))
                U = per_case[('fast', kbe)][ci]['U_anc']
                ip = float(dl @ ep)
                cos = ip / (D * E) if D > 0 and E > 0 else None
                ca = eBE < E and abs(eCN - E) <= .2 * abs(eBE - E)
                cb = cos is not None and cos <= -.3
                cc = E ** 2 - eBE ** 2 >= .5 * D ** 2
                ga = eBE < E - U and abs(eCN - E) + U <= .2 * (E - U - eBE)
                gb = ip + U * (D + E) + U ** 2 <= -.3 * (D + U) * (E + U)
                gc = max(0., E - U) ** 2 - eBE ** 2 >= .5 * (D + U) ** 2
                verdict = ('confirmed' if (ca and cb and cc and ga and gb and gc) else
                           'consistent with cancellation, not resolved against anchor uncertainty' if (ca and cb and cc)
                           else 'cancellation not supported')
                allc[ck] = dict(t_star=ts, e_BE=eBE, e_CN=eCN, e_inf=E, delta_BE=D, cos=cos, U_anc=U, a=ca, b=cb, c=cc,
                                guards=[ga, gb, gc], verdict=verdict)
            h3[tag] = dict(worst_case=worst_case, primary=allc[worst_case], cases=allc,
                           confirmed_count=sum(v['verdict'] == 'confirmed' for v in allc.values()))
        out['hypotheses']['H3'] = h3
    # ---- H5 (A1.9, A2.11): FOM matching inside the common panel ----
    fr = {}
    for r in res['fom'].get('rows', []):
        fr.setdefault(r['run'], []).append(r)
    cal = res['fom'].get('calibration', {})
    fom = {}
    for k_, rs in fr.items():
        eST = [x['e_ST'] for x in rs]
        eTX = [x['e_TX'] for x in rs]
        fom[k_] = dict(worst_ST=max(eST), worst_TX=max(eTX), median_ST=float(np.median(eST)), verified=all(x['verified'] for x in rs),
                       status=cal.get(k_.split('|', 1)[1], {}).get('status'))
    out['fom'] = fom
    com = res['timing'].get('common')
    if com:
        inv = com['invocations']
        idx = {}
        for x in inv:
            idx.setdefault(x['B'], {})[(x['case_index'], x['rep'])] = x['ratio']
        comp_inv = [f'acc|{rkey(f, s2, ff)}' for f, s2 in [('LSPG', 'BE'), ('LSPG', 'CN'), ('LSPG', 'CNR'), ('LSPG', 'BDF2'), ('GAL', 'CN')]
                    for ff in (0.5, 1, 2, 5, 10)]
        h5 = {}
        for tag in ('ST', 'TX'):
            thr = thr_worst
            matches = {}
            for ak, x in arms.items():
                if not x['eligible'] or not ak.endswith('prod') or x.get('ratio_common') is None:
                    continue
                elig_f = [(fk, fv) for fk, fv in fom.items() if fv['verified'] and fv['status'] in ('calibrated', 'fallback_tight')
                          and (fv[f'worst_{tag}'] <= x[f'worst_{tag}'] or abs(fv[f'worst_{tag}'] - x[f'worst_{tag}']) <= thr)]
                if not elig_f:
                    matches[ak] = dict(matched=None)
                    continue
                best = None
                for fk, fv in elig_f:
                    num = idx.get(fk, {})
                    den = idx.get(ak, {}) if not (ak.startswith(cfg['common_baseline'] + '|LSPG|BE|0.005|prod')) else {kk: 1. for kk in num}
                    pairs = [num[kk] / den[kk] for kk in num if kk in den]
                    if not pairs:
                        continue
                    sp = float(np.median(pairs))
                    fm = float(np.median(list(num.values())))
                    if best is None or fm < best[2]:
                        best = (fk, sp, fm)
                matches[ak] = dict(matched=best[0], speedup=best[1]) if best else dict(matched=None)
            comb = [(v['speedup'], k_) for k_, v in matches.items() if v.get('matched')]
            compv = [(v['speedup'], k_) for k_, v in matches.items() if v.get('matched') and k_ in comp_inv]
            bc, bv = (max(comb) if comb else None), (max(compv) if compv else None)
            h5[tag] = dict(best_combined=bc, best_comparator=bv, matches=matches,
                           passed=bool(bc and bv and bc[0] >= 1.2 * bv[0]))
        out['hypotheses']['H5'] = h5
    dest = Path(a.out or HERE / 'checks' / f"analysis-{run.parents[1].name}-{sc['refs']}.json")
    dest.write_text(json.dumps(out, indent=1, default=float) + '\n')
    print(dest)
    print(json.dumps(dict(selection=out['selection'].get('global'), H=list(out['hypotheses'])), indent=1, default=float))


if __name__ == '__main__':
    main()
