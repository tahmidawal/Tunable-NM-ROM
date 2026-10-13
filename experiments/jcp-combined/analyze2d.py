"""Offline analysis of a `c2grid` result (DESIGN 6, A1.6, A1.8, A1.9, A2.6, A2.7, A2.11, A3.1-A3.3, A4.4, A5.3, A5.9).

Inputs: <run>/result.json, W_<setting>.npz, a scores file from score.py (gate status must be 'pass' and bound to this
result.json by sha256). All accuracy numbers come from the scores file (ROM and FOM), so a new reference is one switch:
--scores. Selection: computed here only against the selection reference (2d_first_order) and then frozen
(checks/FROZEN-SELECTION.json, written by freeze.py); with --frozen the frozen arms are used unchanged and only
reference-dependent matching (H5) is recomputed, labelled "reference-dependent rematching".
Tie policy (A1.8, A2.6): per case 2 eps_ref(case); medians 2 median(eps_ref); worst 2 max(eps_ref). eps_ref is the
heuristic temporal tie threshold max_t ||ST - TX|| / n0 unless the registry entry supplies verified per-case
uncertainty 'eta' (then 2 eps = eta).
Smoke results are refused unless --allow-smoke (machinery test only; never a verdict, A5.5).
All accuracy numbers PROVISIONAL (first-order references).

    python analyze2d.py --run runs/c2grid/archive/output --scores checks/scores-c2grid-2d_first_order.json [--frozen F]
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
CANDIDATES = [('LSPG', 'BDF2'), ('LSPG', 'CNR'), ('GAL', 'CN')]          # fixed tie order (A2.6)
CNAMES = [f'{f}-{s}' for f, s in CANDIDATES]
DIAL = ['R128', 'R256', 'R384', 'R512']
BE0 = 'LSPG|BE|0.005|prod'
ANC = ('GAL|BDF2|0.000625|tight', 'GAL|BDF2|0.0003125|tight', 'GAL|BDF2|0.000625|tighter', 'GAL|BDF2|0.0003125|tighter')


def rkey(form, sc, f, lev='prod'):
    return f'{form}|{sc}|{DT0 * f:.8g}|{lev}'


def beaten(a, b, thr):
    """a is beaten by b: b strictly smaller by more than thr."""
    return b < a - thr


def stats_timing(panel):
    by = {}
    for inv in panel['invocations']:
        by.setdefault(inv['B'], []).append(inv['ratio'])
    return {k: dict(median=float(np.median(v)), q25=float(np.quantile(v, .25)), q75=float(np.quantile(v, .75)), n=len(v))
            for k, v in by.items()}


def pareto(points):
    """Ordinary numerical Pareto front of (cost, err, id): no other point with cost <= and err <= and one strict (A3.1)."""
    best = {}
    for c, e, i in sorted(points):                      # exact duplicates collapse to the lowest arm id (A3.1)
        best.setdefault((c, e), (c, e, i))
    pts = list(best.values())
    return [p for p in pts if not any((q[0] <= p[0] and q[1] <= p[1] and (q[0] < p[0] or q[1] < p[1])) for q in pts)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--scores', required=True)
    ap.add_argument('--frozen')
    ap.add_argument('--certified-scores')
    ap.add_argument('--allow-smoke', action='store_true')
    ap.add_argument('--out')
    a = ap.parse_args()
    run = Path(a.run)
    res = json.loads((run / 'result.json').read_text())
    sc = json.loads(Path(a.scores).read_text())
    if res['config'].get('smoke') and not a.allow_smoke:
        raise SystemExit('smoke result: refused (A5.5)')
    rsha = hashlib.sha256((run / 'result.json').read_bytes()).hexdigest()
    assert sc['result_sha256'] == rsha, 'scores not of this run'
    if sc['refs'] == '2d_first_order':
        assert sc['gate']['status'] == 'pass', ('G-score gate not passed', sc['gate'])
    else:                     # new reference: reproduction is certified once against the original reference (A1.8)
        assert a.certified_scores, 'a new-reference score file needs --certified-scores (original-reference pass)'
        cert = json.loads(Path(a.certified_scores).read_text())
        assert cert['refs'] == '2d_first_order' and cert['gate']['status'] == 'pass' and cert['result_sha256'] == rsha
        assert a.frozen, 'a new reference is a rescoring: the frozen selection must be used (A1.8)'
    S, SF = sc['scores'], sc['fom_scores']
    cfg = res['config']
    settings = [s['name'] for s in cfg['settings']]
    cohorts = res['cohorts']
    cases = [(coh, c) for coh in cfg['cohorts'] for c in
             (cfg.get('case_subset', {}).get(coh) if cfg.get('case_subset', {}).get(coh) is not None
              else range(cohorts[coh]['n']))]
    import score
    ent = score.REG[sc['refs']]
    Q, G = score.bank2d(max(res['setup'][s]['Rp'] for s in settings))
    n0 = {k: float(np.linalg.norm(Q.e.initial(256, np.asarray(cohorts[k[0]]['physical'][k[1]])))) for k in cases}
    refs = {k: score.load_ref2d(ent, *k) for k in cases}
    if 'eta' in ent:
        # verified case-wise uncertainty: comparisons by interval overlap with endpoint propagation (A2.6). The scalar
        # threshold machinery below is then replaced: 2 eps_case := eta_x + eta_y is the overlap condition for two
        # values carrying the same reference uncertainty; medians / worst values use the endpoint-propagated half widths.
        eta = {k: float(ent['eta'][f'{k[0]}|{k[1]}']) for k in cases}
        eps = dict(eta)                                      # per case: |x - y| <= 2 eta  <=>  [x +- eta] overlaps [y +- eta]
        tie_kind = 'verified uncertainty (E0): interval overlap, endpoint propagation'
    else:
        assert all('ST' in refs[k] and 'S' in refs[k] for k in cases), 'no ST/S pair and no verified eta: no tie threshold'
        eps = {k: max(float(np.linalg.norm(refs[k]['ST'][j] - refs[k]['S'][j])) for j in range(1, 6)) / 15. / n0[k]
               for k in cases}
        tie_kind = 'heuristic temporal tie threshold'
    ev = list(eps.values())
    thr = dict(median=2 * float(np.median(ev)), worst=2 * max(ev))     # heuristic thresholds (A2.6)
    ETA = 'eta' in ent

    def agg_interval(vals, kind):
        """Verified uncertainty: endpoints propagated through the median / max of the per-case intervals (A2.6, A8.7)."""
        lo = [max(0., v - eps[kc]) for v, kc in zip(vals, cases)]
        hi = [v + eps[kc] for v, kc in zip(vals, cases)]
        f = np.median if kind == 'median' else np.max
        return float(f(lo)), float(f(hi))
    rows = {(r['setting'], r['run'], r['cohort'], r['case']): r for r in res['rows']}
    out = dict(run=str(run), refs=sc['refs'], provisional=True, tie=dict(kind=tie_kind, **thr),
               eps_ref={f'{k[0]}|{k[1]}': v for k, v in eps.items()}, arms={}, timing={}, selection={}, hypotheses={})
    panels = {k: dict(valid=v.get('valid'), worst_subject_drift=v.get('worst_subject_drift'), hashes_ok=v.get('hashes_ok'),
                      subjects=stats_timing(v), invocations=v['invocations']) for k, v in res['timing'].items()}
    out['timing'] = {k: {kk: v[kk] for kk in ('valid', 'worst_subject_drift', 'hashes_ok')} for k, v in panels.items()}
    per_case, Wc = {}, {}
    for s in settings:
        z = np.load(run / f'W_{s}.npz')
        W, keys, cks = z['W'], [str(k) for k in z['runs']], [str(x) for x in z['cases']]
        kix = {k: i for i, k in enumerate(keys)}
        Wc[s] = (W, kix, cks)
        for ci, ck in enumerate(cks):
            coh, c = ck.split('|')
            c = int(c)
            k_ = (coh, c)
            Ud = lambda key: score.decode2d(G, W[ci, kix[key]]) / n0[k_]
            A8, A16, A8x, A16x = (Ud(x) for x in ANC)
            anc_ok = all(rows[(s, x, coh, c)]['verified'] for x in ANC)
            Ut = [float(np.linalg.norm(A8[j] - A16[j]) + np.linalg.norm(A8[j] - A8x[j]) + np.linalg.norm(A16[j] - A16x[j]))
                  for j in range(6)]
            for k in keys:
                U = Ud(k)
                dper = [float(np.linalg.norm(U[j] - A8[j])) for j in range(6)]
                sk = f'{s}|{k}|{coh}|{c}'
                per_case[(s, k, k_)] = dict(verified=rows[(s, k, coh, c)]['verified'], anchor=max(dper[1:]),
                                            U_anc=max(Ut[1:]), U_t=Ut, anchor_qualified=anc_ok,
                                            resolved=bool(anc_ok and max(dper[1:]) > 3 * max(Ut[1:])),
                                            **{f'e_{t}': S[sk][t]['worst'] for t in S[sk]})
        for k in keys:
            vals = [per_case[(s, k, kc)] for kc in cases]
            e_ = dict(cases=len(vals), verified=sum(v['verified'] for v in vals))
            e_['eligible'] = e_['verified'] == len(vals)
            for t in ('ST', 'S', 'TX'):
                xs = [v[f'e_{t}'] for v in vals]
                e_[f'worst_{t}'], e_[f'median_{t}'] = max(xs), float(np.median(xs))
            rs = [v['anchor'] for v in vals if v['resolved']]
            e_.update(anchor_median_resolved=float(np.median(rs)) if rs else None, anchor_resolved=len(rs),
                      anchor_median_all=float(np.median([v['anchor'] for v in vals])))
            e_['ratio_setting'] = (dict(median=1., q25=1., q75=1., n=0) if k == BE0 else
                                   panels.get(f'setting|{s}', {}).get('subjects', {}).get(f'{s}|{k}'))
            e_['ratio_common'] = (dict(median=1., q25=1., q75=1., n=0) if (s == cfg['common_baseline'] and k == BE0) else
                                  panels.get('common', {}).get('subjects', {}).get(f'{s}|{k}'))
            out['arms'][f'{s}|{k}'] = e_
    arms = out['arms']
    common_ok = bool(panels.get('common', {}).get('valid'))

    def vals(s_, k_, tag):
        return [per_case[(s_, k_, kc)][f'e_{tag}'] for kc in cases]

    def bt(A, B, tag, kind):
        """True iff arm A = (setting, key) is BEATEN by arm B on the cohort `kind` (median / worst) of e_tag. Verified
        uncertainty: B's propagated upper endpoint below A's lower endpoint (A8.7); heuristic: scalar threshold (A2.6)."""
        va, vb = (vals(*A, tag) if isinstance(A, tuple) else A), (vals(*B, tag) if isinstance(B, tuple) else B)
        if ETA:
            return agg_interval(vb, kind)[1] < agg_interval(va, kind)[0]
        f = np.median if kind == 'median' else np.max
        return float(f(vb)) < float(f(va)) - thr[kind]

    # ---------------------------------------------------------------- selection (A3.2, A2.6) or frozen
    if a.frozen:
        frz = json.loads(Path(a.frozen).read_text())['2d']                     # the frozen schema written by freeze.py
        sel, glob = frz['per_scheme_setting'], frz['global']['scheme']
        out['selection'] = {'frozen': str(a.frozen), 'frozen_sha256': hashlib.sha256(Path(a.frozen).read_bytes()).hexdigest(),
                            'per_scheme_setting': sel, 'global': frz['global']}
    else:
        assert sc['refs'] == '2d_first_order', 'selection is made only against the selection reference; use --frozen'
        sel = {}
        for s in settings:
            base = arms[f'{s}|{BE0}']
            pvalid = bool(panels.get(f'setting|{s}', {}).get('valid'))
            for form, scn in CANDIDATES:
                name = f'{s}|{form}-{scn}'
                if not pvalid:
                    sel[name] = dict(accurate_equal_cost=None, fast_equal_accuracy=None, reason='timing-invalid')
                    continue
                cand = []
                for f in cfg['grid']['dt_factors']:
                    k = rkey(form, scn, f)
                    x = arms.get(f'{s}|{k}')
                    if x is None or not x['eligible'] or x.get('ratio_setting') is None:
                        continue
                    cand.append(dict(key=k, f=f, cost=x['ratio_setting']['median'], med=x['median_TX'], worst=x['worst_TX'],
                                     medST=x['median_ST'], worstST=x['worst_ST']))
                feas = [c for c in cand if c['cost'] <= 1.0]
                unb = [c for c in feas if not any(bt((s, c['key']), (s, d['key']), 'TX', 'median') for d in feas)]
                aec = min(unb, key=lambda c: (c['cost'], -c['f'])) if unb else None
                bk_ = (s, BE0)
                fe = [c for c in cand if not any(bt((s, c['key']), bk_, tg, kd) for tg in ('ST', 'TX') for kd in ('worst', 'median'))]
                fea = min(fe, key=lambda c: (c['cost'], c['med'], -c['f'])) if fe else None
                sel[name] = dict(accurate_equal_cost=aec, fast_equal_accuracy=fea)
        points = {k: 0 for k in CNAMES}
        sums = {k: 0. for k in CNAMES}
        complete = {k: True for k in CNAMES}
        winners = {}
        for s in DIAL:
            meds = {k: sel[f'{s}|{k}']['accurate_equal_cost']['med'] for k in CNAMES
                    if s in settings and sel[f'{s}|{k}']['accurate_equal_cost']}
            keys_ = {k: sel[f'{s}|{k}']['accurate_equal_cost']['key'] for k in meds}
            for k in CNAMES:
                if k in meds:
                    sums[k] += meds[k]
                else:
                    complete[k] = False
            w = [k for k in meds if not any(bt((s, keys_[k]), (s, keys_[o]), 'TX', 'median') for o in meds)]
            winners[s] = w
            for k in w:
                points[k] += 1
        if max(points.values()) == 0:
            glob = None
        else:
            top = [k for k in CNAMES if points[k] == max(points.values())]
            comp = [k for k in top if complete[k]]
            glob = (min(comp, key=lambda k: (sums[k], CNAMES.index(k))) if comp else min(top, key=CNAMES.index))
        out['selection'] = {'per_scheme_setting': sel,
                            'global': dict(points=points, sums=sums, complete=complete, winners=winners, scheme=glob)}

    # ---------------------------------------------------------------- H1 (A1.9 bars; A2.6 ties; reference-limited)
    def paired(s1, k1, s2, k2, tag):
        r_ = []
        for kc in cases:
            x, y = per_case[(s1, k1, kc)][f'e_{tag}'], per_case[(s2, k2, kc)][f'e_{tag}']
            r_.append(0 if abs(x - y) <= 2 * eps[kc] else (1 if y < x else -1))
        A1, A2 = arms[f'{s1}|{k1}'], arms[f'{s2}|{k2}']
        return dict(improved=r_.count(1), worsened=r_.count(-1), ties=r_.count(0), resolved=len(r_) - r_.count(0),
                    cases=len(r_), median_from=A1[f'median_{tag}'], median_to=A2[f'median_{tag}'],
                    worst_from=A1[f'worst_{tag}'], worst_to=A2[f'worst_{tag}'],
                    median_change=A2[f'median_{tag}'] / A1[f'median_{tag}'] - 1.,
                    median_tied=not (bt((s1, k1), (s2, k2), tag, 'median') or bt((s2, k2), (s1, k1), tag, 'median')))
    h1 = {}
    for s1, s2 in zip(DIAL[:-1], DIAL[1:]):
        x1 = (sel.get(f'{s1}|{glob}') or {}).get('accurate_equal_cost') if glob else None
        x2 = (sel.get(f'{s2}|{glob}') or {}).get('accurate_equal_cost') if glob else None
        if not x1 or not x2:
            h1[f'{s1}->{s2}'] = dict(verdict='unavailable')
            continue
        tx, st = paired(s1, x1['key'], s2, x2['key'], 'TX'), paired(s1, x1['key'], s2, x2['key'], 'ST')
        be = {t: paired(s1, BE0, s2, BE0, t) for t in ('TX', 'ST')}
        if tx['resolved'] < tx['cases'] / 2:
            verdict = 'reference-limited'
        elif s1 != 'R384':                       # H1(a): >= 25 % median fall against TX (ST reported as sensitivity)
            verdict = 'pass' if (tx['median_change'] <= -.25 and not tx['median_tied']) else 'fail'
        elif st['resolved'] < st['cases'] / 2:
            verdict = 'reference-limited'
        else:                                    # H1(b): against TX and ST
            ok = all(d['resolved'] > 0 and d['improved'] >= .6 * d['resolved'] and d['median_change'] <= -.10 and
                     not d['median_tied'] and d['worst_to'] - d['worst_from'] <= 0.001 for d in (tx, st))
            verdict = 'pays off' if ok else 'does not pay off'
        h1[f'{s1}->{s2}'] = dict(verdict=verdict, arms=[x1['key'], x2['key']], TX=tx, ST_sensitivity=st, BE_dt0=be)
    out['hypotheses']['H1'] = h1

    # ---------------------------------------------------------------- H2 (A3.1; exact transitions; common panel valid)
    if not common_ok:
        out['hypotheses']['H2'] = dict(verdict='unavailable', reason='common timing panel invalid')
    else:
        isBE = lambda k: k.startswith('LSPG|BE|') and k.endswith('prod')
        isC = lambda k: any(k.startswith(f'{f}|{s2}|') for f, s2 in CANDIDATES) and k.endswith('prod')

        def pts(pred, err):
            return [(x['ratio_common']['median'], x[err], k_) for k_, x in arms.items()
                    if x['eligible'] and x.get('ratio_common') and pred(k_.split('|', 1)[1])]
        fronts, cover = {}, {}
        for err, tk in (('worst_TX', 'worst'), ('median_TX', 'median')):
            PB, PC = pts(isBE, err), pts(isC, err)
            fronts[err] = dict(BE=pareto(PB), second_order=pareto(PC))
            if not PB or not PC:
                cover[err] = dict(verdict='unavailable', reason='empty population')
                continue
            sk_ = lambda k__: tuple(k__.split('|', 1))
            unc = [kq for cq, eq, kq in PB if not any(cp <= cq and not bt(sk_(kp), sk_(kq), 'TX', tk) for cp, ep, kp in PC)]
            cover[err] = dict(covered=len(PB) - len(unc), total=len(PB), uncovered=unc)
        a_pass = all('total' in v and v['covered'] == v['total'] for v in cover.values())
        dial = []
        for s in DIAL:
            x = (sel.get(f'{s}|{glob}') or {}).get('accurate_equal_cost') if glob else None
            ac = arms.get(f'{s}|{x["key"]}') if x else None
            dial.append(dict(setting=s, arm=x['key'] if x else None,
                             cost_common=(ac['ratio_common'] or {}).get('median') if ac and ac.get('ratio_common') else None,
                             median_TX=ac['median_TX'] if ac else None))
        need = dial[:3]
        if any(d['arm'] is None or d['cost_common'] is None for d in need):
            b = 'unavailable'
        else:
            b = all(need[i + 1]['cost_common'] > need[i]['cost_common'] and
                    not bt((need[i + 1]['setting'], need[i + 1]['arm']), (need[i]['setting'], need[i]['arm']), 'TX', 'median')
                    for i in range(2))
        out['hypotheses']['H2'] = dict(fronts=fronts, a=cover, a_pass=a_pass, dial=dial, b=b)

    # ---------------------------------------------------------------- H3 (A1.9, A2.7, A3.3)
    if 'fast' in settings:
        W, kix, cks = Wc['fast']
        kbe, kcn = BE0, 'LSPG|CN|0.005|prod'
        h3 = {}
        for tag in ('ST', 'TX'):
            allc = {}
            for ci, ck in enumerate(cks):
                coh, c = ck.split('|')
                kc = (coh, int(c))
                pc = per_case[('fast', kbe, kc)]
                if not (pc['anchor_qualified'] and pc['verified'] and per_case[('fast', kcn, kc)]['verified']):
                    allc[ck] = dict(verdict='unavailable (unverified BE, CN or anchor trajectories)')
                    continue
                nn = n0[kc]
                Ub, Uc, Ua = (score.decode2d(G, W[ci, kix[k]]) / nn for k in (kbe, kcn, ANC[0]))
                Rn = refs[kc][tag] / nn
                eI = [float(np.linalg.norm(Ua[j] - Rn[j])) for j in range(6)]
                ts = 1 + int(np.argmax(eI[1:]))                 # earliest maximiser
                dl, ep = (Ub[ts] - Ua[ts]).ravel(), (Ua[ts] - Rn[ts]).ravel()
                E, D = float(np.linalg.norm(ep)), float(np.linalg.norm(dl))
                eBE, eCN = float(np.linalg.norm(Ub[ts] - Rn[ts])), float(np.linalg.norm(Uc[ts] - Rn[ts]))
                U = pc['U_t'][ts]                               # anchor uncertainty indicator AT t* (A3.3)
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
                           else ('angle undefined' if cos is None else 'cancellation not supported'))
                allc[ck] = dict(t_star=ts, e_BE=eBE, e_CN=eCN, e_inf=E, delta_BE=D, cos=cos, U_anc_t=U, a=ca, b=cb, c=cc,
                                guards=[ga, gb, gc], verdict=verdict)
            wc = max(cks, key=lambda ck: per_case[('fast', kcn, (ck.split('|')[0], int(ck.split('|')[1])))][f'e_{tag}'])
            h3[tag] = dict(worst_case=wc, primary=allc[wc], cases=allc,
                           confirmed_count=sum(v.get('verdict') == 'confirmed' for v in allc.values()),
                           note='indicator-based; U_anc is not a certified bound; scoped to cancellation at t*')
        out['hypotheses']['H3'] = h3

    # ---------------------------------------------------------------- H5 (A1.9 frozen inventory; common panel valid)
    fom = {}
    for k, v in SF.items():
        kk, coh, c = k.rsplit('|', 2)
        fom.setdefault(kk, []).append(v)
    cal = res['fom'].get('calibration', {})
    fver = {}
    for r in res['fom'].get('rows', []):
        fver.setdefault(r['run'], []).append(r['verified'])
    fv = {kk: dict(worst_ST=max(v['ST']['worst'] for v in vs), worst_TX=max(v['TX']['worst'] for v in vs),
                   verified=all(fver.get(kk, [False])), status=cal.get(kk.split('|', 1)[1], {}).get('status'), cases=len(vs))
          for kk, vs in fom.items()}
    out['fom'] = fv
    if not common_ok:
        out['hypotheses']['H5'] = dict(verdict='unavailable', reason='common timing panel invalid')
    else:
        idx = {}
        for x in panels['common']['invocations']:
            idx.setdefault(x['B'], {})[(x['case_index'], x['rep'])] = x['ratio']
        comp_inv = [f'acc|{rkey(f, s2, ff)}' for f, s2 in [('LSPG', 'BE'), ('LSPG', 'CN'), ('LSPG', 'CNR'), ('LSPG', 'BDF2'),
                                                           ('GAL', 'CN')] for ff in (0.5, 1, 2, 5, 10)]
        basek = f"{cfg['common_baseline']}|{BE0}"
        h5 = {}
        for tag in ('ST', 'TX'):
            matches = {}
            for ak, x in arms.items():
                if not (x['eligible'] and ak.endswith('prod') and (x.get('ratio_common') or ak == basek)):
                    continue
                fvals = lambda fk: [SF[f'{fk}|{kc[0]}|{kc[1]}'][tag]['worst'] for kc in cases]
                ef = [fk for fk, f_ in fv.items() if f_['verified'] and f_['status'] in ('calibrated', 'fallback_tight')
                      and f_['cases'] == len(cases) and not bt(fvals(fk), tuple(ak.split('|', 1)), tag, 'worst')]
                best = None
                for fk in ef:
                    num = idx.get(fk, {})
                    den = {kk: 1. for kk in num} if ak == basek else idx.get(ak, {})
                    keys_ = [kk for kk in num if kk in den]
                    if not keys_ or len(keys_) != len(num) or not all(np.isfinite(num[kk]) and np.isfinite(den[kk]) for kk in keys_):
                        continue
                    fm = float(np.median(list(num.values())))
                    if best is None or fm < best[2]:
                        best = (fk, float(np.median([num[kk] / den[kk] for kk in keys_])), fm)
                matches[ak] = dict(matched=best[0], speedup=best[1]) if best else dict(matched=None)
            comb = [(v['speedup'], k_) for k_, v in matches.items() if v.get('matched')]
            compv = [(v['speedup'], k_) for k_, v in matches.items() if v.get('matched') and k_ in comp_inv]
            bc, bv = (max(comb) if comb else None), (max(compv) if compv else None)
            h5[tag] = dict(best_combined=bc, best_comparator=bv, matches=matches,
                           verdict=('unavailable' if not (bc and bv) else ('pass' if bc[0] >= 1.2 * bv[0] else 'fail')),
                           label=('reference-dependent rematching' if a.frozen else 'selection-reference matching'))
        out['hypotheses']['H5'] = h5
    dest = Path(a.out or HERE / 'checks' / f"analysis-{run.parents[1].name}-{sc['refs']}.json")
    dest.write_text(json.dumps(out, indent=1, default=float) + '\n')
    print(dest)
    print(json.dumps(dict(selection=out['selection'].get('global'),
                          H={k: (list(v) if isinstance(v, dict) else v) for k, v in out['hypotheses'].items()}), default=float))


if __name__ == '__main__':
    main()
