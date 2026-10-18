"""jcp-fom2d offline scoring, calibration recomputation and FOM matching (DESIGN §3, §4, §6). CPU only.

    python match2.py --ladder runs/<a>/archive/output/ladder_<cfg> [--ladder ...] --targets checks/targets-<x>.json \
                     --out checks/match-<x>.json

1. Offline rescoring: every configuration's saved lattice fields are re-scored against E0 R_ST (and R_S where recorded)
   with refstore; must agree with the in-job scores to 1e-12 (absolute, relative-error units).
2. Calibration recomputed from the saved calibration fields (C_<group>.npz) and evidence: t0 converged on every
   calibration case; the loosest level that converged and is within 1e-6 n0 of t0 at every output time on every case;
   must equal the in-job choice.
3. Admissibility (§4): calibrated (FD) / valid (PS); converged with complete evidence and finite score on every case of
   the ladder population; hash-checked finite positive warm timings.
4. Per target (a ROM configuration with per-case E0 errors on its population) and family: joint, median-only and
   worst-only matches (tie rule of §1 on the common E0-eligible cases), screening-cheapest selection, the top-3 joint
   matches for paired confirmation, the same-mesh match (FD, L = target mesh) or the same-mesh unmatched comparator,
   edge and boundary labels; union = cheapest over families.
5. Controls K-syn (synthetic entries) and K-false (falsely matched pairs) per target; a control that does not fire
   blocks the matches of that target.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import fomrun2 as FR
import refstore as RS

FAMILIES = ('UB', 'CC', 'PS')
TOL_CAL = 1e-6


def ties_or_beats(ea, ec, eta):
    """Step 2 / speed3d rule (interval overlap with endpoint propagation) on median and worst."""
    ea, ec, eta = np.asarray(ea, float), np.asarray(ec, float), np.asarray(eta, float)
    return bool(np.median(ea - eta) <= np.median(ec + eta)), bool(np.max(ea - eta) <= np.max(ec + eta))


def load_ladders(paths):
    """-> (configs {key: record}, groups {gkey: record}, fields loader, jobs info)."""
    cfgs, groups, src, jobs = {}, {}, {}, []
    for p in paths:
        p = Path(p)
        r = json.loads((p / 'result.json').read_text())
        assert r.get('complete') is True, f'{p}: incomplete'
        jobs.append(dict(dir=str(p), job_id=r.get('job_id'), gpu=r.get('gpu'), commit=r.get('commit'),
                         store_version=r.get('store_version')))
        for k, v in r['configs'].items():
            assert k not in cfgs, f'duplicate configuration {k}'
            cfgs[k] = v
            src[k] = p
        for k, v in r.get('groups', {}).items():
            assert k not in groups, f'duplicate group {k}'
            groups[k] = dict(v, _dir=str(p))
    gpus = {j['gpu'] for j in jobs}
    return cfgs, groups, src, jobs, gpus


def rescore(cfgs, src):
    """Offline rescoring; returns per-config per-case scores and the worst disagreement with the in-job scores."""
    out, worst = {}, 0.
    for k, v in cfgs.items():
        z = np.load(src[k] / f"F_{k.replace('|', '_')}.npz")
        ids = [str(x) for x in z['case_ids']]
        F = z['fields']
        rec = {}
        for i, cid in enumerate(ids):
            coh, c = cid.split('|')
            pj = next(p for p in v['production'] if p['case'] == cid)
            rec[cid] = {}
            for kind, sj in pj['scores'].items():
                s = RS.score(F[i], 2, coh, int(c), kind)
                d = abs(s['e'] - sj['e']) if sj['e'] is not None else np.inf
                worst = max(worst, d)
                rec[cid][kind] = dict(e=s['e'], eps_ref=s['eps_ref'], eligible=s['eligible'])
        out[k] = rec
    return out, worst


def recalibrate(groups):
    """Recompute every FD group's chosen level from its saved calibration fields and evidence."""
    res = {}
    for gk, g in groups.items():
        d = Path(g['_dir'])
        z = np.load(d / f"C_{gk.replace('|', '_')}.npz")
        levels = [str(x) for x in z['levels']]
        ids = [str(x) for x in z['case_ids']]
        n0 = np.asarray(z['n0'])
        conv = {lev: [r['converged'] for r in g['calibration'][lev]] for lev in levels}
        t0ok = all(conv['0'])
        diffs = {}
        for lev in levels[1:]:
            diffs[lev] = [max(float(np.linalg.norm(z[f'F{lev}'][i][k] - z['F0'][i][k])) / n0[i] for k in range(1, 6))
                          for i in range(len(ids))]
        chosen = None
        if t0ok:
            chosen = '0'
            for lev in sorted(levels[1:], key=lambda s: -int(s)):
                if all(conv[lev]) and max(diffs[lev]) <= TOL_CAL:
                    chosen = lev
                    break
        passing = (['0'] if t0ok else []) + [lv for lv in levels[1:] if all(conv[lv]) and max(diffs[lv]) <= TOL_CAL]
        rec = g.get('recovery')
        prod_level = chosen
        rec_ok = True
        if rec is not None:          # A2.6: the recovery level must be the next tighter level that passed calibration
            tighter = [lv for lv in passing if chosen is not None and int(lv) < int(chosen)]
            rec_ok = bool(tighter) and rec['level'] == max(tighter, key=int)
            prod_level = rec['level'] if rec_ok else None
        res[gk] = dict(chosen=chosen, injob_chosen=g['chosen'], agrees=bool(chosen == g['chosen'] and rec_ok),
                       production_level=prod_level, recovery=rec, max_diff={k: max(v) for k, v in diffs.items()})
    return res


def validate(k, v, sc, recal, population):
    """One admissibility validator for real and synthetic (K-syn) records."""
    sp = FR.parse(k) if not k.startswith('K|') else None
    why = []
    if sp is not None and sp[0] == 'fd':
        gk = v.get('group')
        want = None if gk not in recal else recal[gk]['production_level']
        if gk not in recal or want != sp[4] or not recal[gk]['agrees']:
            why.append('not calibrated at this level')
    ids = [p['case'] for p in v['production']]
    if sorted(ids) != sorted(population):
        why.append('case population')
    if not all(p['converged'] is True for p in v['production']):
        why.append('not converged / invalid on some case')
    if not all(np.isfinite(sc[c]['ST']['e']) for c in ids):
        why.append('non-finite score')
    w = v.get('warm') or {}
    ts = [x['seconds'] for x in w.get('times', [])]
    if not (ts and w.get('hashes_ok') is True and np.isfinite(ts).all() and min(ts) > 0):
        why.append('warm timing')
    return dict(admissible=not why, why=why, cost=float(np.median(ts)) if ts else np.inf,
                family=FR.family(k) if sp is not None else 'K')


def admissibility(cfgs, groups, recal, scores, population):
    return {k: validate(k, v, scores[k], recal, population) for k, v in cfgs.items()}


def common(target, cand_scores):
    """Common E0-eligible cases of the target population; eta must agree."""
    ids = [c for c in target['case_ids'] if target['eligible'][c] and cand_scores[c]['ST']['eligible']]
    et = np.array([target['e'][c] for c in ids])
    eta = np.array([target['eta'][c] for c in ids])
    ef = np.array([cand_scores[c]['ST']['e'] for c in ids])
    assert np.allclose(eta, [cand_scores[c]['ST']['eps_ref'] for c in ids], rtol=0, atol=1e-15), 'eta differs'
    return ef, et, eta, ids


def edges(adm):
    """Ladder edges per family: coarsest mesh / fewest modes and largest step among admissible configurations."""
    out = {}
    for f in FAMILIES:
        sps = [FR.parse(k) for k, a in adm.items() if a['family'] == f and a['admissible']]
        if sps:
            out[f] = dict(min_res=min(s[2] if s[0] == 'fd' else s[1] for s in sps), max_dt=max(s[3] if s[0] == 'fd' else s[2] for s in sps))
    return out


def is_edge(k, ed):
    sp = FR.parse(k)
    f = FR.family(k)
    if f not in ed:
        return False
    res, dt = (sp[2], sp[3]) if sp[0] == 'fd' else (sp[1], sp[2])
    return bool(res == ed[f]['min_res'] or abs(dt - ed[f]['max_dt']) < 1e-15)


def synthetic(target, population, recal):
    """K-syn: two synthetic records through the SAME validator and tie rule: (a) zero cost, perfect, one unconverged case;
    (b) converged, zero cost, errors 10 e_T + 4 eta + 1e-3 on every case."""
    warm = dict(times=[dict(seconds=1e-9)], hashes_ok=True)
    out = {}
    for name, conv0, ef in (('K|unconverged', False, lambda c: 0.), ('K|inaccurate', True,
                            lambda c: 10 * target['e'].get(c, 1.) + 4 * target['eta'].get(c, 0.) + 1e-3)):
        v = dict(production=[dict(case=c, converged=(conv0 or i != 0)) for i, c in enumerate(population)], warm=warm)
        sc = {c: {'ST': dict(e=ef(c), eps_ref=target['eta'].get(c, 0.), eligible=target['eligible'].get(c, False))}
              for c in population}
        a = validate(name, v, sc, recal, population)
        joint = False
        if a['admissible']:
            ef_, et, eta, _ = common(target, sc)
            joint = all(ties_or_beats(ef_, et, eta))
        out[name] = dict(admissible=a['admissible'], why=a['why'], joint=joint, rejected=not (a['admissible'] and joint))
    return out


def match_target(target, adm, scores, ed, population, recal):
    out = {}
    n_el = sum(bool(target['eligible'][c]) for c in target['case_ids'])
    if n_el < 3:                                         # A2.3
        return dict(status='population too small', n_eligible=n_el, controls=dict(fires=False))
    for f in FAMILIES:
        cands = [k for k, a in adm.items() if a['family'] == f and a['admissible']]
        rows = []
        for k in cands:
            ef, et, eta, ids = common(target, scores[k])
            med, wst = ties_or_beats(ef, et, eta)
            rows.append(dict(key=k, cost=adm[k]['cost'], joint=med and wst, median=med, worst=wst, n_common=len(ids),
                             med_e=float(np.median(ef)), worst_e=float(np.max(ef)), edge=is_edge(k, ed)))
        rows.sort(key=lambda r: r['cost'])
        cheapest = rows[0]['key'] if rows else None
        pick = lambda pred: next((r for r in rows if pred(r)), None)
        joint = [r for r in rows if r['joint']]
        res_of = lambda k: FR.parse(k)[2] if FR.parse(k)[0] == 'fd' else FR.parse(k)[1]
        sm = [r for r in rows if res_of(r['key']) == target['mesh']]      # FD L = N; PS same resolution N_PS = N (A1.2)
        sm_joint = next((r for r in sm if r['joint']), None)
        sm_unm = min(sm, key=lambda r: (r['worst_e'], r['cost'])) if (sm and sm_joint is None) else None
        lab = lambda r: None if r is None else dict(r, boundary=r['key'] == cheapest, upper_bound=True)
        out[f] = dict(n_admissible=len(rows), joint=lab(joint[0] if joint else None), top3_joint=[r['key'] for r in joint[:3]],
                      median_only=lab(pick(lambda r: r['median'])), worst_only=lab(pick(lambda r: r['worst'])),
                      same_mesh=lab(sm_joint), same_mesh_unmatched=lab(sm_unm),
                      same_mesh_status=('matched' if sm_joint else ('unmatched' if sm else
                                        ('same-resolution PS not run (projected cost)' if f == 'PS' else 'not in ladder'))),
                      status=('matched' if joint else ('no ladder match' if rows else 'unavailable')))
    js = [(f, out[f]['joint']) for f in FAMILIES if out[f]['joint']]
    sj = [(f, out[f]['same_mesh']) for f in FAMILIES if out[f]['same_mesh']]
    out['same_mesh_union'] = dict(family=min(sj, key=lambda x: x[1]['cost'])[0], joint=min(sj, key=lambda x: x[1]['cost'])[1],
                                  families_available=[f for f in FAMILIES if out[f]['same_mesh_status'] in ('matched', 'unmatched')]) if sj else None
    out['union'] = dict(family=min(js, key=lambda x: x[1]['cost'])[0], joint=min(js, key=lambda x: x[1]['cost'])[1]) if js else None
    # ---- controls ----
    ctl = {}
    ks = synthetic(target, population, recal)
    ctl['ksyn'] = ks
    ctl['ksyn_unconverged_rejected'] = ks['K|unconverged']['rejected']
    ctl['ksyn_inaccurate_rejected'] = ks['K|inaccurate']['rejected']
    # K-false (A1.5): constructed from the target's own errors on its eligible cases; expected outcomes asserted first
    ids_t = [c for c in target['case_ids'] if target['eligible'][c]]
    if len(ids_t) < 3:
        ctl['kfalse'] = dict(n=len(ids_t), fires=False, note='population too small (n < 3): not evaluated')
        ctl['fires'] = False
        out['controls'] = ctl
        return out
    et = np.array([target['e'][c] for c in ids_t])
    eta = np.array([target['eta'][c] for c in ids_t])
    d = 1e-9
    clone_ok = all(ties_or_beats(et, et, eta))
    j = int(np.argmax(et))
    U = float(np.max(et + eta))
    e1 = et.copy()
    e1[j] = U + eta[j] + d
    pre1 = dict(median_unchanged=bool(np.median(e1 - eta) == np.median(et - eta)) if len(et) >= 3 else False,
                worst_exceeds=bool(np.max(e1 - eta) > U))
    m1, w1 = ties_or_beats(e1, et, eta)
    e2 = et + 2 * eta + d
    pre2 = dict(median_exceeds=bool(np.median(e2 - eta) > np.median(et + eta)), worst_exceeds=bool(np.max(e2 - eta) > U))
    m2, w2 = ties_or_beats(e2, et, eta)
    ctl['kfalse'] = dict(n=len(et), clone_matches=clone_ok, worst_case=ids_t[j], pre_worst=pre1, pre_median=pre2,
                         worst_raised=dict(median=m1, worst=w1), all_raised=dict(median=m2, worst=w2),
                         fires=bool(clone_ok and all(pre1.values()) and all(pre2.values()) and m1 and not w1 and not m2 and not w2))
    ctl['fires'] = bool(ctl['ksyn_unconverged_rejected'] and ctl['ksyn_inaccurate_rejected'] and ctl['kfalse']['fires']
                        and 'not converged / invalid on some case' in ks['K|unconverged']['why']
                        and ks['K|inaccurate']['admissible'] and not ks['K|inaccurate']['joint'])
    out['controls'] = ctl
    return out


def pareto(adm, scores, population):
    out = {}
    for f in FAMILIES:
        pts = []
        for k, a in adm.items():
            if a['family'] != f or not a['admissible']:
                continue
            ids = [c for c in population if scores[k][c]['ST']['eligible']]
            e = np.array([scores[k][c]['ST']['e'] for c in ids])
            pts.append(dict(key=k, cost=a['cost'], median=float(np.median(e)), worst=float(np.max(e))))
        front = [p for p in pts if not any(q['cost'] <= p['cost'] and q['median'] <= p['median'] and q['worst'] <= p['worst']
                                           and (q['cost'], q['median'], q['worst']) != (p['cost'], p['median'], p['worst'])
                                           for q in pts)]
        out[f] = dict(points=pts, front=sorted(front, key=lambda p: p['cost']))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ladder', action='append', required=True)
    ap.add_argument('--targets', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cfgs, groups, src, jobs, gpus = load_ladders(a.ladder)
    population = sorted({p['case'] for v in cfgs.values() for p in v['production']})
    scores, worst = rescore(cfgs, src)
    recal = recalibrate(groups)
    adm = admissibility(cfgs, groups, recal, scores, population)
    ed = edges(adm)
    T = json.loads(Path(a.targets).read_text())
    res = dict(jobs=jobs, gpus=sorted(gpus), store_version=RS.store_version(), offline_vs_injob_worst=worst,
               offline_agrees=bool(worst <= 1e-12), calibration=recal, calibration_agrees=all(v['agrees'] for v in recal.values()),
               admissibility=adm, edges=ed, pareto=pareto(adm, scores, population), targets_source=T.get('source'),
               label='every matched speed-up is an upper bound on the claimed ROM speed-up (finite ladder); edge = matched at '
                     'the family\'s coarsest resolution or largest step; boundary = matched at the screening-cheapest admissible member',
               matches={})
    for lab, t in T['targets'].items():
        res['matches'][lab] = match_target(t, adm, scores, ed, population, recal)
    res['controls_all_fire'] = all(m['controls']['fires'] for m in res['matches'].values() if m.get('status') != 'population too small')
    Path(a.out).write_text(json.dumps(res, indent=1, default=float) + '\n')
    print('offline agrees', res['offline_agrees'], 'calibration agrees', res['calibration_agrees'], 'controls', res['controls_all_fire'])
    for lab, m in res['matches'].items():
        print(lab, {f: (m[f]['status'], (m[f]['joint'] or {}).get('key')) for f in FAMILIES}, 'union', (m['union'] or {}).get('family'))


if __name__ == '__main__':
    main()
