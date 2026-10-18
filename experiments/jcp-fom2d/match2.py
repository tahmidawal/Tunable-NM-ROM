"""jcp-fom2d offline scoring, calibration recomputation and C1 FOM matching (DESIGN §3, §4, §6; A1.1, A1.5, A2.3, A2.5,
A2.6, A3.2–A3.4). CPU only. Fails closed: a failed acceptance, a store mismatch, an offline/in-job disagreement or a
control that does not fire leaves every match BLOCKED.

    python match2.py --smoke runs/smk1/archive/output/ladder_smk1 \
                     --ladder runs/<a>/archive/output/ladder_<cfg> [--ladder ...] [--extension-ladder ...] \
                     --targets checks/targets-<x>.json --out checks/match-<x>.json [--population all38|smoke]

1. Acceptance: the smoke result (gates, controls) and every ladder result must have `accepted` true; every job and the
   targets file must use the active E0 store version.
2. Offline rescoring of every saved lattice field against E0 R_ST (R_S where recorded): must agree with the in-job
   scores to 1e-12.
3. Calibration recomputed from the saved calibration fields and evidence (A2.6 recovery rule included).
4. Admissibility from the persisted evidence (not from summary booleans): FD per-stage residuals/Newton counts, stage and
   unit counts; PS finite fields and running maximum; warm screening rows (coverage, identity policy, finite positive).
5. Per target (its population as given: H or X variant, A2.3) and family: every joint / median-only / worst-only /
   same-mesh match with >= 3 common eligible cases; the C1 screening selection (lattice-contract screening cost; NOT
   the A2.1-confirmed C2 selection), coordinate-edge reasons and the boundary label; the same-mesh-unmatched
   lowest-worst-error comparator; union; PS anchors (A2.1/A3.4).
6. Extension trigger (A3.2) from the base ladder only.
7. Controls K-syn and K-false per target with n >= 3 eligible cases.
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
ALL38 = [f'dev6|{j}' for j in range(6)] + [f'val32|{j}' for j in range(32)]


def ties_or_beats(ea, ec, eta):
    """Step 2 / speed3d rule (interval overlap with endpoint propagation) on median and worst."""
    ea, ec, eta = np.asarray(ea, float), np.asarray(ec, float), np.asarray(eta, float)
    return bool(np.median(ea - eta) <= np.median(ec + eta)), bool(np.max(ea - eta) <= np.max(ec + eta))


def load_ladders(paths, ext=()):
    cfgs, groups, src, jobs, base = {}, {}, {}, [], set()
    for p in list(paths) + list(ext):
        p = Path(p)
        r = json.loads((p / 'result.json').read_text())
        jobs.append(dict(dir=str(p), job_id=r.get('job_id'), gpu=r.get('gpu'), commit=r.get('commit'),
                         store_version=r.get('store_version'), complete=r.get('complete') is True,
                         accepted=r.get('accepted') is True, extension=str(p) in [str(x) for x in ext]))
        for k, v in r['configs'].items():
            assert k not in cfgs, f'duplicate configuration {k}'
            cfgs[k] = v
            src[k] = p
            if str(p) not in [str(x) for x in ext]:
                base.add(k)
        for k, v in r.get('groups', {}).items():
            assert k not in groups, f'duplicate group {k}'
            groups[k] = dict(v, _dir=str(p))
    return cfgs, groups, src, jobs, base


def rescore(cfgs, src):
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


def evidence_ok(sp, ev):
    """Recompute the convergence predicate from the persisted evidence (fomrun2.FOMs.converged)."""
    if sp[0] == 'ps':
        um = ev.get('umax')
        return bool(ev.get('finite') is True and um is not None and np.isfinite(um))
    res, nwt = np.asarray(ev.get('res', []), float), np.asarray(ev.get('nwt', []), int)
    _, fam, L, dt, lev = sp
    want = FR.FC.n_stages(dt, FR.FAM[fam][1])
    return bool(ev.get('finite') is True and ev.get('stages') == want == len(res) == len(nwt)
                and ev.get('units') == 2 * int(round(0.25 / dt)) and np.all(np.isfinite(res))
                and np.all(res <= FR.TOLS[lev][0]) and np.all((nwt >= 0) & (nwt <= FR.NEWTON_LIMIT.get(lev, FR.MAX_NEWTON))))


def recalibrate(groups):
    res = {}
    for gk, g in groups.items():
        d = Path(g['_dir'])
        z = np.load(d / f"C_{gk.replace('|', '_')}.npz")
        levels = [str(x) for x in z['levels']]
        n0 = np.asarray(z['n0'])
        _, fam, Ls, dts = gk.split('|')
        conv = {lev: [evidence_ok(('fd', fam, int(Ls[1:]), FR.canon_dt(dts), lev), r['evidence'])
                      for r in g['calibration'][lev]] for lev in levels}
        t0ok = all(conv['0'])
        diffs = {lev: [max(float(np.linalg.norm(z[f'F{lev}'][i][k] - z['F0'][i][k])) / n0[i] for k in range(1, 6))
                       for i in range(len(n0))] for lev in levels[1:]}
        chosen = None
        if t0ok:
            chosen = '0'
            for lev in sorted(levels[1:], key=lambda s: -int(s)):
                if all(conv[lev]) and max(diffs[lev]) <= TOL_CAL:
                    chosen = lev
                    break
        passing = (['0'] if t0ok else []) + [lv for lv in levels[1:] if all(conv[lv]) and max(diffs[lv]) <= TOL_CAL]
        rec = g.get('recovery')
        prod_level, rec_ok = chosen, True
        if rec is not None:
            tighter = [lv for lv in passing if chosen is not None and int(lv) < int(chosen)]
            rec_ok = bool(tighter) and rec['level'] == max(tighter, key=int)
            prod_level = rec['level'] if rec_ok else None
        res[gk] = dict(chosen=chosen, injob_chosen=g['chosen'], agrees=bool(chosen == g['chosen'] and rec_ok),
                       production_level=prod_level, recovery=rec, max_diff={k: max(v) for k, v in diffs.items()})
    return res


def validate(k, v, sc, recal, population, nonbit=()):
    """One admissibility validator for real and synthetic (K-syn) records, from persisted evidence."""
    sp = FR.parse(k) if not k.startswith('K|') else None
    why = []
    if sp is not None and sp[0] == 'fd':
        gk = v.get('group')
        if gk not in recal or recal[gk]['production_level'] != sp[4] or not recal[gk]['agrees']:
            why.append('not calibrated at this level')
    ids = [p['case'] for p in v['production']]
    if sorted(ids) != sorted(population):
        why.append('case population')
    conv = [p['converged'] is True and (sp is None or evidence_ok(sp, p['evidence'])) for p in v['production']]
    if not all(conv):
        why.append('not converged / invalid on some case')
    if not all(c in sc and np.isfinite(sc[c]['ST']['e']) for c in ids):
        why.append('non-finite score')
    w = v.get('warm') or {}
    rows = w.get('times', [])
    ts = [x['seconds'] for x in rows]
    fam = FR.family(k) if sp is not None else 'K'
    row_ok = lambda r: r.get('bitwise', True) is True or (fam in nonbit and r.get('rel', np.inf) <= 1e-12)
    cov = sp is None or len(rows) == int(w.get('expected_rows', len(rows)))
    if not (ts and cov and all(row_ok(r) for r in rows + w.get('warmup', [])) and np.isfinite(ts).all() and min(ts) > 0):
        why.append('warm timing')
    return dict(admissible=not why, why=why, cost=float(np.median(ts)) if ts else np.inf, family=fam)


def common(target, sc):
    ids = [c for c in target['case_ids'] if target['eligible'][c] and sc[c]['ST']['eligible']]
    et = np.array([target['e'][c] for c in ids])
    eta = np.array([target['eta'][c] for c in ids])
    ef = np.array([sc[c]['ST']['e'] for c in ids])
    assert np.allclose(eta, [sc[c]['ST']['eps_ref'] for c in ids], rtol=0, atol=1e-15), 'eta differs'
    return ef, et, eta, ids


def resolution(k):
    sp = FR.parse(k)
    return (sp[2], sp[3]) if sp[0] == 'fd' else (sp[1], sp[2])


def edges(adm, keys):
    out = {}
    for f in FAMILIES:
        rs = [resolution(k) for k in keys if adm[k]['family'] == f and adm[k]['admissible']]
        if rs:
            out[f] = dict(min_res=min(r for r, _ in rs), max_dt=max(d for _, d in rs))
    return out


def edge_reasons(k, ed):
    f = FR.family(k)
    if f not in ed:
        return []
    r, d = resolution(k)
    out = []
    if r == ed[f]['min_res']:
        out.append('resolution edge')
    if abs(d - ed[f]['max_dt']) < 1e-15:
        out.append('step edge (step = output spacing)' if abs(d - 0.05) < 1e-15 else 'step edge')
    return out


def synthetic(target, population, recal):
    warm = dict(times=[dict(seconds=1e-9, bitwise=True)], hashes_ok=True)
    out = {}
    for name, conv0, ef in (('K|unconverged', False, lambda c: 0.), ('K|inaccurate', True,
                            lambda c: 10 * target['e'].get(c, 1.) + 4 * target['eta'].get(c, 0.) + 1e-3)):
        v = dict(production=[dict(case=c, converged=(conv0 or i != 0), evidence={}) for i, c in enumerate(population)],
                 warm=warm)
        sc = {c: {'ST': dict(e=ef(c), eps_ref=target['eta'].get(c, 0.), eligible=target['eligible'].get(c, False))}
              for c in population}
        a = validate(name, v, sc, recal, population)
        joint = False
        if a['admissible']:
            ef_, et, eta, _ = common(target, sc)
            joint = all(ties_or_beats(ef_, et, eta))
        out[name] = dict(admissible=a['admissible'], why=a['why'], joint=joint, rejected=not (a['admissible'] and joint))
    return out


def kfalse(target):
    ids = [c for c in target['case_ids'] if target['eligible'][c]]
    et = np.array([target['e'][c] for c in ids])
    eta = np.array([target['eta'][c] for c in ids])
    d = 1e-9
    j = int(np.argmax(et))                       # first case (population order) with the largest raw error
    U = float(np.max(et + eta))
    e1 = et.copy()
    e1[j] = U + eta[j] + d
    pre1 = dict(worst_exceeds=bool(np.max(e1 - eta) > U), median_within=bool(np.median(e1 - eta) <= np.median(et + eta)))
    e2 = et + 2 * eta + d
    pre2 = dict(median_exceeds=bool(np.median(e2 - eta) > np.median(et + eta)), worst_exceeds=bool(np.max(e2 - eta) > U))
    m0, w0 = ties_or_beats(et, et, eta)
    m1, w1 = ties_or_beats(e1, et, eta)
    m2, w2 = ties_or_beats(e2, et, eta)
    return dict(n=len(ids), worst_case=ids[j], clone=dict(median=m0, worst=w0), pre_worst=pre1, pre_median=pre2,
                worst_raised=dict(median=m1, worst=w1), all_raised=dict(median=m2, worst=w2),
                fires=bool(m0 and w0 and all(pre1.values()) and all(pre2.values()) and m1 and not w1 and not m2 and not w2))


def anchor(k, adm, src):
    """PS anchor (A2.1/A3.4): lattice distance to the same N at the next smaller ladder step (the next larger for the
    smallest), max over outputs and cases, relative to ||u(t0)||."""
    sp = FR.parse(k)
    sib = sorted(FR.parse(x)[2] for x in adm if adm[x]['family'] == 'PS' and FR.parse(x)[1] == sp[1])
    smaller = [d for d in sib if d < sp[2]]
    larger = [d for d in sib if d > sp[2]]
    if not (smaller or larger):
        return dict(status='unavailable (no adjacent ladder step)')
    d2 = max(smaller) if smaller else min(larger)
    k2 = FR.key(('ps', sp[1], d2))
    if not adm[k2]['admissible']:
        return dict(status='unavailable (adjacent member invalid)', other=k2)
    za = np.load(src[k] / f"F_{k.replace('|', '_')}.npz")
    zb = np.load(src[k2] / f"F_{k2.replace('|', '_')}.npz")
    A, B = za['fields'], zb['fields']
    dist = max(float(np.linalg.norm(A[i][t] - B[i][t])) / float(np.linalg.norm(B[i][0])) for i in range(len(A)) for t in range(1, 6))
    return dict(status='ok', steps=[sp[2], d2], other=k2, both_valid=True, distance=dist)


def match_target(target, adm, scores, ed, population, recal, src):
    n_el = sum(bool(target['eligible'][c]) for c in target['case_ids'])
    if n_el < 3:
        return dict(status='population too small', n_eligible=n_el)
    out = dict(status='ok', n_eligible=n_el)
    for f in FAMILIES:
        rows = []
        for k, a in adm.items():
            if a['family'] != f or not a['admissible']:
                continue
            ef, et, eta, ids = common(target, scores[k])
            if len(ids) < 3:
                continue
            med, wst = ties_or_beats(ef, et, eta)
            rows.append(dict(key=k, cost=a['cost'], joint=med and wst, median=med, worst=wst, n_common=len(ids),
                             med_e=float(np.median(ef)), worst_e=float(np.max(ef)), edge=edge_reasons(k, ed)))
        rows.sort(key=lambda r: (r['cost'], r['key']))
        cheapest = rows[0]['key'] if rows else None
        lab = lambda r: None if r is None else dict(r, boundary=r['key'] == cheapest, upper_bound=True,
                                                     selection_stage='C1 lattice screening (not the A2.1 C2 selection)')
        joint = [r for r in rows if r['joint']]
        sm = [r for r in rows if resolution(r['key'])[0] == target['mesh']]
        sm_joint = [r for r in sm if r['joint']]
        sm_unm = min(sm, key=lambda r: (r['worst_e'], r['cost'], r['key'])) if (sm and not sm_joint) else None
        pick = lambda pred: next((r for r in rows if pred(r)), None)
        out[f] = dict(n_admissible=len(rows), joint=lab(joint[0] if joint else None),
                      candidates=dict(joint=[r['key'] for r in joint], median_only=[r['key'] for r in rows if r['median']],
                                      worst_only=[r['key'] for r in rows if r['worst']], same_mesh=[r['key'] for r in sm_joint],
                                      same_mesh_unmatched=[sm_unm['key']] if sm_unm else []),
                      median_only=lab(pick(lambda r: r['median'])), worst_only=lab(pick(lambda r: r['worst'])),
                      same_mesh=lab(sm_joint[0] if sm_joint else None), same_mesh_unmatched=lab(sm_unm),
                      same_mesh_status=('matched' if sm_joint else ('unmatched' if sm else
                                        ('same-resolution PS not run (deferred)' if f == 'PS' else 'not in ladder'))),
                      status=('matched' if joint else ('no ladder match' if rows else 'unavailable')))
        if f == 'PS' and joint:
            out[f]['anchor'] = anchor(joint[0]['key'], adm, src)
    js = [(f, out[f]['joint']) for f in FAMILIES if out[f]['joint']]
    out['union'] = dict(family=min(js, key=lambda x: (x[1]['cost'], x[1]['key']))[0],
                        joint=min(js, key=lambda x: (x[1]['cost'], x[1]['key']))[1]) if js else None
    sj = [(f, out[f]['same_mesh']) for f in FAMILIES if out[f]['same_mesh']]
    out['same_mesh_union'] = dict(family=min(sj, key=lambda x: (x[1]['cost'], x[1]['key']))[0],
                                  joint=min(sj, key=lambda x: (x[1]['cost'], x[1]['key']))[1],
                                  families_run=[f for f in FAMILIES if out[f]['same_mesh_status'] in ('matched', 'unmatched')]) if sj else None
    ks = synthetic(target, population, recal)
    kf = kfalse(target)
    out['controls'] = dict(ksyn=ks, kfalse=kf,
                           fires=bool(ks['K|unconverged']['rejected'] and 'not converged / invalid on some case' in ks['K|unconverged']['why']
                                      and ks['K|inaccurate']['admissible'] and not ks['K|inaccurate']['joint'] and kf['fires']))
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
    ap.add_argument('--smoke', required=True)
    ap.add_argument('--ladder', action='append', required=True)
    ap.add_argument('--extension-ladder', action='append', default=[])
    ap.add_argument('--targets', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--population', default='all38', choices=['all38', 'smoke'])
    ap.add_argument('--nonbitwise-ok-families', default='')
    a = ap.parse_args()
    nonbit = tuple(x for x in a.nonbitwise_ok_families.split(',') if x)
    sm = json.loads((Path(a.smoke) / 'result.json').read_text())
    cfgs, groups, src, jobs, base = load_ladders(a.ladder, a.extension_ladder)
    T = json.loads(Path(a.targets).read_text())
    sv = RS.store_version()
    acceptance = dict(smoke_accepted=sm.get('accepted') is True and sm.get('complete') is True,
                      ladders_complete=all(j['complete'] for j in jobs),
                      ladders_accepted=all(j['accepted'] for j in jobs),
                      store_versions_equal=bool(all(j['store_version'] == sv for j in jobs) and sm.get('store_version') == sv
                                                and T.get('store_version') == sv))
    population = ALL38 if a.population == 'all38' else sorted({p['case'] for v in cfgs.values() for p in v['production']})
    scores, worst = rescore(cfgs, src)
    acceptance['offline_agrees'] = bool(worst <= 1e-12)
    recal = recalibrate(groups)
    acceptance['calibration_agrees'] = all(v['agrees'] for v in recal.values())
    adm = {k: validate(k, v, scores[k], recal, population, nonbit) for k, v in cfgs.items()}
    ed_all = edges(adm, list(adm))
    ed_base = edges(adm, [k for k in adm if k in base])
    res = dict(jobs=jobs, gpus=sorted({j['gpu'] for j in jobs}), store_version=sv, population=a.population,
               offline_vs_injob_worst=worst, calibration=recal, admissibility=adm, edges=ed_all, edges_base=ed_base,
               pareto=pareto(adm, scores, population), targets_source=T.get('source'),
               label='every matched speed-up is an upper bound on the claimed ROM speed-up (finite ladder); C1 selections '
                     'use lattice-contract screening costs and are not the C2 (A2.1) selections', matches={})
    for lab, t in T['targets'].items():
        res['matches'][lab] = match_target(t, adm, scores, ed_all, population, recal, src)
    evald = [m for m in res['matches'].values() if m.get('status') == 'ok']
    acceptance['controls_all_fire'] = bool(evald) and all(m['controls']['fires'] for m in evald)
    res['acceptance'] = acceptance
    res['blocked'] = not all(acceptance.values())
    # A3.2 extension trigger: a family's screening-cheapest joint match (any target) at its base-ladder minimum resolution
    trig = {}
    for f in FAMILIES:
        hits = [lab for lab, m in res['matches'].items() if m.get('status') == 'ok' and m[f]['joint']
                and m[f]['joint']['key'] in base and f in ed_base and resolution(m[f]['joint']['key'])[0] == ed_base[f]['min_res']]
        trig[f] = dict(fires=bool(hits), targets=hits, base_min_resolution=ed_base.get(f, {}).get('min_res'))
    res['extension_trigger'] = trig
    Path(a.out).write_text(json.dumps(res, indent=1, default=float) + '\n')
    print('acceptance', acceptance, 'BLOCKED' if res['blocked'] else 'ok')
    for lab, m in res['matches'].items():
        if m.get('status') != 'ok':
            print(lab, m.get('status'))
            continue
        print(lab, {f: (m[f]['status'], (m[f]['joint'] or {}).get('key')) for f in FAMILIES}, 'union', (m['union'] or {}).get('family'))
    print('extension trigger', {f: v['fires'] for f, v in trig.items()})


if __name__ == '__main__':
    main()
