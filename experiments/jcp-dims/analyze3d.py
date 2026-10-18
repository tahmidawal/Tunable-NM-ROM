"""jcp-dims 3D analysis (DESIGN 2.3, 3, 6.2, 6.4, 6.5, A1.2, A1.5, A2.2, A3.2; CPU only). From the collected d3run archives
(q3r<k>, m3a, m3b, k3, b3f, b3m) and their offline E0 scores (score3.py -> checks/scores-<att>.json):

  * per ladder configuration: Q1 (speed3d eligibility + ADT anchors, DESIGN 3), ladder Q2 (in-job successor distance
    recomputed offline from the archived coefficients with the lattice rows of score3.glat, rho from the job), labels;
  * per adaptive configuration (k3): Q1 (validity, pooled certificate <= 1 %, G-cert, tight / tighter), ladder Q2;
  * rounds members: qualify's Q2-3D label (same predicate; Q1 = speed3d eligibility, cited);
  * m* per (setting, spec, family, draw), width m* (intersection over the registered specs), family m* (all draws),
    point efficiency, cross-family gate (A3.2), B6-3D transitions and M = r control, A3-3D fixed-(M, m) transitions;
  * the qualified 3D configuration list (for t3s) and, after t3s, the H2q selection (dstats.h2q) for t3c.

    JAX_PLATFORMS=cpu python analyze3d.py [--after-screen]
"""
from __future__ import annotations

import offline  # noqa: F401  (CPU, highest, x64)

import argparse
import json
import re
from pathlib import Path

import numpy as np

import dstats as DS

HERE = Path(__file__).resolve().parent
RUNS = HERE / 'runs'
CFG = HERE / 'configs'
SPEED3D = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d')
QUALIFY = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-qualify/experiments/jcp-qualify')
BAR, BAR_TT, RHO = 2.5e-4, 2.5e-5, .116
COMPARATORS = {'C_old': 'M2_512|gl32|fs|BE|0.01|K1', 'C_W768': 'W768|lat32768|fs|BE|0.01|K1',
               'C_W1024': 'W1024|lat65536|fs|BE|0.01|K1'}


def out_dir(att):
    d = RUNS / att / 'archive' / 'output' / f'd3run_{att}'
    return d if (d / 'result.json').exists() else None


def m3(rule):
    f, n = re.match(r'^([a-z]+)(\d+)', rule).groups()
    n = int(n)
    return {'gl': n ** 3, 'smolyak': {8: 2559, 9: 6271, 10: 15103, 11: 35839, 12: 83967, 13: 194559}.get(n)}.get(f, n)


def scores(att):
    p = HERE / 'checks' / f'scores-{att}.json'
    return json.loads(p.read_text())['scores'] if p.exists() else {}


def glat(setting):
    import score3
    return score3.glat(setting)


def offline_dist(W, Wn, setting, n0):
    G = glat(setting)
    D = (np.asarray(W) - np.asarray(Wn)) @ G
    return (np.linalg.norm(D, axis=-1) / np.asarray(n0)[:, None])[:, 1:].max(1)


def n0s(cases):
    import c3common as C3
    tab = C3.C.table(923801, 64)
    return np.array([float(np.linalg.norm(np.asarray(C3.C.initial_interior(65, tab, j)))) for j in cases])


# ------------------------------------------------------------------------------------------------ ladders
def q1_ladder(e):
    steps = e['steps']
    return dict(valid=all(e['valid']), certificate=bool(steps > 0 and e['nonstationary'] <= .01 * steps),
                distance=bool(max(e['dist_adt']) <= BAR), anchors_valid=all(e['adt_valid']) and all(e['adt_tighter_valid']),
                tight_tighter=bool(e['adt_tight_tighter'] and max(e['adt_tight_tighter']) <= BAR_TT))


def q1_adaptive(z):
    ok = dict(valid=bool(np.all(z['valid'])), certificate=bool(np.sum(z['noncert']) <= .01 * z['noncert'].size * z['cert'].shape[1]),
              gcert=bool(np.all(z['gcert'])), distance=bool(np.max(z['dist_tight']) <= BAR),
              anchors_valid=bool(np.all(z['valid_tight']) and np.all(z['valid_tighter'])),
              tight_tighter=bool(np.max(z['dist_tight_tighter']) <= BAR_TT))
    return ok


def collect():
    """All ladder / adaptive configurations: key -> dict(Q1, entries for Q2, e0, setting, rule, spec, M, att)."""
    C, ladders = {}, []
    for att in ('m3a', 'm3b', 'k3', 'b3f', 'b3m'):
        d = out_dir(att)
        if d is None:
            continue
        res = json.loads((d / 'result.json').read_text())
        sc = scores(att)
        W = np.load(d / f"W_{res['mode']}.npz")
        for blk, rb in zip(res['config']['blocks'], res['blocks']):
            if blk['kind'] == 'ladder':
                ents = rb['result']
                for sp in blk['specs']:
                    keys = [f"{blk['setting']}|{rl}|{sp}" for rl in blk['rungs']]
                    ladders.append(dict(att=att, kind='ladder', setting=blk['setting'], spec=sp, rungs=blk['rungs'], keys=keys))
                    for k in keys:
                        if k in ents:
                            e = ents[k]
                            C[k] = dict(att=att, kind='ladder', entry=e, Q1=q1_ladder(e), setting=blk['setting'], spec=sp,
                                        rule=e['rule'], M=e['M'], r=e['Rp'], cases=e['cases'], e0=sc.get(k), W=W[k])
            elif blk['kind'] == 'adaptive':
                for f in blk['forms']:
                    keys = [f"{blk['setting']}|{rl}|ad|{f}|{float(blk['dt']):.8g}" for rl in blk['rungs']]
                    ladders.append(dict(att=att, kind='adaptive', setting=blk['setting'], spec=f'ad|{f}', rungs=blk['rungs'], keys=keys))
                    for k, rl in zip(keys, blk['rungs']):
                        p = d / f"AD_{k.replace('|', '_')}.npz"
                        if p.exists() and k in rb['result']:
                            z = np.load(p)
                            C[k] = dict(att=att, kind='adaptive', entry=rb['result'][k], Q1=q1_adaptive(z), setting=blk['setting'],
                                        spec=f'ad|{f}', rule=rl, M=int(z['M']), r=int(z['Rp']), cases=[int(x) for x in z['case_ids']],
                                        e0=sc.get(k + '|prod'), W=np.asarray(z['W']), rho=np.asarray(z['rho']))
    return C, ladders


def ladder_q2(C, keys, i):
    k = keys[i]
    if i >= len(keys) - 2:
        if i == len(keys) - 2 and k in C and keys[i + 1] in C:
            e = C[k]['entry']
            if e.get('dist_next') is not None and (max(e['dist_next']) > BAR or (e.get('rho') and np.nanmax(np.asarray(e['rho'], float)) > RHO)):
                return dict(label='fail', worst_dist_qp=max(e['dist_next']))
        return dict(label='unverified (ladder top)')
    if keys[i + 1] not in C:
        return dict(label='unverified (successor invalid)')
    e, en = C[k]['entry'], C[keys[i + 1]]['entry']
    if e.get('dist_next') is None:
        return dict(label='unverified (successor invalid)')
    # offline recomputation of the successor distance (score3 lattice rows) on every case
    cases = C[k]['cases']
    off = offline_dist(C[k]['W'], C[keys[i + 1]]['W'], C[k]['setting'], n0s(cases))
    agree = bool(np.allclose(off, e['dist_next'], rtol=1e-9, atol=1e-13))
    rho = np.asarray(e['rho'] if 'rho' in e else C[k].get('rho'), float).ravel()
    rho_f = rho[~np.isnan(rho)]
    chk = None
    if keys[i + 2] in C and en.get('dist_next') is not None:
        idx = [cases.index(j) for j in range(8) if j in cases]
        chk = (bool(all(en['next_valid'][j] for j in idx)), float(max(en['dist_next'][j] for j in idx)))
    lab = DS.q2_label(True, (bool(all(e['next_valid'])), float(max(e['dist_next']))), float(rho_f.max()) if rho_f.size else np.nan, chk)
    if not agree:
        lab = 'blocked (offline recomputation disagrees)'
    return dict(label=lab, worst_dist_qp=float(max(e['dist_next'])), rho_max=float(rho_f.max()) if rho_f.size else None,
                check=chk, offline_agrees=agree)


# ------------------------------------------------------------------------------------------------ rounds
def rounds():
    """Round members (q3r<k>): qualify's Q2-3D predicate (A1.1 / A2.1), without its reproduction gate (the reproduction
    of speed3d archives is checked once in the smoke and in m3 for overlapping configurations)."""
    out = {}
    for k in range(4, 9):
        d = out_dir(f'q3r{k}')
        if d is None:
            continue
        res = json.loads((d / 'result.json').read_text())
        for blk in res['blocks']:
            if blk['kind'] != 'rounds':
                continue
            for key, m in blk['result'].items():
                if key.startswith('control|'):
                    out[f'round{k}|{key}'] = dict(label='control', worst=max(m['dist_qp']))
                    continue
                dq = np.asarray(m['dist_qp'], float)
                rho = np.asarray(m['rho'], float).ravel()
                rho = rho[~np.isnan(rho)]
                chk = (all(m['valid_qpp']), float(max(m['dist_qp_qpp']))) if m.get('qpp') else None
                lab = DS.q2_label(True, (all(m['valid_qp']), float(dq.max())), float(rho.max()) if rho.size else np.nan, chk)
                if not all(m['valid']):
                    lab = 'fail (own runs invalid)'
                out[key] = dict(label=lab, worst_dist_qp=float(dq.max()), rho_max=float(rho.max()) if rho.size else None,
                                check=chk, round=k)
    return out


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--after-screen', action='store_true')
    a = ap.parse_args()
    C, ladders = collect()
    lab = {}
    out = dict(label='E0 R_ST: conditional Richardson estimate (E0 A10; combined A10, A11, retracted A15 carried)',
               configurations={}, mstar={}, width_mstar={}, rounds=rounds(), families={}, B6={}, A3={}, front={}, h2q={})
    for L in ladders:
        labels = []
        for i, k in enumerate(L['keys']):
            if k not in C:
                labels.append('missing')
                continue
            q1 = C[k]['Q1']
            q2 = ladder_q2(C, L['keys'], i)
            qual = bool(all(q1.values()) and q2['label'] == 'pass')
            lab[k] = qual
            out['configurations'][k] = dict(Q1=q1, Q1_pass=all(q1.values()), Q2=q2, qualified=qual, M=C[k]['M'], r=C[k]['r'],
                                            rule=C[k]['rule'], spec=C[k]['spec'], setting=C[k]['setting'],
                                            median=float(np.median([v['e'] for v in C[k]['e0'].values()])) if C[k]['e0'] else None,
                                            worst=float(np.max([v['e'] for v in C[k]['e0'].values()])) if C[k]['e0'] else None)
            labels.append('qualified' if qual else 'not qualified')
        cand = L['rungs'][:-2]
        fam = re.match(r'^([a-z]+)', L['rungs'][0]).group(1)
        draw = re.search(r's(\d+)$', L['rungs'][0])
        lid = f"{L['att']}#{L['setting']}#{L['spec']}#{fam}#s{draw.group(1) if draw else 0}"
        ms = DS.mstar(labels[:len(cand)], [m3(r) for r in cand])
        out['mstar'][lid] = dict(mstar=ms, labels=labels, rungs=L['rungs'], att=L['att'], setting=L['setting'], spec=L['spec'],
                                 family=fam, draw=int(draw.group(1)) if draw else 0)
    # width m* (3M): intersection over the three specs per (setting, family)
    for st in ('M2_512', 'W256', 'W512', 'W768', 'W1024'):
        for fam in ('gl', 'lat'):
            per = {v['spec']: v for v in out['mstar'].values() if v['setting'] == st and v['att'] in ('m3a', 'm3b')
                   and v['family'] == fam}
            if per:
                cand = next(iter(per.values()))['rungs'][:-2]
                out['width_mstar'][f'{st}|{fam}'] = DS.width_mstar({k: v['labels'][:len(cand)] for k, v in per.items()},
                                                                    [m3(r) for r in cand])
    # per-width coverage (6.5)
    qual_keys = [k for k, v in lab.items() if v] + [k for k, v in out['rounds'].items() if v.get('label') == 'pass']
    qd = json.loads((QUALIFY / 'runs' / 'EXPORT.json').read_text())['threeD']['tested']
    qual_keys += [k for k, v in qd.items() if v == 'pass']
    out['coverage'] = {w: sorted({k for k in qual_keys if k.startswith(w + '|') or re.match(rf'^{w}(k\d|M\d+)\|', k)})
                       for w in ('M2_512', 'W256', 'W512', 'W768', 'W1024')}
    out['qualified_3d'] = sorted(set(qual_keys))
    # ---- E0 per case for any key (lane scores first, then speed3d's grid scores)
    def e0_of(k):
        if k in C and C[k]['e0']:
            return {int(j): v for j, v in C[k]['e0'].items()}
        for g in ('g1', 'g2', 'g3'):
            p = SPEED3D / 'checks' / f'scores-{g}.json'
            sc = json.loads(p.read_text())['scores'] if p.exists() else {}
            if k in sc:
                return {int(j): v for j, v in sc[k].items()}
        return None

    def arr(kX, kY):
        a_, b_ = e0_of(kX), e0_of(kY)
        ids = sorted(set(a_) & set(b_))
        return (np.array([a_[j]['e'] for j in ids]), np.array([b_[j]['e'] for j in ids]),
                np.array([a_[j]['eps_ref'] for j in ids]),
                np.array([a_[j]['eligible'] and all(C[kX]['entry'].get('valid', [True])) and all(C[kY]['entry'].get('valid', [True]))
                          if kX in C and kY in C else a_[j]['eligible'] for j in ids]))

    def tr(kX, kY):
        if kX not in C or kY not in C or not C[kX]['e0'] or not C[kY]['e0']:
            return dict(label='inconclusive (missing)', X=kX, Y=kY)
        a_, b_, e_, el = arr(kX, kY)
        lab_, p = DS.transition(a_, b_, e_, el, qualified=(lab.get(kX, False), lab.get(kY, False)))
        return dict(label=lab_, stats=p, X=kX, Y=kY)
    # ---- B6-3D (W512, primary rung lat65536)
    ad = lambda k, f: f'W512k{k}|lat65536|ad|{f}|CN|0.01'
    fsk = lambda k: 'W512|lat65536|fs|CN|0.01|K1' if k == 4 else f'W512k{k}|lat65536|fs|CN|0.01|K1'
    for f in ('LSPG', 'GAL'):
        t_ = {f'{a_}->{b_}': tr(ad(a_, f), ad(b_, f)) for a_, b_ in ((1, 2), (2, 3), (3, 4), (4, 8))}
        c_ = tr(ad(1, f), ad(4, f))
        ctl = DS.control_mr(c_['label'], (lab.get(ad(1, f), False), lab.get(ad(4, f), False)))
        out['B6'][f'W512|{f}-CN adaptive'] = dict(transitions=t_, control_1_to_4=c_, control=ctl,
                                                  range_2_4=DS.range_verdict([t_['2->3']['label'], t_['3->4']['label']]) if ctl == 'fired'
                                                  else ('test-space sensitivity not demonstrated' if ctl == 'did not fire' else 'blocked (control unavailable)'))
    ctl_ad = out['B6']['W512|LSPG-CN adaptive']['control']
    t_ = {f'{a_}->{b_}': tr(fsk(a_), fsk(b_)) for a_, b_ in ((2, 3), (3, 4), (4, 8))}
    out['B6']['W512|fs-CN (control from the adaptive LSPG form)'] = dict(transitions=t_, control=ctl_ad,
        range_2_4=DS.range_verdict([t_['2->3']['label'], t_['3->4']['label']]) if ctl_ad == 'fired' else 'blocked or not demonstrated')
    for k in (1, 2, 3, 4, 8):
        g, l_ = ad(k, 'GAL'), ad(k, 'LSPG')
        if g in C and l_ in C and C[g]['e0'] and C[l_]['e0']:
            a_, b_, e_, el = arr(g, l_)
            out['B6'].setdefault('GAL vs LSPG (adaptive CN)', {})[f'kappa{k}'] = dict(zip(('label', 'stats'), DS.two_way(
                a_, b_, e_, el, qualified=(lab.get(g, False), lab.get(l_, False)), names=('GAL', 'LSPG'))))
    # ---- A3-3D
    fx = lambda w: 'W512|lat65536|fs|CN|0.01|K1' if w == 'W512' else f'{w}M2052|lat65536|fs|CN|0.01|K1'
    ws = ('W256', 'W512', 'W768', 'W1024')
    t1 = [tr(fx(ws[i]), fx(ws[i + 1])) for i in range(3)]
    t2_ = [tr(f'{ws[i]}|lat65536|fs|CN|0.01|K1', f'{ws[i + 1]}|lat65536|fs|CN|0.01|K1') for i in range(3)]
    out['A3'] = {'fixed (M=2052 frozen list, lat65536)': dict(transitions=t1, range=DS.range_verdict([x['label'] for x in t1])),
                 'fixed m lat65536, nominal M': dict(transitions=t2_, range=DS.range_verdict([x['label'] for x in t2_]))}
    # ---- B3-3D families at W512 kappa 4, fs|CN
    fam = {}
    for v in out['mstar'].values():
        if v['setting'] == 'W512' and v['spec'] == 'fs|CN|0.01|K1' and v['att'] in ('m3a', 'b3f', 'b3m'):
            fam.setdefault(v['family'], {})[v['draw']] = v
    fres = {}
    for f, dr in fam.items():
        cand = next(iter(dr.values()))['rungs'][:-2]
        sizes = [m3(r) for r in cand]
        if len(dr) == 1:
            fres[f] = dict(family=next(iter(dr.values()))['mstar'])
        else:
            fres[f] = DS.family_mstar([dr[s_]['labels'][:len(cand)] for s_ in sorted(dr)], sizes)
    gates = {}
    tgt = 'W512|gl64|fs|CN|0.01|K1'
    fg = {}
    for att in ('b3f', 'b3m'):
        d = out_dir(att)
        if d is not None:
            for b in json.loads((d / 'result.json').read_text())['blocks']:
                if b['kind'] == 'fixgate':
                    fg.update(b['result']['rules'])
    for f, dr in fam.items():
        for s_, v in dr.items():
            fin = v['rungs'][-1]
            kf = f'W512|{fin}|fs|CN|0.01|K1'
            avail = kf in C and tgt in C and all(C[kf]['Q1'].values()) and all(C[tgt]['Q1'].values())
            dist = np.inf
            if avail:
                dist = float(offline_dist(C[kf]['W'], C[tgt]['W'], 'W512', n0s(C[kf]['cases'])).max())
            rb = fg.get(fin, {}).get('rho_tgt', [])
            gates[f'{f}|s{s_}'] = dict(rule=fin, gate=DS.cross_family_gate(rb, avail, dist), worst_dist_to_target=dist)
    gates['badlat262144 (control)'] = dict(gate=DS.cross_family_gate(fg.get('badlat262144', {}).get('rho_tgt', []), False, np.inf))
    pe = {}
    nm = sorted(fres)
    for i in range(len(nm)):
        for j in range(i + 1, len(nm)):
            ok = all(v['gate'] == 'pass' for kk, v in gates.items() if kk.split('|')[0] in (nm[i], nm[j]))
            pe[f'{nm[i]} vs {nm[j]}'] = DS.point_efficiency(fres[nm[i]]['family']['m'], fres[nm[j]]['family']['m'], names=(nm[i], nm[j])) \
                if ok else 'inconclusive (gate flagged or unavailable)'
    out['families'] = dict(mstar=fres, gates=gates, point_efficiency=pe)
    # ---- H2q (after the t3s screening panel)
    if a.after_screen:
        d = out_dir('t3s')
        res = json.loads((d / 'result.json').read_text())
        pan = res['timing']['screen']
        ratios = {}
        for x in pan['invocations']:
            ratios.setdefault(x['B'], []).append(x['ratio'])
        CC = {}
        for k in set(out['qualified_3d']) | set(COMPARATORS.values()):
            name = k if k.split('|')[2] != 'ad' else k + '|prod'
            e = e0_of(k)
            if e is None:
                continue
            el = {j: v for j, v in e.items() if v['eligible']}
            q1 = all(C[k]['Q1'].values()) if k in C else True    # rounds / qualify passes: speed3d Q1 (cited)
            cost = float(np.median(ratios[name])) if name in ratios else (1.0 if k == COMPARATORS['C_old'] else None)
            CC[k] = dict(cost=cost, median=float(np.median([v['e'] for v in el.values()])),
                         worst=float(np.max([v['e'] for v in el.values()])), e={j: v['e'] for j, v in el.items()},
                         eta={j: v['eps_ref'] for j, v in el.items()}, q1=q1, screen_valid=bool(pan['valid'] and cost is not None))
        qset = {k for k in out['qualified_3d'] if k in CC}
        conf = {}
        dc = out_dir('t3c')
        if dc is not None:
            rc = json.loads((dc / 'result.json').read_text())
            for lb, t in rc['timing'].items():
                conf[lb.split('|', 1)[1]] = dict(valid=t['valid'], median_ratio=float(np.median([x['ratio'] for x in t['invocations']])))
        for cn, ck in COMPARATORS.items():
            r_ = DS.h2q(CC, ck, qset, conf.get(cn))
            r_.update(comparator=ck, comparator_q2=out['configurations'].get(ck, {}).get('Q2', {}).get('label'))
            out['h2q'][cn] = r_
        out['screen'] = {k: dict(cost=v['cost'], median=v['median'], worst=v['worst']) for k, v in CC.items()}
        pts = {k: dict(c=v['cost'], e=v['median'], eps=float(np.median(list(v['eta'].values())))) for k, v in CC.items()
               if k in qset and v['cost'] is not None}
        ptw = {k: dict(c=v['cost'], e=v['worst'], eps=float(np.median(list(v['eta'].values())))) for k, v in CC.items()
               if k in qset and v['cost'] is not None}
        import qstats as QS
        out['front'] = dict(median=QS.front(pts)[0] if pts else [], worst=QS.front(ptw)[0] if ptw else [],
                            label='qualified subset of the tested candidates (bounded search; never complete)')
    (HERE / 'checks').mkdir(exist_ok=True)
    (HERE / 'checks' / 'rounds3d.json').write_text(json.dumps(dict(labels={k: v['label'] for k, v in out['rounds'].items()
                                                                            if v.get('label') != 'control'}), indent=1) + '\n')
    (HERE / 'checks' / 'analysis-3d.json').write_text(json.dumps(out, indent=1, default=str) + '\n')
    print('3D configs', len(out['configurations']), 'qualified', len(out['qualified_3d']))


if __name__ == '__main__':
    main()
