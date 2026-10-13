"""Offline analysis of a `c3grid` result (DESIGN 6 H4, H6; A1.5, A1.6, A1.9, A2.8, A3.4, A5.3, A5.7, A6.2).

Inputs: <run>/result.json, W_grid.npz and a score.py file for the 3D reference (gate 'pass', bound to this result).
Per setting and arm (spec): reference error worst / median (reference-limited, PROVISIONAL), anchor discrepancy per
case to GAL-BDF2 dt0/8 tight (lattice field distance / ||u0||), anchor uncertainty U = ||anc8 - anc16|| + s8 + s16
(s = tight-vs-tighter), resolved iff discrepancy > 3U and the four anchor runs are verified; solver distance of every
fixed-sweep arm to the adaptive tight solve of the same scheme and step, and the adaptive probe's tight-vs-tighter
sensitivity; timing ratios (per-setting panel vs fs BE dt0; common panel vs the common baseline).
Eligibility (A6.2 + A1.9 H4 bar): all runs valid, adaptive probe verified with tighter sensitivity <= 0.1 x 2.5e-4, and
per case solver distance <= min(2.5e-4, 0.25 x own anchor discrepancy).
H4 (A2.8): comparison population = cases where the comparator (fs BE dt0) is resolved; needs >= 52 of 64; candidate
unresolved cases enter with the conservative bound 3U. H4a: an eligible candidate with dt >= 1.25 dt0, worst and median
anchor <= comparator's, median panel ratio <= 0.6. H4b: an eligible candidate at dt0 with median anchor <= 0.1 x the
comparator's and ratio <= 1.1. Selection: fast-equal-accuracy (H4a's cheapest) and accurate-equal-cost (smallest
median anchor, ratio <= 1). Withheld if G-FS1's CN control is non-discriminating (A5.7) or the panel is invalid.
H6: along W256 -> W512 -> W768 -> W1024 the accurate-equal-cost arm's median reference error falls >= 10 % with >= 60 %
of cases improving (reference-limited, PROVISIONAL).
Budget arms (A1.5): fs BE dt0, the H4a/H4b selected arms, fs LSPG-CN K=1 at dt0 and 2.5 dt0.

    python analyze3d.py --run runs/c3grid/archive/output --scores checks/scores-c3grid-3d_first_order.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DT0 = .01
COMP = 'fs|BE|0.01|K1'
ANC = ('ad|GAL|BDF2|0.00125|tight', 'ad|GAL|BDF2|0.000625|tight', 'ad|GAL|BDF2|0.00125|tighter', 'ad|GAL|BDF2|0.000625|tighter')
DIAL = ['W256', 'W512', 'W768', 'W1024']


def med_ratio(panel, name):
    xs = [i['ratio'] for i in panel['invocations'] if i['B'] == name]
    return float(np.median(xs)) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--scores', required=True)
    ap.add_argument('--allow-smoke', action='store_true')
    ap.add_argument('--out')
    a = ap.parse_args()
    run = Path(a.run)
    res = json.loads((run / 'result.json').read_text())
    sc = json.loads(Path(a.scores).read_text())
    if res['config'].get('smoke') and not a.allow_smoke:
        raise SystemExit('smoke result refused')
    assert sc['gate']['status'] == 'pass' and sc['result_sha256'] == hashlib.sha256((run / 'result.json').read_bytes()).hexdigest()
    import c3common as C3
    cfg = res['config']
    cases = cfg.get('case_subset') or list(range(cfg['cohort_count']))
    tab = C3.C.table(cfg['cohort_seed'], cfg['cohort_count'])
    i65 = C3.lattice65_index(65)
    n0 = np.array([float(np.linalg.norm(np.asarray(C3.C.initial_interior(65, tab, j))[i65])) for j in cases])
    X65 = np.stack(np.meshgrid(*(np.arange(1, 64) / 64.,) * 3, indexing='ij'), -1).reshape(-1, 3)
    Wz = np.load(run / 'W_grid.npz')
    rows = {}
    for r in res['rows']:
        rows.setdefault((r['setting'], r['spec']), {})[r['case']] = r
    out = dict(run=str(run), refs=sc['refs'], provisional='reference-limited (BE reference at dt0/4)', settings={},
               selection={}, budget_arms={}, hypotheses={})
    common = res['timing'].get('common', {})
    for s in cfg['settings']:
        sname, rule = s['name'], s['rule']
        bk = C3.load_bank(s['bank'])
        G = np.asarray(C3.TB.feature_rows(bk['params'], bk['T'][:, :int(s['Rp'])], X65))
        W = {k.split('|', 2)[2]: Wz[k] for k in Wz.files if k.startswith(f'{sname}|{rule}|')}
        F = lambda sk: np.einsum('cki,ip->ckp', W[sk], G)
        Fanc = {k: F(k) for k in ANC}
        dfield = lambda A_, B_: (np.linalg.norm(A_ - B_, axis=2) / n0[:, None])[:, 1:].max(1)
        U = dfield(Fanc[ANC[0]], Fanc[ANC[1]]) + dfield(Fanc[ANC[0]], Fanc[ANC[2]]) + dfield(Fanc[ANC[1]], Fanc[ANC[3]])
        anc_ok = np.array([all(rows[(sname, k)][j]['valid'] for k in ANC) for j in cases])
        panel = res['timing'].get(f'setting|{sname}', {})
        arms = {}
        for sk in W:
            Fs = F(sk)
            disc = dfield(Fs, Fanc[ANC[0]])
            e = np.array([rows[(sname, sk)][j]['e_ref'] for j in cases])
            ent = dict(valid=all(rows[(sname, sk)][j]['valid'] for j in cases),
                       nonstationary=sum(rows[(sname, sk)][j]['nonstationary'] for j in cases),
                       steps=sum(rows[(sname, sk)][j]['steps_n'] for j in cases),
                       worst_ref=float(e.max()), median_ref=float(np.median(e)), per_case_ref=e.tolist(),
                       anchor=disc.tolist(), resolved=(anc_ok & (disc > 3 * U)).tolist(),
                       anchor_median_all=float(np.median(disc)),
                       ratio=(1.0 if sk == COMP else med_ratio(panel, f'{sname}|{rule}|{sk}')),
                       ratio_common=(1.0 if (sname == cfg['common_baseline'] and sk == COMP) else
                                     med_ratio(common, f'{sname}|{rule}|{sk}')))
            if sk.startswith('fs|'):
                p = sk.split('|')
                ad, adx = f'ad|LSPG|{p[1]}|{p[2]}|tight', f'ad|LSPG|{p[1]}|{p[2]}|tighter'
                if ad in W:
                    sd = dfield(Fs, F(ad))
                    sens = float(dfield(F(ad), F(adx)).max()) if adx in W else float('nan')
                    adv = all(rows[(sname, ad)][j]['valid'] for j in cases)
                    bar = np.minimum(2.5e-4, .25 * disc)
                    ent.update(solver_distance=sd.tolist(), solver_worst=float(sd.max()), probe_valid=adv,
                               probe_tighter_sensitivity=sens,
                               eligible_A62=bool(ent['valid'] and adv and float(sd.max()) <= 2.5e-4),
                               eligible_H4=bool(ent['valid'] and adv and np.isfinite(sens) and sens <= .1 * 2.5e-4 and
                                                bool(np.all(sd <= bar))))
            arms[sk] = ent
        S_ = dict(rule=rule, U_anc=U.tolist(), anchor_qualified=anc_ok.tolist(), panel_valid=panel.get('valid'),
                  gfs1=res['settings'][sname]['G_FS1'], arms=arms)
        out['settings'][sname] = S_
        # ---------------- H4 and selection
        comp = arms[COMP]
        pop = np.array(comp['resolved'])
        h4 = dict(population=int(pop.sum()), cases=len(cases))
        withheld = (not res['settings'][sname]['G_FS1']['cn_control_discriminating']) or not panel.get('valid')
        if withheld:
            h4['verdict'] = 'withheld (G-FS1 control non-discriminating or timing panel invalid)'
        elif pop.sum() < 52 * len(cases) / 64:
            h4['verdict'] = 'anchor-limited'
        else:
            cd = np.array(comp['anchor'])[pop]
            cw, cm = float(cd.max()), float(np.median(cd))

            def cand_stats(sk):
                x = arms[sk]
                d = np.where(np.array(x['resolved']), np.array(x['anchor']), 3 * U)[pop]
                return float(d.max()), float(np.median(d))
            fs = [sk for sk, x in arms.items() if sk.startswith('fs|') and sk != COMP and x.get('eligible_H4') and x['ratio'] is not None]
            a_c = [sk for sk in fs if float(sk.split('|')[2]) >= 1.25 * DT0 - 1e-12 and cand_stats(sk)[0] <= cw and
                   cand_stats(sk)[1] <= cm and arms[sk]['ratio'] <= .6]
            b_c = [sk for sk in fs if abs(float(sk.split('|')[2]) - DT0) < 1e-12 and cand_stats(sk)[1] <= .1 * cm and
                   arms[sk]['ratio'] <= 1.1]
            fea = [sk for sk in fs if cand_stats(sk)[0] <= cw and cand_stats(sk)[1] <= cm]
            aec = [sk for sk in fs if arms[sk]['ratio'] <= 1.0]
            h4.update(comparator=dict(worst=cw, median=cm),
                      H4a=dict(verdict='pass' if a_c else 'fail', witnesses=sorted(a_c, key=lambda k: arms[k]['ratio'])),
                      H4b=dict(verdict='pass' if b_c else 'fail', witnesses=sorted(b_c, key=lambda k: cand_stats(k)[1])),
                      candidates={sk: dict(worst=cand_stats(sk)[0], median=cand_stats(sk)[1], ratio=arms[sk]['ratio']) for sk in fs})
            out['selection'][sname] = dict(
                fast_equal_accuracy=(min(fea, key=lambda k: (arms[k]['ratio'], cand_stats(k)[1])) if fea else None),
                accurate_equal_cost=(min(aec, key=lambda k: (cand_stats(k)[1], arms[k]['ratio'])) if aec else None))
        out['hypotheses'].setdefault('H4', {})[sname] = h4
        sel = out['selection'].get(sname, {})
        barms = [COMP, 'fs|CN|0.01|K1', 'fs|CN|0.025|K1'] + [v for v in sel.values() if v]
        out['budget_arms'][sname] = sorted(set(barms))
    # ---------------- H6
    h6 = {}
    for s1, s2 in zip(DIAL[:-1], DIAL[1:]):
        x1, x2 = out['selection'].get(s1, {}).get('accurate_equal_cost'), out['selection'].get(s2, {}).get('accurate_equal_cost')
        if not x1 or not x2:
            h6[f'{s1}->{s2}'] = dict(verdict='unavailable')
            continue
        e1 = np.array(out['settings'][s1]['arms'][x1]['per_case_ref'])
        e2 = np.array(out['settings'][s2]['arms'][x2]['per_case_ref'])
        imp = float(np.mean(e2 < e1))
        mc = float(np.median(e2) / np.median(e1) - 1)
        h6[f'{s1}->{s2}'] = dict(arms=[x1, x2], median_change=mc, improved_fraction=imp,
                                 verdict='extends' if (mc <= -.10 and imp >= .6) else 'does not extend',
                                 label='reference-limited, PROVISIONAL')
    out['hypotheses']['H6'] = h6
    dest = Path(a.out or HERE / 'checks' / f"analysis3d-{run.parents[1].name}.json")
    dest.write_text(json.dumps(out, indent=1, default=float) + '\n')
    print(dest)
    print(json.dumps(dict(selection=out['selection'], H4={k: v.get('verdict', {k2: v.get(k2, {}).get('verdict') for k2 in ('H4a', 'H4b')})
                                                           for k, v in out['hypotheses']['H4'].items()},
                          H6={k: v['verdict'] for k, v in h6.items()}), default=float))


if __name__ == '__main__':
    main()
