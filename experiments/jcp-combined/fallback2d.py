"""DESIGN A14: qualification of the first fallbacks of the 2D selected arms that failed the finalist quadrature check.

For each failing setting s (fast, acc, R256) and each selected arm (accurate-equal-cost at its step, fast-equal-accuracy
at its step), with the arm unchanged and the rule replaced by the frozen first fallback:
  source   fast -> R128 (same R', rule fib4181), acc -> R384 (same R', rule gauss128): c2grid + c2bud(A/B/C);
           R256 -> R256fb (gauss128): c2fbgrid + c2fbbud.
  gates    every run verified on all 38 cases; the timing panel used is valid; the fallback's finalist check passes
           (budget output, all cases, delta_quad <= tau and the converged-rule check); the source grid's first-order
           G-score passes.
  criterion (the arm's ORIGINAL one, against the ORIGINAL deployed BE of s, A3.2 / A14.3):
           accurate arm: paired time ratio <= 1 (fast/acc: ratio of the two median paired ratios in the c2grid common
           panel; R256: the c2fbgrid common panel, whose baseline is the deployed R256 BE, measured directly);
           fast arm: accuracy feasibility, i.e. worst and median ST and TX no worse than the deployed BE's under the
           first-order tie policy of the selection analysis (cost reported only).
Accuracy is scored against the first-order references (the selection reference, PROVISIONAL); E0 numbers are reported
beside them (A13). Writes checks/fallback2d.json.

    python fallback2d.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CK = HERE / 'checks'
SRC = dict(fast=('R128', 'c2grid', 'budget2d-c2budC'), acc=('R384', 'c2grid', 'budget2d-c2budB'),
           R256=('R256fb', 'c2fbgrid', 'budget2d-c2fbbud'))


def load(p):
    return json.loads(Path(p).read_text())


def med_ratio(panel, name):
    xs = [i['ratio'] for i in panel['invocations'] if i['B'] == name]
    return float(np.median(xs)) if xs else None


def stats(sc, setting, key, cases, tag):
    v = [sc['scores'][f'{setting}|{key}|{coh}|{c}'][tag]['worst'] for coh, c in cases]
    return dict(median=float(np.median(v)), worst=float(max(v)))


def main():
    an = load(CK / 'analysis-c2grid-2d_first_order.json')
    fz = load(CK / 'FROZEN-SELECTION.json')['2d']
    g = fz['global']['scheme']
    thr = an['tie']                                                    # first-order heuristic thresholds (A2.6)
    out = dict(scheme=g, tie=thr, label='first-order scoring (selection reference, PROVISIONAL); E0 beside (A13)',
               settings={})
    res_main = load(HERE / 'runs/c2grid/archive/output/result.json')
    for s, (fb, grid, bud) in SRC.items():
        res = load(HERE / f'runs/{grid}/archive/output/result.json')
        sc1 = load(CK / f'scores-{grid}-2d_first_order.json')
        sce = load(CK / f'scores-{grid}-2d_e0.json')
        assert sc1['gate']['status'] == 'pass', (grid, 'first-order G-score')
        cfg = res['config']
        cases = [(coh, c) for coh in cfg['cohorts'] for c in
                 (cfg.get('case_subset', {}).get(coh) if cfg.get('case_subset', {}).get(coh) is not None
                  else range(res['cohorts'][coh]['n']))]
        rows = {(r['setting'], r['run'], r['cohort'], r['case']): r for r in res['rows']}
        bo = load(CK / f'{bud}-2d_first_order.json')
        bgates_ok = all(((v.get('passed') is True) if isinstance(v, dict) else v is True) for v in bo['gates'].values())
        # the ORIGINAL deployed BE of s (in c2grid for fast/acc; in c2fbgrid for R256 at gauss96, identical model)
        be_src = 'c2grid' if s != 'R256' else 'c2fbgrid'
        sc_be = load(CK / f'scores-{be_src}-2d_first_order.json')
        be_key = 'LSPG|BE|0.005|prod'
        be = {t: stats(sc_be, s, be_key, cases, t) for t in ('ST', 'TX')}
        ent = dict(fallback_setting=fb, fallback_rule=next(x['rule'] for x in cfg['settings'] if x['name'] == fb),
                   grid=grid, budget=bud, budget_gates_ok=bgates_ok, arms={})
        for kind, x in fz['per_scheme_setting'][f'{s}|{g}'].items():
            if kind not in ('accurate_equal_cost', 'fast_equal_accuracy') or not x:
                continue
            key = x['key']
            ver = all(rows[(fb, key, coh, c)]['verified'] for coh, c in cases)
            st = {t: stats(sc1, fb, key, cases, t) for t in ('ST', 'TX')}
            e0 = stats(sce, fb, key, cases, 'ST')
            fa = bo['settings'][fb]['arms'][f"{key.split('|')[0]}|{key.split('|')[1]}|{x['f']:g}"]['finalist']
            if s == 'R256':                                             # direct: baseline = deployed R256 BE
                pan = res['timing']['common']
                ratio, how = med_ratio(pan, f'{fb}|{key}'), 'c2fbgrid common panel (baseline: deployed R256 BE), median paired ratio'
                tvalid = bool(pan.get('valid'))
            else:                                                       # ratio of medians in the c2grid common panel
                pan = res_main['timing']['common']
                num = med_ratio(pan, f'{fb}|{key}')
                den = 1. if (s == res_main['config']['common_baseline']) else med_ratio(pan, f'{s}|{be_key}')
                ratio = num / den if (num is not None and den) else None
                how = 'c2grid common panel: ratio of the two median paired ratios (not a directly paired median)'
                tvalid = bool(pan.get('valid'))
            if kind == 'accurate_equal_cost':
                crit = dict(name='paired time ratio vs the original deployed BE <= 1', value=ratio,
                            ok=bool(ratio is not None and ratio <= 1.0))
            else:                                                       # accuracy feasibility (A3.2 tie policy)
                beaten = {f'{t}|{k}': bool(be[t][k] < st[t][k] - thr[k]) for t in ('ST', 'TX') for k in ('median', 'worst')}
                crit = dict(name='worst and median ST and TX no worse than the deployed BE (first-order tie policy)',
                            beaten_by_BE=beaten, ok=not any(beaten.values()), cost_ratio=ratio)
            q = bool(ver and tvalid and fa.get('qualified') and bgates_ok and crit['ok'])
            ent['arms'][kind] = dict(key=key, verified_all=ver, timing_valid=tvalid, timing_method=how, finalist=fa,
                                     first_order=st, e0_ST=e0, deployed_BE=be, criterion=crit, qualified=q)
        out['settings'][s] = ent
    dest = CK / 'fallback2d.json'
    dest.write_text(json.dumps(out, indent=1, default=float) + '\n')
    print(dest)
    for s, e in out['settings'].items():
        for k, a in e['arms'].items():
            print(s, '->', e['fallback_setting'], k, 'qualified', a['qualified'], 'finalist', a['finalist'].get('qualified'),
                  'crit', a['criterion']['ok'], a['criterion'].get('value', a['criterion'].get('cost_ratio')))


if __name__ == '__main__':
    main()
