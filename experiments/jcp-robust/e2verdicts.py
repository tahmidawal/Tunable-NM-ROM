"""jcp-robust E2 cross-bank verdicts (DESIGN 3.4 as amended by A1.10, A2.7, A3.5, A3.6, A4.2). CPU only, no JAX.

Reads checks/e2-<dim>-<bank>.json for bank 0 (incumbent) and the recipe seeds. For every registered verdict:
  seed-robust      all three new seeds give the same definitive label
  seed-dependent   two new seeds give different definitive labels
  inconclusive     otherwise
and, when seed-robust, historical agreement with the incumbent (agrees / differs (recipe/software version, not attributed
to the seed) / incumbent not definitive). Precedence: G-E2.3 failure on the incumbent blocks every verdict of that
dimension; a bank job that is incomplete / not on GPU / fails integrity makes that bank's labels 'missing'.
Seed spread: per (r, scheme, mesh) case-median and case-worst error, min-max over seeds, incumbent value, and a t interval
(df = n-1) for the mean of log10 over the new seeds, labelled. Spread ratio per H-acc-r transition (A2.7).

    python e2verdicts.py 2d [--seeds 1,2,3]
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import t as tdist

HERE = Path(__file__).resolve().parent
DEFINITIVE = {'pass', 'fail', 'improves', 'does not improve', 'monotone', 'not monotone', 'member', 'dominated', 'unqualified',
              'absent', True, False}


def lab3(labels):
    """labels of the new seeds (None = missing)."""
    d = [x for x in labels if x in DEFINITIVE or (isinstance(x, str) and x.startswith('rule:'))]
    if len(d) >= 2 and len(set(map(str, d))) > 1:
        return 'seed-dependent'
    if len(d) == len(labels) and len(set(map(str, d))) == 1:
        return 'seed-robust'
    return 'inconclusive'


def hist(inc, new, robust):
    if robust != 'seed-robust':
        return None
    if not (inc in DEFINITIVE or (isinstance(inc, str) and inc.startswith('rule:'))):
        return 'incumbent not definitive'
    return 'agrees with incumbent' if str(inc) == str(new[0]) else 'differs from incumbent (recipe/software version, not attributed to the seed)'


def keys_2d(r):
    out = {}
    if r is None:
        return out
    for k, v in r['selection'].items():
        out[f'selected-rule|R{k}'] = ('rule:' + v['rule']) if v['rule'] else ('absent' if v['outcome'] == 'none within ladder' else None)
    for k, v in r['h_acc_n'].items():
        out[f'H-acc-N|{k}'] = v.get('outcome')
    for k, v in r['h_acc_r'].items():
        for tr, x in v.items():
            out[f'H-acc-r|{k}|{tr}'] = x.get('outcome') if isinstance(x, dict) else x
    for k, v in r['configurations'].items():
        out[f'Q1Q2|{k}'] = v['qualified']
    for k, v in (r.get('front_1024') or {}).items():
        if isinstance(v, dict):
            out[f'front1024|{k}'] = v.get('state')
    return out


def keys_3d(r):
    out = {}
    if r is None:
        return out
    for sl in r['selection']:
        k = f"selected-rule|W{sl['width']}|{sl['spec']}|{sl['family']}"
        out[k] = ('rule:' + sl['rule']) if sl['rule'] else ('absent' if sl['outcome'] == 'none within ladder' else None)
    for k, v in r['h_acc_r'].items():
        for tr, x in v.items():
            out[f'H-acc-r|{k}|{tr}'] = x.get('outcome') if isinstance(x, dict) else x
    for k, v in r['configurations'].items():
        out[f'Q1Q2|{k}'] = v['qualified']
    for k, v in ((r.get('front_65') or {}).get('points') or {}).items():
        out[f'front65|{k}'] = v.get('state')
    return out


def tint(xs):
    xs = [math.log10(x) for x in xs if x is not None and np.isfinite(x) and x > 0]
    if len(xs) < 2:
        return None
    m, s = float(np.mean(xs)), float(np.std(xs, ddof=1))
    h = tdist.ppf(.975, len(xs) - 1) * s / math.sqrt(len(xs))
    return dict(mean_log10=m, lo=10 ** (m - h), hi=10 ** (m + h), n=len(xs), label=f'n = {len(xs)}, wide by construction')


def main():
    dim = sys.argv[1]
    seeds = [int(x) for x in (sys.argv[sys.argv.index('--seeds') + 1] if '--seeds' in sys.argv else '1,2,3').split(',')]
    load = lambda b: json.loads((HERE / 'checks' / f'e2-{dim}-{b}.json').read_text()) if (HERE / 'checks' / f'e2-{dim}-{b}.json').exists() else None
    inc = load(0)
    new = {s: load(s) for s in seeds}
    blocked = inc is None or not (inc.get('G-E2.3') or {}).get('passed')
    ok = lambda r: r is not None and r.get('integrity') and all(j.get('complete') and j.get('backend') == 'gpu' for j in r['jobs'].values())
    kf = keys_2d if dim == '2d' else keys_3d
    K = {b: (kf(r) if ok(r) else {}) for b, r in [(0, inc)] + list(new.items())}
    allk = sorted(set().union(*[set(v) for v in K.values()]))
    rows = []
    for k in allk:
        lbls = [K[s].get(k) for s in seeds]
        robust = lab3(lbls)
        rows.append(dict(key=k, incumbent=K[0].get(k), seeds=dict(zip(map(str, seeds), lbls)),
                         label=('blocked (G-E2.3)' if blocked else robust),
                         historical=None if blocked else hist(K[0].get(k), lbls, robust)))
    spread = {}
    for b, r in [(0, inc)] + list(new.items()):
        if not ok(r):
            continue
        for k, v in r['configurations'].items():
            spread.setdefault(k, {})[b] = (v['median_e'], v['worst_e'])
    sp = {}
    for k, v in spread.items():
        s_ = [v[s] for s in seeds if s in v]
        sp[k] = dict(incumbent=v.get(0), seeds={str(s): v.get(s) for s in seeds},
                     median_minmax=[min(x[0] for x in s_), max(x[0] for x in s_)] if s_ else None,
                     worst_minmax=[min(x[1] for x in s_), max(x[1] for x in s_)] if s_ else None,
                     median_t=tint([x[0] for x in s_]), worst_t=tint([x[1] for x in s_]))
    ratios = {}
    for b in (seeds if dim == '2d' else []):            # 3D spread ratios: the gl64 panel rows (reported in the e2-3d files)
        r = new[b]
        if not ok(r):
            continue
        for k, v in r['h_acc_r'].items():
            for tr, x in v.items():
                if isinstance(x, dict) and x.get('median_paired_change') is not None:
                    ratios.setdefault(f'{k}|{tr}', {})[b] = x['median_paired_change']
    sr = {}
    for kk, v in ratios.items():
        kind, mesh, sc = kk.split('|')[:3]
        lo = int(kk.split('|')[3].split('->')[0])
        meds = [spread.get(f'{mesh}|R{lo}|{kind}|LSPG_{sc}_1', {}).get(s, (None,))[0] for s in seeds]
        meds = [m_ for m_ in meds if m_ is not None]
        num = (max(meds) - min(meds)) if len(meds) >= 2 else None
        den = float(np.median([abs(x) for x in v.values()])) if v else None
        if num is None or den is None:
            sr[kk] = None
        elif den == 0:
            sr[kk] = 'undefined' if num == 0 else 'inf'
        else:
            sr[kk] = num / den
    out = dict(dim=dim, seeds=seeds, blocked=blocked, rows=rows, spread=sp, spread_ratio=sr,
               counts={l: sum(1 for x in rows if x['label'] == l) for l in ('seed-robust', 'seed-dependent', 'inconclusive')})
    (HERE / 'checks' / f'e2-verdicts-{dim}.json').write_text(json.dumps(out, indent=1, default=float) + '\n')
    print(json.dumps(out['counts']))


if __name__ == '__main__':
    main()
