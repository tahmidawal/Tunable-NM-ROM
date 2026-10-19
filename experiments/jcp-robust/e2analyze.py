"""jcp-robust E2 2D analysis per bank (DESIGN 3.2, 3.4; A1.8, A1.10, A2.5-A2.7, A3.5, A3.6, A4.2). CPU only.

Per bank b (0 = incumbent, 1..5 = recipe seeds): configuration records from the selection job (1024^2, factor 1) and the
verdict jobs (256^2, 4096^2 P1 subset, 1024^2 factors 2/2.5): Q1 + own-family Q2 (numerical qualification), per-case E0 ST
errors; the registered verdicts H-acc-N (P1's p1stats.h_acc_n, unchanged) and H-acc-r (p1stats.h_acc_r_transition /
range); the 1024^2 front over the slots (r, scheme, factor) from the timing job (dominance among qualified candidates,
qstats.front, A3.5 states); G-E2.3 (incumbent replay, A3.6). Writes checks/e2-2d-<b>.json.

    JAX_PLATFORMS=cpu python e2analyze.py <bank> --sel <dir> --v256 <dir> --v4096 <dir> --v1024 <dir> --time <dir>
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

import e1analyze as EA
import e2bank
import p1stats as S
import p1score as PS
from make_configs_e2 import DEPLOYED, RANKS

HERE = Path(__file__).resolve().parent
MESHES = (256, 1024, 4096)


def records(out, want_label_of):
    """{(r, label_kind, arm): configuration record} from one r2acc output; label_kind 'sel' / 'dep'."""
    res = json.loads((out / 'result.json').read_text())
    e2bank.use(res['config'].get('bank'))
    rec = dict(complete=bool(res.get('complete')) and (out / 'COMPLETE').exists(), backend=res['backend'], job_id=res['job_id'],
               bank=res['config'].get('bank'), mesh=int(res['mesh']), cfg={})
    for g in res['config']['groups']:
        r = int(g['Rp'])
        for arm in g['arms']:
            ak = f"{arm['form']}_{arm['scheme']}_{float(arm['factor']):g}"
            av = EA.analyse_arm(out, res, g, ak, r)
            for lab, kind in want_label_of(r, [x['label'] for x in g['rules']]):
                x = av['rules'][lab]
                rec['cfg'][(r, kind, ak)] = dict(alias_of=lab, rule=x['rule'], qualified=x['own_family_qualified'], status=x['status'],
                                                  e=dict(zip(EA.expected_ids(res['config'].get('cases', 'all38')), x['e'])),
                                                  eps=dict(zip(EA.expected_ids(res['config'].get('cases', 'all38')), x['eps'])),
                                                  eligible=dict(zip(EA.expected_ids(res['config'].get('cases', 'all38')), x['eligible'])),
                                                  case_q=dict(zip(EA.expected_ids(res['config'].get('cases', 'all38')), x['case_q'] or [])),
                                                  integrity=av['integrity']['ok'])
    return rec


def h_acc_n(C, r, kind, scheme):
    e = {m: C[m]['cfg'].get((r, kind, f'LSPG_{scheme}_1')) if C.get(m) else None for m in MESHES}
    if e[256] is None or e[1024] is None:
        return dict(outcome='verdict not reached', reason='missing 256^2 or 1024^2 configuration')
    sub16 = set(PS.case_set('sub16'))
    ep = e[1024]['eps']
    pairs = {}
    for m, tag in ((256, 'P256'), (4096, 'P4096')):
        if e[m] is None:
            continue
        cs = [c for c in e[m]['e'] if c in e[1024]['e'] and e[m]['eligible'][c] and e[1024]['eligible'][c] and (m != 4096 or c in sub16)]
        pairs[tag] = (np.array([e[m]['e'][c] for c in cs]), np.array([e[1024]['e'][c] for c in cs]), np.array([ep[c] for c in cs]))
    slopes = None
    if e[4096] is not None:
        cs = [c for c in e[4096]['e'] if c in sub16 and all(c in e[m]['e'] and e[m]['eligible'][c] for m in MESHES)]
        slopes = [np.polyfit(np.log2(MESHES), [e[m]['e'][c] for m in MESHES], 1)[0] for c in cs]
    qual = all(e[m]['qualified'] is True for m in MESHES if e[m] is not None)
    out = S.h_acc_n(pairs, slopes, qualified=qual)
    out['qualified'] = {m: e[m]['qualified'] for m in MESHES if e[m] is not None}
    out['rule'] = {m: e[m]['rule'] for m in MESHES if e[m] is not None}
    out['median_e'] = {m: float(np.median(list(e[m]['e'].values()))) for m in MESHES if e[m] is not None}
    out['worst_e'] = {m: float(np.max(list(e[m]['e'].values()))) for m in MESHES if e[m] is not None}
    return out


def h_acc_r(C, mesh, kind, scheme):
    out = {}
    for lo, hi in ((128, 256), (256, 384)):
        a, b = C[mesh]['cfg'].get((lo, kind, f'LSPG_{scheme}_1')), C[mesh]['cfg'].get((hi, kind, f'LSPG_{scheme}_1'))
        if a is None or b is None:
            out[f'{lo}->{hi}'] = dict(outcome='not reached (no rule)')
            continue
        cs = [c for c in a['e'] if c in b['e'] and a['eligible'][c] and b['eligible'][c]]
        out[f'{lo}->{hi}'] = S.h_acc_r_transition([b['e'][c] - a['e'][c] for c in cs], [a['eps'][c] for c in cs],
                                                  [a['case_q'].get(c, False) for c in cs], [b['case_q'].get(c, False) for c in cs])
        out[f'{lo}->{hi}']['median_paired_change'] = float(np.median([b['e'][c] - a['e'][c] for c in cs])) if cs else None
    out['range_128_384'] = S.h_acc_r_range([v['outcome'] for v in out.values() if isinstance(v, dict) and 'outcome' in v])
    return out


def front(C1024, timing):
    """A3.5 slot states at 1024^2: cost = median B/A of the slot's timed configuration (dtime result), accuracy = median and
    worst E0 error; dominance among qualified candidates only (qstats.dominates semantics: both no worse, one better)."""
    if timing is None:
        return dict(state='missing (no timing job)')
    tres = json.loads((timing / 'result.json').read_text())
    if not (tres.get('complete') and (timing / 'COMPLETE').exists() and tres.get('backend') == 'gpu'):
        return dict(state='missing (timing job incomplete or not on GPU)')
    if (tres['config'].get('bank') or {}).get('ckpt_sha256') != (C1024.get('bank') or {}).get('ckpt_sha256'):
        return dict(state='missing (timing job ran on another bank)')
    # code audit 2 #23: a subject's cost exists only if EVERY registered case has >= 2 valid triplets (valid = drift band
    # and every identity hash, dtime's own predicate); otherwise that subject's cost is missing
    reps_ok = {}
    for inv in tres.get('invocations', []):
        if inv.get('valid'):
            reps_ok.setdefault(inv['B'], {}).setdefault(inv['case'], []).append(float(inv['ratio']))
    want_cases = [f'dev6|{c}' for c in range(6)]
    ba = {}
    for lb, per in reps_ok.items():
        if all(len(per.get(c, [])) >= 2 for c in want_cases):
            ba[lb] = [float(np.median(per[c])) for c in want_cases]
    slots = {}
    for (r, kind, ak), x in C1024['cfg'].items():
        if kind != 'sel':
            continue
        form, sc, fac = ak.split('_')
        lab = f"1024|R{r}|M{4 * r}|{x['rule']}|LSPG|{sc}|{float(fac):g}"
        cost = float(np.median(ba[lab])) if lab in ba else None              # median over cases of per-case medians
        slots[f'R{r}|{sc}|{fac}'] = dict(rule=x['rule'], qualified=x['qualified'], cost=cost,
                                         median_e=float(np.median(list(x['e'].values()))), worst_e=float(np.max(list(x['e'].values()))))
    q = {k: v for k, v in slots.items() if v['qualified'] is True and v['cost'] is not None}
    for k, v in slots.items():
        if v['qualified'] is None or v['cost'] is None:
            v['state'] = 'missing'
        elif v['qualified'] is False:
            v['state'] = 'unqualified'
        else:
            dom = any(o['cost'] <= v['cost'] and o['median_e'] <= v['median_e'] and o['worst_e'] <= v['worst_e'] and
                      (o['cost'] < v['cost'] or o['median_e'] < v['median_e'] or o['worst_e'] < v['worst_e'])
                      for kk, o in q.items() if kk != k)
            v['state'] = 'dominated' if dom else 'member'
    for r in RANKS:
        for sc in ('BE', 'CNR'):
            for fac in ('1', '2', '2.5'):
                slots.setdefault(f'R{r}|{sc}|{fac}', dict(state='absent'))
    return slots


def g_e23(sel_out, v256_out):
    """A3.6: incumbent replay. 1024^2 from the selection job (candidates fib4181 / gauss128), 256^2 R128 from verdict job a."""
    import numpy as _np
    want = EA.expected_ids('all38')
    rows = {}
    for r, lab, arm, nm in ((128, 'fib4181', 'LSPG_BE_1', 'R_R128_BE.npz'), (128, 'fib4181', 'LSPG_CNR_1', 'R_R128_CNR.npz'),
                            (256, 'gauss128', 'LSPG_BE_1', 'R_R256_BE.npz'), (256, 'gauss128', 'LSPG_CNR_1', 'R_R256_CNR.npz'),
                            (384, 'gauss128', 'LSPG_BE_1', 'R_R384_BE.npz'), (384, 'gauss128', 'LSPG_CNR_1', 'R_R384_CNR.npz')):
        z = _np.load(sel_out / f'R_R{r}_{lab}_{arm}.npz')
        rows[f'1024|R{r}|{arm}'] = EA.reproduce(z['W'], [str(x) for x in z['case_ids']], [('p1', 'acc1024', nm)], None, want)
    for arm, nm in (('LSPG_BE_1', 'R_R128_BE.npz'), ('LSPG_CNR_1', 'R_R128_CNR.npz')):
        lab = 'sel' if (v256_out / f'R_R128_sel_{arm}.npz').exists() and \
            str(_np.load(v256_out / f'R_R128_sel_{arm}.npz')['rule']) == 'fib4181' else 'dep'
        z = _np.load(v256_out / f'R_R128_{lab}_{arm}.npz')
        rows[f'256|R128|{arm}'] = EA.reproduce(z['W'], [str(x) for x in z['case_ids']], [('p1', 'acc256', nm)], None, want)
    return dict(rows=rows, passed=all(v['passed'] for v in rows.values()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bank')
    for k in ('sel', 'v256', 'v4096', 'v1024', 'time'):
        ap.add_argument(f'--{k}')
    a = ap.parse_args()
    b = int(a.bank)
    sel = json.loads((HERE / 'checks' / f'select2d-{b}.json').read_text())
    def sel_labels(r, labels):            # selection job: the candidate equal to the selected / deployed rule
        out = []
        pk = sel['selected'][str(r)]['rule']
        if pk in labels:
            out.append((pk, 'sel'))
        if DEPLOYED[r] in labels:          # code audit 2 #25: the deployed identity always exists (alias when equal)
            out.append((DEPLOYED[r], 'dep'))
        return out
    def ver_labels(r, labels):            # verdict jobs: 'sel' and, if the deployed rule differs, 'dep'; else alias
        out = [(l, 'sel') for l in labels if l == 'sel']
        if 'dep' in labels:
            out.append(('dep', 'dep'))
        elif sel['selected'][str(r)]['rule'] == DEPLOYED[r] and 'sel' in labels:
            out.append(('sel', 'dep'))
        return out
    C = {}
    C[1024] = records(Path(a.sel), sel_labels)
    if a.v1024:
        extra = records(Path(a.v1024), ver_labels)
        C[1024]['cfg'].update(extra['cfg'])
        C[1024]['complete'] = C[1024]['complete'] and extra['complete']
    if a.v256:
        C[256] = records(Path(a.v256), ver_labels)
    if a.v4096:
        C[4096] = records(Path(a.v4096), ver_labels)
    rep = dict(bank=b, selection=sel['selected'], jobs={m: dict(job_id=C[m]['job_id'], complete=C[m]['complete'],
                                                              backend=C[m]['backend']) for m in C},
               integrity=all(x['integrity'] for m in C for x in C[m]['cfg'].values()),
               configurations={f'{m}|R{k[0]}|{k[1]}|{k[2]}': dict(rule=v['rule'], qualified=v['qualified'], status=v['status'],
                                                                   median_e=float(np.median(list(v['e'].values()))),
                                                                   worst_e=float(np.max(list(v['e'].values()))))
                               for m in C for k, v in C[m]['cfg'].items()},
               h_acc_n={}, h_acc_r={})
    for kind in ('sel', 'dep'):
        for r in RANKS:
            for sc in ('BE', 'CNR'):
                rep['h_acc_n'][f'{kind}|R{r}|{sc}'] = h_acc_n(C, r, kind, sc)
        for m in C:
            for sc in ('BE', 'CNR'):
                rep['h_acc_r'][f'{kind}|{m}|{sc}'] = h_acc_r(C, m, kind, sc)
    rep['front_1024'] = front(C[1024], Path(a.time) if a.time else None)
    if b == 0:
        rep['G-E2.3'] = g_e23(Path(a.sel), Path(a.v256)) if a.v256 else dict(passed=False, reason='256^2 verdict job missing')
    (HERE / 'checks' / f'e2-2d-{b}.json').write_text(json.dumps(rep, indent=1, default=float) + '\n')
    print(json.dumps({k: v['outcome'] if isinstance(v, dict) and 'outcome' in v else v.get('range_128_384')
                      for k, v in {**rep['h_acc_n'], **rep['h_acc_r']}.items()}, default=str))


if __name__ == '__main__':
    main()
