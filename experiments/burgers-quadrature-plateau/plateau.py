"""Plateau / M* / crossover logic of DESIGN §6–7 and A1 (pure Python + NumPy; used in-job and by the report).

Input is one job's result.json (2D or 3D, same row schema):
  rows      [{arm, family, m, case, finite, nonconv_steps, steps, ST: {e, eligible}, S: {...}, dist_conv?}]
  fom_rows  [{fom, case, converged, ST: {e, eligible}, S: {...}}]
  references {'<case>|<kind>': {eps_ref, eligible, ...}}
  config    {conv_arm, anchor_check_arm, families: {family: [arm names]} ...}
"""
from __future__ import annotations

import numpy as np

BAND = 0.05
ANCHOR_DIST = 1e-3
ANCHOR_REL = 0.01
NONCONV_MAX = 0.01


def _eps(rep, c, kind):
    v = rep['references'][f'{c}|{kind}'].get('eps_ref')
    return float(v) if v is not None and np.isfinite(v) else 0.0


def eligible_cases(rep, kind):
    return sorted(int(k.split('|')[0]) for k, v in rep['references'].items() if k.endswith(f'|{kind}') and v['eligible'])


def by_arm(rep):
    out = {}
    for r in rep['rows']:
        out.setdefault(r['arm'], {})[r['case']] = r
    return out


def valid(rows_of_arm):
    """A1.4: every rollout finite and non-converged steps <= 1 % of all steps."""
    rs = list(rows_of_arm.values())
    if not all(r['finite'] for r in rs):
        return False
    nc = sum(r['nonconv_steps'] for r in rs)
    tot = sum(r['steps'] for r in rs)
    return bool(nc <= NONCONV_MAX * tot and not any(r.get('hard_fail') for r in rs))


def stat(rep, per_case, kind, cases, shift=0):
    """(W, T) of errors {case: e} on `cases`, each e shifted by shift * eps_ref (A1.5 envelopes)."""
    e = np.array([per_case[c] + shift * _eps(rep, c, kind) for c in cases])
    return float(e.max()), float(np.median(e))


def arm_stats(rep, kind='ST'):
    cases = eligible_cases(rep, kind)
    out = {}
    for arm, rc in by_arm(rep).items():
        pc = {c: rc[c][kind]['e'] for c in cases}
        W, T = stat(rep, pc, kind, cases)
        lo, hi = stat(rep, pc, kind, cases, -1), stat(rep, pc, kind, cases, +1)
        allc = sorted(rc)
        Wa, Ta = stat(rep, {c: rc[c][kind]['e'] for c in allc}, kind, allc)
        r0 = next(iter(rc.values()))
        dist = [rc[c].get('dist_conv') for c in allc if rc[c].get('dist_conv') is not None]
        out[arm] = dict(family=r0['family'], m=r0['m'], W=W, T=T, W_lo=lo[0], T_lo=lo[1], W_hi=hi[0], T_hi=hi[1],
                        W_all=Wa, T_all=Ta, n=len(cases), n_all=len(allc), valid=valid(rc),
                        dist_conv=max(dist) if dist else None, argmax=cases[int(np.argmax([pc[c] for c in cases]))])
    return out


def fom_stats(rep, kind='ST'):
    cases = eligible_cases(rep, kind)
    out = {}
    rows = {}
    for r in rep['fom_rows']:
        rows.setdefault(r['fom'], {})[r['case']] = r
    for f, rc in rows.items():
        pc = {c: rc[c][kind]['e'] for c in cases}
        W, T = stat(rep, pc, kind, cases)
        lo, hi = stat(rep, pc, kind, cases, -1), stat(rep, pc, kind, cases, +1)
        out[f] = dict(W=W, T=T, W_lo=lo[0], T_lo=lo[1], W_hi=hi[0], T_hi=hi[1], n=len(cases),
                      converged_all=all(r['converged'] for r in rc.values()), n_cases=len(rc))
    return out


def _band(s, ref, key):
    """'pass' / 'fail' / 'unresolved' for |s/ref - 1| <= BAND under the eps envelopes of both."""
    v, lo, hi = s[key], s[f'{key}_lo'], s[f'{key}_hi']
    r, rlo, rhi = ref[key], ref[f'{key}_lo'], ref[f'{key}_hi']
    nominal = abs(v / r - 1) <= BAND
    worst = max(abs(hi / rlo - 1), abs(lo / rhi - 1))
    best = 0.0 if (lo <= rhi and hi >= rlo) else min(abs(lo / rhi - 1), abs(hi / rlo - 1))
    if worst <= BAND:
        return 'pass'
    if best > BAND:
        return 'fail'
    return 'unresolved' if nominal or best <= BAND else 'fail'


def _le(s, f, key):
    """ROM statistic <= FOM statistic, robust to the envelopes: 'pass' / 'fail' / 'unresolved'."""
    if s[f'{key}_hi'] <= f[f'{key}_lo']:
        return 'pass'
    if s[f'{key}_lo'] > f[f'{key}_hi']:
        return 'fail'
    return 'unresolved'


def suffix_min(ladder, test):
    """Smallest index i such that test(rung_j) passes for every j >= i (unresolved counted as stated by `mode`).
    Returns (m_star_optimistic, m_star_conservative, bracket) with m values; None if even the top rung fails."""
    res = {}
    for mode in ('opt', 'cons'):
        ok = lambda r: (r == 'pass') or (mode == 'opt' and r == 'unresolved')
        i_star = None
        for i in range(len(ladder) - 1, -1, -1):
            if all(ok(test(ladder[j])) for j in range(i, len(ladder))):
                i_star = i
            else:
                break
        res[mode] = i_star
    m = lambda i: None if i is None else ladder[i]['m']
    br = lambda i: None if i is None else ((ladder[i - 1]['m'] if i > 0 else None), ladder[i]['m'])
    return dict(m_star=m(res['cons']), m_star_optimistic=m(res['opt']), bracket=br(res['cons']),
                arm=None if res['cons'] is None else ladder[res['cons']]['arm'])


def ladders(rep, st):
    fam = {}
    for arm, s in st.items():
        fam.setdefault(s['family'], []).append(dict(arm=arm, **s))
    return {f: sorted(v, key=lambda r: r['m']) for f, v in fam.items()}


def anchor_gate(rep, st):
    cfg = rep['config']
    a, b = st[cfg['conv_arm']], st[cfg['anchor_check_arm']]
    rb = by_arm(rep)[cfg['anchor_check_arm']]
    dist = max(r['dist_conv'] for r in rb.values())
    ok = bool(dist <= ANCHOR_DIST and abs(a['W'] / b['W'] - 1) <= ANCHOR_REL and abs(a['T'] / b['T'] - 1) <= ANCHOR_REL
              and a['valid'] and b['valid'])
    return dict(conv=cfg['conv_arm'], check=cfg['anchor_check_arm'], dist=dist, W_ratio=a['W'] / b['W'],
                T_ratio=a['T'] / b['T'], passed=ok)


def plateau(rep, kind='ST'):
    """Global M* per family (primary), family-relative 'within 5 % of the largest rung' (descriptive)."""
    st = arm_stats(rep, kind)
    controls = set(rep['config'].get('controls', []))
    lad = {f: [r for r in v if r['arm'] not in controls] for f, v in ladders(rep, st).items()}
    lad = {f: v for f, v in lad.items() if v and f not in ('control', 'tensor')}
    conv = st[rep['config']['conv_arm']]
    anchor = anchor_gate(rep, st)
    out = dict(kind=kind, anchor=anchor, conv=dict(arm=rep['config']['conv_arm'], W=conv['W'], T=conv['T']), families={})
    for f, v in lad.items():
        def t_glob(r):
            if not r['valid']:
                return 'fail'
            a, b = _band(r, conv, 'W'), _band(r, conv, 'T')
            return 'pass' if a == b == 'pass' else ('fail' if 'fail' in (a, b) else 'unresolved')
        top = v[-1]

        def t_fam(r):
            if not r['valid']:
                return 'fail'
            a, b = _band(r, top, 'W'), _band(r, top, 'T')
            return 'pass' if a == b == 'pass' else ('fail' if 'fail' in (a, b) else 'unresolved')
        g = suffix_min(v, t_glob) if anchor['passed'] else dict(m_star=None, m_star_optimistic=None, bracket=None,
                                                               arm=None, unresolved_anchor=True)
        tight = suffix_min(v, lambda r: 'pass' if (r['valid'] and abs(r['W'] / conv['W'] - 1) <= 0.01
                                                    and abs(r['T'] / conv['T'] - 1) <= 0.01) else 'fail')
        dist = suffix_min(v, lambda r: 'pass' if (r['valid'] and r['dist_conv'] is not None and r['dist_conv'] <= 1e-3)
                          else ('pass' if r['arm'] == rep['config']['conv_arm'] else 'fail'))
        tail = v[-2:]
        tail_ok = bool(len(tail) == 2 and all(r['valid'] and abs(r['W'] / conv['W'] - 1) <= 0.01
                                              and abs(r['T'] / conv['T'] - 1) <= 0.01 for r in tail))
        out['families'][f] = dict(tail_demonstrated=tail_ok, rungs=[dict(arm=r['arm'], m=r['m'], W=r['W'], T=r['T'], valid=r['valid'],
                                              dist_conv=r['dist_conv'], glob=t_glob(r), fam=t_fam(r)) for r in v],
                                  global_=g, family_relative=suffix_min(v, t_fam), band_1pct=tight, dist_1e3=dist,
                                  top_in_global_band=t_glob(top) == 'pass')
    n_tail = sum(v['tail_demonstrated'] for v in out['families'].values())
    out['demonstrated'] = bool(anchor['passed'] and n_tail >= 2)           # A2.1
    out['families_with_tail'] = n_tail
    return out


def matching(rep, kind='ST'):
    """m_x(F, family): smallest rung with ROM <= FOM (W and T) for it and every larger valid rung (A1.3-A1.5)."""
    st = arm_stats(rep, kind)
    fs = fom_stats(rep, kind)
    controls = set(rep['config'].get('controls', []))
    lad = {f: [r for r in v if r['arm'] not in controls] for f, v in ladders(rep, st).items()}
    lad = {f: v for f, v in lad.items() if v and f not in ('control', 'tensor')}
    conv = st[rep['config']['conv_arm']]
    out = {}
    for fn, F in fs.items():
        ent = dict(W=F['W'], T=F['T'], eligible=F['converged_all'],
                   conv_vs_fom=dict(W=_le(conv, F, 'W'), T=_le(conv, F, 'T'), W_ratio=conv['W'] / F['W'],
                                    T_ratio=conv['T'] / F['T']), families={})
        for f, v in lad.items():
            def t(r):
                if not r['valid']:
                    return 'fail'
                a, b = _le(r, F, 'W'), _le(r, F, 'T')
                return 'pass' if a == b == 'pass' else ('fail' if 'fail' in (a, b) else 'unresolved')
            ent['families'][f] = suffix_min(v, t)
        out[fn] = ent
    return out


def latency(rep, contract='rom_lat'):
    return {k.split('|', 1)[1]: v for k, v in rep['timing']['median_ms'].items() if k.split('|', 1)[0] == contract}


def cheapest_plateau_rule(rep, kind='ST', contract='rom_lat'):
    """A1.3: among families with a validated global M*, the rule at M* with the lowest median latency (ties: smaller m)."""
    p = plateau(rep, kind)
    if not p['demonstrated']:
        return None
    lat = latency(rep, contract)
    cands = []
    for f, v in p['families'].items():
        if not v['tail_demonstrated']:
            continue
        a = v['global_'].get('arm')
        if a is not None and a in lat:
            cands.append((lat[a], v['global_']['m_star'], a, f))
    if not cands:
        return None
    cands.sort()
    t, m, a, f = cands[0]
    return dict(arm=a, family=f, m=m, median_ms=t, candidates=[dict(arm=x[2], family=x[3], m=x[1], median_ms=x[0]) for x in cands])


def matched_fom(rep, arm, kind='ST', contract='fom'):
    """Cheapest eligible FOM whose W <= the selected rule's W and T <= its T (robust pass), with its latency."""
    st = arm_stats(rep, kind)[arm]
    lat = latency(rep, contract)
    best = None
    for fn, F in fom_stats(rep, kind).items():
        if not F['converged_all'] or fn not in lat:
            continue
        if _le(F, st, 'W') == 'pass' and _le(F, st, 'T') == 'pass':
            if best is None or lat[fn] < best[1]:
                best = (fn, lat[fn])
    return None if best is None else dict(fom=best[0], median_ms=best[1])
