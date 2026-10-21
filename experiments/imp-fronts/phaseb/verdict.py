"""imp-fronts Phase B decision procedure (DESIGN B0 as replaced by Amendment 3, refined by Amendments 4-5; Amendments 1-2
where not superseded).
Pure Python / NumPy, no I/O except the CLI.

    python phaseb/verdict.py <VERDICT-INPUT.json> [<out.json>]

Per arm, the guard and each endpoint route are evaluated separately as PASS / FAIL / UNAVAILABLE, then (Amendment 3):

  step 1  arm not trained                                                          -> NOT RUN
  step 2  guard evaluable (>= 6 eligible FI test cases) and violated
          (in-range worst floor f_384 > 2 x K1's)                                  -> STOP (guard)
  step 3  every endpoint route FAIL                                                -> STOP (endpoint)
  step 4  guard PASS and at least one route PASS                                   -> CONTINUE
  step 5  otherwise                                                                -> INCONCLUSIVE

Routes (missing prerequisite -> that route UNAVAILABLE):
  (a) floor : >= 5 eligible FN0s test cases, K1 floors on the same cases; PASS iff median f_384(arm) <= median f_384(K1)/3.
  (b) ROM   : >= 6 eligible FC1 test cases, e_hat_K1 finite (median of K1's policy errors over the eligible cases,
              invalid = +inf, regardless of qualification); PASS iff the arm is numerically qualified on EVERY eligible
              FC1 test case and median e(arm) <= e_hat_K1 / 2; not qualified on every case -> FAIL.
  (c) m*    : PASS iff the arm has a certified selected rule with m*_c <= 12 800 (= gauss160 / 2); no certified rule -> FAIL.
  (c) tail  : common set = eligible FI test cases where BOTH the arm and K1 have valid finite terminal policy runs
              (>= 6, else UNAVAILABLE; Amendment 4); PASS iff the arm is qualified on every eligible FI test case and
              median tau_4096(arm) <= median tau_4096(K1) / 3, both medians over the common set.
Guard (Amendments 5, 8): PASS requires finite arm and K1 floors on EVERY eligible FI test case (>= 6); any missing or
non-finite floor -> UNAVAILABLE, except that when K1's floors are complete and the observed arm maximum (finite arm
floors) exceeds 2 x the complete K1 maximum -> FAIL (STOP (guard)).
Numerical qualification (Amendment 9: Q1 and Q2 and step-converged, plus the bank's pooled non-stationary fraction
<= 1 %) is computed upstream; this module consumes the per-case 'qualified' flag. Route (a) (Amendment 5): finite arm and K1 floors on every eligible FN0s test case (>= 5), else UNAVAILABLE.
No certified selected rule (Amendment 4): m* route FAIL; the policy ran from fib4181 ("uncertified start") and the ROM
routes are evaluated normally.
"x-fold below" means arm <= control / x (ties at exactly the factor pass); the guard is violated only if strictly > 2x.

Interpretations frozen here (recorded as ambiguities in the lane report; none can make unqualified evidence succeed):
  I1  the guard needs >= 6 ELIGIBLE FI test cases before anything (Amendment 3 step 2 "evaluable"); below that it is
      UNAVAILABLE even if a violation is visible.
  I4  ROM routes are UNAVAILABLE if the arm's policy job did not run at all (rom_available false, e.g. policy budget
      not funded); within a run, a case with no record counts as not qualified (-> FAIL).
The per-case 2*eps_ref rule (Amendment 1 B0) is reported as resolved wins / ties / losses next to every median comparison;
it never changes the medians or the outcome. "Passes against all three incumbent retrains (K1-K3)" (Amendment 1 B1) is
reported, not gating: the arm's decision-relevant route PASSes and its guard PASSes with each of K1, K2, K3 as control.
"""
from __future__ import annotations

import json
import math
import sys

import numpy as np

PASS, FAIL, UNAV = 'PASS', 'FAIL', 'UNAVAILABLE'
NOT_RUN, STOP_G, STOP_E, CONT, INCONC = 'NOT RUN', 'STOP (guard)', 'STOP (endpoint)', 'CONTINUE', 'INCONCLUSIVE'
MIN = dict(guard=6, floor=5, rom=6, tail=6)
FAC = dict(guard=2., floor=3., rom=2., mstar=2., tail=3.)
MSTAR_K1_NODES = 160 * 160           # gauss160, jcp-robust E2 selection for K1
MSTAR_BAR = MSTAR_K1_NODES / FAC['mstar']   # 12 800
ARM_ROUTES = {'a': ('floor',), 'b': ('rom',), 'c': ('mstar', 'tail')}


def _fin(x):
    return x is not None and isinstance(x, (int, float, np.floating, np.integer)) and math.isfinite(float(x))


def _num(x):
    if x is None:
        return None
    if isinstance(x, str):
        return float('inf') if x in ('inf', 'Infinity') else float(x)
    return float(x)


def _med(v):
    return float(np.median(np.asarray(v, float)))


def resolved(arm_vals, ctrl_vals, eps):
    """Per-case 2*eps_ref rule: |arm - ctrl| <= 2 eps -> tie; arm lower -> win; else loss. Cases with missing eps_ref or
    values are 'unresolved'; both non-finite -> 'both_invalid'."""
    out = dict(wins=0, ties=0, losses=0, unresolved=0, both_invalid=0)
    for c in arm_vals:
        a, b, e = _num(arm_vals.get(c)), _num(ctrl_vals.get(c)), _num((eps or {}).get(c))
        if a is None or b is None or (isinstance(a, float) and math.isnan(a)) or (isinstance(b, float) and math.isnan(b)):
            out['unresolved'] += 1
        elif not math.isfinite(a) and not math.isfinite(b):
            out['both_invalid'] += 1
        elif not math.isfinite(a):
            out['losses'] += 1
        elif not math.isfinite(b):
            out['wins'] += 1
        elif e is None or not math.isfinite(e):
            out['unresolved'] += 1
        elif abs(a - b) <= 2 * e:
            out['ties'] += 1
        elif a < b:
            out['wins'] += 1
        else:
            out['losses'] += 1
    return out


def _floors(bank, cohort, cases, key='f384'):
    fl = (bank or {}).get('floors', {}).get(cohort, {})
    return {c: _num(fl.get(c, {}).get(key)) if isinstance(fl.get(c), dict) else _num(fl.get(c)) for c in cases}


def guard(arm, ctrl, elig, eps=None):
    cases = [str(c) for c in elig.get('FI', [])]
    if len(cases) < MIN['guard']:
        return dict(status=UNAV, why=f'{len(cases)} eligible FI test cases < {MIN["guard"]}')
    fa, fc = _floors(arm, 'FI', cases), _floors(ctrl, 'FI', cases)
    bad = [c for c in cases if not (_fin(fa[c]) and _fin(fc[c]))]
    if bad:            # Amendments 5 + 8: early STOP on incomplete ARM evidence only against the COMPLETE K1 maximum
        k1_complete = all(_fin(fc[c]) for c in cases)
        av = [c for c in cases if _fin(fa[c])]
        if k1_complete and av and max(fa[c] for c in av) > FAC['guard'] * max(fc.values()):
            return dict(status=FAIL, why=f'observed arm maximum > 2 x complete K1 maximum; arm floors missing on {bad}',
                        arm_worst=max(fa[c] for c in av), ctrl_worst=max(fc.values()), n=len(av), missing=bad)
        return dict(status=UNAV, why=f'missing / non-finite floors on {bad} (defect, reported)', missing=bad,
                    k1_complete=k1_complete)
    wa, wc = max(fa.values()), max(fc.values())
    st = FAIL if wa > FAC['guard'] * wc else PASS
    return dict(status=st, arm_worst=wa, ctrl_worst=wc, ratio=wa / wc if wc > 0 else float('inf'), n=len(cases),
                resolved=resolved(fa, fc, (eps or {}).get('FI')))


def route_floor(arm, ctrl, elig, eps=None):
    cases = [str(c) for c in elig.get('FN0s', [])]
    if len(cases) < MIN['floor']:
        return dict(status=UNAV, why=f'{len(cases)} eligible FN0s test cases < {MIN["floor"]}')
    fa, fc = _floors(arm, 'FN0s', cases), _floors(ctrl, 'FN0s', cases)
    bad = [c for c in cases if not (_fin(fa[c]) and _fin(fc[c]))]
    if bad:
        return dict(status=UNAV, why=f'missing / non-finite floors on {bad} (defect, reported)')
    ma, mc = _med(list(fa.values())), _med(list(fc.values()))
    return dict(status=PASS if ma <= mc / FAC['floor'] else FAIL, arm_median=ma, ctrl_median=mc, n=len(cases),
                bar=mc / FAC['floor'], resolved=resolved(fa, fc, (eps or {}).get('FN0s')))


def _certified(bank):
    sel = (bank or {}).get('selection') or {}
    return bool(sel.get('certified')) and _fin(_num(sel.get('nodes')))


def _policy(bank, cohort):
    return ((bank or {}).get('policy') or {}).get(cohort) or {}


def route_rom(arm, ctrl, elig, eps=None):
    cases = [str(c) for c in elig.get('FC1', [])]
    if len(cases) < MIN['rom']:
        return dict(status=UNAV, why=f'{len(cases)} eligible FC1 test cases < {MIN["rom"]}')
    if not arm.get('rom_available', False):
        return dict(status=UNAV, why='arm policy job not run (I4)')
    pk = _policy(ctrl, 'FC1')
    if any(c not in pk for c in cases):
        return dict(status=UNAV, why='control policy records missing')
    ek = {c: (_num(pk[c].get('e')) if pk[c].get('valid', True) and pk[c].get('e') is not None else float('inf'))
          for c in cases}
    ek = {c: (v if v is not None and not math.isnan(v) else float('inf')) for c, v in ek.items()}
    e_hat = _med(list(ek.values()))
    if not math.isfinite(e_hat):
        return dict(status=UNAV, why='e_hat_K1 is not finite', ctrl_median=e_hat)
    pa = _policy(arm, 'FC1')
    qual = {c: bool(pa.get(c, {}).get('qualified', False)) for c in cases}
    ea = {c: (_num(pa[c].get('e')) if c in pa and pa[c].get('e') is not None and pa[c].get('valid', True)
              else float('inf')) for c in cases}
    rs = resolved(ea, ek, (eps or {}).get('FC1'))
    ma = _med(list(ea.values()))
    if not all(qual.values()):
        return dict(status=FAIL, why=f'not numerically qualified on {[c for c in cases if not qual[c]]}',
                    arm_median=ma, ctrl_median=e_hat, n=len(cases), resolved=rs)
    return dict(status=PASS if ma <= e_hat / FAC['rom'] else FAIL, arm_median=ma, ctrl_median=e_hat,
                bar=e_hat / FAC['rom'], n=len(cases), resolved=rs)


def route_mstar(arm, ctrl=None, elig=None, eps=None, bar=MSTAR_BAR):
    if not _certified(arm):
        return dict(status=FAIL, why='no certified rule in the tested ladder')
    m = float(_num(arm['selection']['nodes']))
    return dict(status=PASS if m <= bar else FAIL, arm_nodes=m, bar=bar, rule=arm['selection'].get('rule'))


def route_tail(arm, ctrl, elig, eps=None):
    elig_fi = [str(c) for c in elig.get('FI', [])]
    if not arm.get('rom_available', False):
        return dict(status=UNAV, why='arm policy job not run (I4)')
    pk, pa = _policy(ctrl, 'FI'), _policy(arm, 'FI')
    ok = lambda p, c: c in p and p[c].get('valid', False) and _fin(_num(p[c].get('tau')))
    common = [c for c in elig_fi if ok(pk, c) and ok(pa, c)]          # Amendment 4: both valid finite
    if len(common) < MIN['tail']:
        return dict(status=UNAV, why=f'{len(common)} eligible FI test cases with arm and K1 valid finite runs < {MIN["tail"]}')
    qual = {c: bool(pa.get(c, {}).get('qualified', False)) for c in elig_fi}
    ta = {c: _num(pa[c]['tau']) for c in common}
    tk = {c: _num(pk[c]['tau']) for c in common}
    ma, mk = _med(list(ta.values())), _med(list(tk.values()))
    if not all(qual.values()):
        return dict(status=FAIL, why=f'not numerically qualified on {[c for c in elig_fi if not qual[c]]}',
                    arm_median=ma, ctrl_median=mk, n=len(common))
    return dict(status=PASS if ma <= mk / FAC['tail'] else FAIL, arm_median=ma, ctrl_median=mk, bar=mk / FAC['tail'],
                n=len(common), common_cases=common)


ROUTE_FN = dict(floor=route_floor, rom=route_rom, mstar=route_mstar, tail=route_tail)


def decide(trained, g, routes):
    """Amendment 3 precedence table. g = guard status, routes = {name: status}."""
    if not trained:
        return NOT_RUN, 1
    if g == FAIL:
        return STOP_G, 2
    if routes and all(s == FAIL for s in routes.values()):
        return STOP_E, 3
    if g == PASS and any(s == PASS for s in routes.values()):
        return CONT, 4
    return INCONC, 5


class InputError(ValueError):
    pass


def rule_nodes(rule):
    """Node count of a registered rule name: gaussN -> N^2 (tensor Gauss-Legendre), fibN -> N (Fibonacci lattice)."""
    r = str(rule)
    if r.startswith('gauss') and r[5:].isdigit():
        return int(r[5:]) ** 2
    if r.startswith('fib') and r[3:].isdigit():
        return int(r[3:])
    raise InputError(f'unknown rule name {rule!r}')


def validate(inp):
    """Audit codex-code-B-1: eligible lists are unique fresh TEST ids (j >= 4); a certified selection's node count
    equals its rule's node count; input-supplied routes are refused (Amendment 12: routes fixed in ARM_ROUTES)."""
    for coh, cases in (inp.get('eligible') or {}).items():
        if coh not in ('FI', 'FN0s', 'FC1', 'FC2'):
            raise InputError(f'eligible: unknown cohort {coh}')
        js = [int(c) for c in cases]
        if [str(c) for c in cases] != [str(j) for j in js]:
            raise InputError(f'eligible {coh}: non-integer case ids {cases}')
        if len(set(js)) != len(js):
            raise InputError(f'eligible {coh}: duplicate case ids {cases}')
        if any(j < 4 for j in js):
            raise InputError(f'eligible {coh}: non-test case ids {[j for j in js if j < 4]}')
    banks = dict((inp.get('arms') or {}), **{f'ctrl_{k}': v for k, v in (inp.get('controls') or {}).items()})
    for name, b in banks.items():
        if not b:
            continue
        if 'routes' in b and tuple(b['routes']) != ARM_ROUTES.get(name, tuple(b['routes'])):
            raise InputError(f'{name}: input-supplied routes {b["routes"]} refused (fixed: {ARM_ROUTES.get(name)})')
        sel = b.get('selection') or {}
        if sel.get('certified'):
            if sel.get('rule') is None or not _fin(_num(sel.get('nodes'))):
                raise InputError(f'{name}: certified selection without rule / nodes')
            if rule_nodes(sel['rule']) != int(_num(sel['nodes'])):
                raise InputError(f'{name}: selection nodes {sel["nodes"]} != {rule_nodes(sel["rule"])} of {sel["rule"]}')
    return True


def evaluate(arm_name, arm, ctrl, elig, eps=None, mstar_bar=MSTAR_BAR):
    if not arm or not arm.get('trained', False):
        return dict(outcome=NOT_RUN, step=1, guard=None, routes={})
    if 'routes' in arm and tuple(arm['routes']) != ARM_ROUTES[arm_name]:
        raise InputError(f'arm {arm_name}: input-supplied routes refused')
    g = guard(arm, ctrl, elig, eps)
    routes = {}
    for r in ARM_ROUTES[arm_name]:                   # Amendment 12: never input-supplied
        routes[r] = (route_mstar(arm, bar=mstar_bar) if r == 'mstar' else ROUTE_FN[r](arm, ctrl, elig, eps))
    out, step = decide(True, g['status'], {k: v['status'] for k, v in routes.items()})
    return dict(outcome=out, step=step, guard=g, routes=routes)


def verdict(inp):
    """inp = dict(arms={a|b|c: bank}, controls={K1, K2, K3: bank}, eligible={FI, FN0s, FC1: [cases]}, eps_ref=...)."""
    validate(inp)
    elig = inp.get('eligible', {})
    eps = inp.get('eps_ref', {})
    ctr = inp.get('controls', {})
    out = dict(arms={}, rules=dict(min=MIN, factors=FAC, mstar_bar_nodes=MSTAR_BAR))
    for name in ('a', 'b', 'c'):
        arm = inp.get('arms', {}).get(name)
        res = evaluate(name, arm, ctr.get('K1'), elig, eps)
        per = {}
        for k in ('K1', 'K2', 'K3'):
            c = ctr.get(k)
            if c is None:
                per[k] = dict(available=False)
                continue
            bar = (float(_num(c['selection']['nodes'])) / FAC['mstar']) if _certified(c) else None
            r = evaluate(name, arm, c, elig, eps, mstar_bar=bar if bar is not None else -1.)
            if bar is None and 'mstar' in r['routes']:
                r['routes']['mstar'] = dict(status=UNAV, why=f'{k} has no certified selected rule')
            per[k] = dict(available=True, guard=r['guard']['status'] if r['guard'] else None,
                          routes={kk: v['status'] for kk, v in r['routes'].items()})
        res['passes_against_all_K1_K3'] = bool(res['outcome'] != NOT_RUN and all(
            per[k].get('available') and per[k]['guard'] == PASS and any(s == PASS for s in per[k]['routes'].values())
            for k in ('K1', 'K2', 'K3')))
        res['per_control'] = per
        out['arms'][name] = res
    return out


def _clean(x):
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return 'inf' if x > 0 else ('-inf' if x < 0 else 'nan')
    return x


def main():
    inp = json.loads(open(sys.argv[1]).read())
    out = verdict(inp)
    s = json.dumps(_clean(out), indent=1)
    if len(sys.argv) > 2:
        open(sys.argv[2], 'w').write(s + '\n')
    for k, v in out['arms'].items():
        print(k, v['outcome'], 'guard', v['guard']['status'] if v['guard'] else None,
              {r: x['status'] for r, x in v['routes'].items()}, 'K1-K3:', v['passes_against_all_K1_K3'])


if __name__ == '__main__':
    main()
