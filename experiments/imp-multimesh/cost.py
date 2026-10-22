"""imp-multimesh decision logic (DESIGN 4-6, A1.1, A2.1, A2.3, A3.1, A3.3-A3.5, A3.7). Pure Python / numpy.

Data model (built by load_inputs from the pulled run JSONs, or by the control-8 fixtures):
  ours[X] = dict(status={L: 'QUALIFIED'|'NOT-QUALIFIED'|'UNRESOLVED'}, err={L: {case: e}}, eps={case: eps_ref},
                 t={L: mean query seconds}, setup={L: seconds}, compile={L: seconds})
  ours_offline = seconds (bank data + bank fit + head, per accounting variant)
  comp[arm][L][cfg] = dict(state='MEASURED-Q1'|'MEASURED-FAIL'|'NOT-CONSTRUCTIBLE'|'UNRESOLVED', err={case: e},
                           t=mean query seconds, compile=s, armdata=s, needs=[artefact keys], rollout_ok=bool|None)
  art[key] = seconds (shared construction artefacts; each charged once per workload)
A workload is (frozenset of meshes, q). Verdict functions return 'MATCH' | 'NO-MATCH' | 'UNRESOLVED'.
"""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np

MESHES = (256, 1024, 4096)
QS = (10, 100, 1000)
BAND = (1 / 1.1, 1.1)
CP_ARMS = ('CPR-256', 'CPR-1024', 'CPU-256', 'CPU-1024')
VERDICTS = ('worst', 'worst_eps', 'casewise', 'worst_median', 'rollout')


def workloads():
    out = []
    for k in (1, 2, 3):
        for ms in itertools.combinations(MESHES, k):
            for q in QS:
                out.append((frozenset(ms), q))
    return out


def wname(w):
    return '+'.join(str(m) for m in sorted(w[0])) + f'|q{w[1]}'


# ---------------------------------------------------------------------------------------------- costs ----

REG_CFGS = tuple(f'r{r}|k{k}|CNR' for r in (128, 256, 384) for k in (4, 8))


def status(D, X, L):
    """A2.1/A3.1: a missing status is UNRESOLVED, never a failure."""
    return D['ours'][X]['status'].get(L, 'UNRESOLVED')


def complete_universe(D, cfgs=REG_CFGS, arms=CP_ARMS, meshes=MESHES):
    """Populate every registered (arm, mesh, configuration) that is absent as UNRESOLVED (code audit 1)."""
    for arm in arms:
        D['comp'].setdefault(arm, {})
        for L in meshes:
            D['comp'][arm].setdefault(L, {})
            for k in cfgs:
                D['comp'][arm][L].setdefault(k, dict(state='UNRESOLVED', err={}, t=float('nan'), compile=0., armdata=0.,
                                                     needs=[], rollout_ok=None))
    return D


def _cmp(v, mode):
    """A7.1: compile value under a mode: None = as measured (missing -> nan); a number = that value."""
    if mode is None:
        return float('nan') if v is None else float(v)
    return float(mode)


def ours_cost(D, X, w, var='A_dep', compile_scale=1.0, ours_compile=None):
    o = D['ours'][X]
    c = D['ours_offline'][var]
    for L in w[0]:
        c += o['setup'].get(L, math.nan) + compile_scale * _cmp(o['compile'].get(L), ours_compile) + w[1] * o['t'].get(L, math.nan)
    return c


def comp_cost(D, arm, choice, w, compile_scale=1.0, comp_compile=None):
    """choice: {L: cfg}. Shared artefacts charged once (A2.3, A3.4)."""
    need = set()
    c = 0.
    for L, cfg in choice.items():
        rec = D['comp'][arm][L][cfg]
        need.update(rec['needs'])
        c += rec['armdata'] + compile_scale * _cmp(rec.get('compile'), comp_compile) + w[1] * rec['t']
    return c + sum(D['art'][k] for k in need)


# ----------------------------------------------------------------------------------------- accuracy ----

def pooled(errs):
    return [e for d in errs for e in d.values()]


def admissible(D, X, arm, choice, w, verdict):
    o = D['ours'][X]
    ex = {L: o['err'][L] for L in w[0]}
    ec = {L: D['comp'][arm][L][choice[L]]['err'] for L in w[0]}
    Wx = max(pooled(ex.values()))
    Wc = max(pooled(ec.values()))
    if verdict == 'worst':
        return Wc <= Wx
    if verdict == 'worst_eps':
        return Wc <= Wx + 2 * max(D['eps'].values())
    if verdict == 'casewise':
        return all(ec[L][c] <= ex[L][c] + 2 * D['eps'][c] for L in w[0] for c in ex[L])
    if verdict == 'worst_median':
        return Wc <= Wx and np.median(pooled(ec.values())) <= np.median(pooled(ex.values()))
    if verdict == 'rollout':
        return Wc <= Wx and all(D['comp'][arm][L][choice[L]].get('rollout_ok') for L in w[0])
    raise ValueError(verdict)


def match(D, X, w, verdict='worst', var='A_dep', compile_scale=1.0, arms=CP_ARMS, ours_compile=None,
          comp_compile=None):
    """Three-valued per (w, X) over all CP variants (A2.1, A3.1). Returns (state, witness, best_admissible)."""
    Cx = ours_cost(D, X, w, var, compile_scale, ours_compile)
    if not math.isfinite(Cx):
        return 'UNRESOLVED', None, Cx                    # a missing measurement of ours (e.g. compile) is unresolved
    unresolved = False
    best = None
    for arm in arms:
        if arm not in D['comp']:
            unresolved = True
            continue
        per = []
        for L in sorted(w[0]):
            cfgs = D['comp'][arm].get(L)
            if not cfgs:
                unresolved = True
                per = None
                break
            if any(r['state'] == 'UNRESOLVED' for r in cfgs.values()):
                unresolved = True
            per.append([(L, k) for k, r in cfgs.items() if r['state'] == 'MEASURED-Q1'])
        if per is None:
            continue
        for combo in itertools.product(*per):
            choice = dict(combo)
            if not admissible(D, X, arm, choice, w, verdict):
                continue
            c = comp_cost(D, arm, choice, w, compile_scale, comp_compile)
            if not math.isfinite(c):
                unresolved = True
                continue
            if best is None or c < best[0]:
                best = (c, arm, choice)
    if best is not None and best[0] <= Cx:
        return 'MATCH', best, Cx
    return ('UNRESOLVED' if unresolved else 'NO-MATCH'), best, Cx


def compile_robust(D, X, w, verdict='worst', var='A_dep'):
    """A7.1 adverse assignments with B = 10 x the largest measured compile time."""
    B = 10. * D['compile_max']
    base = match(D, X, w, verdict, var)[0]
    a = match(D, X, w, verdict, var, ours_compile=0., comp_compile=B)[0]
    b = match(D, X, w, verdict, var, ours_compile=B, comp_compile=0.)[0]
    return dict(base=base, adverse_to_match=a, adverse_to_nomatch=b, robust=(a == base == b), B=B)


def stop_rule(D, verdict='worst', var='A_dep', compile_scale=1.0):
    """A2.1 outcome over the 21 workloads and our configurations."""
    table = {}
    any_nomatch = any_unres = False
    vacuous = []
    for w in workloads():
        row = {}
        qual = [X for X in D['ours'] if all(status(D, X, L) == 'QUALIFIED' for L in w[0])]
        unres_x = [X for X in D['ours'] if any(status(D, X, L) == 'UNRESOLVED' for L in w[0])]
        if unres_x:
            any_unres = True
        if not qual and not unres_x:
            vacuous.append(wname(w))
        for X in qual:
            st, best, Cx = match(D, X, w, verdict, var, compile_scale)
            row[X] = dict(state=st, C_ours=Cx, compile=compile_robust(D, X, w, verdict, var) if 'compile_max' in D else None, best=None if best is None else dict(cost=best[0], arm=best[1],
                          choice={str(k): v for k, v in best[2].items()}, ratio=best[0] / Cx,
                          near_tie=bool(BAND[0] <= best[0] / Cx <= BAND[1])))
            any_nomatch |= st == 'NO-MATCH'
            any_unres |= st == 'UNRESOLVED'
        table[wname(w)] = row
    if any_nomatch:
        outcome = 'NOT FIRED'
    elif any_unres:
        outcome = 'NOT EVALUABLE'
    else:
        outcome = 'FIRES'
    return dict(outcome=outcome, vacuous=vacuous, table=table, verdict=verdict, variant=var, compile_scale=compile_scale)


def breakeven(D, X, meshes, arm_best_fn=None, qgrid=None):
    """Lower envelope (A1.9): cheapest CP total (worst-admissible) vs ours over q; returns the crossing q values."""
    qgrid = np.unique(np.round(np.logspace(0, 6, 241))).astype(int) if qgrid is None else qgrid
    sign = []
    for q in qgrid:
        st, best, Cx = match(D, X, (frozenset(meshes), int(q)))
        sign.append(None if best is None else float(best[0] - Cx))
    xs = []
    for i in range(1, len(qgrid)):
        a, b = sign[i - 1], sign[i]
        if a is not None and b is not None and (a > 0) != (b > 0):
            xs.append(int(qgrid[i]))
    return dict(q=qgrid.tolist(), diff=sign, crossings=xs)


# ------------------------------------------------------------------------------------ control 8 fixtures ----

def selftest(D):
    """A3.5 control 8 on fixtures derived from measured records. Returns dict of (expected, got, passed)."""
    import copy
    res = {}
    X = next(iter(D['ours']))
    w = next(w for w in workloads() if all(L in next(iter(D['comp'].values())) for L in w[0]))

    def only_measured(Dx):
        Dy = copy.deepcopy(Dx)
        for arm in Dy['comp']:
            for L in Dy['comp'][arm]:
                Dy['comp'][arm][L] = {k: r for k, r in Dy['comp'][arm][L].items() if r['state'] != 'UNRESOLVED'}
        for arm in CP_ARMS:
            Dy['comp'].setdefault(arm, {L: {} for L in MESHES})
            for L in MESHES:
                Dy['comp'][arm].setdefault(L, {})
        return Dy

    def run(name, Dx, expected, ww=w, XX=X, fn=None):
        got = fn(Dx) if fn else match(Dx, XX, ww, arms=tuple(Dx['comp']))[0]
        res[name] = dict(expected=expected, got=got, passed=got == expected)

    base = only_measured(D)
    # every arm absent at a mesh of w makes the comparison unresolved; for the exhaustive fixtures, keep only arms
    # that are fully measured at the meshes of w
    full = copy.deepcopy(base)
    full['comp'] = {a: v for a, v in full['comp'].items() if all(v.get(L) for L in w[0])}
    Da = copy.deepcopy(full)
    for a in Da['comp'].values():
        for L in a:
            for r in a[L].values():
                r['err'] = {c: float('inf') for c in r['err']}
    run('8a_errors_inf', Da, 'NO-MATCH')
    Db = copy.deepcopy(full)
    Cx = ours_cost(Db, X, w)
    for a in Db['comp'].values():
        for L in a:
            for r in a[L].values():
                r['t'] = (1.5 * Cx + 1) / max(w[1], 1)
    run('8b_costs_inflated', Db, 'NO-MATCH')
    Dc = copy.deepcopy(full)
    for a in Dc['comp'].values():
        for L in a:
            for r in a[L].values():
                if r['state'] == 'MEASURED-Q1':
                    r['state'] = 'MEASURED-FAIL'
    run('8c_q1_false', Dc, 'NO-MATCH')
    # (d) two witnesses: make ours expensive so a witness exists, duplicate it, remove one
    Dd = copy.deepcopy(full)
    Dd['ours_offline']['A_dep'] = 1e12
    st, best, _ = match(Dd, X, w, arms=tuple(Dd['comp']))
    if best is not None:
        arm, choice = best[1], best[2]
        for L, k in choice.items():
            Dd['comp'][arm][L][k + '#dup'] = copy.deepcopy(Dd['comp'][arm][L][k])
            del Dd['comp'][arm][L][k]
        run('8d_one_of_two_witnesses_removed', Dd, 'MATCH')
        De = copy.deepcopy(Dd)
        for a in De['comp']:
            for L in De['comp'][a]:
                for k in list(De['comp'][a][L]):
                    De['comp'][a][L][k]['state'] = 'MEASURED-FAIL'
        L0 = sorted(w[0])[0]
        k0 = next(iter(De['comp'][arm][L0]))
        De['comp'][arm][L0][k0]['state'] = 'UNRESOLVED'
        run('8e_witnesses_removed_one_unresolved', De, 'UNRESOLVED')
    else:
        res['8d_one_of_two_witnesses_removed'] = dict(expected='MATCH', got='no witness in fixture', passed=False)
    # (f) X status UNRESOLVED, with and without another NO-MATCH (competitor errors 0 -> every qualified X MATCH when
    # ours is expensive; errors +inf -> every qualified X NO-MATCH)
    Df = copy.deepcopy(full)
    Df['ours_offline']['A_dep'] = 1e12
    for XX in Df['ours']:
        Df['ours'][XX]['status'] = {L: 'QUALIFIED' for L in Df['ours'][XX]['status']}
    Df['ours'][X]['status'] = {L: 'UNRESOLVED' for L in Df['ours'][X]['status']}
    for a in Df['comp'].values():
        for L in a:
            for r in a[L].values():
                r['err'] = {c: 0. for c in r['err']}
                r['state'] = 'MEASURED-Q1'
    o = stop_rule_fixture(Df)
    res['8f1_x_unresolved_no_nomatch'] = dict(expected='NOT EVALUABLE', got=o, passed=o == 'NOT EVALUABLE')
    others = [XX for XX in Df['ours'] if XX != X]
    Dg = copy.deepcopy(Df)
    for a in Dg['comp'].values():
        for L in a:
            for r in a[L].values():
                r['err'] = {c: float('inf') for c in r['err']}
    o = stop_rule_fixture(Dg)
    exp = 'NOT FIRED' if others else 'NOT EVALUABLE'
    res['8f2_x_unresolved_other_nomatch'] = dict(expected=exp, got=o, passed=o == exp)
    Dh = copy.deepcopy(full)
    for XX in Dh['ours']:
        Dh['ours'][XX]['status'] = {L: 'NOT-QUALIFIED' for L in Dh['ours'][XX]['status']}
    o = stop_rule_fixture(Dh, full_out=True)
    res['8g_vacuous'] = dict(expected='vacuous flagged', got=len(o['vacuous']), passed=len(o['vacuous']) > 0)
    return res


def stop_rule_fixture(D, full_out=False):
    """stop_rule restricted to the workloads whose meshes every present arm covers (fixture universe); per-workload
    vacuity (code audit 1)."""
    covered = [w for w in workloads() if all(all(a.get(L) for L in w[0]) for a in D['comp'].values())]
    any_nomatch = any_unres = False
    vacuous = []
    for w in covered:
        qual = [X for X in D['ours'] if all(status(D, X, L) == 'QUALIFIED' for L in w[0])]
        unres_w = any(any(status(D, X, L) == 'UNRESOLVED' for L in w[0]) for X in D['ours'])
        any_unres |= unres_w
        if not qual and not unres_w:
            vacuous.append(wname(w))
        for X in qual:
            st = match(D, X, w, arms=tuple(D['comp']))[0]
            any_nomatch |= st == 'NO-MATCH'
            any_unres |= st == 'UNRESOLVED'
    out = 'NOT FIRED' if any_nomatch else ('NOT EVALUABLE' if any_unres else 'FIRES')
    return dict(outcome=out, vacuous=vacuous) if full_out else out


def selftest_strict(D):
    """selftest with the registered fixture preconditions enforced (A3.5): a missing precondition is a control
    failure; also the absent-key variants of (f)."""
    import copy
    res = selftest(D)
    need = ('8a_errors_inf', '8b_costs_inflated', '8c_q1_false', '8d_one_of_two_witnesses_removed',
            '8e_witnesses_removed_one_unresolved', '8f1_x_unresolved_no_nomatch', '8f2_x_unresolved_other_nomatch',
            '8g_vacuous')
    for k in need:
        res.setdefault(k, dict(expected='present', got='precondition missing', passed=False))
    if len(D['ours']) < 2:
        res['8f2_x_unresolved_other_nomatch'] = dict(expected='two X', got='precondition missing', passed=False)
    X = next(iter(D['ours']))
    Dk = copy.deepcopy(D)
    Dk['ours'][X]['status'] = {}                      # absent keys, not explicit UNRESOLVED
    Dk['ours_offline']['A_dep'] = 1e12
    for a in Dk['comp'].values():
        for L in a:
            for r in a[L].values():
                r['err'] = {c: 0. for c in r['err']}
                r['state'] = 'MEASURED-Q1'
    o = stop_rule_fixture(Dk)
    res['8f3_absent_status'] = dict(expected='NOT EVALUABLE', got=o, passed=o == 'NOT EVALUABLE')
    return res


def dump(path, obj):
    def conv(x):
        if isinstance(x, (np.floating, np.integer)):
            return x.item()
        if isinstance(x, (set, frozenset, tuple)):
            return list(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        return str(x)
    Path(path).write_text(json.dumps(obj, indent=1, default=conv))


def combos(D, w, arms=CP_ARMS):
    """Every measured Q1 combination over the CP variants for workload w: (arm, choice, cost, worst, median)."""
    out = []
    for arm in arms:
        per = []
        for L in sorted(w[0]):
            per.append([(L, k) for k, r in D['comp'][arm][L].items() if r['state'] == 'MEASURED-Q1'])
        for combo in itertools.product(*per):
            ch = dict(combo)
            e = [x for L, k in ch.items() for x in D['comp'][arm][L][k]['err'].values()]
            out.append((arm, ch, comp_cost(D, arm, ch, w), max(e), float(np.median(e))))
    return out
