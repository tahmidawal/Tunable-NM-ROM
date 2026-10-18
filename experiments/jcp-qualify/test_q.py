"""jcp-qualify CPU control tests (DESIGN 7, A1.3, A1.4, A1.7, A2.5, A3.5). Writes checks/test_q.json; exit 1 on failure.

    JAX_PLATFORMS=cpu /home/tahmid/Dev/.venv/bin/python test_q.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import qstats as QS
import qscore as SC

HERE = Path(__file__).resolve().parent
RES = {}


def check(name, ok, **kw):
    RES[name] = dict(passed=bool(ok), **kw)


def t_dominance():
    pt = lambda c, e, eps=0.: dict(c=c, e=e, eps=eps)
    # (a) cost-only: equal error, 10 % cheaper
    check('dominance cost-only', QS.dominates(pt(.9, 1.), pt(1., 1.)) and not QS.dominates(pt(1., 1.), pt(.9, 1.)))
    # (b) error-only: equal cost, error lower beyond the margin (2 % + 2 eps)
    check('dominance error-only', QS.dominates(pt(1., .95), pt(1., 1.)) and not QS.dominates(pt(1., 1.), pt(1., .95)))
    # (c) tie band: within 5 % cost and within the error margin -> both stay
    P = {'x': pt(1., 1.), 'y': pt(1.04, .99)}
    f, _ = QS.front(P)
    check('dominance tie band keeps both', f == ['x', 'y'], front=f)
    # eps widens the margin: an error-only gain inside 2 eps is a tie
    check('dominance eps margin', not QS.dominates(pt(1., .95, eps=.03), pt(1., 1., eps=.03)))
    # (d) non-transitive chain: B > A, C > B, C does not dominate A (zero eps)
    A, B, C = pt(1., 1.), pt(1.04, .96), pt(1.0816, .9216)
    chain = QS.dominates(B, A) and QS.dominates(C, B) and not QS.dominates(C, A)
    f, dom = QS.front({'A': A, 'B': B, 'C': C})
    check('dominance non-transitive chain recorded pairwise', chain and f == ['C'] and dom['A'] == ['B'], front=f)
    # no mutual domination over a random set
    rng = np.random.default_rng(0)
    pts = {i: pt(float(c), float(e), float(s)) for i, (c, e, s) in enumerate(zip(rng.uniform(.5, 2, 60), rng.uniform(.01, .05, 60), rng.uniform(0, 1e-4, 60)))}
    mutual = any(QS.dominates(pts[i], pts[j]) and QS.dominates(pts[j], pts[i]) for i in pts for j in pts if i != j)
    check('dominance never mutual', not mutual)


def t_order():
    rng = np.random.default_rng(0)
    hs = (1., .5, .25)

    def case(p, C=1e-3, noise=1e-3):
        e = {h: C * h ** p * (1 + noise * rng.standard_normal()) for h in hs}
        diff = e[.5] - e[.25]
        return QS.order_case(e, d_anchor=1e-3 * e[.25], d_solver=1e-3 * diff, d_quad=1e-3 * diff, d_finer=1e-6, valid=True)
    l1 = QS.order_label([case(1) for _ in range(6)])
    l2 = QS.order_label([case(2) for _ in range(6)])
    check('order classifier first order', l1['label'] == 'order 1', label=l1)
    check('order classifier second order', l2['label'].startswith('order 2'), label=l2)
    pre = QS.order_case({1.: 4e-3, .5: 1e-3, .25: 1e-3 / 2 ** 1.5}, 1e-9, 1e-12, 1e-12, 1e-6, True)
    check('order pre-asymptotic not interpretable', not pre['interpretable'], res=pre)
    anc = QS.order_case({1.: 1e-3, .5: 2.5e-4, .25: 6.25e-5}, 1e-5, 1e-12, 1e-12, 1e-6, True)
    check('order anchor-contaminated not interpretable', not anc['interpretable'], res=anc)
    z = QS.order_case({1.: 1e-3, .5: 0., .25: 0.}, 0., 0., 0., 0., True)
    check('order zero error not interpretable', not z['interpretable'], res=z)
    lab = QS.order_label([case(2) for _ in range(4)] + [z, z])
    check('order fewer than 5 interpretable -> not established', lab['label'].startswith('order not established'), label=lab)


def t_gal():
    import qcert as QC
    e = np.array([1e-8, .99e-8, 1.01e-8, np.nan, np.inf, 0.])
    got = QC.certified_gal(e).tolist()
    check('GAL classifier boundaries', got == [True, True, False, False, False, True], got=got)
    tolf = 1e-10
    # positive scale: eta = tolratio * tolf
    tr = np.array([.5, 2.])
    check('GAL G-cert positive scale', QC.gcert_gal(tr * tolf, tr * tolf * 3., np.array([3., 3.]), tr, tolf))
    check('GAL G-cert detects mismatch', not QC.gcert_gal(tr * tolf * 1.01, tr * tolf * 3., np.array([3., 3.]), tr, tolf))
    # zero scale: raw recovered as tolratio * 1e-300
    raw = np.array([0., 1e-290])
    trz = raw / 1e-300
    check('GAL G-cert zero scale', QC.gcert_gal(raw * 1e300, raw, np.array([0., 0.]), trz, tolf))
    check('GAL recorded eta zero scale', np.array_equal(QC.recorded_eta(trz, np.zeros(2), tolf), np.array([0., np.inf])))
    check('GAL G-cert unrecoverable overflow', not QC.gcert_gal(np.array([np.inf]), np.array([1e9]), np.array([0.]),
                                                                np.array([np.inf]), tolf))


def t_swap():
    rec = dict(case_id='dev6|0', time_tag=SC.TIME_TAG, scheme='BE', form='LSPG', dt=.01, dt_expected=.005, requested=('dev6', 0))
    try:
        SC.score(np.zeros((6, 257, 257)), rec)
        ok = False
    except SC.IdentityError:
        ok = True
    check('Delta-t swap raises', ok)
    rec.update(dt=.01, dt_expected=.01)
    try:
        SC.score(np.zeros((6, 257, 257)), rec, 'S')
        ok = False
    except SC.IdentityError:
        ok = True
    check('S at 2 dt0 raises', ok)


if __name__ == '__main__':
    for t in (t_dominance, t_order, t_gal, t_swap):
        try:
            t()
        except Exception as e:  # noqa: BLE001
            check(t.__name__, False, error=repr(e))
    (HERE / 'checks').mkdir(exist_ok=True)
    (HERE / 'checks' / 'test_q.json').write_text(json.dumps(RES, indent=1, default=str) + '\n')
    bad = [k for k, v in RES.items() if not v['passed']]
    print(json.dumps({k: v['passed'] for k, v in RES.items()}, indent=1))
    sys.exit(1 if bad else 0)
