"""jcp-heads CPU controls (DESIGN A1.7, A2.6): classification synthetics, H-cost-N / H-acc-K unit tests, the
duplicated-latent degeneracy fixture, the timing self-test. Fixtures (A2.6): n = 38, eps_ref = 1e-6,
e_Y,i = 0.01 (1 + i/38), costs as exact constants.

    JAX_PLATFORMS=cpu python test_h.py
"""
from __future__ import annotations

import numpy as np

import hstats as HS
import p1stats as PS
import timing as TM

N = 38
EPS = np.full(N, 1e-6)
EY = .01 * (1 + np.arange(N) / N)


def lab(eX, eY=EY):
    return HS.error_label(eX, eY, EPS)


def cost(r):
    return HS.cost_label(np.full(N, r))


def check(name, got, want):
    assert got == want, f'{name}: got {got!r}, want {want!r}'
    print(f'ok  {name}: {got}')


def main():
    # A1.7 dominance synthetics (X = head)
    check('10% better, 25% costlier', HS.dominance(lab(.9 * EY), cost(1.25)), 'neither (trade-off)')
    check('10% better, 25% cheaper', HS.dominance(lab(.9 * EY), cost(.75)), 'X dominates')
    check('X = Y', HS.dominance(lab(EY.copy()), cost(1.)), 'equivalent')
    check('10% better, cost 1.0', HS.dominance(lab(.9 * EY), cost(1.)), 'X dominates')
    e5 = lab(.95 * EY)
    check('5% better: label', e5['label'], 'X more accurate')
    check('5% better: equivalence flag', e5['equivalent_flag'], True)
    check('unqualified side', HS.dominance(lab(.9 * EY), cost(.75), qualified=False), 'unqualified')
    check('empty population', HS.dominance(HS.error_label([], [], []), cost(1.)), 'no population')
    check('cost exactly 1.1 is equal cost', cost(1.1)['label'], 'equal cost')
    # Q-LH-cost
    head = dict(qualified=True, complete=True)
    costlier = {r: dict(qualified=True, cost=cost(1.5), err=lab(.9 * EY)) for r in (4, 8)}
    check('every rung costlier', HS.lh_cost(head, costlier)['at_most_cost'], 'no possibly-affordable rung')
    check('every rung costlier: reason', HS.lh_cost(head, costlier)['at_most_cost_reason'], 'every qualified rung certified costlier')
    unq = {r: dict(qualified=False, cost=cost(.5), err=lab(.9 * EY)) for r in (4, 8)}
    check('no qualified rung: reason', HS.lh_cost(head, unq)['at_most_cost_reason'], 'no numerically qualified linear rung')
    # Codex round-2 counterexample: H_i = i; rung A = H + 1 except cases 20-25 = H - 10; rung B = H; both affordable
    H = np.arange(1, N + 1, dtype=float)
    Ar = H + 1
    Ar[19:25] = H[19:25] - 10
    rungs = {16: dict(qualified=True, cost=cost(.8), err=HS.error_label(H, Ar, EPS)),
             32: dict(qualified=True, cost=cost(.9), err=HS.error_label(H, H.copy(), EPS))}
    out = HS.lh_cost(head, rungs)
    check('counterexample: not "beats every affordable rung"', out['at_most_cost'],
          'linear at no more cost is at least as accurate (rung 32: matches)')
    # Holm
    rj = HS.holm({'a': .001, 'b': .02, 'c': .04})
    check('holm', (rj['a'], rj['b'], rj['c']), (True, True, True))
    rj = HS.holm({'a': .001, 'b': .03, 'c': .04})
    check('holm step-down stop', (rj['a'], rj['b'], rj['c']), (True, False, False))
    # F5
    per = {4: dict(positive_components=True, lh_dim='unresolved', qualified=True, resolved_negative=False),
           8: dict(positive_components=False, lh_dim='Y dominates', qualified=True, resolved_negative=True)}
    check('F5: unresolved Q-LH-dim blocks positive', HS.f5([4, 8], per)['outcome'], 'inconclusive')
    per[4]['lh_dim'] = 'neither (trade-off)'
    check('F5: positive', HS.f5([4, 8], per)['outcome'], 'exploratory validation evidence of an accuracy-cost benefit of the head')
    per[4].update(positive_components=False, resolved_negative=True)
    check('F5: all resolved negative', HS.f5([4, 8], per)['outcome'], 'no benefit shown on validation evidence')
    # H-acc-K unit tests (p1stats): > 25 % unqualified -> inconclusive (unit only, A2.6); clear improvement -> improves
    de = -np.full(N, 1e-3)
    q = np.ones(N, bool)
    check('H-acc-K improves', PS.h_acc_r_transition(de, EPS, q, q)['outcome'], 'improves')
    qq = q.copy()
    qq[:12] = False
    check('H-acc-K > 25% unqualified (unit)', PS.h_acc_r_transition(de, EPS, qq, q)['outcome'], 'inconclusive')
    # H-cost-N must-fail: a 1.5x slower mesh in a flat panel -> fail; flat -> pass (P1 h_cost_n)
    rng = np.random.default_rng(0)
    flat = np.exp(rng.normal(0, .01, (6, 3, 3)))                 # (cases, meshes, repeats)
    slow = flat.copy()
    slow[:, 2] *= 1.5
    valid = np.ones_like(flat, bool)
    out_f = PS.h_cost_n(flat, valid, (256, 1024, 4096), nboot=2000)
    out_s = PS.h_cost_n(slow, valid, (256, 1024, 4096), nboot=2000)
    check('H-cost-N flat', out_f['outcome'], 'pass')
    check('H-cost-N 1.5x fault', out_s['outcome'], 'fail')
    # degeneracy fixture: duplicated latent -> rank-deficient head Jacobian (A1.3)
    import pickle
    from pathlib import Path
    import nphead as NH
    hd = NH.Head(Path(__file__).resolve().parents[1] / 'separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl')
    W0, b0 = hd.layers[0]
    W0 = W0.copy()
    W0[14] = W0[15]
    hd.layers[0] = (W0, b0)
    hd.h_lin = hd.h_lin.copy()
    hd.h_lin[14] = hd.h_lin[15]
    z = np.random.default_rng(1).normal(size=16) * .1
    J = np.empty((512, 16))
    for i in range(16):
        e = np.zeros(16)
        e[i] = 1e-6
        J[:, i] = (hd.c(z + e) - hd.c(z - e)) / 2e-6
    s = np.linalg.svd(J, compute_uv=False)
    check('duplicated latent flagged (sigma_min / sigma_max < 1e-10 within FD noise 1e-8)', bool(s[-1] / s[0] < 1e-8), True)
    TM._selftest()
    print('test_h: all checks passed')


if __name__ == '__main__':
    main()
