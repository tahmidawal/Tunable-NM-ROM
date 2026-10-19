"""jcp-regimes unit gates (DESIGN 8): generator controls (G-seed, G-bounds, G-train, G-ic), reference-verdict controls
(R-order, R-peclet) on synthetic fixtures. CPU, NumPy (E0 verdict.py, engines.py). Run before every job:

    JAX_PLATFORMS=cpu python test_regimes.py      -> prints one line per gate, exit 1 on any failure
"""
from __future__ import annotations

import json
import sys

import numpy as np

import rcohorts as RG
import verdict as V
import cohorts as CH

RESULTS = []
RG_P = ('Ap', 'Am', 'C1', 'C2', 'B1', 'B2')


def gate(name, ok, detail=''):
    RESULTS.append(dict(gate=name, pass_=bool(ok), detail=str(detail)))
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)


def raises(fn):
    try:
        fn()
    except RG.CohortError:
        return True
    return False


def main():
    # G-seed: must refuse low, historical, public and training seeds; accept the registered ones
    bad = [0, 1000, 7090702, 20260927, 923801, 5550001, 2 ** 31 - 1]
    gate('G-seed refuses', all(raises(lambda s=s: RG.check_seed(s)) for s in bad), bad)
    gate('G-seed accepts registered', not any(raises(lambda s=s: RG.check_seed(s)) for d in RG.SEEDS
                                              for s in RG.SEEDS[d].values()))
    # G-bounds: every registered regime row passes; injected out-of-bounds rows are refused
    ok_all = True
    for g in RG.REGIMES_2D:
        try:
            RG.descriptors_2d(g)
        except RG.CohortError as e:
            ok_all = False
            print(e)
    gate('G-bounds registered regimes', ok_all)
    inj = [('Ap', [.5, .5, .1, 1., .05]),           # nu inside the training range
           ('Am', [.5, .5, .1, 1.9, .005]),         # s = 380 > 200 (stiffness moved too)
           ('C1', [.5, .5, .1, .6, .005]),          # s = 120 < 250
           ('C2', [.5, .5, .1, 2.5, .0025]),        # a > 2
           ('N0', [.5, .5, .3, 1., .05]),           # w out of range
           ('N0s', [.5, .5, .1, 1.0, .011]),        # N0s: a < 1.5
           ('N0s', [.5, .5, .1, 1.8, .02]),         # N0s: nu above .0126
           ('B1', [-1., 1., .05, .3, .3, .1, .8, .7, .7, .1, .8, 1.]),     # wrong sign for B1
           ('B2', [-1., 1., .05, .5, .5, .1, .8, .55, .55, .1, .8, 1.])]   # bumps not separated
    gate('G-bounds refuses injected rows', all(raises(lambda g=g, r=r: RG.check_bounds_2d(g, np.array(r))) for g, r in inj),
         [g for g, r in inj if not raises(lambda g=g, r=r: RG.check_bounds_2d(g, np.array(r)))])
    # G-train: N0 rows are not training / dev / val / test rows; an injected training row is refused
    tr = RG.training_rows_2d()
    others = np.concatenate([tr] + [CH.descriptors_2d(n) for n in ('dev6', 'val32', 'test64')])
    d0 = RG.descriptors_2d('N0')
    try:
        RG.assert_disjoint_2d(d0, others)
        disj = True
    except RG.CohortError:
        disj = False
    gate('G-train N0 disjoint', disj)
    try:
        info = RG.assert_all_disjoint(others)
        gate('G-train all regimes disjoint (A2.1)', True, info)
    except RG.CohortError as e:
        gate('G-train all regimes disjoint (A2.1)', False, e)
    # injected within-regime duplicate must be refused by the same routine's pairwise logic
    dup = np.concatenate([d0, d0[:1]])
    gate('G-train refuses injected duplicate', raises(lambda: RG.assert_disjoint_2d(dup[-1:], dup[:-1])))
    # G-pair (A2.5): C1 row j == A- row j in (cx, cy, w, nu), larger a
    am, c1 = RG.descriptors_2d('Am'), RG.descriptors_2d('C1')
    gate('G-pair C1/A- geometry and nu identical, a larger',
         bool(np.array_equal(am[:, [0, 1, 2, 4]], c1[:, [0, 1, 2, 4]]) and np.all(c1[:, 3] > am[:, 3])))
    gate('G-train refuses injected training row', raises(lambda: RG.assert_disjoint_2d(np.concatenate([d0, tr[:1]]), others)))
    # G-ic: single-bump regime ICs equal engines.initial; two-bump peak equals a on the 1025^2 grid
    E2 = CH.E2
    worst = 0.
    for g in ('N0', 'N0s', 'Ap', 'Am', 'C1', 'C2'):
        D = RG.descriptors_2d(g)
        for L in (256, 1024):
            for j in (0, 7, len(D) - 1):
                worst = max(worst, float(np.max(np.abs(RG.initial_2d(L, g, D[j]) - np.asarray(E2.initial(L, D[j]))))))
    gate('G-ic single bump == engines.initial', worst == 0., worst)
    pk = []
    for g in RG.TWO_BUMP:
        D = RG.descriptors_2d(g)
        for j in range(len(D)):
            x = np.arange(RG.S_REF_L + 1) / RG.S_REF_L
            X, Y = np.meshgrid(x, x, indexing='ij')
            pk.append(abs(float(np.max(np.abs(D[j][1] * RG.bump_sum(D[j], X, Y) / D[j][11]))) - D[j][1]))
    gate('G-ic two-bump peak == a', max(pk) <= 1e-12, max(pk))
    neg = [float(np.min(RG.initial_2d(256, 'B2', r_))) for r_ in RG.descriptors_2d('B2')]
    gate('G-ic B2 sign-changing', all(v < 0 for v in neg), min(neg))
    # E0 cohort identity: the Cohort interface of rcohorts gives the same t0 lattice as RC.ic_lattice expects
    c = RG.Cohort(2, 'B2')
    gate('G-ic interface shapes', c.ic(256, 0).shape == (255, 255) and c.full_ic(256, 0).shape == (257, 257))
    # G-ic-ps (A1.9): two-bump spectral IC coefficients == direct non-separable tensor Gauss-Legendre quadrature (N=64)
    import pscore as PS
    worst_ps = 0.
    for g in RG.TWO_BUMP:
        coh = RG.Cohort(2, g)
        for j in (0, 1):
            b = np.asarray(coh.ps_coefs(j, 64))
            X, W = PS.composite_gl(256)
            XX, YY = np.meshgrid(X, X, indexing='ij')
            F = coh.D[j][1] * RG.bump_sum(coh.D[j], XX, YY) / coh.D[j][11]
            k = np.arange(1, 64)
            Sx = 2. * np.sin(np.pi * k[:, None] * X[None, :])          # 2 int f sin per axis (as PS.sine_integrals)
            ref = (Sx * W[None, :]) @ F @ (Sx * W[None, :]).T
            worst_ps = max(worst_ps, float(np.max(np.abs(b - ref)) / np.max(np.abs(ref))))
    gate('G-ic-ps two-bump coefficients == direct quadrature', worst_ps <= 1e-12, worst_ps)
    # S-check fixtures (DESIGN 8, A2.3): separating fixture passes, non-separating fails, coverage rules fire
    import rscore as RS
    r_ = np.random.default_rng(1)
    base = dict(N0=list(r_.uniform(1, 10, 16)), N0s=list(r_.uniform(8, 12, 8)))
    sep_ok = RS.separation(dict(base, Ap=list(r_.uniform(2, 9, 16)), Am=list(r_.uniform(5, 11, 16)),
                                C1=list(r_.uniform(20, 40, 16)), C2=list(r_.uniform(50, 90, 16)),
                                B1=list(r_.uniform(2, 9, 16)), B2=list(r_.uniform(3, 9, 16))), RG_P)
    gate('S-check separating fixture passes', all(sep_ok[g]['passed'] for g in RG_P), {g: sep_ok[g].get('passed') for g in RG_P})
    sep_bad = RS.separation(dict(base, Ap=list(r_.uniform(30, 40, 16)), Am=list(r_.uniform(30, 40, 16)),
                                 C1=list(r_.uniform(1, 3, 16)), C2=list(r_.uniform(1, 2, 16)),
                                 B1=list(r_.uniform(30, 40, 16)), B2=list(r_.uniform(30, 40, 16))), RG_P)
    gate('S-check non-separating fixture fails', not any(sep_bad[g]['passed'] for g in RG_P))
    cov = RS.separation(dict(N0=[1.] * 3, N0s=[1.] * 8, Ap=[1.] * 16), RG_P)
    gate('S-check refuses < 8 N0', all('undetermined' in str(cov[g]) for g in RG_P))
    cov2 = RS.separation(dict(base, Am=[5.], C1=list(r_.uniform(20, 40, 16))), ('C1',))
    gate('S-check refuses C1 with < 8 A-', 'undetermined' in str(cov2['C1']))
    # R-order / R-peclet: synthetic triples through E0 verdict.judge
    rng = np.random.default_rng(0)
    ustar = rng.standard_normal((6, 257, 257))
    C = rng.standard_normal((6, 257, 257))
    t0 = ustar[0].copy()
    lv = []
    for L in (2048, 4096, 8192):
        u = ustar + C * (1. / L) * 1e3
        u[0] = t0
        lv.append(dict(L=L, lat=u, tau=0., steps_ok=True, umax=1.))
    j1 = V.judge(lv, t0, nu=.05)
    gate('R-order first-order triple rejected', j1['outcome'] == 'order-fail', (j1['outcome'], j1['p']))
    lv2 = []
    for L in (2048, 4096, 8192):
        u = ustar + C * (1. / L) ** 2 * 1e6
        u[0] = t0
        lv2.append(dict(L=L, lat=u, tau=0., steps_ok=True, umax=1.))
    j2 = V.judge(lv2, t0, nu=.05)
    gate('R-order second-order triple accepted', j2['outcome'] == 'accepted', (j2['outcome'], j2['p']))
    j3 = V.judge(lv2, t0, nu=1.0 / 2048 / 2.5)       # Peclet = umax h1 / nu = 2.5
    gate('R-peclet rejected', j3['outcome'] == 'peclet-fail', (j3['outcome'], j3['peclet']))
    ok = all(r['pass_'] for r in RESULTS)
    if len(sys.argv) > 1:
        open(sys.argv[1], 'w').write(json.dumps(dict(all_pass=ok, gates=RESULTS), indent=1) + '\n')
    print('ALL-PASS' if ok else 'SOME-FAIL')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
