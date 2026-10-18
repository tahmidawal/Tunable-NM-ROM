"""jcp-dims CPU controls (DESIGN 7: decision-rule synthetics, draw identity, generator tests, certificates).

    JAX_PLATFORMS=cpu python test_dims.py      -> checks/test_dims.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import dstats as DS
import dcert as DC

HERE = Path(__file__).resolve().parent
RES = {}


def check(name, cond):
    RES[name] = bool(cond)
    print(('PASS ' if cond else 'FAIL ') + name, flush=True)


def t_transition():
    n = 38
    eps = np.full(n, 1e-4)
    eX = np.full(n, .01)
    check('transition: all improve -> improves', DS.transition(eX, eX - .002, eps)[0] == 'improves')
    check('transition: all worsen -> does not improve', DS.transition(eX, eX + .002, eps)[0] == 'does not improve')
    eY = eX.copy()
    eY[:15] -= .002                                    # 15 resolved of 38 < 50 %
    check('transition: 40 % resolved -> inconclusive', DS.transition(eX, eY, eps)[0] == 'inconclusive (resolution)')
    check('transition: unqualified -> inconclusive', DS.transition(eX, eX - .002, eps, qualified=(True, False))[0]
          == 'inconclusive (unqualified cell)')
    check('transition: empty -> inconclusive', DS.transition(eX, eX - .002, eps, eligible=np.zeros(n, bool))[0]
          == 'inconclusive (no eligible cases)')
    eY = eX.copy()
    eY[:20] -= .002
    eY[20:] += .0005                                   # majority improve but median change > 0? (20 improve, 18 worsen)
    lab, p = DS.transition(eX, eY, eps)
    check('transition: majority improve, median < 0 needed', lab == ('improves' if p['median_change'] < 0 else 'does not improve'))
    check('range: all improves -> monotone', DS.range_verdict(['improves'] * 3) == 'monotone')
    check('range: one not -> not monotone', DS.range_verdict(['improves', 'does not improve', 'inconclusive (resolution)'])
          == 'not monotone')
    check('range: inconclusive', DS.range_verdict(['improves', 'inconclusive (resolution)']) == 'inconclusive')


def t_two_way():
    n = 38
    eps = np.full(n, 1e-4)
    eL = np.full(n, .01)
    check('two-way: GAL better', DS.two_way(eL - .002, eL, eps, names=('GAL', 'LSPG'))[0] == 'GAL more accurate')
    check('two-way: LSPG better', DS.two_way(eL + .002, eL, eps, names=('GAL', 'LSPG'))[0] == 'LSPG more accurate')
    eG = eL.copy()
    eG[:19] -= .002
    eG[19:] += .002
    check('two-way: split -> neither', DS.two_way(eG, eL, eps, names=('GAL', 'LSPG'))[0] == 'neither')
    check('control M=r: fired', DS.control_mr('improves', (True, True)) == 'fired')
    check('control M=r: adverse', DS.control_mr('does not improve', (True, True)) == 'did not fire')
    check('control M=r: unqualified never fires', DS.control_mr('improves', (False, True)).startswith('unavailable'))


def t_q2_mstar():
    check('q2 pass', DS.q2_label(True, (True, 1e-4), .05, (True, 1e-5)) == 'pass')
    check('q2 fail distance', DS.q2_label(True, (True, 3e-4), .05, (True, 1e-5)) == 'fail')
    check('q2 fail rho', DS.q2_label(True, (True, 1e-4), .2, (True, 1e-5)) == 'fail')
    check('q2 fail nonfinite', DS.q2_label(True, (True, float('nan')), .05, (True, 1e-5)) == 'fail')
    check('q2 successor invalid', DS.q2_label(True, (False, 1e-4), .05, (True, 1e-5)) == 'unverified (successor invalid)')
    check('q2 finer not converged', DS.q2_label(True, (True, 1e-4), .05, (True, 3e-4)) == 'unverified (finer rule not converged)')
    check('q2 finer unavailable', DS.q2_label(True, (True, 1e-4), .05, None) == 'unverified (finer rule unavailable)')
    check('q2 ladder top', DS.q2_label(True, None, .05, None, ladder_top=True) == 'unverified (ladder top)')
    sz = [1, 2, 3, 4]
    Q, N, X = 'qualified', 'not qualified', 'missing'
    check('m* stable', DS.mstar([N, Q, Q, Q], sz)['outcome'] == 'm* = 2 (stable)')
    check('m* non-monotone', DS.mstar([N, Q, N, Q], sz)['outcome'] == 'm* = 2 (non-monotone)')
    check('m* none', DS.mstar([N, N, N, N], sz)['outcome'].startswith('none'))
    check('m* lower missing', DS.mstar([X, Q, Q, Q], sz)['outcome'].startswith('not reached'))
    check('m* larger missing', DS.mstar([N, Q, X, Q], sz)['outcome'] == 'm* = 2 (stability unverified: incomplete ladder)')
    draws = [[N, Q, Q, Q], [N, N, Q, Q], [N, Q, Q, Q], [Q, Q, Q, Q], [N, N, N, N]]
    f = DS.family_mstar(draws, sz)
    check('family m*: all draws (one draw none) -> none', f['family']['m'] is None and f['censored'] == 1)
    draws[4] = [N, N, Q, Q]
    f = DS.family_mstar(draws, sz)
    check('family m*: smallest all-draw candidate', f['family']['m'] == 3 and f['median_draw'] == 2.0)
    check('width m*: intersection', DS.width_mstar({'a': [N, Q, N, Q], 'b': [N, N, Q, Q]}, sz)['m'] == 4)
    check('width m*: missing lower on one spec -> not reached', DS.width_mstar({'BE': [X, Q], 'CN': [Q, Q]}, [1, 2])['outcome']
          .startswith('not reached'))
    draws = [[Q, Q, Q, Q], [N, X, Q, Q], [Q, Q, Q, Q], [Q, Q, Q, Q], [Q, Q, Q, Q]]
    check('family m*: missing lower in one draw -> not reached', DS.family_mstar(draws, sz)['family']['outcome']
          .startswith('not reached'))


def t_families():
    check('efficiency: 0.8 boundary', DS.point_efficiency(80, 100) == 'X needs fewer points')
    check('efficiency: comparable', DS.point_efficiency(81, 100) == 'comparable')
    check('efficiency: one none', DS.point_efficiency(None, 100).startswith('Y qualifies'))
    check('gate: (b) flags first', DS.cross_family_gate([.2], False, 0.) == 'flagged')
    check('gate: (b) inf flags', DS.cross_family_gate([np.inf], True, 0.) == 'flagged')
    check('gate: (a) unavailable', DS.cross_family_gate([.01], False, 0.) == 'gate unavailable (rollout part)')
    check('gate: (a) flags', DS.cross_family_gate([.01], True, 3e-4) == 'flagged')
    check('gate: pass', DS.cross_family_gate([.01], True, 1e-4) == 'pass')
    check('gate: no state evidence -> unavailable', DS.cross_family_gate([], True, 0.) == 'gate unavailable (state part)')
    check('fixture: empty not admitted', not DS.fixture_admitted([]))
    ok = dict(finite=True, target_norm=1., rho_tgt=2.)
    check('fixture admitted', DS.fixture_admitted([ok] * 6))
    check('fixture: inf not admitted', not DS.fixture_admitted([ok] * 5 + [dict(finite=False, target_norm=float('nan'),
                                                                              rho_tgt=float('nan'))]))
    check('fixture: zero target', not DS.fixture_admitted([dict(finite=True, target_norm=0., rho_tgt=float('nan'))]))


def t_shapley():
    a = np.array([1., 2., 3.])                         # additive: f = a . corner
    f = {(i, j, k): float(a @ [i, j, k]) for i in (0, 1) for j in (0, 1) for k in (0, 1)}
    check('shapley additive exact', np.allclose(DS.shapley3(f), a))
    g = {k: v * 10 + 5 for k, v in f.items()}
    rep = DS.shapley_report(f, g, 1e-3, (1., 2.))
    check('shapley shares sum to 1', abs(sum(rep['acc_shares']) - 1) < 1e-12 and abs(sum(rep['cost_shares']) - 1) < 1e-12)
    rep = DS.shapley_report(f, g, 10., (-1., 2.))
    check('shapley shares withheld under thresholds', rep['acc_shares'] is None and rep['cost_shares'] is None)


def t_elasticity():
    rng = np.random.default_rng(0)
    r = rng.choice([128, 256, 384, 512], 40)
    M = r * rng.choice([1, 2, 4, 8], 40)
    m = rng.choice([4096, 16384, 65536], 40)
    c = 1e-6 * r ** .9 * M ** .7 * m ** .5
    e = DS.elasticity(r, M, m, c)
    check('elasticity recovers exponents', e['status'] == 'ok' and np.allclose(e['beta'][1:], [.9, .7, .5]))
    check('elasticity: too few cells', DS.elasticity(r[:5], M[:5], m[:5], c[:5])['status'].startswith('fit unavailable'))
    check('elasticity: collinear', DS.elasticity(r, 4 * r, m, c)['status'].startswith('fit unavailable'))


def t_h2q():
    ids = list(range(10))
    base = lambda cost, err: dict(cost=cost, median=err, worst=err, e={j: err for j in ids}, eta={j: 1e-5 for j in ids},
                                  screen_valid=True, q1=True)
    C = dict(C=base(1., .01), a=base(.5, .009), b=base(.4, .02), c=base(.51, .008))
    r = DS.h2q(C, 'C', {'a', 'b', 'c'})
    check('h2q: cheapest tie-or-beat, 3 % tie -> lower median', r['winner'] == 'c')
    check('h2q: pass', DS.h2q(C, 'C', {'a', 'c'}, dict(valid=True, median_ratio=.5))['verdict'] == 'pass')
    check('h2q: fail wording', DS.h2q(C, 'C', {'a', 'c'}, dict(valid=True, median_ratio=.7))['verdict'].startswith('fail (the selected'))
    check('h2q: panel invalid', DS.h2q(C, 'C', {'a'}, dict(valid=False, median_ratio=.5))['verdict'] == 'inconclusive (panel invalid)')
    check('h2q: none', DS.h2q(C, 'C', {'b'})['verdict'].startswith('no qualified'))
    C2 = dict(C, d=dict(base(.3, .005), eta={j: 2e-5 for j in ids}))
    check('h2q: differing eta refused', _raises(lambda: DS.h2q(C2, 'C', {'d'})))
    C['C']['q1'] = False
    check('h2q: comparator unavailable', DS.h2q(C, 'C', {'a'})['verdict'] == 'inconclusive (comparator unavailable)')


def t_certs():
    check('square cert boundary', bool(DC.certified_square(np.array([1e-7 * 2.]), 2.)[0]) and
          not bool(DC.certified_square(np.array([1.01e-7 * 2.]), 2.)[0]))
    check('square cert zero scale', bool(DC.certified_square(np.array([0.]), 0.)[0]) and
          not bool(DC.certified_square(np.array([1e-30]), 0.)[0]))
    rn = np.array([3e-10, 1e-12])
    tol = 1e-9 * 2.
    tr = rn / (tol + 1e-300)
    check('gcert exact inversion', DC.gcert(rn, tr, tol))
    check('gcert zero tol', DC.gcert(np.array([1e-10]), np.array([1e-10]) / 1e-300, 0.))
    check('gcert overflow fails', not DC.gcert(np.array([1e-10]), np.array([np.inf]), 0.))
    check('gcert mismatch fails', not DC.gcert(rn * 1.01, tr, tol))
    check('square tolerances', DC.tol_of('LSPG', 'tight', square=True) == (1e-7, 1e-13) and DC.tol_of('LSPG', 'tight') == (1e-7, 0.))


def t_rules():
    import rules2d as R2
    a, b = R2.points('fib4181')[0], R2.points('fib4181s0')[0]
    check('2D draw 0 = deployed', np.array_equal(a, b))
    check('2D draws differ', not np.array_equal(R2.points('fib4181s1')[0], a))
    check('2D same seed reproducible', np.array_equal(R2.points('sobol2048s3')[0], R2.points('sobol2048s3')[0]))
    check('2D mc needs a draw', _raises(lambda: R2.points('mc2048')))
    check('2D gauss takes no draw', _raises(lambda: R2.points('gauss64s1')))
    import rules3d_names as RN
    z = np.load(HERE / 'vendor' / 'quad3d' / 'rules' / 'rules.npz')
    e = np.load(HERE / 'rules3d' / 'rules_extra.npz')
    pin = {k[:-2]: (z[k], z[k[:-2] + '_w']) for k in z.files if k.endswith('_X')}
    pin.update({k[:-2]: (e[k], e[k[:-2] + '_w']) for k in e.files if k.endswith('_X')})
    check('3D lattice draw 0 regenerates the pinned rule', np.array_equal(RN.make('lat32768s0', pin)[0], pin['lat32768'][0]))
    check('3D draws differ', not np.array_equal(RN.make('lat8192s1', pin)[0], pin['lat8192'][0]))
    Xa, wa = RN.make('smolyak8', pin)
    Xb, wb = pin['smol8']
    ia, ib = np.lexsort(Xa.T[::-1]), np.lexsort(Xb.T[::-1])
    check('3D smol8 file = smolyak3(8) (points and weights)', Xa.shape == Xb.shape and np.allclose(Xa[ia], Xb[ib], atol=1e-14)
          and np.allclose(wa[ia], wb[ib], atol=1e-15))
    # stage (ii) generator tests (DESIGN 7)
    f = lambda X: np.prod(np.sin(np.pi * X), axis=1)
    exact = (2 / np.pi) ** 3
    errs = []
    for s in range(32):
        X, w = RN.make(f'mc65536s{s}', pin)
        errs.append(w @ f(X) - exact)
    se = f(RN.make('mc65536s0', pin)[0]).std() / np.sqrt(65536)
    check('MC error within 4 standard errors (32 seeds)', np.all(np.abs(errs) < 4 * se))
    X, w = RN.make('hal65536s0', pin)
    check('Halton3 error < MC error', abs(w @ f(X) - exact) < np.median(np.abs(errs)))
    g = lambda X: np.exp(-20 * np.sum((X - .5) ** 2, axis=1))          # smooth Gaussian bump
    from scipy.special import erf
    gx = np.sqrt(np.pi / 20) * erf(np.sqrt(20) * .5)
    eg = [RN.make(f'mc65536s{s}', pin) for s in range(32)]
    errg = [w_ @ g(X_) - gx ** 3 for X_, w_ in eg]
    seg = g(eg[0][0]).std() / np.sqrt(65536)
    check('MC bump error within 4 standard errors (32 seeds)', np.all(np.abs(errg) < 4 * seg))
    Xh, wh = RN.make('hal65536s0', pin)
    check('Halton3 bump error < MC median error', abs(wh @ g(Xh) - gx ** 3) < np.median(np.abs(errg)))
    e2 = []
    for s in range(32):
        X2, w2 = R2.points(f'mc65536s{s}')
        e2.append(w2 @ g(X2) - gx ** 2)
    se2 = g(R2.points('mc65536s0')[0]).std() / np.sqrt(65536)
    check('2D MC bump error within 4 standard errors (32 seeds)', np.all(np.abs(e2) < 4 * se2))
    check('2D MC seeds distinct', not np.array_equal(R2.points('mc2048s0')[0], R2.points('mc2048s1')[0]))
    check('Halton3/MC in (0,1)^3, weights sum 1', np.all((X > 0) & (X < 1)) and abs(w.sum() - 1) < 1e-12)


def t_panel():
    inv = [dict(B='b', ratio=.5, drift=1., hashes_ok=True)] * 4
    ok = DS.panel_accept(inv, {('b', 0): True, ('A', 0): True}, {'b': 4})
    check('panel: certified complete -> valid', ok['valid'])
    check('panel: uncertified subject -> invalid', not DS.panel_accept(inv, {('b', 0): None, ('A', 0): True}, {'b': 4})['valid'])
    check('panel: uncertified A -> invalid', not DS.panel_accept(inv, {('b', 0): True, ('A', 0): False}, {'b': 4})['valid'])
    check('panel: incomplete counts -> invalid', not DS.panel_accept(inv, {('b', 0): True}, {'b': 6})['valid'])
    check('panel: hash fault -> invalid', not DS.panel_accept(inv[:3] + [dict(inv[0], hashes_ok=False)], {('b', 0): True},
                                                             {'b': 4})['valid'])


def _raises(fn):
    try:
        fn()
    except (ValueError, AssertionError):
        return True
    return False


def main():
    for t in (t_transition, t_two_way, t_q2_mstar, t_families, t_shapley, t_elasticity, t_h2q, t_certs, t_rules, t_panel):
        t()
    (HERE / 'checks').mkdir(exist_ok=True)
    (HERE / 'checks' / 'test_dims.json').write_text(json.dumps(dict(results=RES, all_pass=all(RES.values())), indent=1) + '\n')
    print('ALL PASS' if all(RES.values()) else 'FAILURES: ' + ', '.join(k for k, v in RES.items() if not v))


if __name__ == '__main__':
    main()
