"""CPU tests of the jcp-p1-scaling statistics and synthetic controls (DESIGN 6, A1.9, A2.1). Writes checks/test_p1.json.

    python test_p1.py
Each control states its required outcome; the script exits non-zero if any control misbehaves.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import scipy
from scipy import stats

import p1stats as S

HERE = Path(__file__).resolve().parent
RESULTS = {}


def check(name, cond, **info):
    RESULTS[name] = dict(passed=bool(cond), **info)
    print(('PASS ' if cond else 'FAIL ') + name, {k: v for k, v in info.items() if not isinstance(v, (list, dict))})


def t_wilcoxon_vs_scipy():
    rng = np.random.default_rng(1)
    worst = 0.
    n_cmp = 0
    for n in (1, 3, 5, 6, 10, 16, 38, 60):
        for kind in ('cont', 'ties', 'zeros'):
            x = rng.normal(.3, 1., n)
            if kind == 'ties':
                x = np.round(x, 1)
            if kind == 'zeros':
                x[: max(1, n // 4)] = 0.
            for alt in ('greater', 'less', 'two-sided'):
                p, m, w, meth = S.wilcoxon(x, alt)
                if m == 0:
                    continue
                sm = 'exact' if meth == 'exact' else 'asymptotic'
                ref = stats.wilcoxon(x, zero_method='wilcox', correction=False, alternative=alt, method=sm).pvalue
                worst = max(worst, abs(p - ref))
                n_cmp += 1
    check('wilcoxon matches scipy', worst < 1e-9, worst_abs_diff=worst, comparisons=n_cmp, scipy=scipy.__version__)
    # A1.9 facts: n=5 all-zero d -> both one-sided tests reject at 2^-5; all-zero shifted sample -> no rejection
    t5 = S.tost(np.zeros(5))
    check('tost n=5 zeros rejects both sides (all |d+1| tied -> normal approximation)', t5['equivalent'], p=t5['p_lower'])
    t5b = S.tost(np.array([.1, -.2, .05, .3, -.15]))
    check('tost n=5 untied small d rejects both sides (exact 2^-5)', t5b['equivalent'] and abs(t5b['p_lower'] - 2 ** -5) < 1e-15,
          p=t5b['p_lower'])
    t1 = S.tost(np.ones(8))
    check('tost all d=1 upper side does not reject', (not t1['equivalent']) and t1['p_upper'] == 1., p_upper=t1['p_upper'])


def t_synthetic_timing_fault():
    rng = np.random.default_rng(0)
    t = 1.00 * (1 + .01 * rng.standard_normal((6, 3, 3)))
    ok = S.h_cost_n(t, np.ones_like(t, bool), [256, 1024, 4096])
    bad = t.copy()
    bad[:, 2, :] *= 2.
    fl = S.h_cost_n(bad, np.ones_like(t, bool), [256, 1024, 4096])
    check('synthetic timing fault -> fail', fl['outcome'] == 'fail', ci=fl.get('ci'))
    check('synthetic timing unmodified -> pass', ok['outcome'] == 'pass', ci=ok.get('ci'))
    # undefined resamples (A1.9): two cases fully invalid on one mesh
    v = np.ones_like(t, bool)
    v[:2, 1, :] = False
    u = S.h_cost_n(t, v, [256, 1024, 4096])
    check('H-cost-N undefined resamples handled', u['outcome'] in ('pass', 'inconclusive (insufficient valid timings)'),
          outcome=u['outcome'], undefined=u.get('undefined_fraction'))
    v2 = np.ones_like(t, bool)
    v2[:3, 0, :] = False
    check('H-cost-N > 1/3 invalid -> verdict not reached', S.h_cost_n(t, v2, [256, 1024, 4096])['outcome'] == 'verdict not reached')


def t_synthetic_accuracy_trend():
    rng = np.random.default_rng(0)
    n = 16
    e1 = .01 + 1e-4 * rng.standard_normal(n)
    e256 = .005 + 1e-4 * rng.standard_normal(n)
    e4096 = .015 + 1e-4 * rng.standard_normal(n)
    eps = np.zeros(n)
    sl = [np.polyfit(np.log2([256, 1024, 4096]), [a, b, c], 1)[0] for a, b, c in zip(e256, e1, e4096)]
    tr = S.h_acc_n({'P256': (e256, e1, eps), 'P4096': (e4096, e1, eps)}, sl)
    check('synthetic accuracy trend -> fail', tr['outcome'] == 'fail', outcome=tr['outcome'])
    f1 = .01 + 1e-4 * rng.standard_normal(n)
    f256 = .01 + 1e-4 * rng.standard_normal(n)
    f4096 = .01 + 1e-4 * rng.standard_normal(n)
    sl = [np.polyfit(np.log2([256, 1024, 4096]), [a, b, c], 1)[0] for a, b, c in zip(f256, f1, f4096)]
    eq = S.h_acc_n({'P256': (f256, f1, eps), 'P4096': (f4096, f1, eps)}, sl)
    check('synthetic accuracy equal -> pass', eq['outcome'] == 'pass', outcome=eq['outcome'])
    check('H-acc-N empty pair -> inconclusive',
          S.h_acc_n({'P256': ([], [], []), 'P4096': (f4096, f1, eps)}, sl)['outcome'] == 'inconclusive')
    check('H-acc-N zero margin floor finite', np.isfinite(S.margin([0.], [0.])).all())


def t_h_acc_r():
    n = 38
    eps = np.full(n, 1e-6)
    de = np.full(n, -1e-3)
    q = np.ones(n, bool)
    q10 = q.copy()
    q10[:10] = False
    check('H-acc-r 10/38 unqualified -> inconclusive', S.h_acc_r_transition(de, eps, q10, q)['outcome'] == 'inconclusive')
    check('H-acc-r all improve -> improves', S.h_acc_r_transition(de, eps, q, q)['outcome'] == 'improves')
    check('H-acc-r all worsen -> does not improve', S.h_acc_r_transition(-de, eps, q, q)['outcome'] == 'does not improve')
    check('H-acc-r empty -> inconclusive', S.h_acc_r_transition([], [], [], [])['outcome'] == 'inconclusive')
    check('H-acc-r ties -> inconclusive', S.h_acc_r_transition(np.zeros(n), eps, q, q)['outcome'] == 'inconclusive')
    check('range aggregation', S.h_acc_r_range(['improves', 'improves']) == 'monotone'
          and S.h_acc_r_range(['improves', 'does not improve']) == 'not monotone'
          and S.h_acc_r_range(['improves', 'inconclusive']) == 'inconclusive')


def t_h_cost_r():
    rng = np.random.default_rng(0)
    ranks = np.array([32, 64, 128, 256, 384, 512])
    a, b = 2e-5, 1e-2 / 512 ** 2
    T = (a + b * ranks ** 2)[None, None] * (1 + .01 * rng.standard_normal((6, 2, len(ranks))))
    f3 = S.h_cost_r(ranks, T, 3.)
    f2 = S.h_cost_r(ranks, T, 2.)
    check('H-cost-r synthetic r^2 declared 3 -> fail', f3['outcome'] == 'fail', ci=f3.get('ci'))
    check('H-cost-r synthetic r^2 declared 2 -> pass', f2['outcome'] == 'pass', ci=f2.get('ci'))
    flat = np.full((6, 2, len(ranks)), 3e-5) * (1 + .001 * rng.standard_normal((6, 2, len(ranks))))
    check('H-cost-r flat cost -> inconclusive', S.h_cost_r(ranks, flat, 1., nboot=200)['outcome'] == 'inconclusive')
    # operation-count exponents (A1.9 literal AM-4 label) for the registered fixed panel
    import p1comp as PC
    m, M = 9216, 1536
    pf = {c: S.fit_power(ranks, [PC.flops_bytes(c, m, M, r)[0] for r in ranks], measured=False)['p'] for c in ('N', 'J', 'JTJ', 'chol')}
    pb = {c: S.fit_power(ranks, [PC.flops_bytes(c, m, M, r)[1] for r in ranks], measured=False)['p'] for c in ('N', 'J', 'JTJ', 'chol')}
    RESULTS['operation_model_exponents_fixed_panel'] = dict(flops=pf, bytes=pb)
    print('op-model exponents', pf, pb)


def t_q3_rank_deficient():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((1536, 384))
    A[:, 7] = A[:, 3]
    c = S.conditioning(np.linalg.svd(A, compute_uv=False))
    check('Q3 duplicated column -> rank deficit and cond > 1e12', c['deficient'] and c['cond'] > 1e12, condition=c['cond'], rank=c['rank'])


if __name__ == '__main__':
    for f in (t_wilcoxon_vs_scipy, t_synthetic_timing_fault, t_synthetic_accuracy_trend, t_h_acc_r, t_h_cost_r, t_q3_rank_deficient):
        f()
    (HERE / 'checks').mkdir(exist_ok=True)
    (HERE / 'checks' / 'test_p1.json').write_text(json.dumps(RESULTS, indent=1, default=float))
    bad = [k for k, v in RESULTS.items() if isinstance(v, dict) and v.get('passed') is False]
    print('ALL PASSED' if not bad else f'FAILED: {bad}')
    sys.exit(1 if bad else 0)
